"""
自定义Hook：详细损失和统计信息监控
用于BiFusion训练的全面日志输出
"""

import torch
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from typing import Optional, Dict, Any


@HOOKS.register_module()
class DetailedLossMonitorHook(Hook):
    """详细损失和融合统计监控Hook
    
    功能：
    1. 输出所有损失组件的详细数值
    2. 监控融合门控的统计信息
    3. 记录梯度范数和数值稳定性指标
    4. 输出CLIP特征的统计信息
    """
    
    def __init__(self, 
                 log_interval: int = 10,
                 collect_grad_norm: bool = True,
                 collect_fusion_stats: bool = True,
                 collect_clip_stats: bool = True):
        self.log_interval = log_interval
        self.collect_grad_norm = collect_grad_norm
        self.collect_fusion_stats = collect_fusion_stats
        self.collect_clip_stats = collect_clip_stats
        
    def after_train_iter(self, 
                        runner, 
                        batch_idx: int, 
                        data_batch = None, 
                        outputs = None) -> None:
        """训练迭代后的详细日志输出"""
        
        if batch_idx % self.log_interval != 0:
            return
            
        # 获取当前损失
        if outputs and 'log_vars' in outputs:
            log_vars = outputs['log_vars']
            
            # 构建详细损失报告（包含融合统计）
            loss_report = self._build_loss_report(log_vars)
            
            # 获取梯度统计
            grad_stats = self._get_gradient_stats(runner.model) if self.collect_grad_norm else {}
            
            # 获取CLIP统计
            clip_stats = self._get_clip_stats(runner.model) if self.collect_clip_stats else {}
            
            # 输出完整报告（不再需要单独的fusion_stats，已在loss_report中）
            self._print_detailed_report(batch_idx, loss_report, {}, grad_stats, clip_stats)
    
    def _build_loss_report(self, log_vars: Dict) -> Dict[str, float]:
        """构建损失函数详细报告"""
        loss_report = {}
        
        # 主要损失
        if 'loss' in log_vars:
            loss_report['total_loss'] = log_vars['loss']
            
        # 语义损失
        for key in log_vars:
            if 'sem_loss' in key or 'semantic' in key:
                loss_report[f'semantic_{key}'] = log_vars[key]
                
        # 实例损失
        for key in log_vars:
            if any(x in key for x in ['inst_loss', 'cls_loss', 'mask_bce_loss', 'mask_dice_loss', 'score_loss']):
                loss_report[f'instance_{key}'] = log_vars[key]
                
        # CLIP损失
        for key in log_vars:
            if 'clip' in key.lower():
                loss_report[f'clip_{key}'] = log_vars[key]
                
        # 融合统计（直接从log_vars中提取）
        fusion_keys = [
            'avg_confidence',
            'valid_ratio',
            'norm_ratio_2d_over_3d',
            'cos_2d3d_mean',
            'cos_2d3d_mean_ln',
            'feat3d_mean_abs',
            'feat3d_std',
            'feat3d_nonzero_ratio',
            'feat2d_mean_abs',
            'feat2d_std',
            'feat2d_nonzero_ratio',
            'fused_mean_abs',
            'fused_std',
            'grad_norm_feat3d',
            'grad_norm_feat2d',
            'grad_norm_fusion',
            'grad_norm_feat3d_raw',
            'grad_norm_feat2d_raw',
            'grad_norm_fusion_raw',
            'grad_params_feat2d',
            'grad_params_feat3d',
            'grad_params_fusion',
            'grad_params_decoder',
            'grad_ratio_2d_over_3d'
        ]
        for key in fusion_keys:
            if key in log_vars:
                loss_report[key] = log_vars[key]
                
        return loss_report
    
    def _get_gradient_stats(self, model) -> Dict[str, float]:
        """获取梯度统计信息"""
        grad_stats = {}
        
        try:
            total_norm = 0.0
            param_count = 0
            max_grad = 0.0
            
            # 分模块统计梯度
            for name, module in model.named_modules():
                if any(x in name for x in ['bi_encoder', 'decoder', 'criterion']):
                    module_norm = 0.0
                    module_params = 0
                    
                    for param in module.parameters():
                        if param.grad is not None:
                            param_norm = param.grad.data.norm(2)
                            module_norm += param_norm.item() ** 2
                            module_params += 1
                            max_grad = max(max_grad, param.grad.data.abs().max().item())
                    
                    if module_params > 0:
                        grad_stats[f'grad_norm_{name.split(".")[0]}'] = (module_norm ** 0.5)
                        
            # 总体梯度统计
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
                    param_count += 1
                    
            if param_count > 0:
                grad_stats['grad_norm_total'] = (total_norm ** 0.5)
                grad_stats['grad_max'] = max_grad
                grad_stats['grad_param_count'] = param_count
                
        except Exception as e:
            grad_stats['grad_error'] = str(e)[:50]
            
        return grad_stats
    
    def _get_clip_stats(self, model) -> Dict[str, float]:
        """获取CLIP特征统计信息"""
        clip_stats = {}
        
        try:
            if hasattr(model, 'bi_encoder') and hasattr(model.bi_encoder, 'enhanced_2d_encoder'):
                encoder_2d = model.bi_encoder.enhanced_2d_encoder
                
                # CLIP模型参数统计
                if hasattr(encoder_2d, 'clip_visual'):
                    clip_model = encoder_2d.clip_visual
                    trainable_params = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
                    total_params = sum(p.numel() for p in clip_model.parameters())
                    
                    clip_stats['clip_trainable_ratio'] = trainable_params / max(total_params, 1)
                    clip_stats['clip_total_params'] = total_params
                    clip_stats['clip_trainable_params'] = trainable_params
                
        except Exception as e:
            clip_stats['clip_error'] = str(e)[:50]
            
        return clip_stats
    
    def _print_detailed_report(self, batch_idx: int, loss_report: Dict, fusion_stats: Dict, 
                             grad_stats: Dict, clip_stats: Dict):
        """打印详细报告 - fusion_stats已不再使用，因为融合统计现在在loss_report中"""
        
        print(f"\n{'='*80}")
        print(f"📊 详细训练报告 - Iteration {batch_idx}")
        print(f"{'='*80}")
        
        # 损失函数报告  
        if loss_report:
            print(f"🔥 损失函数:")
            # 先显示主要损失
            main_losses = {}
            fusion_losses = {}
            other_losses = {}
            
            fusion_prefixes = [
                'avg_confidence',
                'valid_ratio',
                'norm_ratio_2d_over_3d',
                'cos_2d3d',
                'feat3d_',
                'feat2d_',
                'fused_',
                'grad_norm_feat'
            ]

            for key, value in loss_report.items():
                if not isinstance(value, (int, float)):
                    continue

                if any(key.startswith(prefix) for prefix in fusion_prefixes):
                    fusion_losses[key] = value
                elif any(x in key for x in ['loss', 'semantic', 'instance', 'clip']):
                    main_losses[key] = value
                else:
                    other_losses[key] = value
            
            # 显示主要损失
            for key, value in main_losses.items():
                print(f"  {key:<25}: {value:.6f}")
                
            # 显示融合与分支统计
            if fusion_losses:
                print(f"\n🔀 融合与分支统计:")
                for key, value in fusion_losses.items():
                    print(f"  📊 {key:<30}: {value:.6f}")
                        
            # 显示其他统计
            if other_losses:
                print(f"\n📈 其他统计:")
                for key, value in other_losses.items():
                    print(f"  {key:<25}: {value:.6f}")
                    
        # 梯度统计报告
        if grad_stats:
            print(f"\n📈 梯度统计:")
            for key, value in grad_stats.items():
                if isinstance(value, (int, float)):
                    print(f"  {key:<25}: {value:.6f}")
                    
        # CLIP统计报告
        if clip_stats:
            print(f"\n🎨 CLIP统计:")
            for key, value in clip_stats.items():
                if isinstance(value, (int, float)):
                    print(f"  {key:<25}: {value:.6f}")
                    
        print(f"{'='*80}\n")


@HOOKS.register_module()  
class NaNDetectionHook(Hook):
    """NaN检测Hook - 及时发现数值不稳定"""
    
    def __init__(self, check_interval: int = 1):
        self.check_interval = check_interval
        
    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None):
        """检测NaN/Inf"""
        if batch_idx % self.check_interval != 0:
            return
            
        # 检查损失
        if outputs and 'log_vars' in outputs:
            for key, value in outputs['log_vars'].items():
                if isinstance(value, (int, float)):
                    if torch.isnan(torch.tensor(value)) or torch.isinf(torch.tensor(value)):
                        print(f"🚨 NaN/Inf检测到在 {key}: {value}")
                        
        # 检查模型参数
        nan_params = []
        inf_params = []
        for name, param in runner.model.named_parameters():
            if torch.isnan(param.data).any():
                nan_params.append(name)
            if torch.isinf(param.data).any():
                inf_params.append(name)
                
        if nan_params:
            print(f"🚨 NaN参数检测: {nan_params[:5]}")  # 只显示前5个
        if inf_params:
            print(f"🚨 Inf参数检测: {inf_params[:5]}")  # 只显示前5个
