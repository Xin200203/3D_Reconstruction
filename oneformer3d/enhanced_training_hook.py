"""
Enhanced Training Monitoring Hook for BiFusion
增强的训练监控Hook，输出详细损失、fusion gate统计、投影有效率等
"""

import torch
import torch.nn.functional as F
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
from mmdet3d.registry import HOOKS as MMDET3D_HOOKS
import numpy as np
from typing import Dict, Any, Optional


@HOOKS.register_module()
@MMDET3D_HOOKS.register_module()
class EnhancedTrainingHook(Hook):
    """Enhanced training monitoring hook for BiFusion system.
    
    监控内容：
    1. 详细损失数值分解
    2. Fusion gate统计 (2D/3D权重分配)
    3. 投影有效率 (valid projection rate)
    4. 梯度健康度监控
    5. 特征质量指标
    """
    
    def __init__(self, 
                 log_interval: int = 10,
                 grad_monitor_interval: int = 50,
                 detailed_stats: bool = True):
        super().__init__()
        self.log_interval = log_interval
        self.grad_monitor_interval = grad_monitor_interval
        self.detailed_stats = detailed_stats
        self.iter_count = 0
        
    def before_train_iter(self, runner, batch_idx: int, data_batch: Any = None) -> None:
        """训练迭代前的准备工作"""
        self.iter_count += 1
        
    def after_train_iter(self, runner, batch_idx: int, data_batch: Any = None, 
                        outputs: Optional[dict] = None) -> None:
        """训练迭代后的监控和日志输出"""
        
        if self.iter_count % self.log_interval != 0:
            return
            
        # 获取模型和损失信息
        model = runner.model
        if hasattr(runner, 'log_buffer') and runner.log_buffer.output:
            loss_info = runner.log_buffer.output
        else:
            return
            
        # 1. 详细损失分解
        detailed_losses = self._extract_detailed_losses(loss_info)
        
        # 2. BiFusion统计信息
        fusion_stats = self._extract_fusion_stats(model, outputs)
        
        # 3. 投影有效率统计
        projection_stats = self._extract_projection_stats(model, outputs)
        
        # 4. 梯度健康度监控
        if self.iter_count % self.grad_monitor_interval == 0:
            grad_stats = self._extract_gradient_health(model)
        else:
            grad_stats = {}
            
        # 5. 输出增强日志
        self._log_enhanced_stats(runner, detailed_losses, fusion_stats, 
                               projection_stats, grad_stats)
    
    def _extract_detailed_losses(self, loss_info: Dict) -> Dict[str, float]:
        """提取详细损失数值"""
        detailed_losses = {}
        
        # 基础损失
        if 'loss' in loss_info:
            detailed_losses['total_loss'] = float(loss_info['loss'])
        
        # 语义损失
        for key in ['semantic_loss', 'sem_loss']:
            if key in loss_info:
                detailed_losses['semantic_loss'] = float(loss_info[key])
                break
                
        # 实例损失组件
        instance_keys = {
            'inst_cls_loss': ['inst_cls_loss', 'classification_loss'],
            'inst_bce_loss': ['inst_bce_loss', 'bce_loss', 'mask_bce_loss'],
            'inst_dice_loss': ['inst_dice_loss', 'dice_loss', 'mask_dice_loss'],
            'inst_score_loss': ['inst_score_loss', 'score_loss'],
            'inst_bbox_loss': ['inst_bbox_loss', 'bbox_loss']
        }
        
        for target_key, possible_keys in instance_keys.items():
            for key in possible_keys:
                if key in loss_info:
                    detailed_losses[target_key] = float(loss_info[key])
                    break
        
        # CLIP和辅助损失
        aux_keys = {
            'clip_cons_loss': ['clip_cons_loss', 'clip_loss', 'clip_consistency_loss'],
            'spatial_cons_loss': ['spatial_cons_loss', 'spatial_consistency_loss'],
            'no_view_loss': ['no_view_loss', 'no_view_supervision_loss']
        }
        
        for target_key, possible_keys in aux_keys.items():
            for key in possible_keys:
                if key in loss_info:
                    detailed_losses[target_key] = float(loss_info[key])
                    break
        
        return detailed_losses
    
    def _extract_fusion_stats(self, model, outputs) -> Dict[str, float]:
        """提取BiFusion门控统计信息"""
        fusion_stats = {}
        
        try:
            # 尝试从模型中获取BiFusion模块
            if hasattr(model, 'module'):  # DDP包装
                model = model.module
                
            bi_encoder = None
            if hasattr(model, 'bi_encoder'):
                bi_encoder = model.bi_encoder
            elif hasattr(model, 'backbone') and hasattr(model.backbone, 'bi_encoder'):
                bi_encoder = model.backbone.bi_encoder
                
            if bi_encoder is not None:
                # 🔥 方法1: 直接获取BiFusionEncoder的统计信息
                if hasattr(bi_encoder, 'get_fusion_statistics'):
                    fusion_stats.update(bi_encoder.get_fusion_statistics())
                
                # 🔥 方法2: 从fusion gate模块获取最新统计
                if hasattr(bi_encoder, 'fusion_gate'):
                    gate_module = bi_encoder.fusion_gate
                    if hasattr(gate_module, '_stats_buffer') and gate_module._stats_buffer:
                        latest_stats = gate_module._stats_buffer[-1]
                        fusion_stats.update(latest_stats)
                
                # 🔥 方法3: 查找具体的gate权重（LiteFusionGate或EnhancedFusionGate）
                for name, module in bi_encoder.named_modules():
                    if 'fusion_gate' in name or isinstance(module, (type(bi_encoder.fusion_gate) if hasattr(bi_encoder, 'fusion_gate') else type(None))):
                        # 检查是否有alpha权重（LiteFusionGate的point_mlp输出）
                        if hasattr(module, '_last_alpha'):
                            alpha = module._last_alpha
                            if isinstance(alpha, torch.Tensor):
                                alpha_np = alpha.detach().cpu().numpy()
                                fusion_stats['gate_2d_ratio'] = float(np.mean(alpha_np))
                                fusion_stats['gate_3d_ratio'] = float(np.mean(1 - alpha_np))
                                fusion_stats['gate_std'] = float(np.std(alpha_np))
                        break
                        
                # 🔥 方法4: 从最近的forward输出中获取
                if hasattr(bi_encoder, '_last_forward_stats'):
                    fusion_stats.update(bi_encoder._last_forward_stats)
                    
        except Exception as e:
            # 如果无法获取fusion统计，记录但不中断
            fusion_stats['fusion_error'] = str(e)[:50]
            
        return fusion_stats
    
    def _extract_projection_stats(self, model, outputs) -> Dict[str, float]:
        """提取投影有效率统计"""
        projection_stats = {}
        
        try:
            # 尝试从模型中获取BiFusion模块
            if hasattr(model, 'module'):  # DDP包装
                model = model.module
                
            bi_encoder = None
            if hasattr(model, 'bi_encoder'):
                bi_encoder = model.bi_encoder
            elif hasattr(model, 'backbone') and hasattr(model.backbone, 'bi_encoder'):
                bi_encoder = model.backbone.bi_encoder
            
            # 🔥 方法1: 从BiFusionEncoder的融合统计中获取
            if bi_encoder is not None and hasattr(bi_encoder, '_fusion_stats'):
                fusion_stats = bi_encoder._fusion_stats
                if 'valid_points_ratio' in fusion_stats:
                    projection_stats['valid_projection_rate'] = fusion_stats['valid_points_ratio']
                if 'total_points' in fusion_stats:
                    projection_stats['total_points'] = fusion_stats['total_points']
                    projection_stats['valid_points'] = int(fusion_stats['total_points'] * fusion_stats.get('valid_points_ratio', 1.0))
                    projection_stats['invalid_points'] = fusion_stats['total_points'] - projection_stats['valid_points']
            
            # 🔥 方法2: 从outputs中获取投影相关信息
            if outputs and isinstance(outputs, dict):
                # 查找valid mask相关信息 - 优先查找BiFusion输出
                for key in ['valid_projection_mask', 'valid_mask', 'projection_mask', 'valid_projection', 'conf_2d']:
                    if key in outputs:
                        mask_data = outputs[key]
                        
                        # 处理列表格式（BiFusion返回格式）
                        if isinstance(mask_data, (list, tuple)):
                            if len(mask_data) > 0:
                                # 使用第一个样本的统计（假设batch内统计相似）
                                mask = mask_data[0]
                                if isinstance(mask, torch.Tensor):
                                    if key == 'conf_2d':  # 置信度需要转换为有效掩码
                                        mask = mask.squeeze(-1) > 0.1 if mask.dim() > 1 else mask > 0.1
                                    
                                    valid_rate = float(mask.float().mean())
                                    projection_stats['valid_projection_rate'] = valid_rate
                                    projection_stats['total_points'] = int(mask.numel())
                                    projection_stats['valid_points'] = int(mask.sum())
                                    projection_stats['invalid_points'] = projection_stats['total_points'] - projection_stats['valid_points']
                                    
                                    # 如果是列表，计算所有样本的平均统计
                                    if len(mask_data) > 1:
                                        total_points = 0
                                        total_valid = 0
                                        for m in mask_data:
                                            if isinstance(m, torch.Tensor):
                                                if key == 'conf_2d':
                                                    m = m.squeeze(-1) > 0.1 if m.dim() > 1 else m > 0.1
                                                total_points += m.numel()
                                                total_valid += int(m.sum())
                                        
                                        if total_points > 0:
                                            projection_stats['valid_projection_rate'] = total_valid / total_points
                                            projection_stats['total_points'] = total_points
                                            projection_stats['valid_points'] = total_valid
                                            projection_stats['invalid_points'] = total_points - total_valid
                                            projection_stats['batch_size'] = len(mask_data)
                                    break
                        
                        # 处理单个Tensor格式
                        elif isinstance(mask_data, torch.Tensor):
                            if key == 'conf_2d':  # 置信度需要转换为有效掩码
                                mask_data = mask_data.squeeze(-1) > 0.1 if mask_data.dim() > 1 else mask_data > 0.1
                            
                            valid_rate = float(mask_data.float().mean())
                            projection_stats['valid_projection_rate'] = valid_rate
                            projection_stats['total_points'] = int(mask_data.numel())
                            projection_stats['valid_points'] = int(mask_data.sum())
                            projection_stats['invalid_points'] = projection_stats['total_points'] - projection_stats['valid_points']
                            break
            
            # 🔥 方法3: 尝试从模型中获取最近的投影统计
            if hasattr(model, '_last_projection_stats'):
                stats = model._last_projection_stats
                if isinstance(stats, dict):
                    projection_stats.update(stats)
            
            # 🔥 方法4: 从bi_encoder获取最近forward的投影统计
            if bi_encoder is not None:
                if hasattr(bi_encoder, '_last_valid_mask'):
                    mask = bi_encoder._last_valid_mask
                    if isinstance(mask, torch.Tensor):
                        valid_rate = float(mask.float().mean())
                        projection_stats['valid_projection_rate'] = valid_rate
                        projection_stats['total_points'] = int(mask.numel())
                        projection_stats['valid_points'] = int(mask.sum())
                        projection_stats['invalid_points'] = projection_stats['total_points'] - projection_stats['valid_points']
                        
        except Exception as e:
            projection_stats['projection_error'] = str(e)[:50]
            
        return projection_stats
    
    def _extract_gradient_health(self, model) -> Dict[str, float]:
        """提取梯度健康统计"""
        gradient_stats = {}
        
        try:
            # 获取真实模型（处理DDP包装）
            real_model = model.module if hasattr(model, 'module') else model
            
            # 🔥 核心组件梯度监控
            component_grads = {}
            
            # 监控CLIP相关梯度（如果存在）
            clip_grad_norm = 0.0
            clip_param_count = 0
            for name, param in real_model.named_parameters():
                if param.grad is not None:
                    if 'clip' in name.lower() or 'text_encoder' in name.lower() or 'vision_encoder' in name.lower():
                        clip_grad_norm += param.grad.data.norm(2).item() ** 2
                        clip_param_count += 1
            
            if clip_param_count > 0:
                component_grads['clip_grad_norm'] = (clip_grad_norm ** 0.5)
                component_grads['clip_param_count'] = clip_param_count
                gradient_stats['clip_grad_norm'] = component_grads['clip_grad_norm']
            
            # 监控BiFusion相关梯度
            bifusion_grad_norm = 0.0
            bifusion_param_count = 0
            for name, param in real_model.named_parameters():
                if param.grad is not None:
                    if any(keyword in name.lower() for keyword in ['bi_fusion', 'bifusion', 'bi_encoder', 'fusion_gate', 'lite_fusion']):
                        bifusion_grad_norm += param.grad.data.norm(2).item() ** 2
                        bifusion_param_count += 1
            
            if bifusion_param_count > 0:
                component_grads['bifusion_grad_norm'] = (bifusion_grad_norm ** 0.5)
                component_grads['bifusion_param_count'] = bifusion_param_count
                gradient_stats['bifusion_grad_norm'] = component_grads['bifusion_grad_norm']
            
            # 监控backbone相关梯度
            backbone_grad_norm = 0.0
            backbone_param_count = 0
            for name, param in real_model.named_parameters():
                if param.grad is not None:
                    if any(keyword in name.lower() for keyword in ['backbone', 'encoder', 'decoder']) and not any(exc in name.lower() for exc in ['clip', 'bi_fusion', 'bifusion']):
                        backbone_grad_norm += param.grad.data.norm(2).item() ** 2
                        backbone_param_count += 1
            
            if backbone_param_count > 0:
                component_grads['backbone_grad_norm'] = (backbone_grad_norm ** 0.5)
                component_grads['backbone_param_count'] = backbone_param_count
                gradient_stats['backbone_grad_norm'] = component_grads['backbone_grad_norm']
            
            # 🔥 全局梯度统计
            total_grad_norm = 0.0
            total_param_count = 0
            max_grad_norm = 0.0
            min_grad_norm = float('inf')
            grad_norms = []
            
            for name, param in real_model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.data.norm(2).item()
                    total_grad_norm += grad_norm ** 2
                    total_param_count += 1
                    grad_norms.append(grad_norm)
                    max_grad_norm = max(max_grad_norm, grad_norm)
                    min_grad_norm = min(min_grad_norm, grad_norm)
            
            if total_param_count > 0:
                gradient_stats['total_grad_norm'] = (total_grad_norm ** 0.5)
                gradient_stats['max_grad_norm'] = max_grad_norm
                gradient_stats['min_grad_norm'] = min_grad_norm if min_grad_norm != float('inf') else 0.0
                gradient_stats['param_with_grad_count'] = total_param_count
                
                # 计算梯度分布统计
                if grad_norms:
                    import numpy as np
                    grad_array = np.array(grad_norms)
                    gradient_stats['grad_mean'] = float(np.mean(grad_array))
                    gradient_stats['grad_std'] = float(np.std(grad_array))
                    gradient_stats['grad_median'] = float(np.median(grad_array))
            
            # 🔥 梯度健康警报
            gradient_stats['gradient_health'] = 'healthy'
            
            # 检测NaN或Inf梯度
            has_nan_grad = False
            has_inf_grad = False
            for name, param in real_model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        has_nan_grad = True
                    if torch.isinf(param.grad).any():
                        has_inf_grad = True
            
            if has_nan_grad:
                gradient_stats['gradient_health'] = 'nan_detected'
            elif has_inf_grad:
                gradient_stats['gradient_health'] = 'inf_detected'
            elif gradient_stats.get('total_grad_norm', 0) > 100.0:
                gradient_stats['gradient_health'] = 'exploding'
            elif gradient_stats.get('total_grad_norm', 0) < 1e-8:
                gradient_stats['gradient_health'] = 'vanishing'
            
            # 🔥 组件梯度比例分析
            if gradient_stats.get('total_grad_norm', 0) > 0:
                if 'clip_grad_norm' in gradient_stats:
                    gradient_stats['clip_grad_ratio'] = gradient_stats['clip_grad_norm'] / gradient_stats['total_grad_norm']
                if 'bifusion_grad_norm' in gradient_stats:
                    gradient_stats['bifusion_grad_ratio'] = gradient_stats['bifusion_grad_norm'] / gradient_stats['total_grad_norm']
                if 'backbone_grad_norm' in gradient_stats:
                    gradient_stats['backbone_grad_ratio'] = gradient_stats['backbone_grad_norm'] / gradient_stats['total_grad_norm']
                    
        except Exception as e:
            gradient_stats['gradient_error'] = str(e)[:50]
            
        return gradient_stats
    
    def _log_enhanced_stats(self, runner, detailed_losses: Dict, fusion_stats: Dict,
                          projection_stats: Dict, grad_stats: Dict) -> None:
        """输出增强的训练统计日志"""
        
        # 构建日志消息
        log_msg = f"\n{'='*80}\n"
        log_msg += f"📊 Enhanced Training Stats - Iter {self.iter_count}\n"
        log_msg += f"{'='*80}\n"

        # Optimizer / LR 信息
        lr_stats = {}
        optim_wrapper = getattr(runner, 'optim_wrapper', None)
        if optim_wrapper is not None and hasattr(optim_wrapper, 'optimizer'):
            optimizer = optim_wrapper.optimizer
            for idx, group in enumerate(optimizer.param_groups):
                lr_stats[f'group{idx}'] = float(group.get('lr', 0.0))
        if lr_stats:
            log_msg += "⚙️  Optimizer / LR:\n"
            for name, value in lr_stats.items():
                log_msg += f"  lr/{name:<16}: {value:.6e}\n"
            log_msg += "\n"
        
        # 1. 详细损失信息
        if detailed_losses:
            log_msg += "🔥 Detailed Loss Breakdown:\n"
            for loss_name, loss_value in detailed_losses.items():
                log_msg += f"  {loss_name:<20}: {loss_value:.6f}\n"
            log_msg += "\n"
        
        # 2. BiFusion统计
        if fusion_stats:
            log_msg += "🔀 BiFusion Gate Statistics:\n"
            for stat_name, stat_value in fusion_stats.items():
                if isinstance(stat_value, (int, float)):
                    log_msg += f"  {stat_name:<20}: {stat_value:.4f}\n"
                else:
                    log_msg += f"  {stat_name:<20}: {stat_value}\n"
            log_msg += "\n"
        
        # 3. 投影统计
        if projection_stats:
            log_msg += "📡 Projection Statistics:\n"
            for stat_name, stat_value in projection_stats.items():
                if isinstance(stat_value, (int, float)):
                    if 'rate' in stat_name:
                        log_msg += f"  {stat_name:<20}: {stat_value:.3%}\n"
                    else:
                        log_msg += f"  {stat_name:<20}: {stat_value}\n"
                else:
                    log_msg += f"  {stat_name:<20}: {stat_value}\n"
            log_msg += "\n"
        
        # 4. 梯度健康度 (定期输出)
        if grad_stats:
            log_msg += "📈 Gradient Health Monitor:\n"
            for stat_name, stat_value in grad_stats.items():
                if isinstance(stat_value, (int, float)):
                    if 'norm' in stat_name:
                        log_msg += f"  {stat_name:<20}: {stat_value:.6f}\n"
                    else:
                        log_msg += f"  {stat_name:<20}: {stat_value}\n"
                else:
                    log_msg += f"  {stat_name:<20}: {stat_value}\n"
            log_msg += "\n"
        
        log_msg += f"{'='*80}\n"
        
        # 输出到logger
        runner.logger.info(log_msg)
        
        # 同时记录到tensorboard (如果有)
        if hasattr(runner, 'visualizer') and runner.visualizer is not None:
            for lr_name, lr_val in lr_stats.items():
                runner.visualizer.add_scalar(f'train/lr/{lr_name}', lr_val, self.iter_count)
            for loss_name, loss_value in detailed_losses.items():
                if isinstance(loss_value, (int, float)):
                    runner.visualizer.add_scalar(f'train/detailed_loss/{loss_name}', 
                                                loss_value, self.iter_count)
            
            for stat_name, stat_value in fusion_stats.items():
                if isinstance(stat_value, (int, float)):
                    runner.visualizer.add_scalar(f'train/fusion/{stat_name}', 
                                                stat_value, self.iter_count)
            
            for stat_name, stat_value in projection_stats.items():
                if isinstance(stat_value, (int, float)):
                    runner.visualizer.add_scalar(f'train/projection/{stat_name}', 
                                                stat_value, self.iter_count)
                                                
            for stat_name, stat_value in grad_stats.items():
                if isinstance(stat_value, (int, float)):
                    runner.visualizer.add_scalar(f'train/gradient/{stat_name}', 
                                                stat_value, self.iter_count)
