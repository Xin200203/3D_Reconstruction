import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import contextlib
from typing import List, Dict, Optional, Tuple, Union, cast

import MinkowskiEngine as ME
from mmdet3d.registry import MODELS
from .mink_unet import Res16UNet34C
from types import SimpleNamespace


class EnhancedProjectionHead3D(nn.Module):
    """简化的3D投影头：96维 -> 256维
    
    按照优化指南要求：Linear(96→256) + LayerNorm
    """
    
    def __init__(self,
                 input_dim: int = 96,
                 output_dim: int = 256):
        super().__init__()
        
        # 简化投影：单层Linear + LayerNorm
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),        # 融合特征
            nn.LayerNorm(output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """3D特征投影 (N, 96) -> (N, 256)"""
        # 数值稳定性检查
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print("Warning: NaN/Inf in 3D projection input, clamping")
            x = torch.clamp(x, -10, 10)
        return self.projection(x)


class MaskedSE1D(nn.Module):
    """掩码化SE模块 - 只统计有效点的通道均值"""
    def __init__(self, C, r=16):
        super().__init__()
        self.excite = nn.Sequential(
            nn.Conv1d(C, C//r, 1), nn.ReLU(),
            nn.Conv1d(C//r, C, 1), nn.Sigmoid()
        )
    
    def forward(self, x, valid_mask): 
        # x: (B, C, N), valid_mask: (B, N)
        m = valid_mask.unsqueeze(1).float()             # (B,1,N)
        s = (x * m).sum(-1, keepdim=True)               # (B,C,1)  有效点的通道求和
        cnt = m.sum(-1, keepdim=True).clamp_min(1.0)    # (B,1,1)  有效点计数
        z = s / cnt                                     # (B,C,1)  掩码化均值
        w = self.excite(z)                              # (B,C,1)
        return x * w                                    # 通道重加权


class Head(nn.Module):
    """统一的Head结构 - 2D/3D分支对称使用"""
    def __init__(self, dim=256, hidden=256, p=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden, dim),
            nn.LayerNorm(dim)
        )
    def forward(self, x):
        return self.net(x)


class LiteFusionGate(nn.Module):
    """Lite Fusion Gate - 轻量级融合门控机制
    
    简化版本：点级融合 + 掩码化SE通道注意力，移除分阶段训练逻辑
    参数量约0.12M，远低于原EnhancedGate
    """
    
    def __init__(self, 
                 feat_dim: int = 256,
                 use_masked_se: bool = True):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.use_masked_se = use_masked_se
        
        # 点级融合权重MLP: 添加LayerNorm确保特征稳定性
        self.point_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),  # 256*2 -> 64
            nn.LayerNorm(64),  # 添加归一化层
            nn.ReLU(),
            nn.Dropout(0.1),   # 添加少量dropout防止过拟合
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # 🔧 调整初始化：bias设为正值，鼓励更多使用2D特征
        nn.init.constant_(self.point_mlp[-2].bias, 1.0)  # 初始偏向2D特征
        
        # 🔧 同时调整权重初始化，使用较小的权重避免梯度消失  
        # 只初始化Linear层的权重
        for module in self.point_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
        
        # 掩码化SE通道注意力模块
        if use_masked_se:
            self.se_masked = MaskedSE1D(feat_dim, r=16)
        else:
            # 原版SE模块（备用）
            self.se_module = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(feat_dim, feat_dim // 16, 1),
                nn.ReLU(),
                nn.Conv1d(feat_dim // 16, feat_dim, 1),
                nn.Sigmoid()
            )
        
    def forward(self, 
                f2d: torch.Tensor, 
                f3d: torch.Tensor, 
                valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f2d: 2D特征 (B, N, 256)
            f3d: 3D特征 (B, N, 256) 
            valid_mask: 有效投影掩码 (B, N)
        Returns:
            fused_feat: 融合特征 (B, N, 256)
            confidence: 融合置信度 (B, N, 1)
        """
        B, N, C = f2d.shape
        
        # 1. 特征标准化：确保2D和3D特征在相同数值范围
        f2d_norm = F.normalize(f2d, dim=-1, p=2)  # L2归一化
        f3d_norm = F.normalize(f3d, dim=-1, p=2)  # L2归一化
        
        # 2. 计算点级融合权重α
        feat_concat = torch.cat([f2d_norm, f3d_norm], dim=-1)  # (B, N, 512)
        alpha_raw = self.point_mlp(feat_concat)  # (B, N, 1)
        
        # 2. 应用有效掩码调整：改进invalid点处理策略
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, N, 1)
        
        # 🔧 改进：对invalid点使用更智能的fallback策略
        # 如果大部分点都invalid，说明投影质量差，应该更多依赖3D
        valid_ratio = valid_mask.float().mean(dim=1, keepdim=True)  # (B, 1)
        
        # 动态调整fallback权重：投影质量好时用更多2D特征
        fallback_alpha = 0.3 * valid_ratio.unsqueeze(-1)  # (B, 1, 1) -> (B, N, 1)
        alpha = torch.where(valid_mask_expanded.bool(), alpha_raw, fallback_alpha)
        
        # 3. 点级融合：f_mix = α·f₂D + (1-α)·f₃D  
        # 使用归一化后的特征进行融合
        f_mix = alpha * f2d_norm + (1 - alpha) * f3d_norm  # (B, N, 256)
        
        # 4. 掩码化SE通道重加权
        f_mix_t = f_mix.permute(0, 2, 1)  # (B, 256, N)
        if self.use_masked_se:
            fused_t = self.se_masked(f_mix_t, valid_mask)  # (B, 256, N)
        else:
            # 回退到原版SE
            se_weights = self.se_module(f_mix_t)  # (B, 256, 1)
            fused_t = se_weights * f_mix_t  # (B, 256, N)
        fused_feat = fused_t.permute(0, 2, 1)  # (B, N, 256)
        
        # 返回融合特征和置信度
        confidence = alpha  # 融合权重可作为置信度
        
        return fused_feat, confidence
    
    def compute_fusion_balance_loss(self, alpha: torch.Tensor, valid_mask: torch.Tensor, 
                                   target_ratio: float = 0.4) -> torch.Tensor:
        """计算融合平衡损失，鼓励合理的2D-3D融合比例
        
        Args:
            alpha: 融合权重 (B, N, 1)
            valid_mask: 有效掩码 (B, N)
            target_ratio: 目标2D特征比例，默认0.4（略偏向3D）
            
        Returns:
            balance_loss: 标量损失值
        """
        if not valid_mask.any():
            return torch.tensor(0.0, device=alpha.device, requires_grad=True)
            
        # 只考虑有效点的融合比例
        valid_alpha = alpha[valid_mask.unsqueeze(-1).expand_as(alpha)]
        
        if valid_alpha.numel() == 0:
            return torch.tensor(0.0, device=alpha.device, requires_grad=True)
        
        # 计算当前2D特征平均比例
        current_ratio = valid_alpha.mean()
        
        # L2损失鼓励接近目标比例
        balance_loss = F.mse_loss(current_ratio, torch.tensor(target_ratio, device=alpha.device))
        
        return balance_loss


# Remove FiLM and PE modules - they are no longer used in simplified architecture


@MODELS.register_module()
class BiFusionEncoder(nn.Module):
    """Enhanced Bi-Fusion Encoder combining 2D CLIP visual features and 3D Sparse features."""

    def __init__(self,
                 voxel_size: float = 0.02,
                 use_amp: bool = True,
                 # 🎯 特征域配置（简化为仅支持60×80预计算）
                 feat_space: str = "precomp_60x80",      # 固定为预计算特征
                 use_precomp_2d: bool = True,            # 默认启用预计算特征
                 # 调试模式控制
                 debug: bool = False,
                 **kwargs):  # 接收其他未知参数
        super().__init__()
        
        # 🎯 特征域配置
        self.feat_space = feat_space
        self.use_precomp_2d = use_precomp_2d
        self.debug = debug

        # 🎯 根据特征域设置（简化，只支持60×80预计算）
        if feat_space != "precomp_60x80":
            print(f"警告: 当前仅支持precomp_60x80特征域，自动切换到precomp_60x80")
            feat_space = "precomp_60x80"
        
        # 删除Enhanced CLIP编码器（不再需要）
        # self.enhanced_clip = None
        
        # 3D encoder - 保持原始96维以兼容预训练权重，然后使用投影头到256维
        cfg_backbone = SimpleNamespace(dilations=[1, 1, 1, 1], bn_momentum=0.02, conv1_kernel_size=5)
        self.backbone3d = Res16UNet34C(in_channels=3, out_channels=96, config=cfg_backbone, D=3)
        
        # 3D投影头：96维 -> 256维（简化版本）
        self.proj_3d = EnhancedProjectionHead3D(
            input_dim=96,
            output_dim=256
        )
        
        # 统一的Head结构（2D/3D对称）
        self.head3d = Head(256, 256, p=0.1)
        self.head2d = Head(256, 256, p=0.1)
        
        # 融合机制：使用掩码化SE的LiteFusionGate      
        self.fusion_gate = LiteFusionGate(
            feat_dim=256,
            use_masked_se=True
        )
        
        # 🎯 预计算特征适配器（惰性初始化）
        self.precomp_adapter = None
        
        # 🎯 Alpha回退值（可学习参数）
        
        # 🎯 损失历史记录（用于抖动分析）
        from collections import deque
        self._loss_hist = deque(maxlen=100)

        # 基本运行/调试开关和统计结构
        self.voxel_size = voxel_size
        self.use_amp = use_amp
        self.use_lite_gate = True
        
        # 🎯 标准分辨率与内参配置
        self.W0, self.H0 = 640, 480
        self.standard_scannet_intrinsics = (577.870605, 577.870605, 319.5, 239.5)
        self.warn_valid_ratio = 0.60   # 🔧 进一步降低阈值，减少干扰信息
        self.align_corners = True  # 🚨 修复：与测试脚本的直接索引采样保持一致
        self.max_depth = 20.0
        
        # 🔧 关键修复：禁用外参自动推断，统一使用确定性处理
        self.auto_pose = False  # 强制禁用，按优化指南要求
        self._pose_pick_stats = {'direct': 0, 'inv': 0}
        # 🔧 修复：始终收集融合统计，方便训练监控
        self._collect_fusion_stats = True  # 始终启用，便于监控融合效果
        self._collect_gradient_stats = debug  # 梯度统计仍然受debug控制
        self._fusion_stats = {}
        self._stats_history = []

    def _intrinsics_for_feat(self, Hf: int, Wf: int):
        """统一内参换算函数 - 使用正确的ScanNet内参计算
        Args:
            Hf: 特征图高度 (H)
            Wf: 特征图宽度 (W)
        Returns:
            tuple: (fx_feat, fy_feat, cx_feat, cy_feat)
        """
        fx0, fy0, cx0, cy0 = self.standard_scannet_intrinsics
        # 输出特征尺寸 - 仅debug模式
        if self.debug:
            print(f"🎯 计算特征内参: 特征图尺寸=({Hf},{Wf}) - H×W格式")

        # 🔧 修正：确保缩放方向正确
        # 原始ScanNet: 640×480 (W×H)
        # 特征图: Wf×Hf
        scale_w = Wf / 640.0  # 宽度缩放
        scale_h = Hf / 480.0  # 高度缩放

        # 内参缩放：保持x/y方向对应关系
        fx_feat = fx0 * scale_w  # x方向焦距随宽度缩放
        fy_feat = fy0 * scale_h  # y方向焦距随高度缩放
        cx_feat = cx0 * scale_w  # x方向主点随宽度缩放
        cy_feat = cy0 * scale_h  # y方向主点随高度缩放

        if self.debug:
            print(f"🔧 内参缩放: 宽度缩放={scale_w:.3f}, 高度缩放={scale_h:.3f}")
            print(f"🔧 计算结果: fx={fx_feat:.1f}, fy={fy_feat:.1f}, cx={cx_feat:.1f}, cy={cy_feat:.1f}")

        return (fx_feat, fy_feat, cx_feat, cy_feat)


    def get_pose_pick_stats(self):
        return dict(self._pose_pick_stats)

    def reset_pose_pick_stats(self):
        self._pose_pick_stats = {'direct': 0, 'inv': 0}
    
    def _ensure_precomp_adapter(self, c_in: int):
        """惰性初始化预计算特征适配器：512 → 256"""
        if (self.precomp_adapter is None) or (self.precomp_adapter[0].in_features != c_in):
            # 按照优化指南要求：Linear(512→256) + LayerNorm
            self.precomp_adapter = nn.Sequential(
                nn.Linear(c_in, 256),
                nn.LayerNorm(256)
            ).to(next(self.parameters()).device)
            if self.debug:
                print(f"🔧 初始化预计算适配器: {c_in} → 256 (优化版本)")
    
    def get_grad_stats(self):
        """获取梯度健康度统计"""
        stats = {}
        for name, module in [("head2d", self.head2d), ("head3d", self.head3d), ("gate", self.fusion_gate)]:
            total = 0.0
            cnt = 0
            for p in module.parameters():
                if p.grad is not None:
                    total += p.grad.data.norm().item()
                    cnt += 1
            stats[f"grad_{name}"] = total / max(cnt, 1)
        return stats
    
    def update_loss_stat(self, loss_val: float):
        """更新损失历史记录"""
        self._loss_hist.append(float(loss_val))
    
    def get_loss_var(self):
        """获取损失滑窗方差"""
        if len(self._loss_hist) < 20:
            return None
        arr = torch.tensor(list(self._loss_hist))
        return float(arr.var(unbiased=False))
    
    def _log_key_metrics(self, valid: torch.Tensor, conf: torch.Tensor):
        """简化监控输出：仅输出关键指标"""
        # 🔧 修复：始终输出关键指标，不受debug模式限制
        # if not self.debug:
        #     return  # 非调试模式不输出
            
        with torch.no_grad():
            # 1. Valid比例
            valid_ratio = valid.float().mean().item()
            
            # 2. Fusion gate参数（alpha统计）- 只计算有效点的alpha
            alpha = conf.squeeze(-1) if conf.dim() == 2 else conf  # (N,)
            
            if valid.any():
                # 只统计有效投影点的alpha
                alpha_valid = alpha[valid]
                alpha_mean = float(alpha_valid.mean()) if alpha_valid.numel() else 0.0
                alpha_std = float(alpha_valid.std(unbiased=False)) if alpha_valid.numel() > 1 else 0.0
            else:
                # 没有有效点时的处理
                alpha_mean = 0.0
                alpha_std = 0.0
            
            # 🔧 增强输出格式：包含融合比例统计
            fusion_2d_ratio = alpha_mean  # α表示2D特征权重
            fusion_3d_ratio = 1.0 - alpha_mean  # 1-α表示3D特征权重
            
            print(f"🎯 Valid比例: {valid_ratio:.3f} | Fusion-α: 均值={alpha_mean:.3f}±{alpha_std:.3f}")
            print(f"🎯 融合比例: 2D={fusion_2d_ratio:.3f} | 3D={fusion_3d_ratio:.3f} | 总点数={valid.numel()}")
            
            # 如果valid比例为0，输出调试信息
            if valid_ratio == 0.0:
                print(f"⚠️ DEBUG: valid全为0，总点数={valid.numel()}")
                
            # 🔧 添加融合模式分析
            if valid.any():
                if alpha_mean < 0.2:
                    print(f"📊 融合模式: 主要使用3D特征 (α={alpha_mean:.3f})")
                elif alpha_mean > 0.8:
                    print(f"📊 融合模式: 主要使用2D特征 (α={alpha_mean:.3f})")
                else:
                    print(f"📊 融合模式: 平衡融合 (α={alpha_mean:.3f})")
            
            # 可配置的有效比例警告
            if self.warn_valid_ratio and valid_ratio < self.warn_valid_ratio:
                print(f"⚠️ 有效比例过低: {valid_ratio:.3f} < {self.warn_valid_ratio}")
    
    def _collect_fusion_statistics(self, conf: torch.Tensor, valid: torch.Tensor, 
                                 f2d: torch.Tensor, f3d: torch.Tensor):
        """收集融合门控统计信息 - 🔧 只统计valid点"""
        try:
            with torch.no_grad():
                # 基础统计
                if conf.dim() == 2:  # (N, 1)
                    conf_values = conf.squeeze(-1)  # (N,)
                else:
                    conf_values = conf
                
                # 🔧 关键修复：只统计valid点，避免invalid点污染统计
                if valid.any():
                    # 只使用有效点进行统计
                    valid_conf = conf_values[valid]
                    valid_f2d = f2d[valid]
                    valid_f3d = f3d[valid]
                    
                    # 计算融合比例（基于有效点）
                    fusion_2d_ratio = valid_conf.mean().item()
                    fusion_3d_ratio = 1.0 - fusion_2d_ratio
                    avg_confidence = valid_conf.mean().item()
                    
                    # 特征质量统计（基于有效点）
                    f2d_norm = torch.norm(valid_f2d, dim=-1).mean().item()
                    f3d_norm = torch.norm(valid_f3d, dim=-1).mean().item()
                    
                    # 特征相似度（基于有效点）
                    cos_sim = F.cosine_similarity(valid_f2d, valid_f3d, dim=-1).mean().item()
                    
                    total_valid_points = valid.sum().item()
                else:
                    # 没有有效点的fallback
                    fusion_2d_ratio = 0.0
                    fusion_3d_ratio = 1.0  
                    avg_confidence = 0.0
                    f2d_norm = 0.0
                    f3d_norm = 0.0
                    cos_sim = 0.0
                    total_valid_points = 0
                
                # 有效点比例（相对于总点数）
                valid_points_ratio = valid.float().mean().item()
                
                # 🔧 计算alpha分布统计（基于有效点）
                if valid.any():
                    valid_alpha = conf_values[valid]
                    alpha_mean = float(valid_alpha.mean())
                    alpha_std = float(valid_alpha.std(unbiased=False)) if valid_alpha.numel() > 1 else 0.0
                    alpha_min = float(valid_alpha.min())
                    alpha_max = float(valid_alpha.max())
                else:
                    alpha_mean = avg_confidence  # 使用总体均值作为fallback
                    alpha_std = 0.0
                    alpha_min = 0.0
                    alpha_max = 1.0
                
                # 更新统计信息 - 🔧 包含完整的alpha统计
                self._fusion_stats = {
                    'fusion_2d_ratio': fusion_2d_ratio,
                    'fusion_3d_ratio': fusion_3d_ratio, 
                    'avg_confidence': avg_confidence,
                    'valid_points_ratio': valid_points_ratio,
                    'f2d_norm_avg': f2d_norm,
                    'f3d_norm_avg': f3d_norm,
                    'feature_similarity': cos_sim,
                    'total_points': conf_values.numel(),
                    'total_valid_points': total_valid_points,
                    # 🔧 添加缺失的alpha统计
                    'alpha_mean': alpha_mean,
                    'alpha_std': alpha_std,
                    'alpha_min': alpha_min,
                    'alpha_max': alpha_max,
                    'cos_2d3d_mean': cos_sim,  # 别名，确保兼容性
                    'norm_ratio_2d_over_3d': f2d_norm / max(f3d_norm, 1e-8),  # 避免除零
                    'valid_ratio': valid_points_ratio,  # 别名，确保兼容性
                    'in_feat': 1.0  # 特征输入状态
                }
                
                # 保留历史记录（最多100条）
                self._stats_history.append(self._fusion_stats.copy())
                if len(self._stats_history) > 100:
                    self._stats_history.pop(0)
                    
        except Exception as e:
            if self.debug:
                print(f"Warning: Failed to collect fusion stats: {e}")
    
    def get_fusion_statistics(self):
        """获取融合统计信息"""
        return self._fusion_stats.copy() if self._fusion_stats else {}
    
    def get_fusion_ratios(self):
        """专门获取融合比例统计 - 供Hook使用"""
        if not self._fusion_stats:
            return {'fusion_2d_ratio': 0.0, 'fusion_3d_ratio': 1.0, 'valid_points_ratio': 0.0}
        
        return {
            'fusion_2d_ratio': self._fusion_stats.get('fusion_2d_ratio', 0.0),
            'fusion_3d_ratio': self._fusion_stats.get('fusion_3d_ratio', 1.0), 
            'valid_points_ratio': self._fusion_stats.get('valid_points_ratio', 0.0),
            'avg_confidence': self._fusion_stats.get('avg_confidence', 0.0),
            'feature_similarity': self._fusion_stats.get('feature_similarity', 0.0)
        }
    
    def get_fusion_balance_loss(self):
        """获取融合平衡损失 - 供主损失函数使用"""
        return getattr(self, '_fusion_balance_loss', None)
    
    def get_statistics_summary(self, last_n: int = 10):
        """获取最近N次的统计摘要"""
        if not self._stats_history:
            return {}
            
        recent_stats = self._stats_history[-last_n:]
        summary = {}
        
        for key in recent_stats[0].keys():
            if key != 'total_points':
                values = [stats[key] for stats in recent_stats if key in stats]
                if values:
                    summary[f'{key}_mean'] = sum(values) / len(values)
                    summary[f'{key}_std'] = (sum((x - summary[f'{key}_mean'])**2 for x in values) / len(values))**0.5
        
        return summary

    # 删除了复杂的 _improved_projection_with_geometry 函数，
    # 统一使用 unified_projection_and_sample

    def _pixels_to_grid(self, uv_feat: torch.Tensor,
                        feat_hw: Tuple[int,int],
                        align_corners: bool = True) -> torch.Tensor:
        """
        关键修复：统一grid_sample规范化标准
        把像素坐标 (u,v) 转为 grid_sample 需要的 [-1,1] 归一化坐标。
        - uv_feat: (M,2) 像素坐标（特征图尺度）
        - feat_hw: (H_feat, W_feat)
        - 返回: (1, M, 1, 2) 的 grid
        """
        H, W = feat_hw
        u = uv_feat[:, 0]
        v = uv_feat[:, 1]
        
        if align_corners:
            # 🔧 align_corners=True: 边界为 [0, W-1] [0, H-1]
            # 这样 (0,0) 映射到 (-1,-1), (W-1,H-1) 映射到 (1,1)
            x_norm = 2.0 * u / max(float(W - 1), 1.0) - 1.0
            y_norm = 2.0 * v / max(float(H - 1), 1.0) - 1.0
        else:
            # align_corners=False: 边界为 [0, W) [0, H)
            x_norm = 2.0 * (u + 0.5) / float(W) - 1.0
            y_norm = 2.0 * (v + 0.5) / float(H) - 1.0
            
        grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
        return grid

    def _sample_img_feat(self, feat_map: torch.Tensor,
                         uv_feat: torch.Tensor,
                         valid_mask: torch.Tensor,
                         align_corners: bool = True) -> torch.Tensor:
        """
        从特征图 (1,C,H,W) 采样 N 个点的特征。
        - feat_map: (1, C, H, W)
        - uv_feat:  (N, 2) 像素坐标（特征图尺度）
        - valid_mask: (N,) bool
        - 返回: (N, C)
        """
        assert feat_map.dim() == 4 and feat_map.size(0) == 1
        H, W = feat_map.shape[-2], feat_map.shape[-1]

        # 只对 valid 的点构造 grid，可以减少边界异常
        idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
        if idx.numel() == 0:
            return feat_map.new_zeros((uv_feat.size(0), feat_map.size(1)))

        uv_valid = uv_feat[idx]  # (M,2)
        grid = self._pixels_to_grid(uv_valid, (H, W), align_corners=align_corners)  # 1xMx1x2

        # 确保feat_map和grid有相同的数据类型
        if feat_map.dtype != grid.dtype:
            grid = grid.to(feat_map.dtype)

        # 采样: F.grid_sample(1, C, H, W), (1, M, 1, 2) -> (1, C, 1, M)
        sampled = F.grid_sample(
            feat_map, grid, mode='bilinear',
            align_corners=align_corners
        ).squeeze(3).squeeze(0).T  # (1, C, M) -> (C, M) -> (M, C)

        out = feat_map.new_zeros((uv_feat.size(0), feat_map.size(1)))
        out[idx] = sampled
        return out

    def unified_projection_and_sample(self,
                                      xyz_cam: torch.Tensor,
                                      feat_map: torch.Tensor):
        """
        🔧 核心修复：动态内参换算，解决valid比例为0的问题
        
        核心原则：
        1. 根据当前特征图尺寸动态计算内参，支持任意H×W
        2. 严格的边界和深度检查
        
        Args:
            xyz_cam: (N, 3) 相机坐标系点云
            intr: (4,) 原图内参 [fx, fy, cx, cy] - 已废弃，改用动态计算
            feat_map: (1, C, H, W) 特征图
        Returns:
            sampled_feat: (N, C) 采样特征
            valid_mask: (N,) 有效投影掩码
        """
        # 🎯 关键修复：优先使用传入内参，否则动态计算
        Hf, Wf = feat_map.shape[2], feat_map.shape[3]
        fx_feat, fy_feat, cx_feat, cy_feat = self._intrinsics_for_feat(Hf, Wf)

        # 3D投影：相机坐标系 → 特征图像素坐标
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        
        # 🔍 相机坐标诊断 - 降低输出频率
        if self.debug and torch.rand(1).item() < 0.05:  # 5%概率输出
            print(f"🔍 相机坐标诊断:")
            print(f"   X范围: [{x.min():.3f}, {x.max():.3f}]")
            print(f"   Y范围: [{y.min():.3f}, {y.max():.3f}]") 
            print(f"   Z范围: [{z.min():.3f}, {z.max():.3f}]")
            print(f"   内参: fx={fx_feat:.1f}, fy={fy_feat:.1f}, cx={cx_feat:.1f}, cy={cy_feat:.1f}")
        
        # 🛡️ 严格的深度过滤 - 提高阈值避免投影爆炸
        min_depth = 0.5  # 提高最小深度阈值到0.5m
        valid_z = (z > min_depth) & (z < self.max_depth)
        
        # 只对有效深度的点进行投影计算，无效点设为边界外值
        u_feat = torch.full_like(x, -1.0)  # 无效点设为-1
        v_feat = torch.full_like(y, -1.0)  # 无效点设为-1
        
        # 只对有效深度的点进行投影
        if valid_z.any():
            valid_x, valid_y, valid_z_vals = x[valid_z], y[valid_z], z[valid_z]
            u_valid = fx_feat * (valid_x / valid_z_vals) + cx_feat
            v_valid = fy_feat * (valid_y / valid_z_vals) + cy_feat
            
            # 检查投影结果是否合理（粗略范围检查）
            reasonable_u = (u_valid > -1000) & (u_valid < 1000)  # 允许较大范围但排除极值
            reasonable_v = (v_valid > -1000) & (v_valid < 1000)
            reasonable_proj = reasonable_u & reasonable_v
            
            if self.debug and torch.rand(1).item() < 0.1:
                unreasonable_count = (~reasonable_proj).sum().item()
                if unreasonable_count > 0:
                    print(f"⚠️ 投影异常: {unreasonable_count}/{len(u_valid)} 点投影坐标异常")
                    print(f"   U异常范围: [{u_valid[~reasonable_u].min():.1f}, {u_valid[~reasonable_u].max():.1f}]" if (~reasonable_u).any() else "   U正常")
                    print(f"   V异常范围: [{v_valid[~reasonable_v].min():.1f}, {v_valid[~reasonable_v].max():.1f}]" if (~reasonable_v).any() else "   V正常")
            
            # 只保留合理的投影结果
            final_valid_mask = valid_z.clone()
            final_valid_mask[valid_z] = reasonable_proj
            
            u_feat[final_valid_mask] = u_valid[reasonable_proj]
            v_feat[final_valid_mask] = v_valid[reasonable_proj]
            
            # 更新有效深度掩码
            valid_z = final_valid_mask
        
        # 边界检查：与测试脚本保持一致
        valid_u = (u_feat >= 0) & (u_feat < Wf)
        valid_v = (v_feat >= 0) & (v_feat < Hf)
        
        # 综合有效性判定
        valid = valid_z & valid_u & valid_v
        
        # 🚨 关键调试：检查投影坐标分布（临时启用）
        total_points = len(z)
        depth_valid = valid_z.sum().item()
        boundary_valid = valid.sum().item()
        
        # 坐标统计
        u_min, u_max = u_feat.min().item(), u_feat.max().item()
        v_min, v_max = v_feat.min().item(), v_feat.max().item()
        z_min, z_max = z.min().item(), z.max().item()
        
        if total_points > 0 and boundary_valid < total_points * 0.8:  # 有效率低于80%时输出
            print(f"🔍 投影坐标诊断({Hf}×{Wf}):")
            print(f"   总点数: {total_points}")
            print(f"   深度范围: [{z_min:.3f}, {z_max:.3f}]m, 有效深度: {depth_valid}/{total_points} ({100*depth_valid/total_points:.1f}%)")
            print(f"   U坐标范围: [{u_min:.1f}, {u_max:.1f}], 目标[0, {Wf})")
            print(f"   V坐标范围: [{v_min:.1f}, {v_max:.1f}], 目标[0, {Hf})")
            print(f"   最终有效: {boundary_valid}/{total_points} ({100*boundary_valid/total_points:.1f}%)")
                    
        # 特征采样 - 使用align_corners确保一致性
        uv_feat = torch.stack([u_feat, v_feat], dim=-1)  # (N, 2)
        sampled_feat = self._sample_img_feat(feat_map, uv_feat, valid, align_corners=self.align_corners)
        
        return sampled_feat, valid

    def _process_single(self, points: torch.Tensor, img: torch.Tensor, cam_meta: Dict, sample_idx: int = 0):
        """处理单帧数据，使用简化的数据流"""
        # 提取基础信息
        xyz = points[:, :3].contiguous()  
        dev = xyz.device
        dtype = xyz.dtype
        
        # 🔧 优化pose解析 - 直接提取当前样本的pose
        pose_matrix = None
        if isinstance(cam_meta, dict) and 'pose' in cam_meta:
            pose_data = cam_meta['pose']
            if isinstance(pose_data, list) and len(pose_data) > sample_idx:
                # PKL文件中的pose是list，选择当前样本对应的pose
                pose_matrix = pose_data[sample_idx]
            elif isinstance(pose_data, (list, tuple, np.ndarray)) and len(pose_data) == 1:
                # 单个pose的情况
                pose_matrix = pose_data[0] if isinstance(pose_data, (list, tuple)) else pose_data
            else:
                # 直接使用pose_data
                pose_matrix = pose_data
        
        if self.debug:
            print(f"🔍 样本{sample_idx} pose矩阵类型: {type(pose_matrix)}")
            if pose_matrix is not None:
                if hasattr(pose_matrix, 'shape') and not isinstance(pose_matrix, (list, tuple)):
                    print(f"🔍 pose矩阵形状: {pose_matrix.shape}")
                elif isinstance(pose_matrix, (list, tuple)):
                    print(f"🔍 pose矩阵长度: {len(pose_matrix)}")

        # 🎯 坐标转换：世界坐标 → 相机坐标
        if pose_matrix is None:
            # 没有pose矩阵，直接使用原始坐标
            xyz_cam_proj = xyz.clone()
            if self.debug:
                print(f"⚠️ 没有找到pose矩阵，使用原始坐标")
        else:
            try:
                # 确保pose矩阵为torch张量
                if not isinstance(pose_matrix, torch.Tensor):
                    T_matrix = torch.as_tensor(pose_matrix, dtype=dtype, device=dev)
                else:
                    T_matrix = pose_matrix.to(dtype=dtype, device=dev)

                # 🔍 调试：检查矩阵属性
                if self.debug:
                    print(f"🔍 T_matrix形状: {T_matrix.shape}")
                    print(f"🔍 T_matrix设备: {T_matrix.device}, 类型: {T_matrix.dtype}")
                    det = torch.det(T_matrix).item()
                    print(f"🔍 T_matrix行列式: {det}")
                    if torch.isnan(T_matrix).any() or torch.isinf(T_matrix).any():
                        print(f"⚠️ T_matrix包含NaN/Inf值")

                # pose是C2W格式，求逆得到W2C变换矩阵
                W2C = torch.inverse(T_matrix)

                # 齐次坐标变换：世界坐标 → 相机坐标
                xyz1 = torch.cat([xyz, torch.ones(xyz.shape[0], 1, device=dev, dtype=dtype)], dim=1)
                xyz_cam_proj = (xyz1 @ W2C.t())[:, :3]

                # 调试输出
                if self.debug:
                    z_cam = xyz_cam_proj[:, 2]
                    neg_z = (z_cam < 0).sum().item()
                    print(f"坐标转换完成: {xyz_cam_proj.shape}, 负深度点={neg_z}")

            except (RuntimeError, torch.linalg.LinAlgError) as e:
                # 矩阵求逆失败，直接使用原始坐标
                xyz_cam_proj = xyz.clone()
                if self.debug:
                    print(f"坐标转换异常，使用原始坐标: {e}")
                    print(f"🔍 异常详情: {type(e).__name__}: {str(e)}")
        
        # 3D分支始终使用世界坐标
        xyz_world = xyz

        # 3D分支：MinkUNet → 96d → Proj3D(96→256, LN inside) → Head3D(256→256, LN inside) → (不做L2)
        coords_int = torch.round(xyz_world / self.voxel_size).to(torch.int32)
        coords = torch.cat([torch.zeros(coords_int.size(0), 1, dtype=torch.int32, device=coords_int.device),
                             coords_int], dim=1)
        feats = points[:, 3:6].contiguous()
        field = ME.TensorField(coordinates=coords, features=feats)
        sparse_tensor = field.sparse()
        feat3d_sparse = self.backbone3d(sparse_tensor)
        
        # 关键修复：使用slice操作将稀疏特征映射回原始点云
        feat3d = feat3d_sparse.slice(field).features
        
        # 验证特征数量匹配（现在应该匹配了）
        if feat3d.shape[0] != points.shape[0]:
            raise RuntimeError(f"3D features shape mismatch: got {feat3d.shape[0]}, expected {points.shape[0]}")
        
        # 3D投影头：96维 -> 256维 (内含LN，不额外做L2)
        feat3d = self.proj_3d(feat3d.float())  # (N, 96) -> (N, 256), 确保float类型
        
        # 统一Head：不做L2归一化
        f3d = self.head3d(feat3d)  # (N, 256)

        # 🎯 2D特征处理：投影采样或零特征fallback
        if xyz_cam_proj is None:
            # 相机投影失败，使用零特征
            print(f"⚠️ 相机投影失败，使用零特征")
            feat2d_raw = f3d.new_zeros((f3d.shape[0], 256))
            valid = f3d.new_zeros((f3d.shape[0],), dtype=torch.bool)
            f2d = self.head2d(feat2d_raw)

        elif isinstance(cam_meta, dict) and cam_meta.get("clip_pix") is not None:
            if self.debug:
                print(f"🎯 使用预计算CLIP特征进行投影采样")
            # 有有效投影和CLIP特征，进行投影采样
            clip_data = cam_meta["clip_pix"]
            
            # 如果clip_data是list，根据sample_idx选择对应的特征
            if isinstance(clip_data, list) and len(clip_data) > sample_idx:
                selected_clip = clip_data[sample_idx]
            elif isinstance(clip_data, (list, tuple)) and len(clip_data) == 1:
                selected_clip = clip_data[0]
            else:
                selected_clip = clip_data
            
            # 确保selected_clip是tensor
            if isinstance(selected_clip, torch.Tensor):
                feat_map = selected_clip.to(device=dev, dtype=dtype)
            else:
                feat_map = torch.as_tensor(selected_clip, device=dev, dtype=dtype)
                
            feat_map = feat_map.float().unsqueeze(0)

            # 投影采样
            feat2d_raw, valid = self.unified_projection_and_sample(
                xyz_cam=xyz_cam_proj,
                feat_map=feat_map)

            # 通道适配：512 → 256
            if feat2d_raw.shape[-1] != 256:
                self._ensure_precomp_adapter(feat2d_raw.shape[-1])
                if self.precomp_adapter is not None:
                    feat2d_raw = self.precomp_adapter(feat2d_raw)

            f2d = self.head2d(feat2d_raw)

        else:
            # 缺少CLIP特征，使用零特征
            print(f"⚠️ 缺少CLIP特征，使用零特征")
            feat2d_raw = f3d.new_zeros((f3d.shape[0], 256))
            valid = f3d.new_zeros((f3d.shape[0],), dtype=torch.bool)
            f2d = self.head2d(feat2d_raw)

        # 融合特征
        f2d_batch = f2d.unsqueeze(0)
        f3d_batch = f3d.unsqueeze(0)
        valid_batch = valid.unsqueeze(0)
        
        fused_batch, conf_batch = self.fusion_gate(f2d_batch, f3d_batch, valid_batch)
        fused = fused_batch.squeeze(0)
        conf = conf_batch.squeeze(0)
        
        # L2归一化
        fused = F.normalize(fused, dim=-1)
        
        # 统计信息收集
        if self._collect_fusion_stats:
            self._collect_fusion_statistics(conf, valid, f2d, f3d)
        self._log_key_metrics(valid, conf)

        # 🔥 计算并保存融合平衡损失（用于主损失函数）
        if self.training:
            fusion_balance_loss = self.fusion_gate.compute_fusion_balance_loss(
                conf, valid, target_ratio=0.4
            )
            # 保存到全局变量中，供损失函数获取
            globals()['_current_fusion_balance_loss'] = fusion_balance_loss
        else:
            globals()['_current_fusion_balance_loss'] = None

        return fused, conf, valid

    def forward(self, points_list, imgs, cam_info):
        """简化的forward函数：批量处理3D-2D融合"""
        
        # 1. 输入格式标准化
        if torch.is_tensor(points_list):
            points_list = list(points_list) if points_list.dim() == 3 else [points_list]
        if torch.is_tensor(imgs):
            imgs = list(imgs) if imgs.dim() == 4 else [imgs]
        
        batch_size = len(points_list)
        if len(imgs) != batch_size:
            raise RuntimeError(f"输入长度不匹配: points({len(points_list)}) != imgs({len(imgs)})")
        
        # 2. cam_info标准化
        if cam_info is None or isinstance(cam_info, dict):
            cam_info = [cam_info] * batch_size
        elif len(cam_info) == 1:
            cam_info = cam_info * batch_size
        elif len(cam_info) != batch_size:
            raise RuntimeError(f"cam_info长度({len(cam_info)})与batch_size({batch_size})不匹配")
        
        # 3. 逐样本处理
        feat_fusion_list, conf_list, valid_mask_list = [], [], []
        
        for idx, (pts, img, meta) in enumerate(zip(points_list, imgs, cam_info)):
            # 简化meta信息处理：PKL文件是帧级组织，直接复制
            meta_std = meta if meta is not None else {}
            
            # 处理单个样本，传递样本索引
            fused, conf, valid_mask = self._process_single(pts, img, meta_std, idx)
            
            feat_fusion_list.append(fused)
            conf_list.append(conf)
            valid_mask_list.append(valid_mask)
        
        return {
            'feat_fusion': feat_fusion_list,
            'conf_2d': conf_list,
            'valid_projection_mask': valid_mask_list
        }
