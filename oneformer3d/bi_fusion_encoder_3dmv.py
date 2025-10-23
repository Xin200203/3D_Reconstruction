import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import contextlib
from typing import List, Dict, Optional, Tuple, Union, cast
import warnings

import MinkowskiEngine as ME
import copy
from mmdet3d.registry import MODELS
from .mink_unet import Res16UNet34C
from .projection_utils import unified_projection_and_sample
from types import SimpleNamespace
from collections import deque, defaultdict

@MODELS.register_module()
class Conv3DFusionModule(nn.Module):
    """3DMV式3D卷积融合模块
    
    仿照3DMV架构设计，通过3D卷积实现空间一致性的2D-3D特征融合：
    - features3d: 处理3D几何特征，96维 → 64维
    - features2d: 处理投影后的2D特征，256维 → 32维  
    - features_fusion: 融合两种特征，96维(64+32) → 128维
    
    与原点级融合相比，3D卷积能更好地利用空间邻域信息进行特征融合
    """
    
    def __init__(self, 
                 feat3d_dim: int = 96,      # 3D特征维度（MinkUNet输出）
                 feat2d_dim: int = 256,     # 2D特征维度（CLIP投影后）
                 output_dim: int = 128,     # 最终输出维度
                 enable_debug: bool = False,
                 collect_gradient_stats: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        
        self.feat3d_dim = feat3d_dim
        self.feat2d_dim = feat2d_dim
        self.output_dim = output_dim
        self.enable_debug = enable_debug
        self.collect_gradient_stats = collect_gradient_stats
        self.dropout = float(max(0.0, min(1.0, dropout)))
        
        # 仿照3DMV的features3d：处理3D几何特征 (96 → 64维)
        self.features3d = nn.Sequential(
            # 第一阶段：特征扩展和空间感知
            ME.MinkowskiConvolution(feat3d_dim, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.02),
            ME.MinkowskiReLU(True),
            # 1x1x1精炼卷积：提取更抽象的特征表示
            ME.MinkowskiConvolution(64, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiDropout(self.dropout),
            
            # 第二阶段：保持64维，进一步特征抽象
            ME.MinkowskiConvolution(64, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(64, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiDropout(self.dropout)
        )
        
        # 仿照3DMV的features2d：处理投影后的2D特征 (256 → 32维)
        self.features2d = nn.Sequential(
            # 第一阶段：维度压缩 256 → 64
            ME.MinkowskiConvolution(feat2d_dim, 64, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.02),
            ME.MinkowskiReLU(True),
            # 1x1x1精炼卷积
            ME.MinkowskiConvolution(64, 64, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(64, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiDropout(self.dropout),
            
            # 第二阶段：进一步压缩 64 → 32
            ME.MinkowskiConvolution(64, 32, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(32, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(32, 32, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(32, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiDropout(self.dropout)
        )
        
        # 3D-only 阶段的通道扩展：将 3D 分支 64 通道扩展到 96 通道（head64 + shadow32）
        self.expand3d_64to96 = nn.Sequential(
            ME.MinkowskiConvolution(64, 96, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(96, momentum=0.02),
            ME.MinkowskiReLU(True)
        )

        # 仿照3DMV的features：多模态特征融合 (96维=64+32 → 128维)
        self.features_fusion = nn.Sequential(
            # 融合阶段：处理concatenated特征
            ME.MinkowskiConvolution(96, 128, kernel_size=3, stride=1, dimension=3),
            ME.MinkowskiBatchNorm(128, momentum=0.02),
            ME.MinkowskiReLU(True),
            # 1x1x1精炼卷积：深层特征抽象
            ME.MinkowskiConvolution(128, 128, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(128, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiConvolution(128, output_dim, kernel_size=1, dimension=3),
            ME.MinkowskiBatchNorm(output_dim, momentum=0.02),
            ME.MinkowskiReLU(True),
            ME.MinkowskiDropout(self.dropout)
        )
        
        self._last_monitor = {}
        self._last_feats = None
        self._grad_feature_norms = {}
        self._prev_grad_stats = {}

        if self.enable_debug:
            print(f"🔧 初始化Conv3DFusionModule: 3D({feat3d_dim}→64) + 2D({feat2d_dim}→32) → 融合({96}→{output_dim})")

    def forward(self, feat3d_sparse: ME.SparseTensor, feat2d_sparse: ME.SparseTensor) -> ME.SparseTensor:
        """
        3D卷积融合前向传播
        
        Args:
            feat3d_sparse: ME.SparseTensor，3D特征 (N, feat3d_dim)
            feat2d_sparse: ME.SparseTensor，2D特征 (N, feat2d_dim)
        Returns:
            fused_sparse: ME.SparseTensor，融合特征 (N, output_dim)
        """
        if self.enable_debug:
            print(f"🔍 Conv3D融合输入: 3D特征{feat3d_sparse.features.shape}, 2D特征{feat2d_sparse.features.shape}")
        
        # 分别处理3D和2D特征：模仿3DMV的双分支设计
        f3d_processed = self.features3d(feat3d_sparse)      # 96 → 64维
        f3d_feats = f3d_processed.features                  # (N, 64)

        # 将 3D 64 通道扩展到 96 通道（head64 + shadow32）
        proj96_sparse = self.expand3d_64to96(f3d_processed)  # (N, 96)
        proj96_feats = proj96_sparse.features
        head64 = proj96_feats[:, :64]
        shadow32 = proj96_feats[:, 64:]

        # 读取 alpha（若未设置则视为 0.0）
        alpha = float(getattr(self, 'alpha_for_blend', 0.0))

        # 仅当 alpha > 0 时才计算 2D 分支，避免 Phase A 额外开销
        if alpha > 0.0:
            f2d_processed = self.features2d(feat2d_sparse)  # (N, 32)
            f2d_feats = f2d_processed.features
        else:
            f2d_processed = None
            f2d_feats = None

        if self.enable_debug:
            print(f"🔍 分支处理后: 3D特征{f3d_feats.shape}, 2D特征{f2d_feats.shape}")

        monitor = {}
        with torch.no_grad():
            monitor['feat3d_mean_abs'] = f3d_feats.abs().mean().item()
            monitor['feat3d_std'] = f3d_feats.std().item()
            monitor['feat3d_nonzero_ratio'] = (f3d_feats.abs() > 1e-3).float().mean().item()

            if f2d_feats is not None:
                monitor['feat2d_mean_abs'] = f2d_feats.abs().mean().item()
                monitor['feat2d_std'] = f2d_feats.std().item()
                monitor['feat2d_nonzero_ratio'] = (f2d_feats.abs() > 1e-3).float().mean().item()
            else:
                monitor['feat2d_mean_abs'] = 0.0
                monitor['feat2d_std'] = 0.0
                monitor['feat2d_nonzero_ratio'] = 0.0

        if self.collect_gradient_stats:
            prev_norms = getattr(self, '_grad_feature_norms', None)
            self._prev_grad_stats = prev_norms.copy() if prev_norms else {}
            self._grad_feature_norms = {}
        else:
            self._prev_grad_stats = {}
        
        # 特征拼接：在通道维度concat (64+32=96维)
        # 捕捉3D坐标顺序并对齐2D特征（或使用 shadow32）
        coord_manager = f3d_processed.coordinate_manager
        coords3d = f3d_processed.C.float()

        if alpha > 0.0 and f2d_processed is not None:
            try:
                # 将 2D 分支特征按照 3D 活跃坐标顺序对齐
                f2d_aligned = f2d_processed.features_at_coordinates(coords3d)
            except RuntimeError as err:
                if self.enable_debug:
                    print(f"⚠️ features_at_coordinates 异常: {err}")
                f2d_aligned = f3d_processed.features.new_zeros(
                    f3d_processed.features.shape[0], 32)

            if not torch.isfinite(f2d_aligned).all():
                invalid_mask = ~torch.isfinite(f2d_aligned)
                if self.enable_debug:
                    invalid_count = invalid_mask.sum().item()
                    print(f"⚠️ 对齐后的2D特征出现NaN/Inf，已置零，数量: {invalid_count}")
                f2d_aligned = f2d_aligned.masked_fill(invalid_mask, 0)
        else:
            # Phase A 或 alpha=0：不使用 2D 分支
            f2d_aligned = None

        if self.collect_gradient_stats:
            def _capture(name):
                def hook(grad):
                    if grad is None:
                        return
                    if not hasattr(self, '_grad_feature_norms'):
                        self._grad_feature_norms = {}
                    with torch.no_grad():
                        self._grad_feature_norms[f'grad_norm_{name}'] = grad.detach().norm().item()
                return hook

            # 仅在需要梯度时注册hook，避免在eval/无梯度时抛出异常
            if f3d_feats.requires_grad:
                f3d_feats.register_hook(_capture('feat3d'))
            if f2d_aligned is not None and f2d_aligned.requires_grad:
                f2d_aligned.register_hook(_capture('feat2d'))

        if self.collect_gradient_stats:
            def _capture(name):
                key = f'grad_norm_{name}_raw'

                def hook(grad):
                    if grad is None:
                        return
                    if not hasattr(self, '_grad_feature_norms'):
                        self._grad_feature_norms = {}
                    with torch.no_grad():
                        self._grad_feature_norms[key] = grad.detach().norm().item()
                return hook

            if f3d_feats.requires_grad:
                f3d_feats.register_hook(_capture('feat3d'))
            if f2d_aligned is not None and f2d_aligned.requires_grad:
                f2d_aligned.register_hook(_capture('feat2d'))

        # 记录监控信息；具体特征快照在后续构建 tail32 后统一存储
        self._last_monitor = monitor

        # 构造 tail32：Phase A 使用 shadow32；Phase B 使用 shadow32 与 f2d_aligned 的线性混合
        if f2d_aligned is None:
            tail32 = shadow32
        else:
            # 保证形状匹配 (N, 32)
            if f2d_aligned.shape[1] != 32:
                if self.enable_debug:
                    print(f"⚠️ f2d_aligned 通道维不为32，当前 {f2d_aligned.shape[1]}，将截断或补零")
                if f2d_aligned.shape[1] > 32:
                    f2d_aligned = f2d_aligned[:, :32]
                else:
                    pad = f2d_aligned.new_zeros(f2d_aligned.size(0), 32 - f2d_aligned.size(1))
                    f2d_aligned = torch.cat([f2d_aligned, pad], dim=1)
            tail32 = (1.0 - alpha) * shadow32 + alpha * f2d_aligned

        manual_features = torch.cat([head64, tail32], dim=1)
        if self.collect_gradient_stats and manual_features.requires_grad:
            manual_features.register_hook(_capture('fusion'))
        fused_sparse = ME.SparseTensor(
            features=manual_features,
            coordinate_map_key=f3d_processed.coordinate_map_key,
            coordinate_manager=coord_manager
        )

        if self.enable_debug:
            print(f"🔍 手动特征拼接成功: {fused_sparse.features.shape}")

        # 最终融合卷积：96 → output_dim维
        output_sparse = self.features_fusion(fused_sparse)

        self._last_monitor = monitor
        # 记录融合前用于相似度的特征（保持键名不变）。若无2D，则用 shadow32 代替，用于上层统计。
        if f2d_feats is None:
            # 伪造一个与 tail32 同形的特征供上层取用
            f2d_record = tail32.detach()
        else:
            f2d_record = f2d_feats.detach()
        self._last_feats = {'f3d_feats': f3d_feats, 'f2d_feats': f2d_record}

        if self.enable_debug:
            print(f"🔍 Conv3D融合输出: {output_sparse.features.shape}")

        return output_sparse

@MODELS.register_module(name='BiFusionEncoder3DMV')
class BiFusionEncoder(nn.Module):
    """Enhanced Bi-Fusion Encoder combining 2D CLIP visual features and 3D Sparse features.
    
    🔥 新增3DMV式3D卷积融合支持：
    
    架构设计：
    - 传统模式：点级融合（LiteFusionGate）
    - 增强模式：3D卷积融合（Conv3DFusionModule）
    - 混合模式：两种融合方式结合
    
    使用方法：
    1. 纯点级融合（默认）：
       use_conv3d_fusion=False, fusion_mode="point_only"
       
    2. 纯3D卷积融合：
       use_conv3d_fusion=True, fusion_mode="conv3d_only"
       
    3. 混合融合：
       use_conv3d_fusion=True, fusion_mode="hybrid"
    
    核心原理：
    - 3D分支：MinkUNet(96维) → Conv3D处理 → 64维
    - 2D分支：CLIP特征(256维) → Conv3D处理 → 32维  
    - 融合：Concat(96维) → Conv3D → 最终特征
    
    相比点级融合的优势：
    - 空间一致性：利用3D卷积的空间邻域信息
    - 层次融合：在卷积特征层级进行融合，更深入
    - 端到端学习：整个过程可微分，支持梯度反传
    """

    def __init__(self,
                 voxel_size: float = 0.02,
                 use_amp: bool = True,
                 # 🎯 特征域配置（简化为仅支持60×80预计算）
                 feat_space: str = "precomp_60x80",      # 固定为预计算特征
                 use_precomp_2d: bool = True,            # 默认启用预计算特征
                 # 🔥 3D卷积融合配置（专门使用Conv3D）
                 conv3d_output_dim: int = 256,           # 3D卷积融合输出维度，默认256保持兼容
                 conv3d_dropout: float = 0.1,            # 3D卷积融合中的Dropout比例（可为0关闭）
                 # 调试模式控制
                 debug: bool = False,
                 collect_gradient_stats: bool = True,
                 freeze_2d_branch: bool = False,
                 **kwargs):  # 接收其他未知参数
        super().__init__()
        self.freeze_2d_branch = freeze_2d_branch
        # 🔧 修复：如果voxel_size是字典（config传入错误），提取或使用默认值
        if isinstance(voxel_size, dict):
            print(f"⚠️ 警告: voxel_size传入了字典，使用默认值0.02")
            voxel_size = 0.02
        
        # 🎯 特征域配置
        self.feat_space = feat_space
        self.use_precomp_2d = use_precomp_2d
        self.debug = debug
        
        # 🔥 3D卷积融合配置（专门使用Conv3D）
        self.conv3d_output_dim = conv3d_output_dim

        # 🎯 根据特征域设置（简化，只支持60×80预计算）
        if feat_space != "precomp_60x80":
            print(f"警告: 当前仅支持precomp_60x80特征域，自动切换到precomp_60x80")
            feat_space = "precomp_60x80"
        
        # 删除Enhanced CLIP编码器（不再需要）
        # self.enhanced_clip = None
        
        # 3D encoder - 保持原始96维以兼容预训练权重
        cfg_backbone = SimpleNamespace(dilations=[1, 1, 1, 1], bn_momentum=0.02, conv1_kernel_size=5)
        self.backbone3d = Res16UNet34C(in_channels=3, out_channels=96, config=cfg_backbone, D=3)
        
        # 🔥 3D卷积融合模块：专门使用Conv3D融合
        self.alpha_2d = 0.0
        self.conv3d_fusion = Conv3DFusionModule(
            feat3d_dim=96,          # MinkUNet输出维度
            feat2d_dim=256,         # 2D特征维度（适配后）
            output_dim=self.conv3d_output_dim,  # 可配置输出维度
            enable_debug=self.debug,
            collect_gradient_stats=collect_gradient_stats,
            dropout=float(max(0.0, min(1.0, conv3d_dropout)))
        )
        self.align_dim = 64
        self.cos_proj3d = nn.Sequential(
            nn.Linear(64, self.align_dim),
            nn.LayerNorm(self.align_dim)
        )
        self.cos_proj2d = nn.Sequential(
            nn.Linear(32, self.align_dim),
            nn.LayerNorm(self.align_dim)
        )
        if self.debug:
            print(f"🔧 初始化3D卷积融合模块: 输出维度={self.conv3d_output_dim}")
        
        # 🎯 预计算特征适配器（惰性初始化）
        self.precomp_adapter = None
        
        # 🎯 Alpha回退值（可学习参数）
        
        # 🎯 损失历史记录（用于抖动分析）
        self._loss_hist = deque(maxlen=100)

        # 基本运行/调试开关和统计结构
        self.voxel_size = voxel_size
        self.use_amp = use_amp
        self.standard_scannet_intrinsics = (577.870605, 577.870605, 319.5, 239.5)
        self.align_corners = True  # 与投影采样保持一致
        self.max_depth = 20.0
        self._collect_fusion_stats = True
        self._collect_gradient_stats = collect_gradient_stats  # 梯度统计输出独立于debug
        self._fusion_stats = {}
        self._stats_history = []

        self._param_grad_sums = defaultdict(float)
        self._param_grad_groups = {}
        self._registered_param_ids = set()
        self._last_param_grad_norms = {}

        # 🔥 输出配置信息
        self._print_config_summary()

        if self._collect_gradient_stats:
            self._register_grad_param_hooks()

        if self.freeze_2d_branch:
            self._freeze_2d_parameters()

    def _freeze_2d_parameters(self):
        """Freeze 2D projection branch during Phase A."""
        modules_to_freeze = []
        if hasattr(self.conv3d_fusion, 'features2d'):
            modules_to_freeze.append(self.conv3d_fusion.features2d)
        modules_to_freeze.append(self.cos_proj2d)
        for module in modules_to_freeze:
            module.eval()
            for param in module.parameters():
                param.requires_grad = False
        # features_fusion 也会接收到2D支路输出，此处不冻结以保留学习能力。

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_2d_branch and mode:
            # 再次施加冻结，避免外部 train() 调用恢复 2D 分支 BN/权重
            self._freeze_2d_parameters()
        return self
    
    def _print_config_summary(self):
        """打印当前配置摘要"""
        print("=" * 60)
        print("🔥 BiFusionEncoder配置摘要 - 专用3D卷积融合版本")
        print("=" * 60)
        print(f"特征域: {self.feat_space}")
        print(f"使用预计算2D特征: {self.use_precomp_2d}")
        print(f"体素大小: {self.voxel_size}")
        print(f"调试模式: {self.debug}")
        print("-" * 40)
        print("🎯 融合配置:")
        print(f"  融合模式: 专用3D卷积融合")
        print(f"  3D卷积输出维度: {self.conv3d_output_dim}")
        print(f"  3D卷积模块: {'已初始化' if self.conv3d_fusion is not None else '未初始化'}")
        print(f"  梯度监控: {'启用' if self._collect_gradient_stats else '关闭'}")
        print("-" * 40)
        print("📊 架构说明:")
        print("  模式: 3DMV式3D卷积融合")  
        print("  特点: 空间一致性强，利用邻域信息，端到端学习")
        print("  流程: 3D(96维)→64维 + 2D(256维)→32维 → Concat(96维) → 融合输出")
        print("=" * 60)
    
    @classmethod
    def create_conv3d_config(cls, **kwargs):
        """创建3D卷积融合配置的便捷方法
        
        示例:
        # 默认配置（256维输出）
        encoder = BiFusionEncoder.create_conv3d_config()
        
        # 自定义输出维度
        encoder = BiFusionEncoder.create_conv3d_config(
            conv3d_output_dim=128,
            debug=True
        )
        """
        default_config = {
            'conv3d_output_dim': 256,
            'debug': False,
            'collect_gradient_stats': True
        }
        default_config.update(kwargs)
        return cls(**default_config)

    def _create_sparse_tensor_from_features(self, 
                                             features: torch.Tensor, 
                                             coordinates: torch.Tensor,
                                             coord_manager=None) -> ME.SparseTensor:
        """
        将特征和坐标转换为MinkowskiEngine稀疏张量
        
        Args:
            features: (N, C) 特征张量
            coordinates: (N, 3) 坐标张量（世界坐标系）
            coord_manager: 坐标管理器，用于确保稀疏张量兼容性
        Returns:
            ME.SparseTensor: 稀疏张量
        """
        # 坐标量化：世界坐标 → 体素坐标
        coords_int = torch.round(coordinates / self.voxel_size).to(torch.int32)
        
        # 添加batch维度：(N, 3) → (N, 4)，第一列为batch_index=0
        coords_with_batch = torch.cat([
            torch.zeros(coords_int.size(0), 1, dtype=torch.int32, device=coords_int.device),
            coords_int
        ], dim=1)
        
        sparse_kwargs = {
            'features': features.float(),
            'coordinates': coords_with_batch,
            'device': features.device
        }
        if coord_manager is not None:
            sparse_kwargs['coordinate_manager'] = coord_manager

        sparse_tensor = ME.SparseTensor(**sparse_kwargs)
        
        if self.debug:
            print(f"🔧 创建稀疏张量: 特征{features.shape} → 坐标{coords_with_batch.shape}")
        
        return sparse_tensor
    
    def _convert_2d_features_to_sparse(self, 
                                       feat2d: torch.Tensor, 
                                       xyz_world: torch.Tensor,
                                       valid_mask: torch.Tensor,
                                       reference_sparse: ME.SparseTensor) -> ME.SparseTensor:
        """将 2D 特征重新排列为与 3D 稀疏张量一致的坐标。"""
        feat2d_filled = feat2d.clone()
        if feat2d_filled.numel() > 0:
            feat2d_filled[~valid_mask] = 0

        coords_ref = reference_sparse.C  # (M, 4)
        device = feat2d_filled.device
        feature_dim = feat2d_filled.shape[1]

        ordered_features = feat2d_filled.new_zeros((coords_ref.shape[0], feature_dim))
        hit_counts = feat2d_filled.new_zeros((coords_ref.shape[0],), dtype=feat2d_filled.dtype)

        coord_to_idx = {tuple(coord): idx for idx, coord in enumerate(coords_ref.cpu().tolist())}
        coords_full = torch.cat([
            torch.zeros(xyz_world.size(0), 1, dtype=torch.int32, device=xyz_world.device),
            torch.round(xyz_world / self.voxel_size).to(torch.int32)
        ], dim=1)

        coords_full_list = coords_full.cpu().tolist()
        idx_list = [coord_to_idx.get(tuple(coord), -1) for coord in coords_full_list]
        idx_tensor = torch.tensor(idx_list, device=device, dtype=torch.long)

        valid_idx_mask = idx_tensor >= 0
        if not valid_idx_mask.all():
            missing = (~valid_idx_mask).sum().item()
            if missing > 0:
                warnings.warn(f"{missing} points not found in sparse coordinate map; skipped.", stacklevel=2)

        if valid_idx_mask.any():
            idx_tensor_valid = idx_tensor[valid_idx_mask]
            feat_selected = feat2d_filled[valid_idx_mask]
            ordered_features.index_add_(0, idx_tensor_valid, feat_selected)
            hit_counts.index_add_(
                0,
                idx_tensor_valid,
                torch.ones(idx_tensor_valid.shape[0], device=device, dtype=hit_counts.dtype)
            )

        ordered_features = ordered_features / hit_counts.clamp_min(1.0).unsqueeze(-1)

        return ME.SparseTensor(
            features=ordered_features.float(),
            coordinate_manager=reference_sparse.coordinate_manager,
            coordinate_map_key=reference_sparse.coordinate_map_key
        )
    
    def _extract_features_from_sparse(self, 
                                      sparse_tensor: ME.SparseTensor, 
                                      target_coordinates: torch.Tensor,
                                      target_size: int) -> torch.Tensor:
        """
        从稀疏张量中提取目标坐标对应的特征
        
        Args:
            sparse_tensor: ME.SparseTensor 输入稀疏张量
            target_coordinates: (N, 3) 目标坐标（世界坐标系）
            target_size: int 目标特征数量
        Returns:
            torch.Tensor: (target_size, C) 提取的特征
        """
        # 量化目标坐标
        target_coords_int = torch.round(target_coordinates / self.voxel_size).to(torch.int32)
        target_coords_with_batch = torch.cat([
            torch.zeros(target_coords_int.size(0), 1, dtype=torch.int32, device=target_coords_int.device),
            target_coords_int
        ], dim=1)
        
        # 使用features_at_coords方法提取特征
        try:
            extracted = sparse_tensor.features_at_coordinates(target_coords_with_batch.float())
        except Exception as err:
            warnings.warn(f"Failed to look up sparse features: {err}", stacklevel=2)
            extracted = torch.zeros(
                target_size,
                sparse_tensor.features.shape[1],
                device=sparse_tensor.device,
                dtype=sparse_tensor.features.dtype
            )
            return extracted
        
        if extracted.shape[0] != target_size:
            padded = torch.zeros(
                target_size,
                sparse_tensor.features.shape[1],
                device=sparse_tensor.device,
                dtype=sparse_tensor.features.dtype
            )
            copy_len = min(extracted.shape[0], target_size)
            padded[:copy_len] = extracted[:copy_len]
            extracted = padded
        
        return extracted

    def get_pose_pick_stats(self):
        """保留接口，当前实现不统计该信息。"""
        return {}

    def reset_pose_pick_stats(self):
        """保留接口，无需执行额外操作。"""
        return None
    
    #？？
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
            if self._collect_gradient_stats:
                self._register_params_to_group('feat2d', self.precomp_adapter.parameters())

    # ------------------------------------------------------------------
    # 梯度监控辅助函数
    # ------------------------------------------------------------------
    def _make_param_hook(self, group_key: str):
        def _hook(grad: torch.Tensor):
            if grad is None or not self._collect_gradient_stats:
                return
            self._param_grad_sums[group_key] += grad.detach().pow(2).sum().item()

        return _hook

    def _register_params_to_group(self, group_key: str, params):
        if not self._collect_gradient_stats:
            return

        group_list = self._param_grad_groups.setdefault(group_key, [])
        for param in params:
            if (param is None) or (not getattr(param, 'requires_grad', False)):
                continue
            param_id = id(param)
            if param_id in self._registered_param_ids:
                continue
            group_list.append(param)
            param.register_hook(self._make_param_hook(group_key))
            self._registered_param_ids.add(param_id)

    def _register_grad_param_hooks(self):
        # 基础分支
        self._register_params_to_group('feat3d', list(self.backbone3d.parameters()))
        self._register_params_to_group('feat3d', list(self.conv3d_fusion.features3d.parameters()))
        self._register_params_to_group('feat2d', list(self.conv3d_fusion.features2d.parameters()))
        self._register_params_to_group('fusion', list(self.conv3d_fusion.features_fusion.parameters()))

        # 解码器
        if hasattr(self, 'decoder') and self.decoder is not None:
            self._register_params_to_group('decoder', list(self.decoder.parameters()))

    def _pop_param_grad_norms(self) -> Dict[str, float]:
        if not self._collect_gradient_stats:
            return {}

        norms = {}
        for group_key, sq_sum in self._param_grad_sums.items():
            norms[f'grad_params_{group_key}'] = sq_sum ** 0.5 if sq_sum > 0 else 0.0

        self._param_grad_sums = defaultdict(float)
        return norms
    
    def update_loss_stat(self, loss_val: float):
        """更新损失历史记录"""
        self._loss_hist.append(float(loss_val))
    
    def get_loss_var(self):
        """获取损失滑窗方差"""
        if len(self._loss_hist) < 20:
            return None
        arr = torch.tensor(list(self._loss_hist))
        return float(arr.var(unbiased=False))
    
    # 简化的统计方法已集成在_process_single中
    
    def get_fusion_statistics(self):
        """获取融合统计信息"""
        return self._fusion_stats.copy() if self._fusion_stats else {}
    
    def get_fusion_ratios(self):
        """专门获取融合比例统计 - 供Hook使用"""
        if not self._fusion_stats:
            return {}

        keys = [
            'avg_confidence',
            'valid_ratio',
            'norm_ratio_2d_over_3d',
            'cos_2d3d_mean',
            'cos_2d3d_mean_ln'
        ]
        return {k: self._fusion_stats.get(k, 0.0) for k in keys if k in self._fusion_stats}
    
    # 融合平衡损失相关方法已删除 - 专用Conv3D不需要
    
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

    def _extract_pose_matrix(self, cam_meta: Dict, sample_idx: int = 0):
        """从 cam_info 中提取单帧 pose 矩阵（cam2world）。"""
        # !!!!!
        if not isinstance(cam_meta, dict):
            return None

        pose = cam_meta.get('pose')
        if pose is None:
            return None

        if isinstance(pose, (list, tuple)):
            poses = [p for p in pose if p is not None]
            if not poses:
                return None
            index = min(sample_idx, len(poses) - 1)
            pose = poses[index]

        if isinstance(pose, torch.Tensor):
            return pose.to(dtype=torch.float32)
        if isinstance(pose, np.ndarray):
            return torch.from_numpy(pose).float()

        warnings.warn(f"Unsupported pose type {type(pose)}; ignoring pose.", stacklevel=2)
        return None

    def _transform_coordinates(self, xyz: torch.Tensor, pose_matrix: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        """将世界坐标系的点转换到相机坐标系。"""
        if pose_matrix is None:
            return None

        if pose_matrix.shape != (4, 4):
            warnings.warn(f"Unexpected pose shape {pose_matrix.shape}; ignoring pose.", stacklevel=2)
            return None

        pose = pose_matrix.to(device=xyz.device, dtype=xyz.dtype)
        if not torch.isfinite(pose).all():
            warnings.warn("Pose matrix contains NaN/Inf values; ignoring pose.", stacklevel=2)
            return None

        try:
            w2c = torch.inverse(pose)
        except RuntimeError as err:
            warnings.warn(f"Pose inversion failed: {err}; ignoring pose.", stacklevel=2)
            return None

        homo = torch.ones((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
        xyz_cam = torch.cat([xyz, homo], dim=1) @ w2c.t()
        xyz_cam = xyz_cam[:, :3]

        if not torch.isfinite(xyz_cam).all():
            warnings.warn("Projected camera coordinates contain NaN/Inf; ignoring pose.", stacklevel=2)
            return None

        positive_depth_ratio = (xyz_cam[:, 2] > 0).float().mean().item()
        if positive_depth_ratio < 0.1:
            warnings.warn(f"Too few points with positive depth ({positive_depth_ratio:.3f}); ignoring pose.", stacklevel=2)
            return None

        return xyz_cam

    def _process_single(self, points: torch.Tensor, img: List[torch.Tensor], cam_meta: Dict, sample_idx: int = 0):
        """处理单帧 2D-3D 融合流程。"""
        xyz = points[:, :3].contiguous()
        dev = xyz.device
        proj3d_points = None
        proj2d_points = None

        pose_matrix = self._extract_pose_matrix(cam_meta, sample_idx=sample_idx)
        xyz_cam_proj = self._transform_coordinates(xyz, pose_matrix)

        coords_int = torch.round(xyz / self.voxel_size).to(torch.int32)
        coords = torch.cat(
            [torch.zeros(coords_int.size(0), 1, dtype=torch.int32, device=coords_int.device), coords_int],
            dim=1)
        feats = points[:, 3:6].contiguous()
        field = ME.TensorField(coordinates=coords, features=feats)
        feat3d_sparse = self.backbone3d(field.sparse())

        clip_data = cam_meta.get('clip_pix') if isinstance(cam_meta, dict) else None
        if isinstance(clip_data, (list, tuple)):
            clip_candidates = [c for c in clip_data if c is not None]
            if clip_candidates:
                clip_data = clip_candidates[min(sample_idx, len(clip_candidates) - 1)]
            else:
                clip_data = None

        if xyz_cam_proj is None or clip_data is None:
            print("Missing xyz_cam_proj or clip_data; falling back to zero 2D features.")
            if clip_data is None:
                warnings.warn("Missing clip_pix feature; falling back to zero 2D features.", stacklevel=2)
            feat2d_raw = torch.zeros((points.shape[0], 256), device=dev, dtype=torch.float32)
            valid = torch.zeros(points.shape[0], device=dev, dtype=torch.bool)
        else:
            # !!!!
            if isinstance(clip_data, torch.Tensor):
                feat_map = clip_data.to(device=dev, dtype=torch.float32)
            elif isinstance(clip_data, np.ndarray):
                feat_map = torch.from_numpy(clip_data).to(device=dev, dtype=torch.float32)
            else:
                warnings.warn(f"Unsupported clip_pix type {type(clip_data)}; using zero features.", stacklevel=2)
                feat_map = None

            if feat_map is None:
                feat2d_raw = torch.zeros((points.shape[0], 256), device=dev, dtype=torch.float32)
                valid = torch.zeros(points.shape[0], device=dev, dtype=torch.bool)
            else:
                feat2d_raw, valid = unified_projection_and_sample(
                    xyz_cam=xyz_cam_proj,
                    feat_map=feat_map.unsqueeze(0),
                    max_depth=self.max_depth,
                    align_corners=self.align_corners,
                    standard_intrinsics=self.standard_scannet_intrinsics,
                    debug=self.debug,
                    debug_prefix=f'[BiFusion3DMV] sample={sample_idx}'
                )
                # 投影有效率过低告警（可能由位姿/内参与分辨率错配引起）
                try:
                    valid_ratio_local = float(valid.float().mean().item())
                    if valid_ratio_local < 0.1:
                        warnings.warn(
                            f"Low projection valid ratio: {valid_ratio_local:.3f} (sample={sample_idx})."
                            " Check pose/intrinsics/resolution consistency.",
                            stacklevel=2)
                except Exception:
                    pass
                if feat2d_raw.shape[-1] != 256:
                    self._ensure_precomp_adapter(feat2d_raw.shape[-1])
                    feat2d_raw = self.precomp_adapter(feat2d_raw) if self.precomp_adapter else feat2d_raw

        try:
            feat2d_sparse = self._convert_2d_features_to_sparse(
                feat2d_raw,
                xyz,
                valid,
                reference_sparse=feat3d_sparse
            )

            if self.alpha_2d < 1.0:
                scale = float(max(0.0, min(1.0, self.alpha_2d)))
                if scale == 0.0:
                    scaled = feat2d_sparse.features.new_zeros(feat2d_sparse.features.shape)
                else:
                    scaled = feat2d_sparse.features * scale
                feat2d_sparse = ME.SparseTensor(
                    features=scaled,
                    coordinate_map_key=feat2d_sparse.coordinate_map_key,
                    coordinate_manager=feat2d_sparse.coordinate_manager,
                    tensor_stride=feat2d_sparse.tensor_stride
                )

            cos_mean = 0.0
            try:
                feat3d_base = feat3d_sparse.features
                feat2d_base = feat2d_sparse.features
                min_dim = min(feat3d_base.shape[1], feat2d_base.shape[1])
                if min_dim > 0:
                    cos_mean = float(F.cosine_similarity(
                        F.normalize(feat3d_base[:, :min_dim], dim=1),
                        F.normalize(feat2d_base[:, :min_dim], dim=1),
                        dim=1).mean().item())
            except Exception as err:
                warnings.warn(f"Failed to compute feature similarity: {err}", stacklevel=2)

            # 将 alpha 传递给融合模块，用于 Phase A/B 下的 tail32 构造策略
            try:
                self.conv3d_fusion.alpha_for_blend = float(self.alpha_2d)
            except Exception:
                pass
            fused_sparse = self.conv3d_fusion(feat3d_sparse, feat2d_sparse)

            monitor_stats = getattr(self.conv3d_fusion, '_last_monitor', {}).copy()
            feat_dict = getattr(self.conv3d_fusion, '_last_feats', None)
            if feat_dict is not None:
                proj3d_points = self.cos_proj3d(feat_dict['f3d_feats'])
                proj2d_points = self.cos_proj2d(feat_dict['f2d_feats'])
                with torch.no_grad():
                    proj3d_ln = F.layer_norm(proj3d_points.detach(), proj3d_points.shape[-1:])
                    proj2d_ln = F.layer_norm(proj2d_points.detach(), proj2d_points.shape[-1:])
                    monitor_stats['cos_2d3d_mean_ln'] = F.cosine_similarity(proj3d_ln, proj2d_ln, dim=1).mean().item()
            else:
                monitor_stats = monitor_stats or {}

            if self._collect_gradient_stats:
                grad_stats = getattr(self.conv3d_fusion, '_prev_grad_stats', None)
                if grad_stats:
                    monitor_stats.update(grad_stats)
                if getattr(self, '_last_param_grad_norms', None):
                    monitor_stats.update(self._last_param_grad_norms)
                    g2d = self._last_param_grad_norms.get('grad_params_feat2d', 0.0)
                    g3d = self._last_param_grad_norms.get('grad_params_feat3d', 0.0)
                    monitor_stats['grad_ratio_2d_over_3d'] = g2d / (g3d + 1e-12)
            self.conv3d_fusion._last_feats = None

            fused = self._extract_features_from_sparse(fused_sparse, xyz, points.shape[0])

            if fused.shape[-1] != self.conv3d_output_dim:
                if fused.shape[-1] < self.conv3d_output_dim:
                    padding = torch.zeros(
                        fused.shape[0],
                        self.conv3d_output_dim - fused.shape[-1],
                        device=fused.device,
                        dtype=fused.dtype
                    )
                    fused = torch.cat([fused, padding], dim=-1)
                else:
                    fused = fused[:, :self.conv3d_output_dim]

            # 数值稳定性检查：若出现 NaN/Inf，告警并用3D-only回退
            if not torch.isfinite(fused).all():
                warnings.warn("Fused features contain NaN/Inf; falling back to 3D-only features for this sample.", stacklevel=2)
                fallback_3d = self._extract_features_from_sparse(feat3d_sparse, xyz, points.shape[0])
                if fallback_3d.shape[-1] != self.conv3d_output_dim:
                    if fallback_3d.shape[-1] < self.conv3d_output_dim:
                        padding = torch.zeros(
                            fallback_3d.shape[0],
                            self.conv3d_output_dim - fallback_3d.shape[-1],
                            device=fallback_3d.device,
                            dtype=fallback_3d.dtype
                        )
                        fused = torch.cat([fallback_3d, padding], dim=-1)
                    else:
                        fused = fallback_3d[:, :self.conv3d_output_dim]
                else:
                    fused = fallback_3d

            valid_ratio = valid.float().mean().item()
            conf_value = max(0.3, min(0.9, valid_ratio))
            conf = torch.full((points.shape[0], 1), conf_value, device=dev, dtype=torch.float32)

            if self.debug:
                print(f"[BiFusion3DMV] sample={sample_idx} valid_ratio={valid_ratio:.3f} cos_mean={cos_mean:.3f}")

        except Exception as e:
            warnings.warn(f"Conv3D fusion failed; using 3D-only fallback. Details: {e}", stacklevel=2)
            # 用3D主干特征回退，避免将全零特征送入解码器导致预测退化
            fallback_3d = self._extract_features_from_sparse(feat3d_sparse, xyz, points.shape[0])
            # 调整维度至 conv3d_output_dim
            if fallback_3d.shape[-1] != self.conv3d_output_dim:
                if fallback_3d.shape[-1] < self.conv3d_output_dim:
                    padding = torch.zeros(
                        fallback_3d.shape[0],
                        self.conv3d_output_dim - fallback_3d.shape[-1],
                        device=fallback_3d.device,
                        dtype=fallback_3d.dtype
                    )
                    fused = torch.cat([fallback_3d, padding], dim=-1)
                else:
                    fused = fallback_3d[:, :self.conv3d_output_dim]
            else:
                fused = fallback_3d
            conf = torch.full((points.shape[0], 1), 0.5, device=dev, dtype=torch.float32)
            # 关键：当需要构造稀疏张量时，特征行数必须与活跃点数一致
            n_active_fallback = int(feat3d_sparse.features.shape[0])
            proj3d_points = torch.zeros((n_active_fallback, self.align_dim), device=dev, dtype=torch.float32)
            proj2d_points = torch.zeros((n_active_fallback, self.align_dim), device=dev, dtype=torch.float32)
            self.conv3d_fusion._last_monitor = {}
            self.conv3d_fusion._last_feats = None
            self.conv3d_fusion._prev_grad_stats = {}
            if hasattr(self.conv3d_fusion, '_grad_feature_norms'):
                self.conv3d_fusion._grad_feature_norms = {}
            monitor_stats = {}
            self._last_param_grad_norms = {}

        # 记录融合特征原始幅值，便于监控
        fused_pre_norm = fused.detach()

        # 简化的统计信息收集（同时记录 pre-gate 与 post-gate 指标，post-gate 能反映 α 对2D分支的实际抑制程度）
        if self._collect_fusion_stats:
            try:
                valid_ratio = valid.float().mean().item()
                feat2d_norm = feat2d_raw.norm(dim=-1).clamp_min(1e-6).mean().item()
                
                feat3d_norm = feat3d_sparse.features.norm(dim=-1).clamp_min(1e-6).mean().item()
                norm_ratio = feat2d_norm / max(feat3d_norm, 1e-6)

                # 记录post-gate（经过 α 门控后的）2D范数与比值
                try:
                    feat2d_post = feat2d_sparse.features
                    feat2d_norm_post = feat2d_post.norm(dim=-1).clamp_min(1e-6).mean().item()
                    norm_ratio_post = feat2d_norm_post / max(feat3d_norm, 1e-6)
                except Exception:
                    feat2d_norm_post = 0.0
                    norm_ratio_post = 0.0

                with torch.no_grad():
                    monitor_stats['fused_mean_abs_raw'] = fused_pre_norm.abs().mean().item()
                    monitor_stats['fused_std_raw'] = fused_pre_norm.std().item()
                    monitor_stats['fused_norm_mean_raw'] = fused_pre_norm.norm(dim=-1).mean().item()

                self._fusion_stats = {
                    'valid_ratio': valid_ratio,
                    'valid_points_ratio': valid_ratio,
                    'avg_confidence': conf_value,
                    'norm_ratio_2d_over_3d': norm_ratio,
                    'norm_ratio_2d_over_3d_post': norm_ratio_post,
                    'cos_2d3d_mean': cos_mean,
                    'norm_2d_mean': feat2d_norm,
                    'norm_2d_mean_post': feat2d_norm_post,
                    'norm_3d_mean': feat3d_norm
                }
                self._fusion_stats.update(monitor_stats)
                
                if self.debug:
                    print(f"📊 融合统计: 有效比例={valid_ratio:.3f}, 2D特征范数={feat2d_norm:.3f}")
            except Exception as e:
                if self.debug:
                    print(f"⚠️ 统计收集失败: {e}")
        
        # 在构建稀疏张量前，确保特征长度与坐标映射一致；如不一致，直接报错
        n_active = int(feat3d_sparse.features.shape[0])
        if proj3d_points is None or proj3d_points.shape[0] != n_active:
            got = -1 if proj3d_points is None else int(proj3d_points.shape[0])
            warnings.warn(
                f"proj3d_points invalid for SparseTensor (got rows={got}, active={n_active});"
                " filling zeros aligned to active coordinates.",
                stacklevel=2)
            proj3d_points = torch.zeros((n_active, self.align_dim), device=dev, dtype=torch.float32)
        proj3d_sparse = ME.SparseTensor(
            features=proj3d_points,
            coordinate_map_key=feat3d_sparse.coordinate_map_key,
            coordinate_manager=feat3d_sparse.coordinate_manager,
            tensor_stride=feat3d_sparse.tensor_stride
        )
        proj3d_points = self._extract_features_from_sparse(proj3d_sparse, xyz, points.shape[0])

        if proj2d_points is None or proj2d_points.shape[0] != n_active:
            got = -1 if proj2d_points is None else int(proj2d_points.shape[0])
            warnings.warn(
                f"proj2d_points invalid for SparseTensor (got rows={got}, active={n_active});"
                " filling zeros aligned to active coordinates.",
                stacklevel=2)
            proj2d_points = torch.zeros((n_active, self.align_dim), device=dev, dtype=torch.float32)
        proj2d_sparse = ME.SparseTensor(
            features=proj2d_points,
            coordinate_map_key=feat3d_sparse.coordinate_map_key,
            coordinate_manager=feat3d_sparse.coordinate_manager,
            tensor_stride=feat3d_sparse.tensor_stride
        )
        proj2d_points = self._extract_features_from_sparse(proj2d_sparse, xyz, points.shape[0])

        return fused, conf, valid, proj3d_points, proj2d_points


    def set_alpha_2d(self, value: float) -> None:
        """Set 2D branch gating value between 0 and 1."""
        self.alpha_2d = float(max(0.0, min(1.0, value)))

    def forward(self, points_list, imgs, cam_info):
        """简化的forward函数：批量处理3D-2D融合"""
        # 1. 输入格式标准化
        if self.debug:
            print(f"🔍 forward输入概览 | points_list: {type(points_list)} | imgs: {type(imgs)} | cam_info: {type(cam_info)}")

        if self._collect_gradient_stats:
            self._last_param_grad_norms = self._pop_param_grad_norms()
        else:
            self._last_param_grad_norms = {}

        if not isinstance(points_list, list):
            raise TypeError(f"points_list must be list[Tensor], got {type(points_list)}")
        if not isinstance(imgs, list):
            raise TypeError(f"imgs must be list[Tensor], got {type(imgs)}")
        if not isinstance(cam_info, list):
            raise TypeError(f"cam_info must be list[dict], got {type(cam_info)}")

        batch_size = len(points_list)
        if len(imgs) != batch_size:
            if len(imgs) == 1:
                single_img = imgs[0]
                if torch.is_tensor(single_img):
                    imgs = [single_img.clone() for _ in range(batch_size)]
                else:
                    imgs = [copy.deepcopy(single_img) for _ in range(batch_size)]
            else:
                raise RuntimeError(f"points({batch_size}) and imgs({len(imgs)}) length mismatch")
        if len(cam_info) != batch_size:
            if len(cam_info) == 1:
                single_meta = cam_info[0]
                cam_info = [copy.deepcopy(single_meta) for _ in range(batch_size)]
            else:
                raise RuntimeError(f"cam_info({len(cam_info)}) and points({batch_size}) length mismatch")
        
        # 3. 逐样本处理
        feat_fusion_list, conf_list, valid_mask_list = [], [], []
        proj3d_list, proj2d_list = [], []
        
        for idx, (pts, img, meta) in enumerate(zip(points_list, imgs, cam_info)):
            # 简化meta信息处理：PKL文件是帧级组织，直接复制
            meta_std = meta if meta is not None else {}
            
            # 处理单个样本，传递样本索引
            fused, conf, valid_mask, proj3d_pts, proj2d_pts = self._process_single(pts, img, meta_std, idx)
            
            feat_fusion_list.append(fused)
            conf_list.append(conf)
            valid_mask_list.append(valid_mask)
            proj3d_list.append(proj3d_pts)
            proj2d_list.append(proj2d_pts)
        
        return {
            'feat_fusion': feat_fusion_list,
            'conf_2d': conf_list,
            'valid_projection_mask': valid_mask_list,
            'proj_3d_points': proj3d_list,
            'proj_2d_points': proj2d_list,
            'alpha_2d': float(self.alpha_2d)
        }
