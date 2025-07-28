import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math
from typing import List, Dict, Optional, Tuple

import MinkowskiEngine as ME
from mmdet3d.registry import MODELS
from .mink_unet import Res16UNet34C
from .tiny_sa import TinySAModule, TinySA2D
from .clip_utils import (
    freeze_clip_except_last_blocks as _freeze_clip_except_last_blocks,
    build_uv_index as _build_uv_index,
    sample_img_feat as _sample_img_feat
)
from types import SimpleNamespace


def build_geo_pe(xyz_world: torch.Tensor, bbox_size: torch.Tensor,
                 pose_delta: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    """Assemble 64-d geometric positional encoding.
    xyz_world: (N,3) world coordinates
    bbox_size: (N,3) w,h,l
    pose_delta: (9,) repeat to N (R6 + t3)
    height: (N,1)
    return: (N,64)
    """
    N = xyz_world.shape[0]
    # base 3
    feats = [xyz_world]
    # sin/cos 48d (8 freq per axis)
    freq = torch.pow(2, torch.arange(8, device=xyz_world.device, dtype=xyz_world.dtype)) * math.pi
    sin_list = []
    for f in freq:
        sin_list.append(torch.sin(xyz_world * f))
        sin_list.append(torch.cos(xyz_world * f))
    feats.append(torch.cat(sin_list, dim=-1))  # (N,3*2*8)
    feats.append(bbox_size)  # 3
    feats.append(pose_delta.unsqueeze(0).repeat(N, 1))  # 9
    feats.append(height)  # 1
    return torch.cat(feats, dim=-1)  # (N,64)


class EnhancedCLIPEncoder(nn.Module):
    """改进的CLIP编码器，使用前几层Transformer blocks"""
    
    def __init__(self,
                 clip_pretrained: str = 'openai',
                 num_layers: int = 6,
                 freeze_conv1: bool = False,
                 freeze_early_layers: bool = True,
                 target_resolution: int = 224):
        super().__init__()
        
        # 加载CLIP模型
        self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
            'ViT-B-16', pretrained=clip_pretrained
        )
        self.clip_visual = self.clip_model.visual
        self.num_layers = num_layers
        self.target_resolution = target_resolution
        
        # 智能冻结策略
        self._setup_freezing(freeze_conv1, freeze_early_layers)
        
        # 输出投影层
        self.spatial_proj = nn.Conv2d(768, 256, kernel_size=1)
        self.global_proj = nn.Linear(768, 256)
        
    def _setup_freezing(self, freeze_conv1: bool, freeze_early_layers: bool):
        """智能冻结策略"""
        for name, param in self.clip_visual.named_parameters():
            if 'conv1' in name:
                param.requires_grad = not freeze_conv1
            elif 'positional_embedding' in name or 'class_embedding' in name:
                param.requires_grad = not freeze_conv1
            elif 'ln_pre' in name:
                param.requires_grad = not freeze_conv1
            elif 'transformer.resblocks' in name:
                layer_idx = int(name.split('.')[2])
                if freeze_early_layers and layer_idx < 3:
                    param.requires_grad = False
                elif layer_idx < self.num_layers:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            else:
                param.requires_grad = False
                
        # 打印冻结状态
        total_params = sum(p.numel() for p in self.clip_visual.parameters())
        trainable_params = sum(p.numel() for p in self.clip_visual.parameters() if p.requires_grad)
        print(f"Enhanced CLIP: {trainable_params:,}/{total_params:,} "
              f"参数可训练 ({trainable_params/total_params*100:.1f}%)")
    
    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        B = images.shape[0]
        
        # Resize到CLIP标准尺寸
        if images.shape[-2:] != (self.target_resolution, self.target_resolution):
            images = F.interpolate(images, size=(self.target_resolution, self.target_resolution), 
                                 mode='bilinear', align_corners=False)
        
        # Patch embedding
        x = self.clip_visual.conv1(images)  # (B, 768, 14, 14)
        spatial_raw = x
        
        # Reshape for transformer
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)  # (B, 196, 768)
        
        # Add class token and positional embedding
        class_token = self.clip_visual.class_embedding.to(x.dtype) + torch.zeros(
            x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
        )
        x = torch.cat([class_token, x], dim=1)  # (B, 197, 768)
        x = x + self.clip_visual.positional_embedding.to(x.dtype)
        x = self.clip_visual.ln_pre(x)
        
        # 通过前num_layers层Transformer
        x = x.permute(1, 0, 2)  # NLD -> LND
        for i in range(self.num_layers):
            x = self.clip_visual.transformer.resblocks[i](x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 重建空间特征
        patch_tokens = x[:, 1:, :].permute(0, 2, 1).reshape(B, 768, 14, 14)
        fused_spatial = patch_tokens + spatial_raw  # 残差连接
        spatial_feat = self.spatial_proj(fused_spatial)  # (B, 256, 14, 14)
        
        # 全局特征
        cls_token = x[:, 0, :]  # (B, 768)
        global_feat = self.global_proj(cls_token)  # (B, 256)
        
        return spatial_feat, global_feat


class EnhancedFusionGate(nn.Module):
    """增强的融合Gate机制"""
    
    def __init__(self, 
                 feat_dim: int = 96,
                 use_spatial_attention: bool = True,
                 spatial_k: int = 16):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.use_spatial_attention = use_spatial_attention
        self.spatial_k = spatial_k
        
        # 基础Gate
        self.base_gate = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力模块
        if use_spatial_attention:
            self.spatial_attn = nn.Sequential(
                nn.Conv1d(feat_dim * 2, 64, 1),
                nn.ReLU(),
                nn.Conv1d(64, 64, 1),
                nn.ReLU(),
                nn.Conv1d(64, 1, 1),
                nn.Sigmoid()
            )
        
        # 几何一致性模块
        self.geo_encoder = nn.Sequential(
            nn.Linear(6, 32),  # xyz + normal
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        self.consistency_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2 + 16, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Gate融合
        num_gates = 2 if use_spatial_attention else 1
        self.gate_fusion = nn.Sequential(
            nn.Linear(num_gates + 1, 16),  # base + spatial + geometry
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        # 置信度预测
        self.confidence_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def _estimate_normals(self, xyz: torch.Tensor, k: int = 8) -> torch.Tensor:
        """简单的法向量估计"""
        N = xyz.shape[0]
        device = xyz.device
        
        with torch.no_grad():
            dist = torch.cdist(xyz, xyz)
            _, knn_idx = torch.topk(dist, k+1, dim=1, largest=False)
            knn_idx = knn_idx[:, 1:]  # 去掉自己
        
        neighbors = xyz[knn_idx]  # (N, k, 3)
        center = xyz.unsqueeze(1)  # (N, 1, 3)
        centered = neighbors - center  # (N, k, 3)
        cov = torch.bmm(centered.transpose(1, 2), centered)  # (N, 3, 3)
        
        try:
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            normals = eigenvectors[:, :, 0]  # 最小特征值对应的向量
        except:
            normals = torch.randn(N, 3, device=device)
        
        normals = F.normalize(normals, dim=1)
        return normals
    
    def forward(self, 
                f2d: torch.Tensor, 
                f3d: torch.Tensor,
                xyz: torch.Tensor,
                valid_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            f2d: (N, C) 2D特征
            f3d: (N, C) 3D特征
            xyz: (N, 3) 3D坐标
            valid_mask: (N,) 投影有效性
            
        Returns:
            fused_feat: (N, C) 融合特征
            confidence: (N, 1) 融合置信度
        """
        N, C = f2d.shape
        
        # 基础Gate
        base_input = torch.cat([f2d, f3d], dim=1)
        base_gate = self.base_gate(base_input)  # (N, 1)
        
        gates = [base_gate]
        
        # 空间注意力Gate
        if self.use_spatial_attention:
            with torch.no_grad():
                dist = torch.cdist(xyz, xyz)
                _, knn_idx = torch.topk(dist, self.spatial_k, dim=1, largest=False)
            
            f2d_neighbors = f2d[knn_idx]  # (N, k, C)
            f3d_neighbors = f3d[knn_idx]  # (N, k, C)
            f2d_local = f2d_neighbors.mean(dim=1)  # (N, C)
            f3d_local = f3d_neighbors.mean(dim=1)  # (N, C)
            
            fusion_input = torch.cat([f2d + f2d_local, f3d + f3d_local], dim=1)  # (N, 2C)
            fusion_input = fusion_input.unsqueeze(0).transpose(1, 2)  # (1, 2C, N)
            spatial_gate = self.spatial_attn(fusion_input).transpose(1, 2).squeeze(0)  # (N, 1)
            gates.append(spatial_gate)
        
        # 几何一致性Gate (不受valid_mask影响，仅作为几何先验)
        normals = self._estimate_normals(xyz)
        geo_feat = torch.cat([xyz, normals], dim=1)  # (N, 6)
        geo_encoded = self.geo_encoder(geo_feat)  # (N, 16)
        consistency_input = torch.cat([f2d, f3d, geo_encoded], dim=1)
        geometry_gate = self.consistency_mlp(consistency_input)  # (N, 1)
        # 几何Gate不受valid_mask直接影响，而是作为几何先验
        gates.append(geometry_gate)
        
        # 融合多个Gate
        gate_concat = torch.cat(gates, dim=1)  # (N, num_gates)
        final_gate = self.gate_fusion(gate_concat)  # (N, 1)
        
        # 应用有效性约束 - 这里才考虑valid_mask
        valid_weight = valid_mask.float().unsqueeze(1)
        # 对于无效投影点，使用较小的2D权重但不完全清零
        final_gate = final_gate * valid_weight + final_gate * 0.1 * (1 - valid_weight)
        
        # 特征融合
        fused_feat = final_gate * f2d + (1 - final_gate) * f3d
        
        # 置信度估计
        confidence = self.confidence_mlp(base_input)
        # 置信度受valid_mask影响，无效点置信度较低
        confidence = confidence * (valid_weight * 0.9 + 0.1)  # 最低保持10%置信度
        
        return fused_feat, confidence


# 注册到 MMEngine MODELS，便于在配置中直接引用
@MODELS.register_module()
class TinySANeck(nn.Module):
    """Two-layer Tiny Self-Attention neck implemented by stacking TinySAModule.

    Args:
        dim (int): feature dimension.
        num_heads (int): number of attention heads for each TinySA layer.
        radius (float): ball query radius.
        max_k (int): max neighbours per center.
        sample_ratio (float): ratio of sampled center points.
        num_layers (int): number of TinySA layers to stack. Default 2 as in paper spec.
    """
    def __init__(self,
                 dim: int = 128,
                 num_heads: int = 4,
                 radius: float = 0.3,
                 max_k: int = 32,
                 sample_ratio: float = 0.25,
                 num_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([
            TinySAModule(dim=dim,
                          num_heads=num_heads,
                          radius=radius,
                          max_k=max_k,
                          sample_ratio=sample_ratio)
            for _ in range(num_layers)
        ])

    def forward(self, x, feats: Optional[torch.Tensor] = None, voxel_size: float = 0.02):
        """Forward 支持两种输入：

        1. `x` 为 MinkowskiEngine SparseTensor（来自 3D Backbone）。
        2. `x` 为 (N,3) xyz 坐标张量，需同时传入 `feats` (N,C)。
        返回与输入类型一致的数据结构。
        """
        import MinkowskiEngine as ME  # 避免循环依赖

        # Case 1: SparseTensor
        if isinstance(x, ME.SparseTensor):
            sp_tensor = x
            xyz = sp_tensor.coordinates[:, 1:].float() * voxel_size  # 去掉批索引
            feats_in = sp_tensor.features
            updated_feats = self._apply_sa(xyz, feats_in)
            return ME.SparseTensor(
                updated_feats,
                coordinate_map_key=sp_tensor.coordinate_map_key,
                coordinate_manager=sp_tensor.coordinate_manager)

        # Case 2: xyz + feats
        if feats is None:
            raise ValueError('When first argument is xyz Tensor, feats must not be None.')
        return self._apply_sa(x, feats)

    # === 新增内部函数：统一执行 TinySA 堆叠 ===
    def _apply_sa(self, xyz: torch.Tensor, feats: torch.Tensor):
        """Apply stacked TinySA layers.

        Args:
            xyz (Tensor): (N,3) coordinates.
            feats (Tensor): (N,C) features.
        Returns:
            Tensor: (N,C) updated features.
        """
        for sa in self.layers:
            feats = sa(xyz, feats)
        return feats


@MODELS.register_module()
class BiFusionEncoder(nn.Module):
    """Enhanced Bi-Fusion Encoder combining 2D CLIP visual features and 3D Sparse features."""

    def __init__(self,
                 clip_pretrained: str = 'openai',
                 voxel_size: float = 0.02,
                 freeze_blocks: int = 0,  # 控制CLIP冻结层数
                 use_amp: bool = True,
                 use_tiny_sa_2d: bool = False,
                 # Enhanced CLIP配置
                 clip_num_layers: int = 6,
                 freeze_clip_conv1: bool = False,
                 freeze_clip_early_layers: bool = True,
                 # Enhanced Gate配置
                 use_enhanced_gate: bool = True,
                 use_spatial_attention: bool = True,
                 spatial_k: int = 16,
                 # TinySA控制
                 use_tiny_sa_3d: bool = False):  # 新增参数控制是否使用TinySA
        super().__init__()
        
        # Enhanced CLIP编码器
        self.enhanced_clip = EnhancedCLIPEncoder(
            clip_pretrained=clip_pretrained,
            num_layers=clip_num_layers,
            freeze_conv1=freeze_clip_conv1,
            freeze_early_layers=freeze_clip_early_layers
        )
        
        # 2D特征处理
        self.lin2d = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.ln2d = nn.LayerNorm(256)
        
        # 3D encoder - 保持原始96维以兼容预训练权重，然后适配到256维
        cfg_backbone = SimpleNamespace(dilations=[1, 1, 1, 1], bn_momentum=0.02, conv1_kernel_size=5)
        self.backbone3d = Res16UNet34C(in_channels=3, out_channels=96, config=cfg_backbone, D=3)
        backbone_out_dim = 96
        
        # 添加适配层：96维 -> 256维
        self.backbone_adapter = nn.Sequential(
            nn.Linear(96, 256),
            nn.ReLU(),
            nn.LayerNorm(256)
        )
        
        # 条件性地使用TinySA或简单的线性层
        adapted_dim = 256
        self.use_tiny_sa_3d = use_tiny_sa_3d
        
        if use_tiny_sa_3d:
            # 使用TinySA（如果明确启用）
            self.tiny_sa_neck = TinySANeck(dim=adapted_dim, num_heads=8, radius=0.3, max_k=32, sample_ratio=0.25, num_layers=2)
        else:
            # 使用简单的线性层替代TinySA
            self.simple_neck = nn.Sequential(
                nn.Linear(adapted_dim, adapted_dim),
                nn.ReLU(),
                nn.LayerNorm(adapted_dim),
                nn.Linear(adapted_dim, adapted_dim),
                nn.ReLU(),
                nn.LayerNorm(adapted_dim)
            )
            
        self.lin3d = nn.Sequential(nn.Linear(adapted_dim, 256), nn.ReLU())
        self.ln3d = nn.LayerNorm(256)

        # PE mapping
        self.pe_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        # 特征对齐 - 调整维度以匹配256维输出
        self.lin2d_final = nn.Linear(320, 256)  # 256+64 -> 256
        self.lin3d_final = nn.Linear(320, 256)  # 256+64 -> 256

        # 融合机制选择
        self.use_enhanced_gate = use_enhanced_gate
        if use_enhanced_gate:
            self.fusion_gate = EnhancedFusionGate(
                feat_dim=256,
                use_spatial_attention=use_spatial_attention,
                spatial_k=spatial_k
            )
        else:
            # 回退到简单Gate - 调整输入维度以匹配256维特征
            self.gate_mlp = nn.Sequential(
                nn.Linear(512, 128),  # 256*2 -> 128
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        self.voxel_size = voxel_size
        self.use_amp = use_amp

    def build_uv_index(self, xyz_cam, intr, img_shape):
        return _build_uv_index(xyz_cam, intr, img_shape)

    def sample_img_feat(self, feat_map, uv):
        return _sample_img_feat(feat_map, uv)

    def _process_single(self, points: torch.Tensor, img: torch.Tensor, cam_meta: Dict,
                        feat2d_map: Optional[torch.Tensor] = None,
                        clip_global: Optional[torch.Tensor] = None):
        """处理单帧数据，使用增强的CLIP和融合机制"""
        # 提取基础信息
        xyz_cam = points[:, :3]
        
        # 坐标变换
        T_world2cam = None
        if cam_meta.get('extrinsics', None) is not None:
            extr = cam_meta['extrinsics']
            if not torch.is_tensor(extr):
                extr = torch.as_tensor(extr, dtype=xyz_cam.dtype, device=xyz_cam.device)
            
            # 处理可能的批量维度或列表格式
            if isinstance(extr, (list, tuple)):
                extr = torch.tensor(extr, dtype=xyz_cam.dtype, device=xyz_cam.device)
            
            if extr.dim() == 3:  # (B, 3, 4) 或 (B, 4, 4)
                # 取第一个样本，或者选择合适的样本
                extr = extr[0]  # 现在是 (3, 4) 或 (4, 4)
            elif extr.dim() == 1:  # 展平的矩阵，reshape
                if extr.numel() == 12:  # 3x4矩阵
                    extr = extr.view(3, 4)
                elif extr.numel() == 16:  # 4x4矩阵
                    extr = extr.view(4, 4)
            
            if extr.shape == (3, 4):
                extr = torch.cat([extr, extr.new_tensor([[0, 0, 0, 1]])], dim=0)
            T_cam2world = extr
            T_world2cam = torch.inverse(T_cam2world)
            xyz_h_cam = torch.cat([xyz_cam, xyz_cam.new_ones(xyz_cam.size(0), 1)], dim=-1)
            # 确保矩阵乘法结果维度正确
            xyz_world_h = torch.matmul(xyz_h_cam, T_cam2world.T)  # (N, 4)
            xyz_world = xyz_world_h[:, :3]  # (N, 3)
        else:
            xyz_world = xyz_cam

        # 投影坐标
        if T_world2cam is not None:
            # 确保xyz_world是2D张量 (N, 3)
            if xyz_world.dim() != 2 or xyz_world.size(-1) != 3:
                raise ValueError(f"xyz_world should be (N, 3), got {xyz_world.shape}")
            xyz_h_world = torch.cat([xyz_world, xyz_world.new_ones(xyz_world.size(0), 1)], dim=-1)
            xyz_cam_proj_h = torch.matmul(xyz_h_world, T_world2cam.T)  # (N, 4)
            xyz_cam_proj = xyz_cam_proj_h[:, :3]  # (N, 3)
        else:
            xyz_cam_proj = xyz_world

        # 几何PE
        bbox_size = torch.zeros_like(xyz_world)
        pose_delta = torch.zeros(9, device=xyz_world.device, dtype=xyz_world.dtype)
        height = xyz_world[:, 2:3]
        pe = self.pe_mlp(build_geo_pe(xyz_world, bbox_size, pose_delta, height))

        # 3D分支
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
        
        # 应用适配层：96维 -> 256维
        feat3d = self.backbone_adapter(feat3d)
        
        if self.use_tiny_sa_3d:
            feat3d = self.tiny_sa_neck(xyz_world, feat3d)
        else:
            feat3d = self.simple_neck(feat3d)
            
        feat3d = self.lin3d(feat3d)
        feat3d = self.ln3d(feat3d)

        # 2D分支 - 使用Enhanced CLIP
        if feat2d_map is None or clip_global is None:
            with torch.no_grad():
                amp_ctx = torch.cuda.amp.autocast(enabled=self.use_amp and img.is_cuda)
                with amp_ctx:
                    feat2d_map, clip_global = self.enhanced_clip(img.unsqueeze(0))
                    feat2d_map = feat2d_map.squeeze(0)  # Remove batch dim

        # 投影采样 - 处理内参格式（增强容错）
        intr = cam_meta['intrinsics']
        if not torch.is_tensor(intr):
            intr = torch.as_tensor(intr, dtype=xyz_cam.dtype, device=xyz_cam.device)
        
        # 确保intrinsics是1D tensor (4,) - 增强处理逻辑
        if intr.dim() == 2:  # (1, 4) 或 (B, 4)
            if intr.shape[-1] == 4:
                intr = intr[0]  # 取第一个
            elif intr.shape[0] == 4:
                intr = intr[:, 0] if intr.shape[1] == 1 else intr.flatten()
        elif intr.dim() == 0:  # 标量
            # 使用默认ScanNet内参
            intr = torch.tensor([577.870605, 577.870605, 319.5, 239.5], 
                              dtype=xyz_cam.dtype, device=xyz_cam.device)
        elif intr.dim() > 2:  # 多维tensor，尝试展平
            intr = intr.flatten()
        
        # 确保是4个元素，如果不是则使用默认值
        if intr.numel() != 4:
            # 记录异常intrinsics用于调试
            if intr.numel() == 1:
                # 可能是错误的单值，使用默认值
                intr = torch.tensor([577.870605, 577.870605, 319.5, 239.5], 
                                  dtype=xyz_cam.dtype, device=xyz_cam.device)
            elif intr.numel() > 4:
                # 取前4个元素
                intr = intr[:4]
            else:
                # 其他情况，抛出详细错误信息
                raise ValueError(f"intrinsics异常: 期望4个元素[fx,fy,cx,cy], 实际得到{intr.numel()}个元素，"
                               f"值为{intr.tolist() if intr.numel() <= 10 else '太多元素'}, "
                               f"原始形状: {intr.shape}, cam_meta: {cam_meta}")
        
        # 最终验证
        assert intr.numel() == 4, f"内参处理后仍然异常: {intr.shape}"
        
        valid, uv = self.build_uv_index(xyz_cam_proj, intr, feat2d_map.shape[-2:])
        sampled2d = xyz_cam.new_zeros((xyz_cam.shape[0], 256))
        if valid.any():
            # 确保uv和feat2d_map的数据类型匹配
            if uv.dtype != feat2d_map.dtype:
                uv = uv.to(feat2d_map.dtype)
            f2d_vis = self.sample_img_feat(feat2d_map.unsqueeze(0), uv[valid])
            sampled2d[valid] = f2d_vis.to(sampled2d.dtype)  # 确保输出类型一致
        
        feat2d = self.lin2d(sampled2d)
        feat2d = self.ln2d(feat2d)

        # 特征融合
        f2d_final = self.lin2d_final(torch.cat([feat2d, pe], dim=-1))
        f3d_final = self.lin3d_final(torch.cat([feat3d, pe], dim=-1))

        # 使用Enhanced Gate进行融合
        if self.use_enhanced_gate:
            fused, conf = self.fusion_gate(f2d_final, f3d_final, xyz_world, valid)
        else:
            # 回退到简单的gate机制
            gate_input = torch.cat([f2d_final, f3d_final], dim=-1)
            gate = self.gate_mlp(gate_input)
            valid_weight = valid.float().unsqueeze(-1)
            gate = gate * valid_weight + 0.2 * (1 - valid_weight)
            fused = gate * f2d_final + (1 - gate) * f3d_final
            conf = gate

        return fused, conf, pe, clip_global

    def forward(self, points_list, imgs, cam_info):
        """支持 List 或 batched Tensor 输入，统一返回 List 结果。"""
        # 兼容性处理
        if torch.is_tensor(points_list):
            points_list = list(points_list)
        if torch.is_tensor(imgs):
            imgs = list(imgs)
        
        # 检查输入长度
        n_points = len(points_list)
        n_imgs = len(imgs)
        
        # 基本长度检查（数据预处理器应该已经处理了tuple展开）
        if n_points != n_imgs:
            raise RuntimeError(f"Length mismatch after preprocessing: points_list ({n_points}) != imgs ({n_imgs})")
        
        # 处理cam_info格式
        if isinstance(cam_info, dict):
            # 单个字典，复制给所有样本
            cam_info = [cam_info for _ in range(n_points)]
        elif isinstance(cam_info, list):
            # 已经是列表，确保长度匹配
            if len(cam_info) != n_points:
                # 如果长度不匹配，使用第一个元素填充
                first_info = cam_info[0] if cam_info else {}
                cam_info = [first_info for _ in range(n_points)]
        else:
            # 其他格式，使用默认值
            default_info = {'intrinsics': [577.870605, 577.870605, 319.5, 239.5], 'extrinsics': None}
            cam_info = [default_info for _ in range(n_points)]

        # 最终长度验证
        assert len(points_list) == len(imgs) == len(cam_info), \
            f"Final length check failed: points={len(points_list)}, imgs={len(imgs)}, cam_info={len(cam_info)}"

        # 批量CLIP处理（如果图像尺寸一致）
        feat2d_maps, clip_globals = None, None
        try:
            if all(img.shape == imgs[0].shape for img in imgs):
                imgs_batch = torch.stack(imgs, dim=0)
                with torch.no_grad():
                    amp_ctx = torch.cuda.amp.autocast(enabled=self.use_amp and imgs_batch.is_cuda)
                    with amp_ctx:
                        feat2d_maps, clip_globals = self.enhanced_clip(imgs_batch)
        except Exception as e:
            # 批量处理失败，回退到单独处理
            feat2d_maps = clip_globals = None

        # 逐样本处理
        feat_fusion_list, conf_list, pe_list, clip_global_list = [], [], [], []
        for idx, (pts, img, meta) in enumerate(zip(points_list, imgs, cam_info)):
            try:
                # 确保meta是字典格式
                if not isinstance(meta, dict):
                    meta = {'intrinsics': [577.870605, 577.870605, 319.5, 239.5], 'extrinsics': None}
                
                # 确保intrinsics存在
                if 'intrinsics' not in meta:
                    meta['intrinsics'] = [577.870605, 577.870605, 319.5, 239.5]  # ScanNet默认内参
                
                fmap = feat2d_maps[idx:idx+1] if feat2d_maps is not None else None
                cglb = clip_globals[idx] if clip_globals is not None else None
                fused, conf, pe, clip_global = self._process_single(pts, img, meta, 
                                                                  fmap.squeeze(0) if fmap is not None else None, 
                                                                  cglb)
                feat_fusion_list.append(fused)
                conf_list.append(conf)
                pe_list.append(pe)
                clip_global_list.append(clip_global)
            
            except Exception as e:
                raise e  # 重新抛出异常，不要跳过样本
        
        return {
            'feat_fusion': feat_fusion_list,
            'conf_2d': conf_list,
            'pe_xyz': pe_list,
            'clip_global': clip_global_list
        } 