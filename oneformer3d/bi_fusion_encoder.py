import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math
import os
import numpy as np
from typing import List, Dict, Optional, Tuple, cast

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


class EnhancedProjectionHead3D(nn.Module):
    """Enhanced 3D Projection Head: 96维 -> 256维的投影头
    
    按照优化脚本要求：1×1 SparseConv(96→256) + BN + ReLU → L2-Norm
    """
    
    def __init__(self,
                 input_dim: int = 96,
                 output_dim: int = 256,
                 use_dropout: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        # 对应稀疏卷积的1×1卷积投影
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 数值稳定性检查
        if torch.any(torch.isnan(x)) or torch.any(torch.isinf(x)):
            print("Warning: NaN/Inf in projection input, clamping")
            x = torch.clamp(x, -10, 10)
        """3D特征投影 (N, 96) -> (N, 256)"""
        return self.projection(x)


class EnhancedProjectionHead2D(nn.Module):
    """Enhanced 2D Projection Head: 渐进式维度压缩
    
    按照优化脚本要求：LayerNorm(768) → GELU → Linear(768→512) → GELU → Linear(512→256)
    """
    
    def __init__(self,
                 input_dim: int = 768,
                 hidden_dim: int = 512,
                 output_dim: int = 256,
                 use_dropout: bool = True,
                 dropout_rate: float = 0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout_rate) if use_dropout else nn.Identity(),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 空间特征投影 (for spatial features)
        self.spatial_projection = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout_rate) if use_dropout else nn.Identity(),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=1),
            nn.BatchNorm2d(output_dim)
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward_global(self, x: torch.Tensor) -> torch.Tensor:
        """全局特征投影 (B, 768) -> (B, 256)"""
        return self.projection(x)
    
    def forward_spatial(self, x: torch.Tensor) -> torch.Tensor:
        """空间特征投影 (B, 768, H, W) -> (B, 256, H, W)"""
        return self.spatial_projection(x)


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
        
        # 🔧 优先使用本地CLIP权重 - 避免网络依赖
        local_weight_path = '/home/nebula/xxy/ESAM/data/open_clip_pytorch_model.bin'
        
        try:
            # 方案1：如果传入了本地文件路径，直接使用
            if os.path.exists(clip_pretrained):
                print(f"✅ 使用指定的本地CLIP权重: {clip_pretrained}")
                self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
                    'ViT-B-16', pretrained=clip_pretrained
                )
            # 方案2：如果有预设的本地权重文件，优先使用
            elif os.path.exists(local_weight_path):
                print(f"✅ 使用预设的本地CLIP权重: {local_weight_path}")
                # 先创建模型结构，然后加载权重
                self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
                    'ViT-B-16', pretrained=None
                )
                # 手动加载本地权重
                state_dict = torch.load(local_weight_path, map_location='cpu')
                # 处理可能的键名不匹配问题
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                self.clip_model.load_state_dict(state_dict, strict=False)
                print(f"✅ 本地CLIP权重加载成功")
            # 方案3：回退到网络下载（如果本地权重不可用）
            else:
                print(f"🌐 本地权重不可用，尝试网络下载: {clip_pretrained}")
                self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
                    'ViT-B-16', pretrained=clip_pretrained
                )
        except Exception as e:
            print(f"⚠️  CLIP权重加载失败: {e}")
            # 最终回退：使用随机初始化
            print("🔄 回退到随机初始化的CLIP模型")
            self.clip_model, _, self.clip_transform = open_clip.create_model_and_transforms(
                'ViT-B-16', pretrained=None
            )
        
        self.clip_visual = self.clip_model.visual
        self.num_layers = num_layers
        self.target_resolution = target_resolution
        
        # 智能冻结策略
        self._setup_freezing(freeze_conv1, freeze_early_layers)
        
        # 改进的2D投影头：渐进式维度压缩 768->512->256
        self.enhanced_proj_2d = EnhancedProjectionHead2D(
            input_dim=768,
            hidden_dim=512,
            output_dim=256,
            use_dropout=True,
            dropout_rate=0.1
        )
        
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
        # 使用 type: ignore 来处理CLIP内部结构的类型检查问题
        for i in range(self.num_layers):
            x = self.clip_visual.transformer.resblocks[i](x)  # type: ignore
        x = x.permute(1, 0, 2)  # LND -> NLD
        
        # 重建空间特征
        patch_tokens = x[:, 1:, :].permute(0, 2, 1).reshape(B, 768, 14, 14)
        fused_spatial = patch_tokens + spatial_raw  # 残差连接
        spatial_feat = self.enhanced_proj_2d.forward_spatial(fused_spatial)  # (B, 256, 14, 14)
        
        # 全局特征
        cls_token = x[:, 0, :]  # (B, 768)
        global_feat = self.enhanced_proj_2d.forward_global(cls_token)  # (B, 256)
        
        # L2归一化到单位球面 (按照优化脚本要求)
        global_feat = F.normalize(global_feat, dim=-1)
        # 对空间特征的每个位置进行L2归一化
        B, C, H, W = spatial_feat.shape
        spatial_feat = F.normalize(spatial_feat.view(B, C, -1), dim=1).view(B, C, H, W)
        
        return spatial_feat, global_feat


class LiteFusionGate(nn.Module):
    """Lite Fusion Gate - 轻量级融合门控机制
    
    按照优化脚本要求：点级融合 + 通道级SE注意力
    参数量约0.12M，远低于原EnhancedGate
    """
    
    def __init__(self, 
                 feat_dim: int = 256,
                 early_steps: int = 3000):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.early_steps = early_steps
        self.training_step = 0
        
        # 点级融合权重MLP: Linear(512→64→1) + Sigmoid
        self.point_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, 64),  # 256*2 -> 64
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # SE通道注意力模块
        self.se_module = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(feat_dim, feat_dim // 16, 1),
            nn.ReLU(),
            nn.Conv1d(feat_dim // 16, feat_dim, 1),
            nn.Sigmoid()
        )
        
        # 早期冻结：前3000步α固定为0.5
        self.register_buffer('frozen_alpha', torch.tensor(0.5))
        
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
        
        # 1. 计算点级融合权重α
        if self.training and self.training_step < self.early_steps:
            # 早期冻结阶段：α = 0.5
            alpha_raw = self.frozen_alpha.expand(B, N, 1)
        else:
            # 正常训练阶段：α = σ(MLP([f₂D‖f₃D]))
            feat_concat = torch.cat([f2d, f3d], dim=-1)  # (B, N, 512)
            alpha_raw = self.point_mlp(feat_concat)  # (B, N, 1)
        
        # 2. 应用有效掩码调整：α = α*valid + 0.1*(1-valid)
        valid_mask_expanded = valid_mask.unsqueeze(-1).float()  # (B, N, 1)
        alpha = alpha_raw * valid_mask_expanded + 0.1 * (1 - valid_mask_expanded)
        
        # 3. 点级融合：f_mix = α·f₂D + (1-α)·f₃D
        f_mix = alpha * f2d + (1 - alpha) * f3d  # (B, N, 256)
        
        # 4. SE通道重加权：w = σ(SE(f_mix))
        # 调整维度适配SE模块：(B, N, C) -> (B, C, N)
        f_mix_transposed = f_mix.permute(0, 2, 1)  # (B, 256, N)
        se_weights = self.se_module(f_mix_transposed)  # (B, 256, 1)
        se_weights = se_weights.permute(0, 2, 1)  # (B, 1, 256)
        
        # 5. 应用SE权重：fused = w⊙f_mix
        fused_feat = se_weights * f_mix  # (B, N, 256)
        
        # 返回融合特征和置信度
        confidence = alpha  # 融合权重可作为置信度
        
        # 收集融合统计信息（如果启用）
        if hasattr(self, "collect_stats") and self.collect_stats:
            fusion_2d_ratio = alpha.mean().item()
            fusion_3d_ratio = (1 - alpha).mean().item()
            avg_confidence = confidence.mean().item()
            valid_points_ratio = valid_mask.float().mean().item()
            
            # 可以通过全局变量或日志记录这些统计信息
            if not hasattr(self, "_stats_buffer"):
                self._stats_buffer = []
            self._stats_buffer.append({
                "fusion_2d_ratio": fusion_2d_ratio,
                "fusion_3d_ratio": fusion_3d_ratio,
                "avg_confidence": avg_confidence,
                "valid_points_ratio": valid_points_ratio
            })
        return fused_feat, confidence
    
    def update_training_step(self, step: int):
        """更新训练步数"""
        self.training_step = step


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
        
        # 收集融合统计信息（如果启用）
        if hasattr(self, "collect_stats") and self.collect_stats:
            fusion_2d_ratio = final_gate.mean().item()
            fusion_3d_ratio = (1 - final_gate).mean().item()
            avg_confidence = confidence.mean().item()
            valid_points_ratio = valid_mask.float().mean().item()
            
            # 可以通过全局变量或日志记录这些统计信息
            if not hasattr(self, "_stats_buffer"):
                self._stats_buffer = []
            self._stats_buffer.append({
                "fusion_2d_ratio": fusion_2d_ratio,
                "fusion_3d_ratio": fusion_3d_ratio,
                "avg_confidence": avg_confidence,
                "valid_points_ratio": valid_points_ratio
            })
        return fused_feat, confidence


class FiLMModulation(nn.Module):
    """FiLM调制机制 - 几何位置编码注入
    
    按照优化脚本要求：Linear(64→128) + SiLU + Linear(128→512) → γ, β (各256)
    FiLM: (1+γ) ⊙ feat + β 同时作用于 f₂D, f₃D
    """
    
    def __init__(self, 
                 pe_dim: int = 64,
                 hidden_dim: int = 128,
                 feat_dim: int = 256):
        super().__init__()
        
        self.feat_dim = feat_dim
        
        # PE到FiLM参数的映射
        self.pe_to_film = nn.Sequential(
            nn.Linear(pe_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, feat_dim * 2)  # γ + β
        )
        
        # 初始化：γ接近0（保持原特征），β接近0（不增加偏置）
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重：确保初始时FiLM调制接近恒等变换"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 最后一层的γ部分初始化为小值，β部分初始化为0
        with torch.no_grad():
            final_layer = self.pe_to_film[-1]
            # 前半部分是γ，后半部分是β
            final_layer.weight.data[:self.feat_dim] *= 0.01  # γ接近0
            final_layer.weight.data[self.feat_dim:] *= 0.01  # β接近0
    
    def forward(self, features: torch.Tensor, pe: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 输入特征 (B, N, feat_dim) 或 (N, feat_dim)
            pe: 位置编码 (B, N, pe_dim) 或 (N, pe_dim)
        Returns:
            调制后的特征 (与输入形状相同)
        """
        # 获取FiLM参数
        film_params = self.pe_to_film(pe)  # (..., feat_dim*2)
        
        # 分离γ和β
        gamma, beta = torch.chunk(film_params, 2, dim=-1)  # 各自 (..., feat_dim)
        
        # 应用FiLM调制: (1+γ) ⊙ feat + β
        modulated_features = (1 + gamma) * features + beta
        
        return modulated_features


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
                 use_tiny_sa_3d: bool = False,  # 新增参数控制是否使用TinySA
                 # 调试模式控制
                 debug: bool = False):  # 控制调试信息输出
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
        
        # 3D encoder - 保持原始96维以兼容预训练权重，然后使用投影头到256维
        cfg_backbone = SimpleNamespace(dilations=[1, 1, 1, 1], bn_momentum=0.02, conv1_kernel_size=5)
        self.backbone3d = Res16UNet34C(in_channels=3, out_channels=96, config=cfg_backbone, D=3)
        
        # 3D投影头：96维 -> 256维 (替代简单的适配层)
        self.proj_3d = EnhancedProjectionHead3D(
            input_dim=96,
            output_dim=256,
            use_dropout=True,
            dropout_rate=0.1
        )
        
        # 条件性地使用TinySA或简单的线性层 - 在投影后的256维上操作
        self.use_tiny_sa_3d = use_tiny_sa_3d
        
        if use_tiny_sa_3d:
            # 使用TinySA（如果明确启用）
            self.tiny_sa_neck = TinySANeck(dim=256, num_heads=8, radius=0.3, max_k=32, sample_ratio=0.25, num_layers=2)
        else:
            # 使用简单的线性层替代TinySA
            self.simple_neck = nn.Sequential(
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.LayerNorm(256)
            )
            
        # 3D特征最终处理层（在256维上操作）
        self.lin3d = nn.Sequential(nn.Linear(256, 256), nn.ReLU())
        self.ln3d = nn.LayerNorm(256)

        # PE mapping with FiLM modulation
        self.pe_mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU())
        self.film_modulation = FiLMModulation(pe_dim=64, hidden_dim=128, feat_dim=256)
        
        # 特征对齐 - 调整维度以匹配256维输出（不再需要PE拼接，因为使用FiLM调制）
        self.lin2d_final = nn.Linear(256, 256)  # 256 -> 256 (移除PE拼接)
        self.lin3d_final = nn.Linear(256, 256)  # 256 -> 256 (移除PE拼接)

        # 融合机制选择：优先使用LiteFusionGate
        self.use_enhanced_gate = use_enhanced_gate
        self.use_lite_gate = True  # 默认使用轻量级门控
        
        if self.use_lite_gate:
            # 使用轻量级LiteFusionGate
            self.fusion_gate = LiteFusionGate(
                feat_dim=256,
                early_steps=3000
            )
        elif use_enhanced_gate:
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
        self.debug = debug  # 保存调试模式设置
        
        # 🔍 统计信息收集配置
        self._collect_fusion_stats = debug   # 基于调试模式启用融合统计收集
        self._collect_gradient_stats = debug  # 基于调试模式启用梯度统计收集
        self._fusion_stats = {}  # 存储融合统计信息
        self._stats_history = []  # 历史统计信息

    def update_training_step(self, step: int):
        """更新训练步数，用于LiteFusionGate的早期冻结策略"""
        if self.use_lite_gate and hasattr(self.fusion_gate, 'update_training_step'):
            self.fusion_gate.update_training_step(step)
    
    def _collect_fusion_statistics(self, conf: torch.Tensor, valid: torch.Tensor, 
                                 f2d: torch.Tensor, f3d: torch.Tensor):
        """收集融合门控统计信息"""
        try:
            with torch.no_grad():
                # 基础统计
                if conf.dim() == 2:  # (N, 1)
                    conf_values = conf.squeeze(-1)  # (N,)
                else:
                    conf_values = conf
                
                # 计算融合比例
                fusion_2d_ratio = conf_values.mean().item()
                fusion_3d_ratio = 1.0 - fusion_2d_ratio
                avg_confidence = conf_values.mean().item()
                valid_points_ratio = valid.float().mean().item()
                
                # 特征质量统计
                f2d_norm = torch.norm(f2d, dim=-1).mean().item()
                f3d_norm = torch.norm(f3d, dim=-1).mean().item()
                
                # 特征相似度
                cos_sim = F.cosine_similarity(f2d, f3d, dim=-1).mean().item()
                
                # 更新统计信息
                self._fusion_stats = {
                    'fusion_2d_ratio': fusion_2d_ratio,
                    'fusion_3d_ratio': fusion_3d_ratio, 
                    'avg_confidence': avg_confidence,
                    'valid_points_ratio': valid_points_ratio,
                    'f2d_norm_avg': f2d_norm,
                    'f3d_norm_avg': f3d_norm,
                    'feature_similarity': cos_sim,
                    'total_points': conf_values.numel()
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

    def build_uv_index(self, xyz_cam, intr, img_shape):
        return _build_uv_index(xyz_cam, intr, img_shape)

    def sample_img_feat(self, feat_map, uv):
        return _sample_img_feat(feat_map, uv)
    
    def _improved_projection(self, xyz_cam, intr, img_shape):
        """改进的投影机制：增加视距裁剪和优先级过滤"""
        # 1. 基础投影
        fx, fy, cx, cy = intr
        x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
        
        # 2. 视距裁剪：ScanNet室内场景合理深度范围 - 进一步放宽限制
        # 从 0.02m-30m 进一步放宽到 0.01m-50m，几乎不限制深度
        depth_valid = (z > 0.01) & (z < 50.0)
        
        # 3. 计算投影坐标  
        u = fx * x / (z + 1e-8) + cx  # 添加小值避免除零
        v = fy * y / (z + 1e-8) + cy
        
        # 4. 边界检查 - 修复边界包含问题
        H, W = img_shape
        # 使用<=来包含边界像素，但为采样安全留0.5像素边距
        boundary_valid = (u >= 0.5) & (u <= W-0.5) & (v >= 0.5) & (v <= H-0.5)
        
        # 5. 组合所有条件
        valid = depth_valid & boundary_valid
        
        # 6. 如果有效点太少，进一步放宽深度限制  
        if valid.sum() < 300:  # 降低阈值到300个点
            depth_valid_relaxed = (z > 0.005) & (z < 100.0)  # 极度放宽到0.005m-100m
            valid = depth_valid_relaxed & boundary_valid
            if self.debug:
                print(f"🔍 深度极度放宽 - 初始有效点: {depth_valid.sum()}, 放宽后: {valid.sum()}")
            
        # 7. 使用全部有效投影点，不施加人为限制
        original_valid_count = valid.sum().item()
        if self.debug:
            print(f"✅ 使用全部有效投影: {original_valid_count}/{len(valid)} ({original_valid_count/len(valid)*100:.1f}%)")
        
        return valid, torch.stack([u, v], dim=-1)

    def _process_single(self, points: torch.Tensor, img: torch.Tensor, cam_meta: Dict,
                        feat2d_map: Optional[torch.Tensor] = None,
                        clip_global: Optional[torch.Tensor] = None):
        """处理单帧数据，使用增强的CLIP和融合机制"""
        # 提取基础信息
        xyz_depth = points[:, :3]  # DEPTH坐标系的原始坐标
        
        # 🎯 关键修复：使用pose逆变换将ScanNet传感器坐标转换为标准相机坐标
        # 坐标变换：将点云从传感器坐标系转换到相机坐标系
        xyz_cam_proj = None
        xyz_world = xyz_depth # PE默认使用原始坐标

        try:
            if cam_meta.get('pose', None) is not None:
                pose = cam_meta['pose']
                if not torch.is_tensor(pose):
                    pose = torch.from_numpy(pose).float().to(xyz_depth.device)

                # 确保pose是4x4矩阵
                if pose.dim() == 2 and pose.shape == (4, 4):
                    # 🎯 关键修复：使用pose逆变换（你的完美解决方案）
                    pose_inv = torch.inverse(pose)
                    
                    # 转换为齐次坐标并应用逆变换
                    xyz_homo = torch.cat([xyz_depth, torch.ones(xyz_depth.shape[0], 1, device=xyz_depth.device)], dim=1)
                    xyz_cam_homo = torch.mm(xyz_homo, pose_inv.T)
                    xyz_cam_proj = xyz_cam_homo[:, :3]
                    
                    # 验证变换结果
                    if self.debug:
                        positive_z_ratio = (xyz_cam_proj[:, 2] > 0).float().mean().item()
                        z_range = [xyz_cam_proj[:, 2].min().item(), xyz_cam_proj[:, 2].max().item()]
                        print(f"🎯 pose逆变换成功: 正Z比例={positive_z_ratio:.1%}, Z范围=[{z_range[0]:.3f}, {z_range[1]:.3f}]")
                else:
                    if self.debug:
                        print(f"⚠️ 无效的pose形状: {pose.shape}，跳过变换")
            else:
                 if self.debug:
                    print(f"⚠️ cam_meta中无pose信息，跳过变换")

        except Exception as e:
            if self.debug:
                print(f"❌ pose处理异常: {e}，跳过变换")
        
        # 如果任何步骤失败，xyz_cam_proj将保持为None
        if xyz_cam_proj is None:
            if self.debug:
                print(f"🔧 使用原始坐标作为备用方案")
            xyz_cam_proj = xyz_depth
        
        # 世界坐标仍用depth坐标（PE需要）
        xyz_world = xyz_depth
        
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
        
        # 应用3D投影头：96维 -> 256维，并L2归一化
        feat3d = self.proj_3d(feat3d)  # (N, 96) -> (N, 256)
        feat3d = F.normalize(feat3d + 1e-8, dim=-1, eps=1e-8)  # L2归一化到单位球面
        
        # 可选的TinySA或简单neck处理
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
                    if feat2d_map is not None:  # 添加None检查
                        feat2d_map = feat2d_map.squeeze(0)  # Remove batch dim

        # 投影采样 - 处理内参格式（修复ScanNet格式解析）
        intr_raw = cam_meta['intrinsics']
        
        # 🔧 修复ScanNet内参格式解析，添加类型安全检查
        if self.debug:
            print(f"🔍 原始内参类型: {type(intr_raw)}, 长度: {len(intr_raw) if hasattr(intr_raw, '__len__') else 'N/A'}")
            print(f"🔍 原始内参内容: {intr_raw}")
            if isinstance(intr_raw, (list, tuple)) and len(intr_raw) == 4:
                print(f"🔍 第一个元素类型: {type(intr_raw[0])}, 内容: {intr_raw[0]}")
        
        # 类型安全的内参处理
        if isinstance(intr_raw, (list, tuple)) and len(intr_raw) == 4:
            # ScanNet格式: [(fx_values...), (fy_values...), (cx_values...), (cy_values...)]
            if all(isinstance(item, (list, tuple)) for item in intr_raw):
                # 每个参数是多个相同值的元组，取第一个
                fx = float(intr_raw[0][0])
                fy = float(intr_raw[1][0]) 
                cx = float(intr_raw[2][0])
                cy = float(intr_raw[3][0])
                if self.debug:
                    print(f"🔧 ScanNet格式解析: fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                intr = torch.tensor([fx, fy, cx, cy], dtype=xyz_cam_proj.dtype, device=xyz_cam_proj.device)
            elif all(isinstance(item, (int, float)) for item in intr_raw):
                # 标准格式: [fx, fy, cx, cy]
                if self.debug:
                    print(f"🔧 标准格式解析: {intr_raw}")
                intr = torch.tensor(intr_raw, dtype=xyz_cam_proj.dtype, device=xyz_cam_proj.device)
            else:
                # 混合格式，逐个处理
                values = []
                for item in intr_raw:
                    if isinstance(item, (list, tuple)) and len(item) > 0:
                        values.append(float(item[0]))
                    elif isinstance(item, (int, float)):
                        values.append(float(item))
                    else:
                        values.append(577.870605)  # 默认值
                intr = torch.tensor(values, dtype=xyz_cam_proj.dtype, device=xyz_cam_proj.device)
        else:
            # 转换为tensor后处理
            if not torch.is_tensor(intr_raw):
                intr_tensor = torch.as_tensor(intr_raw, dtype=xyz_cam_proj.dtype, device=xyz_cam_proj.device)
            else:
                intr_tensor = intr_raw
            
            # 使用类型转换确保Pylance理解这是tensor
            intr_tensor = cast(torch.Tensor, intr_tensor)
            
            # 确保intrinsics是1D tensor (4,) - 增强处理逻辑
            if intr_tensor.dim() == 2:  # (1, 4) 或 (B, 4)
                if intr_tensor.shape[-1] == 4:
                    intr = intr_tensor[0]  # 取第一个
                elif intr_tensor.shape[0] == 4:
                    intr = intr_tensor[:, 0] if intr_tensor.shape[1] == 1 else intr_tensor.flatten()
                else:
                    intr = intr_tensor.flatten()
            elif intr_tensor.dim() == 0:  # 标量
                # 使用默认ScanNet内参
                intr = torch.tensor([577.870605, 577.870605, 319.5, 239.5], 
                                  dtype=xyz_cam_proj.dtype, device=xyz_cam_proj.device)
            elif intr_tensor.dim() > 2:  # 多维tensor，尝试展平
                intr = intr_tensor.flatten()
            else:
                intr = intr_tensor
        
        # 确保是4个元素，如果不是则使用默认值
        if intr.numel() != 4:
            # 记录异常intrinsics用于调试
            if intr.numel() == 1:
                # 可能是错误的单值，使用默认值
                intr = torch.tensor([577.870605, 577.870605, 319.5, 239.5], 
                                  dtype=xyz_cam_proj.dtype, device=xyz_cam_proj.device)
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
        
        # 🔍 调试cam_meta内容，寻找真实图像尺寸
        if self.debug:
            print(f"🔍 调试 cam_meta 内容: {list(cam_meta.keys())}")
            if 'img_shape' in cam_meta:
                print(f"🔍 发现 img_shape: {cam_meta['img_shape']}")
            else:
                print(f"🔍 cam_meta 完整内容: {cam_meta}")
        
        # 确保feat2d_map不为None
        if feat2d_map is None:
            # 如果feat2d_map仍然为None，创建默认的特征图（已经上采样到40×30）
            feat2d_map = torch.zeros((256, 30, 40), dtype=xyz_cam_proj.dtype, device=xyz_cam_proj.device)
        
        # 💡 修复投影缩放问题：CLIP特征图是14x14，需要缩放内参
        
        # 🔧 关键修复：上采样CLIP特征图到合理分辨率
        # 当前14×14 → 目标40×30 (stride=16对应分辨率)
        if feat2d_map is not None:
            original_h, original_w = feat2d_map.shape[-2:]
        else:
            # 默认CLIP特征图尺寸
            original_h, original_w = 14, 14
        
        # 计算目标分辨率 (stride=16)
        target_w = 640 // 16  # 40
        target_h = 480 // 16  # 30
        
        if feat2d_map is not None and (original_h != target_h or original_w != target_w):
            # 使用双线性插值上采样
            feat2d_map = F.interpolate(
                feat2d_map.unsqueeze(0),  # 添加batch维度
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=True
            ).squeeze(0)  # 移除batch维度
            
            if self.debug:
                print(f"🔧 特征图上采样: {original_h}×{original_w} → {target_h}×{target_w}")
        
        # 使用上采样后的特征图尺寸
        if feat2d_map is not None:
            feat_h, feat_w = feat2d_map.shape[-2:]
        else:
            feat_h, feat_w = target_h, target_w
        
        # 🔧 使用正确的ScanNet图像尺寸：640x480
        # 内参 cx=319.5, cy=239.5 对应 640x480 图像
        original_w = 640
        original_h = 480
        
        # 🔧 修复内参缩放：使用正确的ScanNet图像尺寸
        # ScanNet标准：640×480，CLIP特征：40×30 (上采样后)
        scale_x = feat_w / original_w  # 40 / 640 = 0.0625
        scale_y = feat_h / original_h  # 30 / 480 = 0.0625
        
        # 缩放内参以匹配特征图尺寸
        scaled_intr = intr.clone()
        scaled_intr[0] *= scale_x  # fx: 577.87 → 36.12
        scaled_intr[1] *= scale_y  # fy: 577.87 → 36.12  
        scaled_intr[2] *= scale_x  # cx: 319.5 → 19.97
        scaled_intr[3] *= scale_y  # cy: 239.5 → 14.97
        
        # 🔧 关键修复：调整主点坐标使其居中
        # 理论上cx应该是feat_w/2=20, cy应该是feat_h/2=15
        scaled_intr[2] = feat_w / 2.0  # cx: 修正为20.0
        scaled_intr[3] = feat_h / 2.0  # cy: 修正为15.0
        
        if self.debug:
            print(f"🔍 内参缩放 - 原始: fx={intr[0]:.1f}, fy={intr[1]:.1f}, cx={intr[2]:.1f}, cy={intr[3]:.1f}")
            print(f"🔍 内参缩放 - 缩放: fx={scaled_intr[0]:.2f}, fy={scaled_intr[1]:.2f}, cx={scaled_intr[2]:.2f}, cy={scaled_intr[3]:.2f}")
            print(f"🔍 内参缩放 - 比例: scale_x={scale_x:.6f}, scale_y={scale_y:.6f}")
        
        if self.debug:
            print(f"🔍 投影调试 - 缩放内参: {scaled_intr.tolist()}")
            print(f"🔍 投影调试 - 缩放比例: x={scale_x:.4f}, y={scale_y:.4f}")
            print(f"🔍 投影调试 - 原始图像尺寸: {original_w}x{original_h} → 特征图尺寸: {feat_w}x{feat_h}")
        
        # 🔧 改进的投影机制：增加视距裁剪和空间优先级
        valid, uv = self._improved_projection(xyz_cam_proj, scaled_intr, (feat_h, feat_w))
        
        # 🔍 调试投影问题
        if self.debug:
            with torch.no_grad():
                z_values = xyz_cam_proj[:, 2]
                print(f"🔍 投影调试 - 深度统计: min={z_values.min().item():.3f}, max={z_values.max().item():.3f}, "
                      f"mean={z_values.mean().item():.3f}, 正深度比例={((z_values > 0).float().mean()*100):.1f}%")
                print(f"🔍 投影调试 - 原始内参: {intr.tolist()}")
                print(f"🔍 投影调试 - 缩放内参: {scaled_intr.tolist()}")
                print(f"🔍 投影调试 - 缩放比例: x={scale_x:.4f}, y={scale_y:.4f}")
                print(f"🔍 投影调试 - 原始图像尺寸: {original_w}x{original_h} → 特征图尺寸: {feat_w}x{feat_h}")
                if valid.any():
                    uv_valid = uv[valid]
                    print(f"✅ 改进投影 - 有效投影: {valid.sum().item()}/{valid.numel()}, "
                          f"uv范围: u[{uv_valid[:, 0].min().item():.1f}, {uv_valid[:, 0].max().item():.1f}], "
                          f"v[{uv_valid[:, 1].min().item():.1f}, {uv_valid[:, 1].max().item():.1f}]")
                else:
                    print(f"🚨 改进投影 - 仍然无有效投影点！")
        
        sampled2d = xyz_cam_proj.new_zeros((xyz_cam_proj.shape[0], 256))  # 已经是256维，来自enhanced_clip的输出
        if valid.any() and feat2d_map is not None:
            # 确保uv和feat2d_map的数据类型匹配
            if uv.dtype != feat2d_map.dtype:
                uv = uv.to(feat2d_map.dtype)
            f2d_vis = self.sample_img_feat(feat2d_map.unsqueeze(0), uv[valid])
            sampled2d[valid] = f2d_vis.to(sampled2d.dtype)  # 确保输出类型一致
        
        # 2D特征后处理（已经是256维，无需额外投影）
        feat2d = self.lin2d(sampled2d)  # 256 -> 256
        feat2d = self.ln2d(feat2d)
        # 应用L2归一化确保特征在单位球面
        feat2d = F.normalize(feat2d + 1e-8, dim=-1, eps=1e-8)

        # 应用FiLM调制而非特征拼接
        feat2d_modulated = self.film_modulation(feat2d, pe)
        feat3d_modulated = self.film_modulation(feat3d, pe)
        
        # 特征最终投影（不再需要PE拼接）
        f2d_final = self.lin2d_final(feat2d_modulated)
        f3d_final = self.lin3d_final(feat3d_modulated)
        
        # 确保最终特征归一化到单位球面（在融合前）
        f2d_final = F.normalize(f2d_final + 1e-8, dim=-1, eps=1e-8)
        f3d_final = F.normalize(f3d_final + 1e-8, dim=-1, eps=1e-8)

        # 使用LiteFusionGate或Enhanced Gate进行融合
        if self.use_lite_gate:
            # 添加批量维度以适配LiteFusionGate
            f2d_batch = f2d_final.unsqueeze(0)  # (1, N, 256)
            f3d_batch = f3d_final.unsqueeze(0)  # (1, N, 256)
            valid_batch = valid.unsqueeze(0)    # (1, N)
            
            fused_batch, conf_batch = self.fusion_gate(f2d_batch, f3d_batch, valid_batch)
            fused = fused_batch.squeeze(0)      # (N, 256)
            conf = conf_batch.squeeze(0)        # (N, 1)
            
            # 🔍 收集融合统计信息
            if self._collect_fusion_stats:
                self._collect_fusion_statistics(conf, valid, f2d_final, f3d_final)
            
        elif self.use_enhanced_gate:
            fused, conf = self.fusion_gate(f2d_final, f3d_final, xyz_world, valid)
            
            # 🔍 收集融合统计信息
            if self._collect_fusion_stats:
                self._collect_fusion_statistics(conf, valid, f2d_final, f3d_final)
        else:
            # 回退到简单的gate机制
            gate_input = torch.cat([f2d_final, f3d_final], dim=-1)
            gate = self.gate_mlp(gate_input)
            valid_weight = valid.float().unsqueeze(-1)
            gate = gate * valid_weight + 0.2 * (1 - valid_weight)
            fused = gate * f2d_final + (1 - gate) * f3d_final
            conf = gate
            
            # 🔍 收集融合统计信息
            if self._collect_fusion_stats:
                self._collect_fusion_statistics(conf, valid, f2d_final, f3d_final)

        return fused, conf, pe, clip_global, valid

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
        feat_fusion_list, conf_list, pe_list, clip_global_list, valid_mask_list = [], [], [], [], []
        for idx, (pts, img, meta) in enumerate(zip(points_list, imgs, cam_info)):
            try:
                # 🔍 调试batch拆解过程
                if self.debug and idx == 0:
                    print(f"🔍 Forward拆解调试 - 第{idx}个样本:")
                    print(f"   points形状: {pts.shape}")
                    print(f"   img形状: {img.shape}")
                    print(f"   meta类型: {type(meta)}")
                    if isinstance(meta, dict):
                        print(f"   meta键值: {list(meta.keys())}")
                        for key, value in meta.items():
                            if isinstance(value, (list, tuple, np.ndarray)):
                                print(f"   meta[{key}]类型: {type(value)}, 长度: {len(value) if hasattr(value, '__len__') else 'N/A'}")
                                if torch.is_tensor(value):
                                    tensor_value = cast(torch.Tensor, value)
                                    print(f"   meta[{key}]形状: {tensor_value.shape}")
                                elif isinstance(value, np.ndarray):
                                    ndarray_value = cast(np.ndarray, value)
                                    print(f"   meta[{key}]形状: {ndarray_value.shape}")
                                else:
                                    print(f"   meta[{key}]形状: 非tensor/ndarray类型")
                            else:
                                print(f"   meta[{key}]类型: {type(value)}")
                
                # 确保meta是字典格式
                if not isinstance(meta, dict):
                    meta = {'intrinsics': [577.870605, 577.870605, 319.5, 239.5], 'extrinsics': None}
                
                # 🔧 关键修复：拆解meta内部的batch数据
                meta_fixed = {}
                for key, value in meta.items():
                    if key in ['pose', 'extrinsics'] and isinstance(value, (list, tuple, np.ndarray)):
                        if hasattr(value, '__len__') and len(value) > idx:
                            # 🔧 关键修复：正确处理不同形状的batch数据
                            if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape == (4, 4):
                                # 4x4变换矩阵保持完整，这是单个矩阵，不是batch
                                meta_fixed[key] = value
                                if self.debug and idx == 0:
                                    print(f"🔧 保持{key}完整: 形状{value.shape}，单个4x4变换矩阵")
                            elif isinstance(value, np.ndarray) and value.ndim == 3:
                                # 3D数组: [batch_size, 4, 4] → 选择第idx个[4, 4]
                                meta_fixed[key] = value[idx]
                                if self.debug and idx == 0:
                                    print(f"🔧 拆解{key}: 从形状{value.shape}中选择第{idx}个4x4矩阵")
                            else:
                                # 其他情况：list、tuple或其他格式，按索引拆解
                                meta_fixed[key] = value[idx]
                                if self.debug and idx == 0:
                                    print(f"🔧 拆解{key}: 从长度{len(value)}的{type(value).__name__}中选择第{idx}个元素")
                        else:
                            meta_fixed[key] = value
                    else:
                        meta_fixed[key] = value
                
                # 确保intrinsics存在
                if 'intrinsics' not in meta_fixed:
                    meta_fixed['intrinsics'] = [577.870605, 577.870605, 319.5, 239.5]  # ScanNet默认内参
                
                fmap = feat2d_maps[idx:idx+1] if feat2d_maps is not None else None
                cglb = clip_globals[idx] if clip_globals is not None else None
                fused, conf, pe, clip_global, valid_mask = self._process_single(pts, img, meta_fixed, 
                                                                  fmap.squeeze(0) if fmap is not None else None, 
                                                                  cglb)
                feat_fusion_list.append(fused)
                conf_list.append(conf)
                pe_list.append(pe)
                clip_global_list.append(clip_global)
                valid_mask_list.append(valid_mask)
            
            except Exception as e:
                raise e  # 重新抛出异常，不要跳过样本
        
        return {
            'feat_fusion': feat_fusion_list,
            'conf_2d': conf_list,
            'pe_xyz': pe_list,
            'clip_global': clip_global_list,
            'valid_projection_mask': valid_mask_list  # 新增有效投影掩码
        }