import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip
import math
from typing import List, Dict

import MinkowskiEngine as ME
from mmdet3d.registry import MODELS
from .mink_unet import Res16UNet34C
from .tiny_sa import TinySAModule, TinySA2D
from .clip_utils import (
    freeze_clip_except_last_blocks as _freeze_clip_except_last_blocks,
    build_uv_index as _build_uv_index,
    sample_img_feat as _sample_img_feat
)


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


class FusionGate(nn.Module):
    def __init__(self, in_channels: int = 192, out_channels: int = 96):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid())

    def forward(self, f2d, f3d):
        gate = self.mlp(torch.cat([f2d, f3d], dim=-1))
        return gate * f2d + (1 - gate) * f3d, gate.mean(dim=-1, keepdim=True)


class TinySANeck(nn.Module):
    """Two-layer self-attention neck for lowest res point features."""

    def __init__(self, dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.sa1 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.sa2 = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out, _ = self.sa1(x, x, x)
        x = self.norm(out + x)
        out, _ = self.sa2(x, x, x)
        x = self.norm(out + x)
        x = self.norm(self.ffn(x) + x)
        return x


@MODELS.register_module()
class BiFusionEncoder(nn.Module):
    """Bi-Fusion Encoder combining 2D CLIP visual features and 3D Sparse features."""

    def __init__(self,
                 clip_pretrained: str = 'openai',
                 voxel_size: float = 0.02,
                 freeze_blocks: int = 0,  # 默认完全冻结 CLIP
                 use_amp: bool = True,
                 use_tiny_sa_2d: bool = False):
        super().__init__()
        # 2D ViT
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained=clip_pretrained)
        self.clip_visual = clip_model.visual
        _freeze_clip_except_last_blocks(self.clip_visual, freeze_blocks)
        # reduce dimension to 256 after PixelShuffle handled outside (H/8)
        self.conv_reduce = nn.Conv2d(768 // 4, 256, kernel_size=1)
        # 2D linear to 128
        self.lin2d = nn.Sequential(nn.Linear(256, 128), nn.ReLU())
        self.use_tiny_sa_2d = use_tiny_sa_2d
        if use_tiny_sa_2d:
            self.tiny_sa_2d = TinySA2D(dim=256, num_heads=4)

        # 3D encoder without memory
        self.backbone3d = Res16UNet34C(in_channels=3, out_channels=128, config=dict(dilations=[1,1,1,1]), D=3)
        self.tiny_sa = TinySAModule(dim=128, num_heads=4, radius=0.3, max_k=32, sample_ratio=0.25)
        self.lin3d = nn.Sequential(nn.Linear(128, 128), nn.ReLU())

        # pe mapping
        self.pe_mlp = nn.Sequential(nn.Linear(64, 32), nn.ReLU())
        # dimension alignment
        self.lin2d96 = nn.Linear(160, 96)
        self.lin3d96 = nn.Linear(160, 96)
        # fusion gate
        self.fuse_gate = FusionGate()
        self.voxel_size = voxel_size
        self.use_amp = use_amp

    def build_uv_index(self, xyz_cam, intr, img_shape):
        return _build_uv_index(xyz_cam, intr, img_shape)

    def sample_img_feat(self, feat_map, uv):
        return _sample_img_feat(feat_map, uv)

    def _process_single(self, points: torch.Tensor, img: torch.Tensor, cam_meta: Dict):
        """处理单帧/单批数据，返回融合特征结果。"""
        # ==== 提取基础信息 ====
        xyz_cam = points[:, :3]
        # 若给出外参，则转换到 world 坐标系
        if cam_meta.get('extrinsics', None) is not None:
            extr = cam_meta['extrinsics']
            if not torch.is_tensor(extr):
                extr = torch.as_tensor(extr, dtype=xyz_cam.dtype, device=xyz_cam.device)
            # 保证形状 (4,4)
            if extr.shape == (3, 4):
                extr = torch.cat([extr, extr.new_tensor([[0, 0, 0, 1]])], dim=0)
            xyz_h = torch.cat([xyz_cam, xyz_cam.new_ones(xyz_cam.size(0), 1)], dim=-1)  # (N,4)
            xyz_world = (torch.inverse(extr) @ xyz_h.T).T[:, :3]
        else:
            xyz_world = xyz_cam

        # ===== 几何 PE =====
        bbox_size = torch.zeros_like(xyz_world)
        pose_delta = torch.zeros(9, device=xyz_world.device, dtype=xyz_world.dtype)
        height = xyz_world[:, 2:3]
        pe = self.pe_mlp(build_geo_pe(xyz_world, bbox_size, pose_delta, height))  # (N,32)

        # ===== 3D branch =====
        coords = torch.round(xyz_cam / self.voxel_size)
        feats = points[:, 3:6]
        field = ME.TensorField(coordinates=coords, features=feats)
        sparse_tensor = field.sparse()
        feat3d = self.backbone3d(sparse_tensor).slice(field).features  # (N,128)
        feat3d = self.tiny_sa(xyz_cam, feat3d)
        feat3d = self.lin3d(feat3d)

        # ===== 2D branch =====
        with torch.no_grad():
            amp_ctx = torch.cuda.amp.autocast(enabled=self.use_amp and img.is_cuda)
            with amp_ctx:
                clip_feat = self.clip_visual(img.unsqueeze(0))  # (1,768,h/16,w/16)
                up = F.pixel_shuffle(clip_feat, 2)            # (1,192,h/8,w/8)
                feat2d_map = self.conv_reduce(up)              # (1,256,h/8,w/8)
                if self.use_tiny_sa_2d:
                    feat2d_map = self.tiny_sa_2d(feat2d_map)
        clip_global = clip_feat.mean(dim=[2, 3]).squeeze(0)  # (768,)

        intr = cam_meta['intrinsics']  # fx,fy,cx,cy tensor / list
        if not torch.is_tensor(intr):
            intr = torch.as_tensor(intr, dtype=xyz_cam.dtype, device=xyz_cam.device)
        valid, uv = self.build_uv_index(xyz_cam, intr, feat2d_map.shape[-2:])
        sampled2d = xyz_cam.new_zeros((xyz_cam.shape[0], 256))
        if valid.any():
            f2d_vis = self.sample_img_feat(feat2d_map, uv[valid])
            sampled2d[valid] = f2d_vis
        feat2d = self.lin2d(sampled2d)  # (N,128)

        # ===== 融合 =====
        f2d_cat = torch.cat([feat2d, pe], dim=-1)  # 160
        f3d_cat = torch.cat([feat3d, pe], dim=-1)
        f2d96 = self.lin2d96(f2d_cat)
        f3d96 = self.lin3d96(f3d_cat)
        fused, conf = self.fuse_gate(f2d96, f3d96)
        return fused, conf, pe, clip_global

    def forward(self, points_list, imgs, cam_info):
        """支持 List 或 batched Tensor 输入，统一返回 List 结果。"""
        # --- 兼容性处理 ---
        if torch.is_tensor(points_list):  # (B,N,6)
            points_list = list(points_list)
        if torch.is_tensor(imgs):         # (B,3,H,W)
            imgs = list(imgs)
        if isinstance(cam_info, dict):    # 单 dict → 复制 B 份
            cam_info = [cam_info for _ in range(len(points_list))]

        feat_fusion_list, conf_list, pe_list, clip_global_list = [], [], [], []
        for pts, img, meta in zip(points_list, imgs, cam_info):
            fused, conf, pe, clip_global = self._process_single(pts, img, meta)
            feat_fusion_list.append(fused)
            conf_list.append(conf)
            pe_list.append(pe)
            clip_global_list.append(clip_global)
        return {
            'feat_fusion': feat_fusion_list,
            'conf_2d': conf_list,
            'pe_xyz': pe_list,
            'clip_global': clip_global_list
        } 