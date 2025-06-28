import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

class TinySAModule(nn.Module):
    """Tiny Self-Attention with ball-query downsample & nearest upsample.

    Args:
        dim (int): feature dimension
        num_heads (int): attention heads
        radius (float): ball query radius in meters
        max_k (int): maximum neighbors per center
        sample_ratio (float): ratio of points kept as centers (0‒1)
    """
    def __init__(self, dim: int = 128, num_heads: int = 4, radius: float = 0.3, max_k: int = 32, sample_ratio: float = 0.25):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.radius = radius
        self.max_k = max_k
        self.sample_ratio = sample_ratio
        self.scale = (dim // num_heads) ** -0.5
        self.proj_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj_out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor):
        """Args:
            xyz: (N,3) tensor of coordinates in meters.
            feats: (N,C) tensor of features.
        Returns:
            (N,C) updated features (residual added).
        """
        device = xyz.device
        N = xyz.size(0)
        # 1. sample centers via random or fps
        M = max(1, int(N * self.sample_ratio))
        idx_center = torch.randperm(N, device=device)[:M]
        center_xyz = xyz[idx_center]  # (M,3)
        center_feat = feats[idx_center]  # (M,C)

        # 2. radius search (center to all points) -> neighbor mask
        dist2 = torch.cdist(center_xyz, xyz)  # (M,N)
        nbr_mask = dist2 < self.radius
        # ensure at least one neighbor (include self)
        nbr_mask[torch.arange(M, device=device), idx_center] = True

        # 3. attention per center (loop for memory efficiency)
        updated_center = torch.empty_like(center_feat)
        for i in range(M):
            nbr_idx = torch.nonzero(nbr_mask[i])[:, 0]
            if nbr_idx.numel() > self.max_k:
                nbr_idx = nbr_idx[:self.max_k]
            qkv = self.proj_qkv(torch.cat([center_feat[i:i+1], feats[nbr_idx]], dim=0))  # (1+k,3C)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q[0:1]  # (1,C)
            k = k[1:]
            v = v[1:]
            # reshape heads
            q = q.view(self.num_heads, -1, self.dim // self.num_heads)  # (h,1,d)
            k = k.view(-1, self.num_heads, self.dim // self.num_heads).transpose(0,1)  # (h,k,d)
            v = v.view(-1, self.num_heads, self.dim // self.num_heads).transpose(0,1)  # (h,k,d)
            attn = (q * self.scale) @ k.transpose(-2, -1)  # (h,1,k)
            attn = F.softmax(attn, dim=-1)
            out = (attn @ v).transpose(0,1).reshape(1, self.dim)  # (1,C)
            updated_center[i] = out.squeeze(0)
        updated_center = self.proj_out(updated_center)
        center_feat = center_feat + self.norm(updated_center)
        center_feat = center_feat + self.norm(self.ffn(center_feat))

        # 4. upsample to all points via nearest center
        nearest_center_idx = dist2.argmin(dim=0)  # (N,)
        output_feats = center_feat[nearest_center_idx]  # (N,C)
        # residual connection
        return feats + output_feats 

class TinySA2D(nn.Module):
    """Tiny Self-Attention for 2D feature map (flatten–MHSA–reshape).

    Args:
        dim (int): channel dimension.
        num_heads (int): attention heads.
    """
    def __init__(self, dim: int = 256, num_heads: int = 4):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.ReLU(), nn.Linear(dim * 4, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor):
        """x: (B,C,H,W)"""
        B, C, H, W = x.shape
        tokens = x.flatten(2).permute(0, 2, 1)  # (B,HW,C)
        out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.norm1(tokens + out)
        tokens = self.norm2(tokens + self.ffn(tokens))
        x = tokens.permute(0, 2, 1).view(B, C, H, W)
        return x 