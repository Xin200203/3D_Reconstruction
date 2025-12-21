#!/usr/bin/env python3
import argparse
from pathlib import Path
import math

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


def pca_color_from_nxc(x: torch.Tensor,
                       brightness: float = 1.2,
                       center: bool = True) -> torch.Tensor:
    """
    x: [N, C] tensor
    return: [N, 3] tensor, each channel in [0,1]
    """
    x = x.float()
    # 每个位置做 L2 归一化，更符合 DINO 的用法
    x = F.normalize(x, dim=1)

    C = x.shape[1]
    q = min(6, C)
    # 低秩 PCA（Concerto 的做法）
    U, S, V = torch.pca_lowrank(x, center=center, q=q, niter=5)  # V: [C,q]
    proj = x @ V  # [N, q]

    if q >= 6:
        color = proj[:, :3] * 0.6 + proj[:, 3:6] * 0.4
    elif q >= 3:
        color = proj[:, :3]
    else:
        pad = torch.zeros(x.shape[0], 3 - q, device=x.device, dtype=x.dtype)
        color = torch.cat([proj, pad], dim=1)

    # 每个通道做 min-max 归一化
    min_val, _ = color.min(dim=0, keepdim=True)
    max_val, _ = color.max(dim=0, keepdim=True)
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (color - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color


def save_img_from_chw(feat_chw: torch.Tensor, out_path: Path):
    """
    feat_chw: [C,H,W]
    """
    C, H, W = feat_chw.shape
    x = feat_chw.reshape(C, -1).T   # [N,C]
    color = pca_color_from_nxc(x)   # [N,3]
    img = color.view(H, W, 3).cpu().numpy()
    img = np.clip(img, 0.0, 1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(out_path, img)
    print(f"  saved {out_path}   shape={H}x{W}, C={C}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat-path", required=True,
                        help="path to saved DINO/CLIP feature (.pt/.pth)")
    parser.add_argument("--out-dir", default="vis_pca_2d",
                        help="output directory")
    args = parser.parse_args()

    feat_path = Path(args.feat_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    obj = torch.load(feat_path, map_location="cpu")
    if isinstance(obj, dict):
        # 常见 key 尝试一下
        for k in ["feat", "clip_feat", "clip_pix", "dino_feat", "x"]:
            if k in obj and torch.is_tensor(obj[k]):
                feat = obj[k]
                print(f"Using dict[{k}] as feature")
                break
        else:
            raise ValueError(f"dict does not contain a tensor feature, keys={list(obj.keys())}")
    else:
        feat = obj

    feat = feat.float()
    print(f"Loaded feature shape={tuple(feat.shape)}, dtype={feat.dtype}")

    # 处理 batch 维
    if feat.dim() == 4:
        # 假设 [B,C,H,W] 或 [B,H,W,C]，只取第 0 个
        feat = feat[0]
        print(f"Using first batch -> shape={tuple(feat.shape)}")

    if feat.dim() == 3:
        d0, d1, d2 = feat.shape
        print(f"3D feature, trying all permutations as [C,H,W], original shape={feat.shape}")

        # 三种排列假设：原样 / 把 dim1 当 H / 把 dim2 当 H
        perms = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
        for p in perms:
            chw = feat.permute(p).contiguous()
            C, H, W = chw.shape
            if H < 4 or W < 4:
                continue
            out_path = out_dir / f"pca_perm{p[0]}{p[1]}{p[2]}.png"
            print(f"perm {p} -> [C,H,W]={chw.shape}")
            save_img_from_chw(chw, out_path)

    elif feat.dim() == 2:
        # [N_tokens, C]：尝试找一个接近方形的 (H,W)
        N, C = feat.shape
        root = int(math.sqrt(N))
        H = None
        for delta in range(0, 20):
            for h in (root - delta, root + delta):
                if h >= 4 and h <= N // 4 and N % h == 0:
                    H = h
                    break
            if H is not None:
                break
        if H is None:
            H = root
        W = N // H
        print(f"2D token feature: N={N}, C={C} -> using H={H}, W={W}")
        color = pca_color_from_nxc(feat)
        img = color[:H * W].view(H, W, 3).cpu().numpy()
        img = np.clip(img, 0.0, 1.0)
        out_path = out_dir / f"pca_tokens_{H}x{W}.png"
        plt.imsave(out_path, img)
        print(f"  saved {out_path}")
    else:
        raise ValueError(f"Unsupported feature dim={feat.dim()}")


if __name__ == "__main__":
    main()
