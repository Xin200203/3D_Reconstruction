#!/usr/bin/env python3
"""
Quick visualization: project per-point DINO/CLIP features to 3D point cloud colors.

Example:
python vis_demo/vis_dino_point_color.py \
  --scene scene0000_00 --frame 200 \
  --data-root data/scannet200-sv \
  --out-dir vis_demo
"""

import argparse
from pathlib import Path
from typing import Tuple, Tuple as Tup

import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F

from oneformer3d.projection_utils import (
    SCANET_INTRINSICS,
    project_points_to_uv,
    sample_img_feat,
)


def load_points(bin_path: Path) -> np.ndarray:
    """Load .bin point cloud saved as float32 (N,6): xyz + rgb."""
    arr = np.fromfile(bin_path, dtype=np.float32)
    if arr.size % 6 != 0:
        raise RuntimeError(f"Unexpected point file size: {bin_path}, size={arr.size}")
    pts = arr.reshape(-1, 6)
    return pts


def load_clip_feat(path: Path) -> torch.Tensor:
    """Load DINO/CLIP feature (.pt). Accept tensor or dict{'pix': tensor}."""
    data = torch.load(path, map_location="cpu")
    if isinstance(data, dict):
        if "pix" in data and torch.is_tensor(data["pix"]):
            data = data["pix"]
        else:
            # pick first tensor value if exists
            for v in data.values():
                if torch.is_tensor(v):
                    data = v
                    break
    if not torch.is_tensor(data):
        data = torch.as_tensor(data)
    if data.dim() == 4:
        # if batched, take first
        data = data[0]
    if data.dim() != 3:
        raise RuntimeError(f"clip/dino feat has unexpected shape {tuple(data.shape)}")
    return data


def feat_to_rgb(feat: np.ndarray) -> np.ndarray:
    """Concerto-style PCA color mapping from high-dim features to RGB.

    This mirrors ``pca_color_from_nxc`` used in the 2D notebook:
      - per-vector L2 normalize
      - torch.pca_lowrank with q<=6
      - if q>=6: mix PC1-3 and PC4-6 with 0.6/0.4
      - per-channel min-max to [0,1], with slight brightness scaling.
    """
    if feat.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    x = torch.from_numpy(feat).float()  # (N, C)
    x = F.normalize(x, dim=1)
    C = x.shape[1]
    q = min(6, C)

    # low-rank PCA
    U, S, V = torch.pca_lowrank(x, center=True, q=q, niter=5)  # V: (C, q)
    proj = x @ V  # (N, q)

    if q >= 6:
        color = proj[:, :3] * 0.6 + proj[:, 3:6] * 0.4
    elif q >= 3:
        color = proj[:, :3]
    else:
        pad = torch.zeros(x.shape[0], 3 - q, device=x.device, dtype=x.dtype)
        color = torch.cat([proj, pad], dim=1)

    # per-channel min-max to [0,1], with brightness factor
    brightness = 1.2
    min_val, _ = color.min(dim=0, keepdim=True)
    max_val, _ = color.max(dim=0, keepdim=True)
    div = torch.clamp(max_val - min_val, min=1e-6)
    color = (color - min_val) / div * brightness
    color = color.clamp(0.0, 1.0)
    return color.cpu().numpy().astype(np.float32)


def build_colored_pcd(xyz: np.ndarray, rgb: np.ndarray) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb)
    return pcd


def project_feat_to_points(
    xyz: torch.Tensor,
    clip_feat: torch.Tensor,
    pose_cam2world: torch.Tensor,
    max_depth: float = 20.0,
    intrinsics: Tuple[float, float, float, float] = SCANET_INTRINSICS,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D feat map and sample features.

    Returns:
        feat2d: (N, C) sampled features.
        valid: (N,) bool mask indicating points truly seen by the camera.
    """
    # world -> cam
    pose_cam2world = pose_cam2world.to(dtype=xyz.dtype, device=xyz.device)
    w2c = torch.linalg.inv(pose_cam2world)
    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
    xyz_cam = (w2c @ xyz_h.t()).t()[:, :3]

    Hf, Wf = clip_feat.shape[-2], clip_feat.shape[-1]
    uv, valid = project_points_to_uv(
        xyz_cam, (Hf, Wf), max_depth=max_depth, standard_intrinsics=intrinsics
    )
    # grid_sample 不支持 half on CPU；统一到 xyz.device 且用 float32
    clip_feat = clip_feat.to(device=xyz.device, dtype=torch.float32)
    uv = uv.to(device=xyz.device, dtype=clip_feat.dtype)
    valid = valid.to(device=xyz.device)
    sampled = sample_img_feat(
        clip_feat.unsqueeze(0), uv, valid, align_corners=False
    )  # (N,C)
    return sampled.cpu().numpy(), valid.cpu().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="scene0000_00", help="scene id")
    parser.add_argument("--frame", type=int, default=200, help="frame index")
    parser.add_argument("--data-root", default="data/scannet200-sv", help="dataset root for points/poses")
    parser.add_argument("--out-dir", default="vis_demo", help="output directory")
    parser.add_argument(
        "--feat-path",
        default="/home/nebula/xxy/3D_Reconstruction/vis_demo/200_dinov2_feat.pt",
        help="path to pre-extracted DINO feature (.pt, CxH xW); overrides default clip_feat path",
    )
    parser.add_argument("--max-depth", type=float, default=20.0, help="max depth for projection")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame_str = str(args.frame)
    points_path = data_root / "points" / f"{args.scene}_{frame_str}.bin"
    pose_path = data_root / "pose_centered" / args.scene / f"{frame_str}.npy"
    # 默认 clip 路径（保持兼容），如提供 --feat-path 则优先使用
    default_clip_path = data_root / "clip_feat" / args.scene / f"{frame_str}.pt"
    feat_path = Path(args.feat_path) if args.feat_path is not None else default_clip_path

    if not points_path.exists():
        raise FileNotFoundError(points_path)
    if not pose_path.exists():
        raise FileNotFoundError(pose_path)
    if not feat_path.exists():
        raise FileNotFoundError(feat_path)

    pts = load_points(points_path)
    xyz = torch.from_numpy(pts[:, :3])
    clip_feat = load_clip_feat(feat_path)
    pose = torch.from_numpy(np.load(pose_path)).float()

    feat2d, valid = project_feat_to_points(
        xyz=xyz,
        clip_feat=clip_feat,
        pose_cam2world=pose,
        max_depth=args.max_depth,
    )
    valid = valid.astype(bool).reshape(-1)
    if valid.sum() < 10:
        print(f"[warn] Only {valid.sum()} valid projected points out of {valid.size}.")
    # Only use valid points to compute PCA mapping, then fill back
    rgb_dino = np.zeros((feat2d.shape[0], 3), dtype=np.float32)
    if valid.any():
        rgb_valid = feat_to_rgb(feat2d[valid])
        rgb_dino[valid] = rgb_valid

    # Save colored point cloud
    pcd_dino = build_colored_pcd(pts[:, :3], rgb_dino)
    out_ply = out_dir / f"{args.scene}_{frame_str}_dino_rgb.ply"
    o3d.io.write_point_cloud(str(out_ply), pcd_dino)
    print(f"Saved DINO-colored point cloud to {out_ply}")

    # Optionally also save original color for reference
    if pts.shape[1] >= 6:
        rgb_orig = pts[:, 3:6].copy()
        rgb_orig = np.clip(rgb_orig / 255.0, 0.0, 1.0)
        pcd_orig = build_colored_pcd(pts[:, :3], rgb_orig)
        out_ply_orig = out_dir / f"{args.scene}_{frame_str}_orig_rgb.ply"
        o3d.io.write_point_cloud(str(out_ply_orig), pcd_orig)
        print(f"Saved original-color point cloud to {out_ply_orig}")


if __name__ == "__main__":
    main()
