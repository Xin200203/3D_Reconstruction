#!/usr/bin/env python
import torch
import torch.nn.functional as F
from typing import Tuple

SCANET_INTRINSICS = (577.870605, 577.870605, 319.5, 239.5)
BASE_IMAGE_SIZE = (480.0, 640.0)
MIN_DEPTH = 0.3


def _scale_intrinsics(standard_intrinsics: Tuple[float, float, float, float],
                      feat_hw: Tuple[int, int]) -> Tuple[float, float, float, float]:
    fx0, fy0, cx0, cy0 = standard_intrinsics
    Hf, Wf = feat_hw
    base_h, base_w = BASE_IMAGE_SIZE
    scale_w = Wf / base_w
    scale_h = Hf / base_h
    fx_feat = fx0 * scale_w
    fy_feat = fy0 * scale_h
    cx_feat = cx0 * scale_w
    cy_feat = cy0 * scale_h
    return fx_feat, fy_feat, cx_feat, cy_feat


def pixels_to_grid(uv_feat: torch.Tensor,
                   feat_hw: Tuple[int, int],
                   align_corners: bool = True) -> torch.Tensor:
    H, W = feat_hw
    u = uv_feat[:, 0]
    v = uv_feat[:, 1]
    if align_corners:
        x_norm = 2.0 * u / max(float(W - 1), 1.0) - 1.0
        y_norm = 2.0 * v / max(float(H - 1), 1.0) - 1.0
    else:
        x_norm = 2.0 * (u + 0.5) / float(W) - 1.0
        y_norm = 2.0 * (v + 0.5) / float(H) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).view(1, -1, 1, 2)
    return grid


def sample_img_feat(feat_map: torch.Tensor,
                     uv_feat: torch.Tensor,
                     valid_mask: torch.Tensor,
                     align_corners: bool = True) -> torch.Tensor:
    assert feat_map.dim() == 4 and feat_map.size(0) == 1
    H, W = feat_map.shape[-2], feat_map.shape[-1]
    idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return feat_map.new_zeros((uv_feat.size(0), feat_map.size(1)))
    uv_valid = uv_feat[idx]
    grid = pixels_to_grid(uv_valid, (H, W), align_corners=align_corners)
    if grid.dtype != feat_map.dtype:
        grid = grid.to(feat_map.dtype)
    sampled = F.grid_sample(
        feat_map, grid, mode='bilinear', align_corners=align_corners
    ).squeeze(3).squeeze(0).T
    out = feat_map.new_zeros((uv_feat.size(0), feat_map.size(1)))
    out[idx] = sampled
    return out


def project_points_to_uv(xyz_cam: torch.Tensor,
                         feat_hw: Tuple[int, int],
                         max_depth: float,
                         standard_intrinsics: Tuple[float, float, float, float] = SCANET_INTRINSICS,
                         debug: bool = False,
                         debug_prefix: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    fx_feat, fy_feat, cx_feat, cy_feat = _scale_intrinsics(standard_intrinsics, feat_hw)
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    valid_z = (z > MIN_DEPTH) & (z < max_depth)
    ratio_x = torch.zeros_like(x)
    ratio_y = torch.zeros_like(y)
    valid_ratio_mask = valid_z & (torch.abs(z) > torch.finfo(z.dtype).eps)
    if valid_ratio_mask.any():
        ratio_x[valid_ratio_mask] = x[valid_ratio_mask] / z[valid_ratio_mask]
        ratio_y[valid_ratio_mask] = y[valid_ratio_mask] / z[valid_ratio_mask]
    u_feat = fx_feat * ratio_x + cx_feat
    v_feat = fy_feat * ratio_y + cy_feat
    u_feat = torch.where(valid_z, u_feat, torch.full_like(u_feat, -1.0))
    v_feat = torch.where(valid_z, v_feat, torch.full_like(v_feat, -1.0))
    Wf = float(feat_hw[1])
    Hf = float(feat_hw[0])
    valid_u = (u_feat >= 0) & (u_feat < Wf)
    valid_v = (v_feat >= 0) & (v_feat < Hf)
    valid = valid_z & valid_u & valid_v
    if debug:
        total_points = len(z)
        depth_valid = int(valid_z.sum().item())
        boundary_valid = int(valid.sum().item())
        u_min, u_max = float(u_feat.min().item()), float(u_feat.max().item())
        v_min, v_max = float(v_feat.min().item()), float(v_feat.max().item())
        prefix = f"{debug_prefix} " if debug_prefix else ""
        print(f"{prefix}projection stats: depth_ok={depth_valid}/{total_points}, "
              f"boundary_ok={boundary_valid}/{total_points}, "
              f"u_range=[{u_min:.1f}, {u_max:.1f}], v_range=[{v_min:.1f}, {v_max:.1f}]")
    uv_feat = torch.stack([u_feat, v_feat], dim=-1)
    return uv_feat, valid


def unified_projection_and_sample(xyz_cam: torch.Tensor,
                                  feat_map: torch.Tensor,
                                  max_depth: float,
                                  align_corners: bool = True,
                                  standard_intrinsics: Tuple[float, float, float, float] = SCANET_INTRINSICS,
                                  debug: bool = False,
                                  debug_prefix: str = "") -> Tuple[torch.Tensor, torch.Tensor]:
    Hf, Wf = feat_map.shape[2], feat_map.shape[3]
    uv_feat, valid = project_points_to_uv(
        xyz_cam,
        (Hf, Wf),
        max_depth=max_depth,
        standard_intrinsics=standard_intrinsics,
        debug=debug,
        debug_prefix=debug_prefix
    )
    sampled = sample_img_feat(feat_map, uv_feat, valid, align_corners=align_corners)
    return sampled, valid


def splat_to_grid(uv: torch.Tensor,
                  z: torch.Tensor,
                  feats: torch.Tensor,
                  valid: torch.Tensor,
                  H: int,
                  W: int,
                  mode: str = 'bilinear',
                  depth_tol: float = 0.05) -> Tuple[torch.Tensor, torch.Tensor]:
    """Rasterize点特征到像素网格。

    Args:
        uv: (N, 2) 像素坐标 (u, v)。
        z:  (N,) 深度值。
        feats: (N, C) 点特征。
        valid: (N,) bool，表示该点是否有效。
        H, W: 输出特征图高/宽。
        mode: 'bilinear'（默认）或 'zbuf'。
        depth_tol: z-buffer 模式下的深度容差。

    Returns:
        Tuple[Tensor, Tensor]: (C, H, W) 的融合特征，以及 (1, H, W) 的覆盖权重。
    """
    if feats.numel() == 0:
        return feats.new_zeros((feats.shape[1], H, W)), feats.new_zeros((1, H, W))

    device = feats.device
    dtype = feats.dtype
    C = feats.shape[1]

    valid = valid if valid is not None else torch.ones_like(z, dtype=torch.bool)
    idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if idx.numel() == 0:
        return feats.new_zeros((C, H, W)), feats.new_zeros((1, H, W))

    uv_valid = uv[idx]
    z_valid = z[idx]
    feats_valid = feats[idx]

    if mode.lower() not in ('bilinear', 'zbuf'):
        raise ValueError(f"Unsupported splat mode: {mode}")

    if mode.lower() == 'bilinear':
        x = uv_valid[:, 0]
        y = uv_valid[:, 1]
        x0 = torch.floor(x)
        y0 = torch.floor(y)
        dx = x - x0
        dy = y - y0

        x0 = x0.clamp(0, W - 1).long()
        y0 = y0.clamp(0, H - 1).long()
        x1 = (x0 + 1).clamp(0, W - 1)
        y1 = (y0 + 1).clamp(0, H - 1)

        w00 = (1.0 - dx) * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w10 = dx * (1.0 - dy)
        w11 = dx * dy

        grids = []
        covers = []
        accum_feat = feats_valid.new_zeros((H * W, C))
        accum_cover = feats_valid.new_zeros((H * W,))

        weights = (w00, w01, w10, w11)
        xs = (x0, x0, x1, x1)
        ys = (y0, y1, y0, y1)

        flat_size = H * W
        for w, xx, yy in zip(weights, xs, ys):
            w = w.clamp(min=0.0)
            flat_idx = (yy * W + xx).long()
            accum_feat.index_add_(0, flat_idx, feats_valid * w.unsqueeze(1))
            accum_cover.index_add_(0, flat_idx, w)

        F2D = accum_feat.t().reshape(C, H, W)
        cover = accum_cover.view(1, H, W)
        return F2D, cover

    # z-buffer
    x = uv_valid[:, 0]
    y = uv_valid[:, 1]
    cols = torch.round(x).clamp(0, W - 1).long()
    rows = torch.round(y).clamp(0, H - 1).long()
    flat_idx = (rows * W + cols).long()

    flat_size = H * W
    device = feats_valid.device

    if hasattr(torch.Tensor, 'scatter_reduce_'):
        depth_init = torch.full((flat_size,), float('inf'), device=device, dtype=z_valid.dtype)
        depth_min = depth_init.scatter_reduce(0, flat_idx, z_valid, reduce='amin', include_self=True)
    else:
        depth_min_cpu = torch.full((flat_size,), float('inf'), device='cpu', dtype=z_valid.dtype)
        flat_cpu = flat_idx.cpu()
        z_cpu = z_valid.cpu()
        for pixel, depth in zip(flat_cpu.tolist(), z_cpu.tolist()):
            if depth < depth_min_cpu[pixel]:
                depth_min_cpu[pixel] = depth
        depth_min = depth_min_cpu.to(z_valid.device)

    min_depth = depth_min[flat_idx]
    depth_mask = torch.isfinite(min_depth) & ((z_valid - min_depth).abs() <= depth_tol)

    if not depth_mask.any():
        return feats.new_zeros((C, H, W)), feats.new_zeros((1, H, W))

    flat_idx = flat_idx[depth_mask]
    feats_valid = feats_valid[depth_mask]
    depth_counts = torch.ones_like(z_valid[depth_mask])

    accum_feat = feats_valid.new_zeros((flat_size, C))
    accum_cover = feats_valid.new_zeros((flat_size,))

    accum_feat.index_add_(0, flat_idx, feats_valid)
    accum_cover.index_add_(0, flat_idx, depth_counts)

    cover = accum_cover.view(1, H, W)
    F2D = (accum_feat / accum_cover.clamp_min(1.0).unsqueeze(1)).t().reshape(C, H, W)
    return F2D, cover
