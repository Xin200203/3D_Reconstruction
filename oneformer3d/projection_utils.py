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
