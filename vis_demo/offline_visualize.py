"""
Offline single-scene visualization using DataConverter.

Exports 3D segmentation colored point cloud (PLY) and 2D reprojection
(segmentation overlay, feature heatmap, cover map) per frame if camera info exists.

Minimal dependencies: relies on existing online_demo.DataConverter and
projection_utils for projection and splatting.
"""
from __future__ import annotations

import os
import argparse
import json
import copy
from typing import Tuple, Optional, List

import numpy as np
import torch
from PIL import Image

from mmengine.dataset import Compose, pseudo_collate
from mmdet3d.apis import init_model

# Ensure project root is on sys.path when running as a script via relative path
import os as _os, sys as _sys
_CUR_DIR = _os.path.dirname(_os.path.abspath(__file__))
_PROJ_ROOT = _os.path.dirname(_CUR_DIR)
if _PROJ_ROOT not in _sys.path:
    _sys.path.insert(0, _PROJ_ROOT)

# Reuse scene constructor from the existing demo
from vis_demo.online_demo import DataConverter

# Projection and splatting utilities
from oneformer3d.projection_utils import (
    project_points_to_uv,
    splat_to_grid,
    sample_img_feat,
    SCANET_INTRINSICS,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _masks_to_point_labels(masks: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Convert instance masks (I, P) and scores (I,) into per-point labels (P,).

    Later instances in the sorted order overwrite earlier ones, so the highest
    scoring instance owning a point wins. Points belonging to no instance get -1.
    """
    assert masks.dim() == 2 and scores.dim() == 1
    num_inst, num_pts = masks.shape
    if num_inst == 0 or num_pts == 0:
        return masks.new_full((num_pts,), -1, dtype=torch.long)

    order = torch.argsort(scores)  # ascending; overwrite from low to high
    labels = masks.new_full((num_pts,), -1, dtype=torch.long)
    for i in order:
        mi = masks[i]
        if mi.dtype != torch.bool:
            mi = mi.bool()
        labels[mi] = int(i)
    return labels.long()


def _palette_for_instances(k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    pal = (rng.rand(max(k, 1) + 1, 3) * 255).astype(np.float32)
    pal[-1] = 200.0  # last reserved for background/empty
    return pal


def _save_ply(points_xyz: np.ndarray, colors_rgb_uint8: np.ndarray, path: str) -> None:
    """Save point cloud to ASCII PLY with per-vertex RGB.
    points_xyz: (N,3) float32/float64
    colors_rgb_uint8: (N,3) uint8 or float in [0,255]
    """
    assert points_xyz.ndim == 2 and points_xyz.shape[1] == 3
    assert colors_rgb_uint8.ndim == 2 and colors_rgb_uint8.shape[1] == 3
    N = points_xyz.shape[0]
    cols = colors_rgb_uint8
    if cols.dtype != np.uint8:
        cols = np.clip(cols, 0, 255).astype(np.uint8)
    header = """ply
format ascii 1.0
element vertex {N}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
""".replace("{N}", str(N))
    with open(path, 'w') as f:
        f.write(header)
        for p, c in zip(points_xyz, cols):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def _maybe_sample(points_xyz: np.ndarray, colors_rgb_uint8: np.ndarray, ratio: float, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """Uniformly sample point-color pairs by ratio in (0,1]."""
    if ratio >= 0.9999 or points_xyz.shape[0] <= 0:
        return points_xyz, colors_rgb_uint8
    n = points_xyz.shape[0]
    m = max(1, int(n * ratio))
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(n, size=m, replace=False))
    return points_xyz[idx], colors_rgb_uint8[idx]


def _overlay_seg_on_image(img: np.ndarray, seg_map: np.ndarray, palette: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay segmentation map (H,W) of instance ids onto RGB image (H,W,3)."""
    h, w = img.shape[:2]
    seg_vis = palette[np.clip(seg_map, -1, palette.shape[0]-1)]  # (-1 uses last color)
    seg_vis = seg_vis.reshape(h, w, 3)
    out = (alpha * seg_vis + (1.0 - alpha) * img.astype(np.float32)).astype(np.uint8)
    return out


def _normalize_intrinsics_from_caminfo(cam_info: dict) -> Tuple[float, float, float, float]:
    intr = cam_info.get('intrinsics', None)
    if intr is None:
        return SCANET_INTRINSICS
    # intr may be [fx, fy, cx, cy] or a 3x3
    if isinstance(intr, (list, tuple)) and len(intr) == 4:
        fx, fy, cx, cy = float(intr[0]), float(intr[1]), float(intr[2]), float(intr[3])
        return fx, fy, cx, cy
    if isinstance(intr, (list, tuple, np.ndarray)):
        arr = np.asarray(intr)
        if arr.shape == (3, 3):
            fx, fy, cx, cy = float(arr[0, 0]), float(arr[1, 1]), float(arr[0, 2]), float(arr[1, 2])
            return fx, fy, cx, cy
    return SCANET_INTRINSICS


def _to_numpy_img(path: str) -> np.ndarray:
    return np.array(Image.open(path))


def _pca_colors(feat: np.ndarray) -> np.ndarray:
    """Project features (N,D) to RGB via PCA, normalized to [0,255]."""
    if feat.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)
    X = feat.astype(np.float32)
    X -= X.mean(0, keepdims=True)
    # robust SVD-based PCA
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    Y = X @ Vt[:3].T
    Y -= Y.min(axis=0, keepdims=True)
    denom = (Y.max(axis=0, keepdims=True) - Y.min(axis=0, keepdims=True))
    denom[denom < 1e-6] = 1.0
    Y = (Y / denom * 255.0).clip(0, 255)
    return Y.astype(np.uint8)


def _scalar_gray(vals: np.ndarray) -> np.ndarray:
    """Map scalar array (N,) to grayscale RGB [0,255]. NaN/Inf safe."""
    v = np.asarray(vals, dtype=np.float32)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    v -= v.min() if v.size else 0.0
    rng = v.max() - (v.min() if v.size else 0.0)
    if rng < 1e-6:
        g = np.zeros_like(v)
    else:
        g = (v / rng) * 255.0
    rgb = np.stack([g, g, g], axis=1)
    return rgb.astype(np.uint8)


def _resolve_clip_root(scene_idx: str,
                       explicit: Optional[str],
                       data_root: Optional[str]) -> Tuple[Optional[str], List[str]]:
    """Resolve a clip feature root that contains clip_feat/<scene_idx>.

    Returns the resolved root (may be None) and the list of candidates tried."""
    seen: set[str] = set()
    candidates: List[str] = []

    def _add(path: Optional[str]) -> None:
        if not path:
            return
        abs_path = os.path.abspath(path)
        if abs_path not in seen:
            seen.add(abs_path)
            candidates.append(abs_path)

    _add(explicit)
    _add(data_root)
    if data_root:
        _add(os.path.dirname(os.path.abspath(data_root)))
    # project root siblings
    parent = os.path.dirname(_PROJ_ROOT)
    _add(parent)
    _add(os.path.join(parent, 'dataset'))
    _add(os.path.join(_PROJ_ROOT, 'data'))

    tried: List[str] = []
    for root in candidates:
        clip_dir = os.path.join(root, 'clip_feat', scene_idx)
        tried.append(root)
        if os.path.isdir(clip_dir):
            return root, tried
    return explicit or data_root, tried


def main() -> None:
    parser = argparse.ArgumentParser(description='Offline single-scene visualization (DataConverter path)')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--scene_idx', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='./vis_results')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--grid_hw', type=int, nargs=2, default=None, help='2D grid H W for reprojection')
    parser.add_argument('--m3d', type=str, default='seg', help='3D mode: seg|feat|alpha|valid|all')
    parser.add_argument('--m2d', type=str, default='all', help='2D mode: seg|feat|cover|all|none')
    parser.add_argument('--frame_ids', type=int, nargs='*', default=None, help='Subset of frame indices to visualize (e.g., 0 3 5)')
    parser.add_argument('--points_sample_ratio', type=float, default=1.0, help='Downsample ratio for PLY export (0,1]')
    parser.add_argument('--data_root', type=str, default=None, help='Override dataset root; if set, overrides config dataset.data_root')
    parser.add_argument('--clip_root', type=str, default=None, help='Override clip feature root for LoadClipFeature (if different from data_root)')
    args = parser.parse_args()

    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)
    model.map_to_rec_pcd = False
    cfg = model.cfg

    # Compose pipeline + build single-scene sample
    pipeline_cfg = copy.deepcopy(cfg.test_pipeline)
    # 优先使用 CLI 的 --data_root，其次使用 val_dataloader.dataset.data_root，最后 cfg.data_root
    data_root = args.data_root
    if data_root is None:
        try:
            ds = cfg.val_dataloader.dataset
            if isinstance(ds, dict):
                data_root = ds.get('data_root', None)
            else:
                data_root = getattr(ds, 'data_root', None)
        except Exception:
            data_root = getattr(cfg, 'data_root', None)
    clip_root, clip_candidates = _resolve_clip_root(args.scene_idx, args.clip_root, data_root)
    if clip_root is not None:
        for t in pipeline_cfg:
            if isinstance(t, dict) and t.get('type') == 'LoadClipFeature':
                t['data_root'] = clip_root
    test_pipeline = Compose(pipeline_cfg)
    data_converter = DataConverter(root_path=data_root, cfg=cfg)
    data = data_converter.process_single_scene(args.scene_idx)
    # Ensure clip feature paths exist for LoadClipFeature
    lidar_path = data.get('lidar_points', {}).get('lidar_path', '')
    frame_token = os.path.splitext(os.path.basename(lidar_path))[0]
    if frame_token.startswith(args.scene_idx + '_'):
        frame_token = frame_token[len(args.scene_idx) + 1:]
    clip_rel_path = os.path.join('clip_feat', args.scene_idx, f'{frame_token}.pt')
    data['clip_feat_path'] = clip_rel_path
    data['clip_feat_paths'] = [clip_rel_path]
    if clip_root:
        clip_abs = os.path.join(clip_root, clip_rel_path)
    elif data_root:
        clip_abs = os.path.join(data_root, clip_rel_path)
    else:
        clip_abs = None
    if clip_abs and not os.path.exists(clip_abs):
        print(f"[offline_visualize] Warning: clip feature not found at {clip_abs}")
    img_paths: List[str] = data.get('img_paths', [])
    if not img_paths and 'img_path' in data:
        img_paths = [data['img_path']]
    data_list = [test_pipeline(data)]
    collate_data = pseudo_collate(data_list)

    with torch.no_grad():
        results = model.test_step(collate_data)

    assert isinstance(results, list) and len(results) == 1
    result = results[0]

    # Prepare output dir
    out_dir = os.path.join(args.save_dir, args.scene_idx)
    _ensure_dir(out_dir)

    # Extract points, instance masks/scores
    inputs = data_list[0]['inputs']
    raw_points = inputs['points']
    if isinstance(raw_points, torch.Tensor):
        points_all = raw_points
    elif isinstance(raw_points, list):
        if len(raw_points) == 0:
            raise AssertionError('Empty points list from pipeline')
        if isinstance(raw_points[0], torch.Tensor):
            points_all = torch.stack(raw_points, dim=0)
        else:
            points_all = torch.as_tensor(raw_points)
    else:
        points_all = torch.as_tensor(raw_points)

    if points_all.dim() == 2:
        points_all = points_all.unsqueeze(0)
    elif points_all.dim() != 3:
        raise AssertionError('Expect points tensor with dim 2 or 3, got shape {}'.format(points_all.shape))

    if points_all.size(-1) < 3:
        raise AssertionError('Expect points shape (T,N,C>=3)')

    points_all = points_all.cpu()
    T, N = int(points_all.shape[0]), int(points_all.shape[1])
    pts_xyz = points_all[:, :, :3].reshape(-1, 3)  # (T*N, 3)

    pred_seg = result.pred_pts_seg
    inst_masks_raw = pred_seg.pts_instance_mask[0]  # (I, P)
    inst_scores_raw = pred_seg.instance_scores  # (I,)
    inst_masks = torch.as_tensor(inst_masks_raw)
    inst_scores = torch.as_tensor(inst_scores_raw)
    if inst_masks.dtype != torch.bool:
        inst_masks = inst_masks.bool()
    labels_1d = _masks_to_point_labels(inst_masks, inst_scores)  # (P,)

    # Palette and colors for 3D
    num_inst = int(inst_scores.numel())
    palette = _palette_for_instances(num_inst)
    labels_np = labels_1d.cpu().numpy().astype(np.int64)
    colors_3d = palette[np.clip(labels_np, -1, num_inst)]  # (P,3)

    # Save 3D segmentation
    ply_path = os.path.join(out_dir, '3d_seg.ply')
    pts_xyz_np = pts_xyz.numpy()
    pts_xyz_s, colors_3d_s = _maybe_sample(pts_xyz_np, colors_3d.astype(np.uint8), max(0.0, min(1.0, float(args.points_sample_ratio))))
    _save_ply(pts_xyz_s, colors_3d_s, ply_path)

    # Optional 3D feature/alpha/valid exports (BiFusion)
    m3d = args.m3d.lower()
    want_feat3d = m3d in ('feat', 'all')
    want_alpha = m3d in ('alpha', 'all')
    want_valid = m3d in ('valid', 'all')

    enc_out = getattr(model, '_encoder_out', None)
    if isinstance(enc_out, dict) and any([want_feat3d, want_alpha, want_valid]):
        # fused features list, conf/valid lists
        fused_list = enc_out.get('feat_fusion', None)
        conf_list = enc_out.get('conf_2d', None)
        valid_list = enc_out.get('valid_projection_mask', None)
        # Flatten per-sample tensors to (T*N, ...)
        def _to_np(flat_like: Optional[List[torch.Tensor]], take_dim: Optional[int] = None) -> Optional[np.ndarray]:
            if not isinstance(flat_like, list) or len(flat_like) == 0:
                return None
            x = flat_like[0]
            if isinstance(x, torch.Tensor):
                arr = x.detach().cpu().numpy()
            else:
                arr = np.asarray(x)
            return arr
        fused_np = _to_np(fused_list)
        conf_np = _to_np(conf_list)
        valid_np = _to_np(valid_list)
        # Ensure lengths align with points
        num_pts_total = T * N
        if fused_np is not None and fused_np.shape[0] == num_pts_total and want_feat3d:
            colors_fused = _pca_colors(fused_np)
            pts_s, col_s = _maybe_sample(pts_xyz_np, colors_fused, max(0.0, min(1.0, float(args.points_sample_ratio))))
            _save_ply(pts_s, col_s, os.path.join(out_dir, '3d_feat_fused.ply'))
        if conf_np is not None and conf_np.shape[0] == num_pts_total and want_alpha:
            conf_flat = conf_np.reshape(-1)
            colors_alpha = _scalar_gray(conf_flat)
            pts_s, col_s = _maybe_sample(pts_xyz_np, colors_alpha, max(0.0, min(1.0, float(args.points_sample_ratio))))
            _save_ply(pts_s, col_s, os.path.join(out_dir, '3d_alpha.ply'))
        if valid_np is not None and valid_np.shape[0] == num_pts_total and want_valid:
            valid_flat = valid_np.reshape(-1).astype(np.float32)
            colors_valid = _scalar_gray(valid_flat)
            pts_s, col_s = _maybe_sample(pts_xyz_np, colors_valid, max(0.0, min(1.0, float(args.points_sample_ratio))))
            _save_ply(pts_s, col_s, os.path.join(out_dir, '3d_valid.ply'))

    # 3D feat_2d export: sample per-point 2D features from clip_pix when available
    clip_any = inputs.get('clip_pix', None)
    def _clip_for_frame(idx: int):
        if isinstance(clip_any, list):
            if len(clip_any) == 0:
                return None
            t = clip_any[idx] if idx < len(clip_any) else clip_any[0]
            return t
        return clip_any
    # Collect per-point 2D features for all frames if clip is available
    have_clip = False
    try:
        if _clip_for_frame(0) is not None:
            have_clip = True
    except Exception:
        have_clip = False
    if have_clip and (m3d in ('feat', 'all')):
        # Accumulate features as (T*N, C2)
        feat2d_list = []
        for i in range(T):
            clip_i = _clip_for_frame(i)
            if clip_i is None:
                feat2d_list.append(np.zeros((N, 0), dtype=np.float32))
                continue
            # clip_i: (C,Hc,Wc)
            if isinstance(clip_i, torch.Tensor):
                clip_t = clip_i
            else:
                clip_t = torch.as_tensor(clip_i)
            if clip_t.dim() != 3:
                feat2d_list.append(np.zeros((N, 0), dtype=np.float32))
                continue
            Cc, Hc, Wc = int(clip_t.shape[0]), int(clip_t.shape[1]), int(clip_t.shape[2])
            # Recompute uv for this feature map size
            cam_meta = None
            cam_info_field = inputs.get('cam_info', None)
            if isinstance(cam_info_field, list):
                if i < len(cam_info_field):
                    cm = cam_info_field[i]
                    cam_meta = cm[0] if isinstance(cm, list) and len(cm) > 0 and isinstance(cm[0], dict) else (cm if isinstance(cm, dict) else None)
            elif isinstance(cam_info_field, dict):
                cam_meta = cam_info_field
            if not isinstance(cam_meta, dict):
                feat2d_list.append(np.zeros((N, 0), dtype=np.float32))
                continue
            pose_mat = cam_meta.get('pose', cam_meta.get('extrinsics', None))
            if pose_mat is None:
                feat2d_list.append(np.zeros((N, 0), dtype=np.float32))
                continue
            intrinsics_4 = _normalize_intrinsics_from_caminfo(cam_meta)
            pose_t = torch.as_tensor(pose_mat, dtype=points_all.dtype, device=points_all.device)
            W2C = torch.inverse(pose_t)
            pts_i = points_all[i, :, :3]
            ones = torch.ones((pts_i.shape[0], 1), dtype=pts_i.dtype, device=pts_i.device)
            xyz1 = torch.cat([pts_i, ones], dim=1)
            xyz_cam = (xyz1 @ W2C.t())[:, :3]
            uv_i, valid_i = project_points_to_uv(xyz_cam, (Hc, Wc), max_depth=20.0, standard_intrinsics=intrinsics_4)
            # Sample features at uv
            fmap = clip_t.unsqueeze(0).to(dtype=uv_i.dtype, device=uv_i.device)
            feats_i = sample_img_feat(fmap, uv_i, valid_i, align_corners=True)  # (N,Cc)
            feat2d_list.append(feats_i.cpu().numpy())
        if len(feat2d_list) == T:
            feat2d_all = np.concatenate(feat2d_list, axis=0)  # (T*N, Cc)
            if feat2d_all.shape[1] > 0:
                colors_feat2d = _pca_colors(feat2d_all)
                pts_s, col_s = _maybe_sample(pts_xyz_np, colors_feat2d, max(0.0, min(1.0, float(args.points_sample_ratio))))
                _save_ply(pts_s, col_s, os.path.join(out_dir, '3d_feat_2d.ply'))

    # Attempt 2D reprojection (if requested)
    if args.m2d.lower() != 'none' and len(img_paths) > 0:
        # Grid size
        if args.grid_hw is not None:
            H, W = int(args.grid_hw[0]), int(args.grid_hw[1])
        else:
            H, W = 60, 80
            try:
                if hasattr(cfg.model, 'two_d_losses') and 'grid_hw' in cfg.model.two_d_losses:
                    H, W = map(int, cfg.model.two_d_losses['grid_hw'])
            except Exception:
                pass

        # Restore per-frame labels for reprojection
        labels_2d = labels_np.reshape(T, N)

        # Load images for overlay
        images = [ _to_numpy_img(p) for p in img_paths ]
        np.save(os.path.join(out_dir, 'images.npy'), np.stack(images, axis=0))

        # cam_info may be a list (per-frame) or a single list with one dict
        cam_info_field = inputs.get('cam_info', None)
        cam_infos: List[Optional[dict]]
        if isinstance(cam_info_field, list):
            # unwrap nested [ [dict] ] → [dict]
            cam_infos = []
            for item in cam_info_field:
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                    cam_infos.append(item[0])
                elif isinstance(item, dict):
                    cam_infos.append(item)
                else:
                    cam_infos.append(None)
        elif isinstance(cam_info_field, dict):
            cam_infos = [cam_info_field] * T
        else:
            cam_infos = [None] * T

        # Iterate frames using per-frame pose/intrinsics if available
        frame_indices = list(range(T)) if args.frame_ids is None else [fi for fi in args.frame_ids if 0 <= fi < T]
        for i in frame_indices:
            cam_meta = cam_infos[i] if i < len(cam_infos) else None
            if not isinstance(cam_meta, dict):
                continue
            pose_mat = cam_meta.get('pose', cam_meta.get('extrinsics', None))
            if pose_mat is None:
                continue
            intrinsics_4 = _normalize_intrinsics_from_caminfo(cam_meta)
            pose_t = torch.as_tensor(pose_mat, dtype=points_all.dtype, device=points_all.device)
            W2C = torch.inverse(pose_t)

            max_depth = getattr(model, 'max_depth', 20.0) if hasattr(model, 'max_depth') else 20.0
            std_intr = intrinsics_4

            pts_i = points_all[i, :, :3]
            ones = torch.ones((pts_i.shape[0], 1), dtype=pts_i.dtype, device=pts_i.device)
            xyz1 = torch.cat([pts_i, ones], dim=1)
            xyz_cam = (xyz1 @ W2C.t())[:, :3]

            uv, valid = project_points_to_uv(
                xyz_cam,
                (H, W),
                max_depth=max_depth,
                standard_intrinsics=std_intr,
            )
            z = xyz_cam[:, 2]

            # 2D segmentation reprojection using z-buffer per instance id
            lab_i = torch.as_tensor(labels_2d[i], device=uv.device, dtype=torch.long)
            valid_l = valid & (lab_i >= 0)
            if valid_l.any():
                K = int(lab_i.max().item()) + 1
                one_hot = torch.zeros((pts_i.shape[0], K), device=uv.device, dtype=uv.dtype)
                one_hot[valid_l, lab_i[valid_l]] = 1.0
                F2D_seg, _ = splat_to_grid(uv, z, one_hot, valid_l, H, W, mode='zbuf')  # (K,H,W)
                seg_map = torch.argmax(F2D_seg, dim=0).cpu().numpy().astype(np.int32)  # (H,W)
            else:
                seg_map = np.full((H, W), -1, dtype=np.int32)

            # Resize to original image for overlay
            img_i = images[i]
            seg_img = Image.fromarray(seg_map.astype(np.int32), mode='I')
            seg_resized = np.array(seg_img.resize((img_i.shape[1], img_i.shape[0]), resample=Image.NEAREST))
            overlay = _overlay_seg_on_image(img_i, seg_resized, _palette_for_instances(num_inst), alpha=0.5)
            Image.fromarray(overlay).save(os.path.join(out_dir, f'2d_{i:03d}_seg_overlay.png'))

            # Feature reprojection (prefer fused features; fallback to instance-color proxy)
            want_feat = args.m2d.lower() in ('feat', 'all')
            want_cover = args.m2d.lower() in ('cover', 'all')
            if want_feat or want_cover:
                # Try fused features from encoder_out
                feat_proxy_np: Optional[np.ndarray] = None
                enc_out = getattr(model, '_encoder_out', None)
                if isinstance(enc_out, dict) and 'feat_fusion' in enc_out and isinstance(enc_out['feat_fusion'], list) and len(enc_out['feat_fusion']) > 0:
                    fused = enc_out['feat_fusion'][0]
                    if isinstance(fused, torch.Tensor):
                        feat_all = fused.detach().to(uv.device)
                    else:
                        feat_all = torch.as_tensor(fused, device=uv.device)
                    # Per-frame slice
                    feat_proxy = feat_all[i * N:(i + 1) * N]  # (N,D)
                    # Reduce to 3 channels by PCA (CPU ok for offline)
                    try:
                        X = feat_proxy.float().cpu().numpy()
                        X = X - X.mean(0, keepdims=True)
                        U, S, Vt = np.linalg.svd(X, full_matrices=False)
                        W = (X @ Vt[:3].T)
                        # Normalize to [0,255]
                        W = W - W.min(axis=0, keepdims=True)
                        W = W / (W.max(axis=0, keepdims=True) + 1e-6)
                        feat_proxy_np = (W * 255.0).astype(np.float32)
                    except Exception:
                        feat_proxy_np = None
                if feat_proxy_np is None:
                    # Fallback: instance-colors
                    feat_proxy_np = _palette_for_instances(num_inst)[np.clip(lab_i.cpu().numpy(), -1, num_inst)]  # (N,3)
                feat_proxy_t = torch.from_numpy(feat_proxy_np).to(uv.device, dtype=uv.dtype)
                feat_proxy_t = torch.where(valid_l.unsqueeze(1), feat_proxy_t, torch.zeros_like(feat_proxy_t))
                F2D_feat, cover = splat_to_grid(uv, z, feat_proxy_t, valid_l, H, W, mode='bilinear')  # (3,H,W) or (C,H,W)
                # If channels > 3, take first 3 for visualization
                if F2D_feat.shape[0] > 3:
                    F2D_feat_vis = F2D_feat[:3]
                else:
                    F2D_feat_vis = F2D_feat
                feat_vis = F2D_feat_vis.permute(1, 2, 0).clamp(0, 255).cpu().numpy().astype(np.uint8)
                feat_vis = np.array(Image.fromarray(feat_vis).resize((img_i.shape[1], img_i.shape[0]), resample=Image.BILINEAR))
                if want_feat:
                    Image.fromarray(feat_vis).save(os.path.join(out_dir, f'2d_{i:03d}_feat.png'))
                if want_cover:
                    cover_np = cover.squeeze(0).cpu().numpy()
                    # normalize cover to [0,255]
                    cmin, cmax = float(np.nanmin(cover_np)), float(np.nanmax(cover_np))
                    denom = (cmax - cmin) if (cmax - cmin) > 1e-6 else 1.0
                    cover_img = ((cover_np - cmin) / denom * 255.0).astype(np.uint8)
                    cover_img = np.array(Image.fromarray(cover_img).resize((img_i.shape[1], img_i.shape[0]), resample=Image.BILINEAR))
                    Image.fromarray(cover_img).save(os.path.join(out_dir, f'2d_{i:03d}_cover.png'))

                # Optional: export raw clip feature map visualization if available
                clip_i = _clip_for_frame(i)
                if clip_i is not None and isinstance(clip_i, (torch.Tensor, np.ndarray)):
                    clip_np = clip_i.detach().cpu().numpy() if isinstance(clip_i, torch.Tensor) else np.asarray(clip_i)
                    if clip_np.ndim == 3:
                        Cc, Hc, Wc = clip_np.shape
                        X = clip_np.reshape(Cc, -1).T  # (Hc*Wc, Cc)
                        if Cc >= 3:
                            Xc = X.astype(np.float32)
                            Xc -= Xc.mean(0, keepdims=True)
                            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
                            Y = Xc @ Vt[:3].T
                            Y -= Y.min(axis=0, keepdims=True)
                            denom = (Y.max(axis=0, keepdims=True) - Y.min(axis=0, keepdims=True))
                            denom[denom < 1e-6] = 1.0
                            Y = (Y / denom * 255.0)
                            clip_rgb = Y.reshape(Hc, Wc, 3).astype(np.uint8)
                        else:
                            # replicate channels
                            g = (X[:, 0] - X[:, 0].min()) / (X[:, 0].ptp() + 1e-6) * 255.0
                            clip_rgb = np.stack([g, g, g], axis=1).reshape(Hc, Wc, 3).astype(np.uint8)
                        clip_rgb = np.array(Image.fromarray(clip_rgb).resize((img_i.shape[1], img_i.shape[0]), resample=Image.BILINEAR))
                        Image.fromarray(clip_rgb).save(os.path.join(out_dir, f'2d_{i:03d}_clip.png'))

        # Save meta for traceability
        with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
            json.dump({
                'scene_idx': args.scene_idx,
                'img_paths': img_paths,
                'grid_hw': [H, W],
                'num_instances': int(num_inst),
                'clip_feat_path': clip_rel_path,
                'clip_root': clip_root,
                'clip_root_candidates': clip_candidates,
            }, f, indent=2)


if __name__ == '__main__':
    main()
