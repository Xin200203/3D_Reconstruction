#!/usr/bin/env python3
"""
Visualize a single ScanNet200-SV frame (point cloud + GT + predictions).

This script is meant for quick qualitative debugging:
- Loads one sample by its frame token (e.g., scene0000_00_1000)
- Runs inference with a given config/checkpoint
- Visualizes and/or exports OBJ files for:
  - superpoints
  - semantic segmentation (GT vs Pred)
  - instance segmentation (GT vs Pred)
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import Optional, Tuple


def _add_project_root_to_syspath() -> str:
    tools_dir = os.path.dirname(os.path.abspath(__file__))
    proj_root = os.path.dirname(tools_dir)
    if proj_root not in sys.path:
        sys.path.insert(0, proj_root)
    return proj_root


_PROJ_ROOT = _add_project_root_to_syspath()


def _parse_sample_id(sample_id: str, scene_id: Optional[str], frame_id: Optional[str]) -> str:
    if sample_id:
        return sample_id
    if not scene_id or frame_id is None:
        raise ValueError("Provide either --sample-id or both --scene-id and --frame-id.")
    return f"{scene_id}_{frame_id}"


def _infer_scene_and_frame(sample_id: str) -> Tuple[str, str]:
    # Expected format: scene0000_00_1000
    # scene prefix is first 12 chars: scene0000_00
    if len(sample_id) < 14 or sample_id[12] != "_":
        # fallback: split by last underscore
        parts = sample_id.split("_")
        if len(parts) < 3:
            raise ValueError(f"Unrecognized sample id format: {sample_id}")
        return "_".join(parts[:-1]), parts[-1]
    return sample_id[:12], sample_id[13:]


def _resolve_data_root(cfg, override: Optional[str]) -> str:
    if override:
        return os.path.abspath(override)
    for attr in ("data_root",):
        v = getattr(cfg, attr, None)
        if v:
            return os.path.abspath(v)
    try:
        ds = cfg.test_dataloader.dataset
        if isinstance(ds, dict) and ds.get("data_root"):
            return os.path.abspath(ds["data_root"])
    except Exception:
        pass
    try:
        ds = cfg.val_dataloader.dataset
        if isinstance(ds, dict) and ds.get("data_root"):
            return os.path.abspath(ds["data_root"])
    except Exception:
        pass
    raise ValueError("Cannot resolve data_root; pass --data-root explicitly.")


def _build_data_info_for_sv(sample_id: str, data_root: str) -> dict:
    # Local imports (these require the user's ML env).
    import numpy as np

    from tools.update_infos_to_v2 import get_empty_standard_data_info, clear_data_info_unused_keys

    pts_name = f"{sample_id}.bin"
    sem_name = f"{sample_id}.bin"
    ins_name = f"{sample_id}.bin"
    sp_name = f"{sample_id}.bin"

    scene_id, frame_id = _infer_scene_and_frame(sample_id)
    img_rel = os.path.join("2D", scene_id, "color", f"{frame_id}.jpg")
    pose_path = os.path.join(data_root, "pose_centered", scene_id, f"{frame_id}.npy")

    temp = get_empty_standard_data_info()
    temp["sample_idx"] = sample_id
    temp["lidar_points"]["num_pts_feats"] = 6
    temp["lidar_points"]["lidar_path"] = pts_name
    temp["pts_semantic_mask_path"] = sem_name
    temp["pts_instance_mask_path"] = ins_name
    temp["super_pts_path"] = sp_name
    temp["img_path"] = img_rel

    if os.path.exists(pose_path):
        temp["pose"] = np.load(pose_path)

    temp, _ = clear_data_info_unused_keys(temp)
    return temp


def _fix_and_check_paths(data_info: dict, data_root: str, data_prefix: dict) -> dict:
    """Best-effort fixups for absolute paths, then sanity-check existence."""
    def _fix(path: str, prefix_key: Optional[str] = None) -> str:
        if not path:
            return path
        if os.path.isabs(path) and os.path.exists(path):
            return path
        candidates = []
        if prefix_key and prefix_key in data_prefix and data_prefix[prefix_key]:
            candidates.append(os.path.join(data_root, data_prefix[prefix_key], path))
        candidates.append(os.path.join(data_root, path))
        for cand in candidates:
            if os.path.exists(cand):
                return cand
        return path

    # lidar points
    lp = data_info.get("lidar_points", {})
    if isinstance(lp, dict) and "lidar_path" in lp:
        lp["lidar_path"] = _fix(lp["lidar_path"], "pts")
        data_info["lidar_points"] = lp

    # masks + superpoints + image
    if "pts_semantic_mask_path" in data_info:
        data_info["pts_semantic_mask_path"] = _fix(data_info["pts_semantic_mask_path"], "pts_semantic_mask")
    if "pts_instance_mask_path" in data_info:
        data_info["pts_instance_mask_path"] = _fix(data_info["pts_instance_mask_path"], "pts_instance_mask")
    if "super_pts_path" in data_info:
        data_info["super_pts_path"] = _fix(data_info["super_pts_path"], "sp_pts_mask")
    if "img_path" in data_info:
        data_info["img_path"] = _fix(data_info["img_path"], None)

    # Hard checks (points/masks/superpoints must exist).
    missing = []
    if isinstance(lp, dict) and lp.get("lidar_path") and not os.path.exists(lp["lidar_path"]):
        missing.append(f"points: {lp['lidar_path']}")
    for k in ("pts_semantic_mask_path", "pts_instance_mask_path", "super_pts_path"):
        if k in data_info and data_info.get(k) and not os.path.exists(data_info[k]):
            missing.append(f"{k}: {data_info[k]}")
    if missing:
        raise FileNotFoundError("Missing required files:\n- " + "\n- ".join(missing))
    return data_info


def _load_raw_points(data_root: str, sample_id: str) -> "object":
    import numpy as np

    pts_path = os.path.join(data_root, "points", f"{sample_id}.bin")
    if not os.path.exists(pts_path):
        raise FileNotFoundError(f"Points file not found: {pts_path}")
    pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 6)
    return pts


def _pred_instance_to_point_labels(pred_mask, pred_scores):
    """Convert instance masks + scores to per-point instance id labels.

    Returns int64 labels (P,), with background as -1.
    """
    import torch

    if isinstance(pred_mask, (list, tuple)):
        pred_mask = pred_mask[0]
    if isinstance(pred_scores, (list, tuple)):
        pred_scores = pred_scores[0]

    mask_t = pred_mask
    if not torch.is_tensor(mask_t):
        mask_t = torch.as_tensor(mask_t)
    scores_t = pred_scores
    if not torch.is_tensor(scores_t):
        scores_t = torch.as_tensor(scores_t)

    if mask_t.dim() == 3 and mask_t.shape[0] == 1:
        mask_t = mask_t[0]

    # Expected: (I,P). If (P,I), transpose.
    if mask_t.dim() != 2:
        raise ValueError(f"Unexpected pred instance mask shape: {tuple(mask_t.shape)}")
    if mask_t.shape[0] != scores_t.numel() and mask_t.shape[1] == scores_t.numel():
        mask_t = mask_t.t().contiguous()

    # Ensure boolean-ish mask
    if mask_t.dtype != torch.bool:
        mask_t = mask_t > 0.5

    order = torch.argsort(scores_t.flatten())  # low -> high, overwrite so high wins
    mask_sorted = mask_t[order]

    labels = mask_sorted[0].to(torch.int64) - 1  # background=-1, first inst=0
    for i in range(1, mask_sorted.shape[0]):
        labels[mask_sorted[i]] = i
    return labels.cpu().numpy()

def _as_numpy_int(x):
    import numpy as np
    try:
        import torch
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x).astype(np.int64)


def _write_obj_xyzrgb(points_xyz, colors_rgb, out_path: str) -> None:
    pts = points_xyz.astype("float32")
    colors = colors_rgb.astype("float32")
    if pts.shape[1] != 3:
        pts = pts[:, :3]
    if colors.shape[1] != 3:
        colors = colors[:, :3]
    if pts.shape[0] != colors.shape[0]:
        raise ValueError("points/colors length mismatch for OBJ export")
    out = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for i in range(pts.shape[0]):
            p = pts[i]
            c = colors[i].astype(int)
            f.write("v %f %f %f %d %d %d\n" % (p[0], p[1], p[2], c[0], c[1], c[2]))


def _write_diff_objs(points_xyz, colors_rgb, diff_mask, out_dir: str, prefix: str) -> None:
    _write_obj_xyzrgb(points_xyz, colors_rgb, os.path.join(out_dir, f"{prefix}_diff.obj"))
    if diff_mask is not None:
        _write_obj_xyzrgb(
            points_xyz[diff_mask],
            colors_rgb[diff_mask],
            os.path.join(out_dir, f"{prefix}_diff_errors.obj"),
        )


def _make_semseg_diff_colors(gt_sem, pred_sem, ignore_index: Optional[int]):
    import numpy as np

    gt_sem = _as_numpy_int(gt_sem)
    pred_sem = _as_numpy_int(pred_sem)
    diff_mask = gt_sem != pred_sem
    if ignore_index is not None:
        valid = gt_sem != ignore_index
        diff_mask = diff_mask & valid
    colors = np.tile(np.array([120, 120, 120], dtype=np.float32), (gt_sem.shape[0], 1))
    if ignore_index is not None:
        colors[gt_sem == ignore_index] = np.array([30, 30, 30], dtype=np.float32)
    colors[diff_mask] = np.array([255, 212, 0], dtype=np.float32)
    return colors, diff_mask


def _make_insseg_diff_colors(gt_ins, pred_ins, bg_value: int = -1):
    import numpy as np

    gt_ins = _as_numpy_int(gt_ins)
    pred_ins = _as_numpy_int(pred_ins)
    # Highlight only background vs instance presence (no instance matching).
    gt_bg = gt_ins == bg_value
    pred_bg = pred_ins == bg_value
    false_pos = gt_bg & ~pred_bg
    false_neg = ~gt_bg & pred_bg
    both_bg = gt_bg & pred_bg
    colors = np.tile(np.array([140, 140, 140], dtype=np.float32), (gt_ins.shape[0], 1))
    colors[both_bg] = np.array([30, 30, 30], dtype=np.float32)
    colors[false_pos] = np.array([255, 0, 183], dtype=np.float32)
    colors[false_neg] = np.array([0, 245, 255], dtype=np.float32)
    diff_mask = false_pos | false_neg
    return colors, diff_mask


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize single ScanNet200-SV frame (GT + pred).")
    parser.add_argument("--config", required=True, type=str, help="Config file, e.g. configs/ESAM_CA/ESAM_sv_scannet200_CA_dino.py")
    parser.add_argument("--checkpoint", required=True, type=str, help="Checkpoint path")
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument("--data-root", default=None, type=str, help="Override cfg.data_root")

    parser.add_argument("--sample-id", default="", type=str, help="e.g. scene0000_00_1000")
    parser.add_argument("--scene-id", default=None, type=str, help="e.g. scene0000_00 (used with --frame-id)")
    parser.add_argument("--frame-id", default=None, type=str, help="e.g. 1000 (used with --scene-id)")

    parser.add_argument("--out-dir", default="work_dirs/vis_single_frame", type=str)
    parser.add_argument("--show", action="store_true", help="Open Open3D windows")
    parser.add_argument("--snapshot", action="store_true", help="Save Open3D screenshots (when --show is on)")
    parser.add_argument("--tasks", default="all", choices=["all", "super", "sem", "ins"], help="Which results to export/visualize")
    args = parser.parse_args()

    # Local imports (these require the user's ML env).
    import numpy as np
    import torch

    from mmengine.dataset import Compose, pseudo_collate
    from mmdet3d.apis import init_model
    from mmdet3d.registry import DATASETS
    from mmdet3d.utils import register_all_modules

    # Ensure all registries are ready (mmdet3d + our custom modules).
    register_all_modules()
    import oneformer3d  # noqa: F401

    sample_id = _parse_sample_id(args.sample_id, args.scene_id, args.frame_id)

    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)
    if hasattr(model, "map_to_rec_pcd"):
        model.map_to_rec_pcd = False
    cfg = model.cfg
    data_root = _resolve_data_root(cfg, args.data_root)

    # Build dataset ONLY for parse_data_info (seg_label_mapping, abs paths, etc).
    ds_cfg = copy.deepcopy(cfg.val_dataloader.dataset)
    if isinstance(ds_cfg, dict):
        ds_cfg["data_root"] = data_root
    dataset = DATASETS.build(ds_cfg)

    test_pipeline = Compose(copy.deepcopy(cfg.test_pipeline))

    data_info = _build_data_info_for_sv(sample_id, data_root)
    data_info = dataset.parse_data_info(data_info)
    data_info = _fix_and_check_paths(data_info, data_root, getattr(dataset, "data_prefix", {}))

    raw_points = _load_raw_points(data_root, sample_id)

    packed = test_pipeline(data_info)
    collate_data = pseudo_collate([packed])

    with torch.no_grad():
        pred_sample = model.test_step(collate_data)[0]

    out_root = os.path.join(args.out_dir, sample_id)
    os.makedirs(out_root, exist_ok=True)

    # Import visualization helpers from demo/
    from demo.show_result import show_seg_result

    # Superpoints (GT only, shown as both gt/pred for convenience)
    if args.tasks in ("all", "super"):
        sp = _as_numpy_int(packed["data_samples"].eval_ann_info["sp_pts_mask"])
        np.random.seed(0)
        palette = (np.random.rand(int(sp.max()) + 2, 3) * 255).astype(np.float32)
        palette[-1] = 200.0
        show_seg_result(
            raw_points,
            sp,
            sp,
            out_dir=out_root,
            filename="superpoints",
            palette=palette,
            ignore_index=None,
            show=args.show,
            snapshot=args.snapshot,
        )

    # Semantic segmentation
    if args.tasks in ("all", "sem"):
        gt_sem = _as_numpy_int(packed["data_samples"].eval_ann_info["pts_semantic_mask"])
        pred_sem = _as_numpy_int(pred_sample.pred_pts_seg.pts_semantic_mask[0])
        np.random.seed(0)
        palette = (np.random.rand(201, 3) * 255).astype(np.float32)
        palette[-1] = 200.0
        show_seg_result(
            raw_points,
            gt_sem,
            pred_sem,
            out_dir=out_root,
            filename="semseg",
            palette=palette,
            ignore_index=200,
            show=args.show,
            snapshot=args.snapshot,
        )
        semseg_dir = os.path.join(out_root, "semseg")
        sem_colors, sem_diff = _make_semseg_diff_colors(gt_sem, pred_sem, ignore_index=200)
        _write_diff_objs(raw_points[:, :3], sem_colors, sem_diff, semseg_dir, "semseg")

    # Instance segmentation
    if args.tasks in ("all", "ins"):
        gt_ins = _as_numpy_int(packed["data_samples"].eval_ann_info["pts_instance_mask"])
        pred_ins_mask = pred_sample.pred_pts_seg.pts_instance_mask[0]
        pred_ins_score = pred_sample.pred_pts_seg.instance_scores
        pred_ins = _pred_instance_to_point_labels(pred_ins_mask, pred_ins_score)

        np.random.seed(0)
        max_label = int(max(int(np.max(gt_ins)), int(np.max(pred_ins))) + 2)
        palette = (np.random.rand(max_label, 3) * 255).astype(np.float32)
        palette[-1] = 200.0
        show_seg_result(
            raw_points,
            gt_ins,
            pred_ins,
            out_dir=out_root,
            filename="insseg",
            palette=palette,
            ignore_index=None,
            show=args.show,
            snapshot=args.snapshot,
        )
        insseg_dir = os.path.join(out_root, "insseg")
        ins_colors, ins_diff = _make_insseg_diff_colors(gt_ins, pred_ins, bg_value=-1)
        _write_diff_objs(raw_points[:, :3], ins_colors, ins_diff, insseg_dir, "insseg")


if __name__ == "__main__":
    main()
