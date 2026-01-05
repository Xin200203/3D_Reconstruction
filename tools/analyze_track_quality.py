#!/usr/bin/env python3
"""
Track/ID diagnostics for ESAM online runs.

Consumes baseline_stats.json (and optionally baseline_stats_summary.json) produced by UnifiedSegMetric.
Requires new fields for full power:
  - baseline_stats[*].scene_level_dup.gt_vis_ids
  - baseline_stats[*].scene_level_dup.best_iou_per_gt
  - baseline_stats[*].scene_level_dup.hit_cnt_per_gt
  - baseline_stats[*].frames[*].det_to_merge.pred_match (store_det_to_gt=True)
  - baseline_stats[*].frames[*].online.matched_track_ids (OnlineMerge stable track ids)
  - baseline_stats[*].frames[*].pre_pool.gt.gt_vis_ids (store_gt_vis_ids=True) for per-GT visibility
  - baseline_stats[*].frames[*].killed_by_inst_thr.useful_gt_hist (store_killed_useful_gt_hist=True)

If some fields are missing, it will still run and report what is unavailable.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _percentile(xs: List[float], q: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    idx = int(round((len(xs) - 1) * q))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


@dataclass
class GTTrackStats:
    gt_id: int
    best_iou: float
    hit_cnt_05: int
    hit_cnt_01: int
    num_tracks_associated: int
    assoc_coverage: float
    id_switch: int
    fragmentation: int
    killed_useful_frames: int


def _contiguous_segments(seq: List[Optional[int]]) -> int:
    """Count contiguous non-None segments in a sequence."""
    seg = 0
    in_seg = False
    for x in seq:
        if x is None:
            in_seg = False
            continue
        if not in_seg:
            seg += 1
            in_seg = True
    return seg


def _id_switches(seq: List[Optional[int]]) -> int:
    prev = None
    sw = 0
    for x in seq:
        if x is None:
            continue
        if prev is None:
            prev = x
            continue
        if x != prev:
            sw += 1
            prev = x
    return sw


def analyze_scene(scene: Dict[str, Any], *, iou_lo: float, iou_hi: float) -> Tuple[Dict[str, Any], List[GTTrackStats]]:
    scene_id = scene.get("scene_id", "unknown")
    frames = scene.get("frames", [])
    sdup = scene.get("scene_level_dup", {}) or {}

    gt_vis_ids = sdup.get("gt_vis_ids")
    best_iou_per_gt = sdup.get("best_iou_per_gt")
    hit_cnt_per_gt = sdup.get("hit_cnt_per_gt", {}) or {}

    out_meta: Dict[str, Any] = {
        "scene_id": scene_id,
        "num_frames": int(scene.get("num_frames", len(frames))),
        "has_scene_gt_vis_ids": isinstance(gt_vis_ids, list),
        "has_scene_best_iou": isinstance(best_iou_per_gt, list),
        "has_scene_hit_cnt": isinstance(hit_cnt_per_gt, dict) and len(hit_cnt_per_gt) > 0,
        "has_frame_det_to_gt": False,
        "has_frame_gt_vis_ids": False,
        "has_online_track_ids": False,
        "has_killed_useful_hist": False,
    }

    if not (isinstance(gt_vis_ids, list) and isinstance(best_iou_per_gt, list) and len(gt_vis_ids) == len(best_iou_per_gt)):
        return out_meta, []

    # Heuristic: some datasets encode instance ids as <base> + local_id (e.g. 3000+).
    # baseline_stats per-frame GT often uses local_id; align to scene-level by adding base when needed.
    gt_ids_int = [int(x) for x in gt_vis_ids if isinstance(x, (int, float, str))]
    gt_base = 0
    if gt_ids_int:
        mn = min(gt_ids_int)
        if mn >= 1000:
            gt_base = (mn // 1000) * 1000

    # hit counts
    thr05_key = "thr_0.50"
    thr01_key = "thr_0.10"
    hit05 = hit_cnt_per_gt.get(thr05_key)
    hit01 = hit_cnt_per_gt.get(thr01_key)
    if not (isinstance(hit05, list) and isinstance(hit01, list) and len(hit05) == len(gt_vis_ids) and len(hit01) == len(gt_vis_ids)):
        hit05 = [0] * len(gt_vis_ids)
        hit01 = [0] * len(gt_vis_ids)

    # per-GT visibility across frames (requires per-frame gt_vis_ids)
    vis_frames: Dict[int, List[int]] = defaultdict(list)
    # per-GT associated track id sequence across visible frames
    assoc_seq: Dict[int, List[Optional[int]]] = {int(g): [] for g in gt_vis_ids}
    assoc_track_set: Dict[int, set] = {int(g): set() for g in gt_vis_ids}

    # killed-useful frames per GT (count frames where gt appears in hist)
    killed_useful_frames: Dict[int, int] = defaultdict(int)

    for fr in frames:
        fi = int(fr.get("frame_i", -1))
        # frame visible GT ids (optional)
        frame_vis_ids = None
        try:
            frame_vis_ids = fr.get("pre_pool", {}).get("gt", {}).get("gt_vis_ids")
            if isinstance(frame_vis_ids, list):
                out_meta["has_frame_gt_vis_ids"] = True
        except Exception:
            frame_vis_ids = None
        if isinstance(frame_vis_ids, list) and gt_base > 0:
            try:
                # If frame ids look like local indices, shift into global space.
                if frame_vis_ids and max(int(x) for x in frame_vis_ids) < 1000:
                    frame_vis_ids = [gt_base + int(x) for x in frame_vis_ids]
            except Exception:
                pass
        if isinstance(frame_vis_ids, list):
            for gid in frame_vis_ids:
                if int(gid) in assoc_seq:
                    vis_frames[int(gid)].append(fi)

        # killed useful hist
        hist = fr.get("killed_by_inst_thr", {}).get("useful_gt_hist")
        if isinstance(hist, dict):
            out_meta["has_killed_useful_hist"] = True
            for k in hist.keys():
                try:
                    gid = int(k)
                    if gt_base > 0 and gid < 1000:
                        gid = gt_base + gid
                    killed_useful_frames[gid] += 1
                except Exception:
                    continue

        # det_to_merge per-det GT match arrays
        det_pm = fr.get("det_to_merge", {}).get("pred_match", {})
        if isinstance(det_pm, dict):
            out_meta["has_frame_det_to_gt"] = True
        det_best_gt = det_pm.get("best_gt_id")
        det_best_iou = det_pm.get("best_iou")

        online = fr.get("online", {})
        matched_det_idx = online.get("matched_det_idx")
        matched_track_ids = online.get("matched_track_ids")
        if isinstance(matched_track_ids, list):
            out_meta["has_online_track_ids"] = True

        if not (isinstance(det_best_gt, list) and isinstance(det_best_iou, list) and isinstance(matched_det_idx, list) and isinstance(matched_track_ids, list)):
            continue
        if len(matched_det_idx) != len(matched_track_ids):
            continue

        # Build per-GT association for this frame: choose the strongest matched det for each GT (by det best_iou).
        best_for_gt: Dict[int, Tuple[float, int]] = {}
        for det_i, tid in zip(matched_det_idx, matched_track_ids):
            if det_i is None:
                continue
            det_i = int(det_i)
            if det_i < 0 or det_i >= len(det_best_gt):
                continue
            gid = int(det_best_gt[det_i]) if det_best_gt[det_i] is not None else -1
            if gt_base > 0 and gid >= 0 and gid < 1000:
                gid = gt_base + gid
            if gid < 0 or gid not in assoc_seq:
                continue
            iou = float(det_best_iou[det_i]) if det_best_iou[det_i] is not None else 0.0
            # Only consider weak-overlap as association evidence.
            if iou < float(iou_lo):
                continue
            prev = best_for_gt.get(gid)
            if prev is None or iou > prev[0]:
                best_for_gt[gid] = (iou, int(tid))

        # Append to sequences for GTs visible in this frame; otherwise skip (keeps seq aligned to visibility).
        if isinstance(frame_vis_ids, list):
            for gid in frame_vis_ids:
                gid = int(gid)
                if gid not in assoc_seq:
                    continue
                if gid in best_for_gt:
                    tid = best_for_gt[gid][1]
                    assoc_seq[gid].append(tid)
                    assoc_track_set[gid].add(tid)
                else:
                    assoc_seq[gid].append(None)

    gt_rows: List[GTTrackStats] = []
    for idx, gid in enumerate(gt_vis_ids):
        gid = int(gid)
        best_iou = float(best_iou_per_gt[idx])
        hit05_i = int(hit05[idx]) if idx < len(hit05) else 0
        hit01_i = int(hit01[idx]) if idx < len(hit01) else 0
        seq = assoc_seq.get(gid, [])
        vis_n = len(seq)
        assoc_n = sum(1 for x in seq if x is not None)
        assoc_cov = float(assoc_n / vis_n) if vis_n > 0 else float("nan")
        gt_rows.append(
            GTTrackStats(
                gt_id=gid,
                best_iou=best_iou,
                hit_cnt_05=hit05_i,
                hit_cnt_01=hit01_i,
                num_tracks_associated=len(assoc_track_set.get(gid, set())),
                assoc_coverage=assoc_cov,
                id_switch=_id_switches(seq),
                fragmentation=_contiguous_segments(seq),
                killed_useful_frames=int(killed_useful_frames.get(gid, 0)),
            )
        )

    # Aggregate gray-zone
    gray = [r for r in gt_rows if (r.best_iou >= float(iou_lo) and r.best_iou < float(iou_hi))]
    out_meta["gt_total"] = len(gt_rows)
    out_meta["gt_gray"] = len(gray)
    if gray:
        out_meta["gray_assoc_cov_p50"] = _percentile([r.assoc_coverage for r in gray if r.assoc_coverage == r.assoc_coverage], 0.5)
        out_meta["gray_num_tracks_assoc_p50"] = _percentile([float(r.num_tracks_associated) for r in gray], 0.5)
        out_meta["gray_frag_p50"] = _percentile([float(r.fragmentation) for r in gray], 0.5)
        out_meta["gray_idsw_p50"] = _percentile([float(r.id_switch) for r in gray], 0.5)
    return out_meta, gt_rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online-workdir", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--iou-lo", type=float, default=0.1)
    ap.add_argument("--iou-hi", type=float, default=0.5)
    args = ap.parse_args()

    wd = Path(args.online_workdir)
    bs_path = wd / "baseline_stats" / "baseline_stats.json"
    if not bs_path.exists():
        raise SystemExit(f"missing: {bs_path}")
    bs = _read_json(bs_path)
    if not isinstance(bs, list):
        raise SystemExit("baseline_stats.json must be a list")

    out_dir = Path(args.out_dir) if args.out_dir else (wd / "track_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    scene_summaries: List[Dict[str, Any]] = []
    gray_gt_rows: List[Dict[str, Any]] = []
    all_gt_rows: List[Dict[str, Any]] = []

    for s in bs:
        meta, rows = analyze_scene(s, iou_lo=args.iou_lo, iou_hi=args.iou_hi)
        scene_summaries.append(meta)
        for r in rows:
            row = r.__dict__.copy()
            row["scene_id"] = meta["scene_id"]
            all_gt_rows.append(row)
            if r.best_iou >= args.iou_lo and r.best_iou < args.iou_hi:
                gray_gt_rows.append(row)

    (out_dir / "scene_summary.json").write_text(json.dumps(scene_summaries, indent=2, ensure_ascii=False))
    (out_dir / "gt_rows_all.json").write_text(json.dumps(all_gt_rows, indent=2, ensure_ascii=False))
    (out_dir / "gt_rows_grayzone.json").write_text(json.dumps(gray_gt_rows, indent=2, ensure_ascii=False))

    # Print a minimal console summary.
    n_ok = sum(1 for s in scene_summaries if s.get("has_scene_best_iou") and s.get("has_online_track_ids"))
    print(f"scenes: {len(scene_summaries)}; scenes_with_best_iou+track_ids: {n_ok}")
    gray_total = sum(int(s.get('gt_gray', 0)) for s in scene_summaries)
    gt_total = sum(int(s.get('gt_total', 0)) for s in scene_summaries)
    print(f"gt_total={gt_total} gray(0.1-0.5)={gray_total} rate={gray_total/max(gt_total,1):.3f}")
    print(f"wrote: {out_dir}/scene_summary.json")


if __name__ == "__main__":
    main()
