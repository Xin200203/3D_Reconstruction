#!/usr/bin/env python3
"""
Analyze why online scene-level AP drops vs single-frame, using existing
baseline_stats / online_monitor dumps.

This script does NOT require rerunning evaluation. It consumes:
  - <online_workdir>/baseline_stats/baseline_stats_summary.json
  - <online_workdir>/baseline_stats/baseline_stats.json
  - <online_workdir>/online_monitor/online_monitor_summary.json
  - <online_workdir>/online_monitor/online_monitor.json
Optionally:
  - <single_frame_diag>/instance_error_diagnostics.json

Outputs:
  - stdout: concise report with evidence
  - --out-csv: per-scene table (for deeper digging)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _get_quantile(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * q))
    idx = max(0, min(idx, len(values_sorted) - 1))
    return float(values_sorted[idx])


def _spearman(xs: List[float], ys: List[float]) -> float:
    assert len(xs) == len(ys)
    n = len(xs)
    if n < 3:
        return float("nan")

    def rank(vs: List[float]) -> List[float]:
        pairs = sorted([(v, i) for i, v in enumerate(vs)], key=lambda t: t[0])
        ranks = [0.0] * len(vs)
        i = 0
        while i < len(pairs):
            j = i
            while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
                j += 1
            r = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[pairs[k][1]] = r
            i = j + 1
        return ranks

    rx = rank(xs)
    ry = rank(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    denx = math.sqrt(sum((r - mx) ** 2 for r in rx))
    deny = math.sqrt(sum((r - my) ** 2 for r in ry))
    if denx == 0 or deny == 0:
        return float("nan")
    return float(num / (denx * deny))


def _safe_get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


@dataclass
class SceneRow:
    scene_id: str
    num_frames: int
    n_gt_scene: int
    scene_hit0_iou05: float
    scene_hit_ge2_iou05: float
    scene_hit0_iou01: float
    scene_hit_ge2_iou01: float
    det_hit0_iou05_mean: float
    det_hit_ge2_iou05_mean: float
    det_hit0_iou01_mean: float
    det_hit_ge2_iou01_mean: float
    pre_hit0_iou05_mean: float
    pre_hit_ge2_iou05_mean: float
    pre_hit0_iou01_mean: float
    pre_hit_ge2_iou01_mean: float
    inst_thr_killed_useful_mean: float
    mem_full_p95: float
    mem_kept_p95: float
    topk_drop_p95: float
    inflation_p95: float


def _mean_of(values: List[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _p95_of(values: List[float]) -> float:
    return _get_quantile(values, 0.95)


def _collect_scene_rows(
    baseline_stats: List[Dict[str, Any]],
    online_monitor: List[Dict[str, Any]],
) -> List[SceneRow]:
    om_by_scene = {s["scene_id"]: s for s in online_monitor}
    rows: List[SceneRow] = []

    for s in baseline_stats:
        scene_id = s["scene_id"]
        frames = s["frames"]
        om = om_by_scene.get(scene_id, {})
        om_frames = om.get("frames", [])

        # NOTE: threshold keys like "thr_0.50" contain a dot, so we must NOT
        # traverse them via a generic "a.b.c" path splitter.
        def collect_metric(source: str, thr: str, key: str) -> List[float]:
            out: List[float] = []
            for f in frames:
                src = f.get(source)
                if not isinstance(src, dict):
                    continue
                dup = src.get("dup_iou")
                if not isinstance(dup, dict):
                    continue
                thr_dict = dup.get(thr)
                if not isinstance(thr_dict, dict):
                    continue
                v = thr_dict.get(key)
                if v is None:
                    continue
                out.append(float(v))
            return out

        # frame-level dup/miss
        det_hit0_iou05 = collect_metric("det_to_merge", "thr_0.50", "hit0")
        det_hit_ge2_iou05 = collect_metric("det_to_merge", "thr_0.50", "hit_ge2")
        det_hit0_iou01 = collect_metric("det_to_merge", "thr_0.10", "hit0")
        det_hit_ge2_iou01 = collect_metric("det_to_merge", "thr_0.10", "hit_ge2")

        pre_hit0_iou05 = collect_metric("pre_pool", "thr_0.50", "hit0")
        pre_hit_ge2_iou05 = collect_metric("pre_pool", "thr_0.50", "hit_ge2")
        pre_hit0_iou01 = collect_metric("pre_pool", "thr_0.10", "hit0")
        pre_hit_ge2_iou01 = collect_metric("pre_pool", "thr_0.10", "hit_ge2")

        # inst_thr killed useful
        inst_killed_useful: List[float] = []
        for f in frames:
            v = _safe_get(f, "killed_by_inst_thr.killed_useful_any05")
            if v is not None:
                inst_killed_useful.append(float(v))

        # scene-level dup/miss
        sdup = s.get("scene_level_dup", {})
        thr05 = sdup.get("thr_0.50", {})
        thr01 = sdup.get("thr_0.10", {})
        n_gt_scene = int(thr05.get("n_gt", thr01.get("n_gt", 0)) or 0)

        # online monitor distributions (per-scene)
        mem_full = [float(ff.get("mem_size_full", 0)) for ff in om_frames]
        mem_kept = [float(ff.get("mem_size_kept", 0)) for ff in om_frames]
        topk_drop = [float(ff.get("topk_drop", 0)) for ff in om_frames]

        inflation = []
        for f in frames:
            gt_mem = _safe_get(f, "gt.gt_mem")
            mem_full_t = _safe_get(f, "online.mem_size_full")
            if gt_mem is None or mem_full_t is None:
                continue
            gt_mem = float(gt_mem)
            mem_full_t = float(mem_full_t)
            if gt_mem > 0:
                inflation.append(mem_full_t / gt_mem)

        rows.append(
            SceneRow(
                scene_id=scene_id,
                num_frames=int(s.get("num_frames", len(frames))),
                n_gt_scene=n_gt_scene,
                scene_hit0_iou05=float(thr05.get("hit0", float("nan"))),
                scene_hit_ge2_iou05=float(thr05.get("hit_ge2", float("nan"))),
                scene_hit0_iou01=float(thr01.get("hit0", float("nan"))),
                scene_hit_ge2_iou01=float(thr01.get("hit_ge2", float("nan"))),
                det_hit0_iou05_mean=_mean_of(det_hit0_iou05),
                det_hit_ge2_iou05_mean=_mean_of(det_hit_ge2_iou05),
                det_hit0_iou01_mean=_mean_of(det_hit0_iou01),
                det_hit_ge2_iou01_mean=_mean_of(det_hit_ge2_iou01),
                pre_hit0_iou05_mean=_mean_of(pre_hit0_iou05),
                pre_hit_ge2_iou05_mean=_mean_of(pre_hit_ge2_iou05),
                pre_hit0_iou01_mean=_mean_of(pre_hit0_iou01),
                pre_hit_ge2_iou01_mean=_mean_of(pre_hit_ge2_iou01),
                inst_thr_killed_useful_mean=_mean_of(inst_killed_useful),
                mem_full_p95=_p95_of(mem_full),
                mem_kept_p95=_p95_of(mem_kept),
                topk_drop_p95=_p95_of(topk_drop),
                inflation_p95=_p95_of(inflation),
            )
        )

    return rows


def _print_key_findings(
    online_dir: Path,
    baseline_summary: Dict[str, Any],
    online_summary: Dict[str, Any],
    scene_rows: List[SceneRow],
    single_frame_diag: Optional[Dict[str, Any]] = None,
) -> None:
    print("== Inputs ==")
    print(f"online_workdir: {online_dir}")
    print()

    # overall: show the “gates”
    stage_counts = baseline_summary.get("stage_counts", {})
    stage_drops = baseline_summary.get("stage_drops", {})
    print("== Supply Gates (per-frame) ==")
    for k in [
        "after_topk",
        "after_nms",
        "after_geom_merge",
        "after_inst_thr",
        "after_npoint_thr",
        "after_copy_suppress",
    ]:
        if k in stage_counts:
            print(f"{k}: mean={stage_counts[k]['mean']:.3f}  p95={stage_counts[k]['p95']:.3f}")
    for k in ["drop_inst_thr", "drop_npoint_thr", "drop_nms", "drop_geom_merge", "drop_copy_suppress"]:
        if k in stage_drops:
            print(f"{k}: mean={stage_drops[k]['mean']:.3f}  p95={stage_drops[k]['p95']:.3f}")

    print()
    print("== Killed Useful (IoU/Cov >= 0.5 among killed) ==")
    killed = baseline_summary.get("killed_useful_iou05", {})
    inst_any = _safe_get(killed, "inst_thr.any.mean")
    inst_iou = _safe_get(killed, "inst_thr.iou.mean")
    npt_any = _safe_get(killed, "npoint_thr.any.mean")
    print(f"inst_thr: killed_useful_any05.mean={inst_any:.3f}  killed_useful_iou05.mean={inst_iou:.3f}")
    print(f"npoint_thr: killed_useful_any05.mean={npt_any:.3f}")

    print()
    print("== Dup/Miss (GT view) ==")
    dup05 = baseline_summary.get("dup_gt_iou05", {})
    dup01 = baseline_summary.get("dup_gt_iou01", {})
    print(f"pre_pool hit0@0.5 mean={dup05['pre_pool_hit0']['mean']:.4f} ; det_to_merge hit0@0.5 mean={dup05['det_to_merge_hit0']['mean']:.4f}")
    print(f"pre_pool hit_ge2@0.5 mean={dup05['pre_pool_hit_ge2']['mean']:.4f} ; det_to_merge hit_ge2@0.5 mean={dup05['det_to_merge_hit_ge2']['mean']:.4f}")
    print(f"pre_pool hit_ge2@0.1 mean={dup01['pre_pool_hit_ge2']['mean']:.4f} ; det_to_merge hit_ge2@0.1 mean={dup01['det_to_merge_hit_ge2']['mean']:.4f}")

    print()
    print("== Scene-level Dup/Miss (final map vs GT) ==")
    s05 = baseline_summary.get("scene_dup_iou05", {})
    s01 = baseline_summary.get("scene_dup_iou01", {})
    print(f"scene hit0@0.5 mean={s05['hit0']['mean']:.4f}  p95={s05['hit0']['p95']:.4f}")
    print(f"scene hit_ge2@0.5 mean={s05['hit_ge2']['mean']:.4f}  p95={s05['hit_ge2']['p95']:.4f}")
    print(f"scene hit_ge2@0.1 mean={s01['hit_ge2']['mean']:.4f}  p95={s01['hit_ge2']['p95']:.4f}  (fragmentation/ambiguous overlap flag)")

    print()
    print("== Online Dynamics ==")
    for k in ["det_to_merge", "matched", "birth", "mem_size_full", "mem_size_kept", "topk_drop"]:
        if k in online_summary:
            s = online_summary[k]
            print(f"{k}: mean={s['mean']:.3f}  p95={s['p95']:.3f}")
    infl = baseline_summary.get("inflation", {})
    print(f"inflation=mem_full/gt_mem: mean={infl.get('mean', float('nan')):.3f}  p95={infl.get('p95', float('nan')):.3f}")

    print()
    # correlations (per scene)
    valid = [r for r in scene_rows if r.n_gt_scene > 0 and math.isfinite(r.scene_hit0_iou05)]
    if valid:
        hit0 = [r.scene_hit0_iou05 for r in valid]
        topk = [r.topk_drop_p95 for r in valid]
        infl = [r.inflation_p95 for r in valid]
        n_gt = [float(r.n_gt_scene) for r in valid]
        det_hit0 = [r.det_hit0_iou05_mean for r in valid]
        corr_topk = _spearman(hit0, topk)
        corr_infl = _spearman(hit0, infl)
        corr_ngt = _spearman(hit0, n_gt)
        corr_supply = _spearman(hit0, det_hit0)
        print("== Correlations to scene miss hit0@0.5 (Spearman) ==")
        print(f"corr(hit0_scene, topk_drop_p95) = {corr_topk:.3f}")
        print(f"corr(hit0_scene, inflation_p95) = {corr_infl:.3f}")
        print(f"corr(hit0_scene, n_gt_scene) = {corr_ngt:.3f}")
        print(f"corr(hit0_scene, det_to_merge_hit0@0.5_mean) = {corr_supply:.3f}")

        # worst scenes
        worst = sorted(valid, key=lambda r: r.scene_hit0_iou05, reverse=True)[:10]
        print("\n== Worst 10 scenes by scene hit0@0.5 ==")
        for r in worst:
            print(
                f"{r.scene_id}: hit0@0.5={r.scene_hit0_iou05:.3f} "
                f"n_gt={r.n_gt_scene} topk_drop_p95={r.topk_drop_p95:.0f} "
                f"infl_p95={r.inflation_p95:.2f} det_hit0@0.5_mean={r.det_hit0_iou05_mean:.3f}"
            )

    if single_frame_diag is not None:
        print()
        print("== Single-frame diagnostics (reference) ==")
        miss = single_frame_diag.get("miss_rates", {})
        diag = single_frame_diag.get("diagnosis", {})
        print(f"single-frame gt_best_iou<0.5_rate_ge100 = {miss.get('gt_best_iou_0_0p5_rate_ge_100'):.4f}")
        print(f"single-frame hit0@0.1_rate_ge100 = {miss.get('gt_hit_zero_rate_iou_lo_ge_100'):.4f}")
        print(f"single-frame gt_hit_ge2_rate (dup@0.5) = {diag.get('rates',{}).get('gt_hit_ge2_rate'):.4f}")
        print("note: online scene hit0@0.5 is higher and dup@0.5 is far lower => online is not failing because of strict-IoU duplicates; it is losing recall/IoU quality across time.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--online-workdir", type=str, required=True)
    ap.add_argument("--single-frame-diagnostics", type=str, default=None)
    ap.add_argument("--out-csv", type=str, default=None)
    args = ap.parse_args()

    online_dir = Path(args.online_workdir)
    baseline_summary = _read_json(online_dir / "baseline_stats" / "baseline_stats_summary.json")
    online_summary = _read_json(online_dir / "online_monitor" / "online_monitor_summary.json")
    baseline_stats = _read_json(online_dir / "baseline_stats" / "baseline_stats.json")
    online_monitor = _read_json(online_dir / "online_monitor" / "online_monitor.json")

    single_frame_diag = None
    if args.single_frame_diagnostics:
        single_frame_diag = _read_json(Path(args.single_frame_diagnostics))

    rows = _collect_scene_rows(baseline_stats, online_monitor)
    _print_key_findings(online_dir, baseline_summary, online_summary, rows, single_frame_diag)

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow([field.name for field in SceneRow.__dataclass_fields__.values()])
            for r in rows:
                w.writerow([getattr(r, field.name) for field in SceneRow.__dataclass_fields__.values()])
        print()
        print(f"Wrote per-scene table: {out}")


if __name__ == "__main__":
    main()
