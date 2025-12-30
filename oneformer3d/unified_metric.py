import torch
import numpy as np
import os
import json
from pathlib import Path
from typing import Sequence, Optional
from mmengine.logging import MMLogger

from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS
from mmdet3d.evaluation import panoptic_seg_eval, seg_eval
from .instance_seg_eval import instance_seg_eval, instance_cat_agnostic_eval

@METRICS.register_module()
class UnifiedSegMetric(SegMetric):
    """Metric for instance, semantic, and panoptic evaluation.
    The order of classes must be [stuff classes, thing classes, unlabeled].

    Args:
        thing_class_inds (List[int]): Ids of thing classes.
        stuff_class_inds (List[int]): Ids of stuff classes.
        min_num_points (int): Minimal size of mask for panoptic segmentation.
        id_offset (int): Offset for instance classes.
        sem_mapping (List[int]): Semantic class to gt id.
        inst_mapping (List[int]): Instance class to gt id.
        metric_meta (Dict): Analogue of dataset meta of SegMetric. Keys:
            `label2cat` (Dict[int, str]): class names,
            `ignore_index` (List[int]): ids of semantic categories to ignore,
            `classes` (List[str]): class names.
        logger_keys (List[Tuple]): Keys for logger to save; of len 3:
            semantic, instance, and panoptic.
        eval_mode (str): Evaluation mode. Must be 'auto', 'multi_class', or 'cat_agnostic'.
    """

    def __init__(self,
                 thing_class_inds,
                 stuff_class_inds,
                 min_num_points,
                 id_offset,
                 sem_mapping,   
                 inst_mapping,
                 metric_meta,
                 logger_keys=[('miou',),
                              ('all_ap', 'all_ap_50%', 'all_ap_25%'), 
                              ('pq',)],
                 eval_mode: str = 'auto',
                 diagnostics: Optional[dict] = None,
                 online_monitor: Optional[dict] = None,
                 **kwargs):
        self.thing_class_inds = thing_class_inds
        self.stuff_class_inds = stuff_class_inds
        self.min_num_points = min_num_points
        self.id_offset = id_offset
        self.metric_meta = metric_meta
        # self.logger_keys = logger_keys
        self.logger_keys = [('all_ap', 'all_ap_50%', 'all_ap_25%')]
        self.sem_mapping = np.array(sem_mapping)
        self.inst_mapping = np.array(inst_mapping)
        assert eval_mode in ('auto', 'multi_class', 'cat_agnostic'), \
            "eval_mode must be 'auto', 'multi_class', or 'cat_agnostic'"
        self.eval_mode = eval_mode
        self.diagnostics = diagnostics or {}
        self.online_monitor = online_monitor or {}
        super().__init__(**kwargs)

    @staticmethod
    def _to_numpy_int(x) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.int64)

    @staticmethod
    def _to_numpy_float(x) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        return np.asarray(x, dtype=np.float32)

    @staticmethod
    def _to_numpy_bool(x) -> np.ndarray:
        if torch.is_tensor(x):
            x = x.detach().cpu().numpy()
        arr = np.asarray(x)
        if arr.dtype != np.bool_:
            arr = arr.astype(np.bool_)
        return arr

    @staticmethod
    def _resolve_diagnostics_dir(logger: MMLogger, out_dir: str) -> str:
        # Prefer logger's work dir if available; otherwise fall back to cwd.
        base = None
        for attr in ("log_dir", "work_dir", "output_dir"):
            base = getattr(logger, attr, None)
            if base:
                break
        if not base:
            base = os.getcwd()
        if os.path.isabs(out_dir):
            return out_dir
        return os.path.join(str(base), out_dir)

    @staticmethod
    def _summarize_online_monitor(monitors: Sequence[dict]) -> dict:
        """Summarize online behavior stats over all scenes/frames.

        Expected input format is a list of dicts with key "frames", where each
        frame is a dict of scalar stats (det_to_merge/matched/birth/mem sizes).
        """
        def _collect(key: str) -> np.ndarray:
            vals = []
            for s in monitors:
                frames = s.get("frames", [])
                if not isinstance(frames, list):
                    continue
                for fr in frames:
                    if isinstance(fr, dict) and key in fr:
                        v = fr.get(key)
                        if isinstance(v, (int, float, np.number)):
                            vals.append(float(v))
            return np.asarray(vals, dtype=np.float32)

        def _pack(x: np.ndarray) -> dict:
            if x.size == 0:
                return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0}
            return {
                "n": int(x.size),
                "mean": float(np.mean(x)),
                "median": float(np.median(x)),
                "p90": float(np.percentile(x, 90)),
                "p95": float(np.percentile(x, 95)),
            }

        det = _collect("det_to_merge")
        matched = _collect("matched")
        birth = _collect("birth")
        mem_full = _collect("mem_size_full")
        mem_kept = _collect("mem_size_kept")
        topk_drop = _collect("topk_drop")

        # Derived rates (frame-level).
        if det.size and matched.size and matched.size == det.size:
            match_rate = np.divide(matched, np.maximum(det, 1.0))
        else:
            match_rate = np.asarray([], dtype=np.float32)
        if det.size and birth.size and birth.size == det.size:
            birth_rate = np.divide(birth, np.maximum(det, 1.0))
        else:
            birth_rate = np.asarray([], dtype=np.float32)

        n_scenes = int(len(monitors))
        n_frames = int(det.size) if det.size else int(sum(len(s.get("frames", [])) for s in monitors if isinstance(s.get("frames", None), list)))

        return {
            "counts": {
                "scenes": n_scenes,
                "frames": n_frames,
            },
            "det_to_merge": _pack(det),
            "matched": _pack(matched),
            "birth": _pack(birth),
            "match_rate": _pack(match_rate),
            "birth_rate": _pack(birth_rate),
            "mem_size_full": _pack(mem_full),
            "mem_size_kept": _pack(mem_kept),
            "topk_drop": _pack(topk_drop),
        }

    @staticmethod
    def _diagnose_bottleneck(
        *,
        pred_purity: np.ndarray,
        pred_strong_gt_cnt: np.ndarray,
        gt_best_iou: np.ndarray,
        gt_hit_cnt: np.ndarray,
        purity_low_thr: float,
        purity_high_thr: float,
        iou_mid_lo: float,
        iou_mid_hi: float,
    ) -> dict:
        # Rates used by the heuristic rules.
        purity_low_rate = float(np.mean(pred_purity < purity_low_thr)) if pred_purity.size else 0.0
        purity_high_rate = float(np.mean(pred_purity > purity_high_thr)) if pred_purity.size else 0.0
        multi_gt_strong_rate = float(np.mean(pred_strong_gt_cnt >= 2)) if pred_strong_gt_cnt.size else 0.0
        multi_gt_strong_lowpur_rate = float(
            np.mean((pred_purity < purity_low_thr) & (pred_strong_gt_cnt >= 2))
        ) if pred_purity.size else 0.0

        gt_best_iou_mid_rate = float(
            np.mean((gt_best_iou >= iou_mid_lo) & (gt_best_iou < iou_mid_hi))
        ) if gt_best_iou.size else 0.0
        gt_hit_ge2_rate = float(np.mean(gt_hit_cnt >= 2)) if gt_hit_cnt.size else 0.0

        # Soft scores; pick the largest as the primary bottleneck.
        sticky_score = purity_low_rate * (multi_gt_strong_lowpur_rate + 1e-6)
        boundary_score = purity_high_rate * (gt_best_iou_mid_rate + 1e-6)
        dup_score = gt_hit_ge2_rate

        scores = {
            "sticky_underseg": float(sticky_score),
            "boundary_alignment": float(boundary_score),
            "duplicate_instances": float(dup_score),
        }
        primary = max(scores, key=scores.get)

        suggestions = {
            "sticky_underseg": [
                "box prompt / 2D det guidance",
                "depth ownership / z-buffer assignment",
                "3D split (geometric cut / graph split / boundary cues)",
            ],
            "boundary_alignment": [
                "finer 2D→3D projection sampling",
                "boundary/contour losses",
                "local geometric constraints",
                "increase point density / TSDF completion",
            ],
            "duplicate_instances": [
                "mask NMS / matrix NMS tuning",
                "query de-dup / diversity regularization",
                "score re-calibration",
                "merge strategy (spatial + feature consistency)",
            ],
        }

        return {
            "primary": primary,
            "scores": scores,
            "rates": {
                "purity_low_rate": purity_low_rate,
                "purity_high_rate": purity_high_rate,
                "multi_gt_strong_rate": multi_gt_strong_rate,
                "multi_gt_strong_lowpur_rate": multi_gt_strong_lowpur_rate,
                "gt_best_iou_mid_rate": gt_best_iou_mid_rate,
                "gt_hit_ge2_rate": gt_hit_ge2_rate,
            },
            "suggestions": suggestions[primary],
        }

    def _run_instance_error_diagnostics(
        self,
        *,
        logger: MMLogger,
        gt_instance_masks: Sequence[np.ndarray],
        pred_instance_masks: Sequence[np.ndarray],
        pred_instance_scores: Optional[Sequence[np.ndarray]],
        out_dir: str,
        iou_thr: float = 0.5,
        iou_lo_thr: float = 0.1,
        purity_low_thr: float = 0.85,
        purity_high_thr: float = 0.90,
        iou_mid_lo: float = 0.6,
        iou_mid_hi: float = 0.8,
        max_pred_per_scene: Optional[int] = None,
        gt_size_thr: int = 100,
        oracle_ks: Optional[Sequence[int]] = None,
        rank_bins: Optional[Sequence[int]] = None,
    ) -> None:
        out_dir = self._resolve_diagnostics_dir(logger, out_dir)
        Path(out_dir).mkdir(parents=True, exist_ok=True)

        pred_purity = []
        pred_coverage = []
        pred_best_iou = []
        pred_strong_gt_cnt = []

        gt_best_iou_all = []
        gt_hit_cnt_all = []
        gt_hit_cnt_lo_all = []
        gt_sizes_all = []

        # Oracle@K and best-IoU rank diagnostics (require per-pred scores).
        oracle_ks_sorted = sorted({int(k) for k in (oracle_ks or (20, 30, 50, 100)) if int(k) > 0})
        rank_bins_sorted = sorted({int(b) for b in (rank_bins or (10, 20, 50, 100)) if int(b) > 0})
        gt_best_rank_all = []  # 1-based rank in score-sorted preds; 0 if undefined.
        gt_oracle_best_iou_by_k = {k: [] for k in oracle_ks_sorted}
        n_scenes_oracle_used = 0
        n_gt_oracle_used = 0

        n_scenes_used = 0
        n_pred_used = 0
        n_gt_used = 0
        gt_small_total = 0
        gt_large_total = 0
        gt_small_miss = 0
        gt_large_miss = 0
        gt_small_best_zero = 0
        gt_large_best_zero = 0
        gt_small_best_low = 0
        gt_large_best_low = 0
        gt_small_miss_lo = 0
        gt_large_miss_lo = 0

        for scene_i, (gt_ids_raw, pred_masks_raw) in enumerate(zip(gt_instance_masks, pred_instance_masks)):
            gt_ids = self._to_numpy_int(gt_ids_raw).reshape(-1)
            valid = gt_ids >= 0
            if not np.any(valid):
                continue
            # Map GT instance ids to dense [0..G-1] to avoid sparse max-id artifacts.
            unique_ids = np.unique(gt_ids[valid])
            if unique_ids.size == 0:
                continue
            # unique_ids is sorted; map by searchsorted.
            mapped = np.full_like(gt_ids, -1)
            mapped[valid] = np.searchsorted(unique_ids, gt_ids[valid])

            pred_masks = self._to_numpy_bool(pred_masks_raw)
            if pred_masks.ndim != 2:
                continue

            # Normalize shape to (Npred, Npts).
            if pred_masks.shape[1] != gt_ids.shape[0] and pred_masks.shape[0] == gt_ids.shape[0]:
                pred_masks = pred_masks.T
            if pred_masks.shape[1] != gt_ids.shape[0]:
                continue

            if max_pred_per_scene is not None and pred_masks.shape[0] > int(max_pred_per_scene):
                # Keep top-K by score (if provided), otherwise keep the first K.
                if pred_instance_scores is not None and scene_i < len(pred_instance_scores):
                    scores = pred_instance_scores[scene_i]
                    scores = np.asarray(scores).reshape(-1)
                    if scores.shape[0] == pred_masks.shape[0]:
                        keep = np.argsort(scores)[-int(max_pred_per_scene):]
                        pred_masks = pred_masks[keep]
                    else:
                        pred_masks = pred_masks[: int(max_pred_per_scene)]
                else:
                    pred_masks = pred_masks[: int(max_pred_per_scene)]

            # Per-GT stats for this scene (dense ids).
            G = int(unique_ids.size)
            gt_sizes = np.bincount(mapped[valid], minlength=G).astype(np.int64)
            gt_best_iou = np.zeros((G,), dtype=np.float32)
            gt_hit_cnt = np.zeros((G,), dtype=np.int64)
            gt_hit_cnt_lo = np.zeros((G,), dtype=np.int64)

            # Per-GT best-pred tracking for rank diagnostics.
            best_pred_idx = np.full((G,), -1, dtype=np.int32)
            best_pred_score = np.full((G,), -1e9, dtype=np.float32)

            scene_scores = None
            if pred_instance_scores is not None and scene_i < len(pred_instance_scores):
                s = np.asarray(pred_instance_scores[scene_i]).reshape(-1)
                if s.shape[0] == pred_masks.shape[0]:
                    scene_scores = s.astype(np.float32, copy=False)

            for pi in range(pred_masks.shape[0]):
                pm = pred_masks[pi] & valid
                pred_size = int(pm.sum())
                if pred_size <= 0:
                    continue

                labels = mapped[pm]
                if labels.size == 0:
                    continue
                inter = np.bincount(labels, minlength=G).astype(np.int64)
                union = (pred_size + gt_sizes - inter).astype(np.float32)
                union = np.maximum(union, 1.0)
                iou = inter.astype(np.float32) / union

                best_g = int(np.argmax(iou))
                best_i = float(iou[best_g])

                purity = float(inter[best_g] / max(pred_size, 1))
                coverage = float(inter[best_g] / max(int(gt_sizes[best_g]), 1))

                pred_purity.append(purity)
                pred_coverage.append(coverage)
                pred_best_iou.append(best_i)
                pred_strong_gt_cnt.append(int(np.sum(iou >= float(iou_thr))))

                # Track, for each GT, which prediction achieves the best IoU.
                # IMPORTANT: compare against the *previous* best IoU (before updating).
                if scene_scores is not None:
                    score_pi = float(scene_scores[pi])
                    better_iou = iou > (gt_best_iou + 1e-12)
                    tie_iou = np.abs(iou - gt_best_iou) <= 1e-12
                    better = better_iou | (tie_iou & (score_pi > best_pred_score))
                    if np.any(better):
                        best_pred_idx[better] = int(pi)
                        best_pred_score[better] = score_pi
                else:
                    # If no per-instance scores are provided, still track by IoU only.
                    better = iou > (gt_best_iou + 1e-12)
                    if np.any(better):
                        best_pred_idx[better] = int(pi)

                gt_best_iou = np.maximum(gt_best_iou, iou)
                gt_hit_cnt += (iou >= float(iou_thr)).astype(np.int64)
                gt_hit_cnt_lo += (iou >= float(iou_lo_thr)).astype(np.int64)

                n_pred_used += 1

            # Oracle@K (requires scores) computed on top-score predictions.
            if scene_scores is not None and oracle_ks_sorted:
                order = np.argsort(scene_scores)[::-1]
                rank_map = np.empty((pred_masks.shape[0],), dtype=np.int32)
                rank_map[order] = np.arange(pred_masks.shape[0], dtype=np.int32)

                # Best-IoU rank for each GT (1-based).
                gt_best_rank = np.zeros((G,), dtype=np.int32)
                has_best = best_pred_idx >= 0
                if np.any(has_best):
                    gt_best_rank[has_best] = rank_map[best_pred_idx[has_best]] + 1
                gt_best_rank_all.extend(gt_best_rank.tolist())

                # Oracle curve: max IoU per GT within top-K predictions by score.
                max_k = int(min(int(order.size), int(max(oracle_ks_sorted))))
                oracle_running = np.zeros((G,), dtype=np.float32)
                snapshots = {}
                snap_ks = sorted({min(int(k), int(order.size)) for k in oracle_ks_sorted})
                snap_set = set(snap_ks)
                for oi in range(max_k):
                    pi = int(order[oi])
                    pm = pred_masks[pi] & valid
                    pred_size = int(pm.sum())
                    if pred_size <= 0:
                        continue
                    labels = mapped[pm]
                    if labels.size == 0:
                        continue
                    inter = np.bincount(labels, minlength=G).astype(np.int64)
                    union = (pred_size + gt_sizes - inter).astype(np.float32)
                    union = np.maximum(union, 1.0)
                    iou = inter.astype(np.float32) / union
                    oracle_running = np.maximum(oracle_running, iou)
                    if (oi + 1) in snap_set and (oi + 1) not in snapshots:
                        snapshots[oi + 1] = oracle_running.copy()

                # Append per-GT oracle best IoU for requested Ks (mapped to available K).
                for k in oracle_ks_sorted:
                    k_eff = int(min(int(k), int(order.size)))
                    if k_eff <= 0:
                        arr = np.zeros((G,), dtype=np.float32)
                    else:
                        arr = snapshots.get(k_eff, oracle_running)
                    gt_oracle_best_iou_by_k[k].extend(arr.tolist())

                n_scenes_oracle_used += 1
                n_gt_oracle_used += int(G)

            gt_best_iou_all.extend(gt_best_iou.tolist())
            gt_hit_cnt_all.extend(gt_hit_cnt.tolist())
            gt_hit_cnt_lo_all.extend(gt_hit_cnt_lo.tolist())
            gt_sizes_all.extend(gt_sizes.tolist())
            n_gt_used += int(G)
            n_scenes_used += 1
            # Miss rate by GT size bucket.
            if gt_sizes.size:
                miss = gt_hit_cnt == 0
                small = gt_sizes < int(gt_size_thr)
                best_zero = gt_best_iou == 0
                best_low = (gt_best_iou > 0) & (gt_best_iou < 0.5)
                miss_lo = gt_hit_cnt_lo == 0
                gt_small_total += int(small.sum())
                gt_large_total += int((~small).sum())
                gt_small_miss += int((miss & small).sum())
                gt_large_miss += int((miss & ~small).sum())
                gt_small_best_zero += int((best_zero & small).sum())
                gt_large_best_zero += int((best_zero & ~small).sum())
                gt_small_best_low += int((best_low & small).sum())
                gt_large_best_low += int((best_low & ~small).sum())
                gt_small_miss_lo += int((miss_lo & small).sum())
                gt_large_miss_lo += int((miss_lo & ~small).sum())

        pred_purity_np = np.asarray(pred_purity, dtype=np.float32)
        pred_coverage_np = np.asarray(pred_coverage, dtype=np.float32)
        pred_best_iou_np = np.asarray(pred_best_iou, dtype=np.float32)
        pred_strong_gt_cnt_np = np.asarray(pred_strong_gt_cnt, dtype=np.int64)
        gt_best_iou_np = np.asarray(gt_best_iou_all, dtype=np.float32)
        gt_hit_cnt_np = np.asarray(gt_hit_cnt_all, dtype=np.int64)
        gt_hit_cnt_lo_np = np.asarray(gt_hit_cnt_lo_all, dtype=np.int64)
        gt_sizes_np = np.asarray(gt_sizes_all, dtype=np.int64)

        gt_hit_zero_rate = float(np.mean(gt_hit_cnt_np == 0)) if gt_hit_cnt_np.size else 0.0
        gt_hit_zero_rate_lo = float(np.mean(gt_hit_cnt_lo_np == 0)) if gt_hit_cnt_lo_np.size else 0.0
        gt_best_iou_zero_rate = float(np.mean(gt_best_iou_np == 0)) if gt_best_iou_np.size else 0.0
        gt_best_iou_low_rate = float(np.mean((gt_best_iou_np > 0) & (gt_best_iou_np < 0.5))) if gt_best_iou_np.size else 0.0
        gt_best_iou_lt_0p5_rate = float(np.mean(gt_best_iou_np < 0.5)) if gt_best_iou_np.size else 0.0
        gt_small_miss_rate = (gt_small_miss / gt_small_total) if gt_small_total > 0 else 0.0
        gt_large_miss_rate = (gt_large_miss / gt_large_total) if gt_large_total > 0 else 0.0
        gt_small_best_zero_rate = (gt_small_best_zero / gt_small_total) if gt_small_total > 0 else 0.0
        gt_large_best_zero_rate = (gt_large_best_zero / gt_large_total) if gt_large_total > 0 else 0.0
        gt_small_best_low_rate = (gt_small_best_low / gt_small_total) if gt_small_total > 0 else 0.0
        gt_large_best_low_rate = (gt_large_best_low / gt_large_total) if gt_large_total > 0 else 0.0
        gt_small_miss_lo_rate = (gt_small_miss_lo / gt_small_total) if gt_small_total > 0 else 0.0
        gt_large_miss_lo_rate = (gt_large_miss_lo / gt_large_total) if gt_large_total > 0 else 0.0

        # Oracle@K and best-IoU rank summaries.
        oracle_summary = {
            "available": bool(n_scenes_oracle_used > 0 and n_gt_oracle_used > 0),
            "ks": oracle_ks_sorted,
            "n_scenes_used": int(n_scenes_oracle_used),
            "n_gt_used": int(n_gt_oracle_used),
            "overall": {},
            f"ge_{int(gt_size_thr)}": {},
            f"lt_{int(gt_size_thr)}": {},
        }
        rank_summary = {
            "available": bool(len(gt_best_rank_all) == int(gt_best_iou_np.size) and len(gt_best_rank_all) > 0),
            "bins": rank_bins_sorted,
            "overall": {},
            f"ge_{int(gt_size_thr)}": {},
            f"lt_{int(gt_size_thr)}": {},
        }

        if oracle_summary["available"]:
            is_ge = gt_sizes_np >= int(gt_size_thr)
            for k in oracle_ks_sorted:
                arr = np.asarray(gt_oracle_best_iou_by_k[k], dtype=np.float32)
                if arr.size != gt_best_iou_np.size:
                    continue
                def _pack(x):
                    return {
                        "mean_best_iou": float(np.mean(x)) if x.size else 0.0,
                        "hit_rate_iou_thr": float(np.mean(x >= float(iou_thr))) if x.size else 0.0,
                        "hit_rate_iou_lo": float(np.mean(x >= float(iou_lo_thr))) if x.size else 0.0,
                    }
                oracle_summary["overall"][str(k)] = _pack(arr)
                oracle_summary[f"ge_{int(gt_size_thr)}"][str(k)] = _pack(arr[is_ge])
                oracle_summary[f"lt_{int(gt_size_thr)}"][str(k)] = _pack(arr[~is_ge])

        if rank_summary["available"]:
            ranks = np.asarray(gt_best_rank_all, dtype=np.int32)
            is_ge = gt_sizes_np >= int(gt_size_thr)

            def _rank_pack(mask):
                r = ranks[mask]
                # Exclude undefined ranks (0) for percentile stats.
                r_pos = r[r > 0]
                out = {
                    "n": int(r.size),
                    "n_rank_defined": int(r_pos.size),
                    "median": float(np.median(r_pos)) if r_pos.size else 0.0,
                    "p90": float(np.percentile(r_pos, 90)) if r_pos.size else 0.0,
                    "p95": float(np.percentile(r_pos, 95)) if r_pos.size else 0.0,
                }
                for b in rank_bins_sorted:
                    out[f"le_{int(b)}"] = float(np.mean(r_pos <= int(b))) if r_pos.size else 0.0
                out["gt_50"] = float(np.mean(r_pos > 50)) if r_pos.size else 0.0
                out["gt_100"] = float(np.mean(r_pos > 100)) if r_pos.size else 0.0
                return out

            rank_summary["overall"] = _rank_pack(np.ones_like(ranks, dtype=bool))
            rank_summary[f"ge_{int(gt_size_thr)}"] = _rank_pack(is_ge)
            rank_summary[f"lt_{int(gt_size_thr)}"] = _rank_pack(~is_ge)

        diag = self._diagnose_bottleneck(
            pred_purity=pred_purity_np,
            pred_strong_gt_cnt=pred_strong_gt_cnt_np,
            gt_best_iou=gt_best_iou_np,
            gt_hit_cnt=gt_hit_cnt_np,
            purity_low_thr=float(purity_low_thr),
            purity_high_thr=float(purity_high_thr),
            iou_mid_lo=float(iou_mid_lo),
            iou_mid_hi=float(iou_mid_hi),
        )

        summary = {
            "counts": {
                "scenes_used": int(n_scenes_used),
                "pred_instances": int(n_pred_used),
                "gt_instances": int(n_gt_used),
            },
            "thresholds": {
                "iou_thr": float(iou_thr),
                "iou_lo_thr": float(iou_lo_thr),
                "purity_low_thr": float(purity_low_thr),
                "purity_high_thr": float(purity_high_thr),
                "iou_mid_lo": float(iou_mid_lo),
                "iou_mid_hi": float(iou_mid_hi),
                "max_pred_per_scene": None if max_pred_per_scene is None else int(max_pred_per_scene),
                "gt_size_thr": int(gt_size_thr),
                "oracle_ks": oracle_ks_sorted,
                "rank_bins": rank_bins_sorted,
            },
            "miss_rates": {
                "gt_hit_zero_rate": gt_hit_zero_rate,
                "gt_hit_zero_rate_iou_lo": gt_hit_zero_rate_lo,
                "gt_best_iou_zero_rate": gt_best_iou_zero_rate,
                "gt_best_iou_0_0p5_rate": gt_best_iou_low_rate,
                "gt_best_iou_lt_0p5_rate": gt_best_iou_lt_0p5_rate,
                f"gt_miss_rate_lt_{int(gt_size_thr)}": float(gt_small_miss_rate),
                f"gt_miss_rate_ge_{int(gt_size_thr)}": float(gt_large_miss_rate),
                f"gt_best_iou_zero_rate_lt_{int(gt_size_thr)}": float(gt_small_best_zero_rate),
                f"gt_best_iou_zero_rate_ge_{int(gt_size_thr)}": float(gt_large_best_zero_rate),
                f"gt_best_iou_0_0p5_rate_lt_{int(gt_size_thr)}": float(gt_small_best_low_rate),
                f"gt_best_iou_0_0p5_rate_ge_{int(gt_size_thr)}": float(gt_large_best_low_rate),
                f"gt_hit_zero_rate_iou_lo_lt_{int(gt_size_thr)}": float(gt_small_miss_lo_rate),
                f"gt_hit_zero_rate_iou_lo_ge_{int(gt_size_thr)}": float(gt_large_miss_lo_rate),
                f"gt_count_lt_{int(gt_size_thr)}": int(gt_small_total),
                f"gt_count_ge_{int(gt_size_thr)}": int(gt_large_total),
            },
            "oracle": oracle_summary,
            "best_iou_rank": rank_summary,
            "diagnosis": diag,
        }
        with open(os.path.join(out_dir, "instance_error_diagnostics.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        np.savez_compressed(
            os.path.join(out_dir, "instance_error_diagnostics_raw.npz"),
            pred_purity=pred_purity_np,
            pred_coverage=pred_coverage_np,
            pred_best_iou=pred_best_iou_np,
            pred_strong_gt_cnt=pred_strong_gt_cnt_np,
            gt_best_iou=gt_best_iou_np,
            gt_hit_cnt=gt_hit_cnt_np,
            gt_hit_cnt_lo=gt_hit_cnt_lo_np,
            gt_size=gt_sizes_np,
            gt_best_rank=np.asarray(gt_best_rank_all, dtype=np.int32) if len(gt_best_rank_all) == gt_best_iou_np.size else np.zeros((0,), dtype=np.int32),
            **{f"gt_oracle_best_iou_at{k}": np.asarray(gt_oracle_best_iou_by_k[k], dtype=np.float32) for k in oracle_ks_sorted},
        )

        # Plots (optional dependency).
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            # 1) Purity vs coverage scatter (one point per prediction).
            plt.figure(figsize=(6, 5))
            if pred_purity_np.size:
                # Color-code by strong overlap count: 0/1/2+
                c = np.clip(pred_strong_gt_cnt_np, 0, 2)
                plt.scatter(pred_coverage_np, pred_purity_np, s=6, c=c, cmap="viridis", alpha=0.6)
                cb = plt.colorbar()
                cb.set_label(f"#GT with IoU≥{iou_thr:.2f} (clipped to 2)")
            plt.xlabel("coverage (IoU-best GT recall)")
            plt.ylabel("purity (IoU-best GT precision)")
            plt.title("Purity/Coverage scatter (per prediction)")
            plt.grid(True, linewidth=0.3, alpha=0.5)
            plt.xlim(0.0, 1.0)
            plt.ylim(0.0, 1.0)
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "purity_coverage_scatter.png"), dpi=200)
            plt.close()

            # 2) Duplicate rate histogram (per GT hit count at IoU≥thr).
            plt.figure(figsize=(6, 4))
            if gt_hit_cnt_np.size:
                max_k = int(min(np.max(gt_hit_cnt_np), 10))
                bins = np.arange(-0.5, max_k + 1.5, 1.0)
                plt.hist(np.clip(gt_hit_cnt_np, 0, max_k + 1), bins=bins, rwidth=0.85)
                plt.xticks(range(0, max_k + 2), [str(i) for i in range(0, max_k + 1)] + ([f"{max_k+1}+"] if max_k < int(np.max(gt_hit_cnt_np)) else []))
            plt.xlabel(f"#pred with IoU≥{iou_thr:.2f} (per GT)")
            plt.ylabel("count")
            plt.title("Duplicate rate (GT hit count distribution)")
            plt.grid(True, linewidth=0.3, alpha=0.5, axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "duplicate_rate_hist.png"), dpi=200)
            plt.close()

            # 3) Best-IoU histogram (per GT).
            plt.figure(figsize=(6, 4))
            if gt_best_iou_np.size:
                plt.hist(gt_best_iou_np, bins=np.linspace(0.0, 1.0, 21), rwidth=0.9)
                plt.axvspan(iou_mid_lo, iou_mid_hi, color="orange", alpha=0.15, label=f"{iou_mid_lo:.1f}–{iou_mid_hi:.1f}")
                plt.legend(loc="upper left")
            plt.xlabel("best IoU over predictions (per GT)")
            plt.ylabel("count")
            plt.title("Best-IoU histogram (GT)")
            plt.grid(True, linewidth=0.3, alpha=0.5, axis="y")
            plt.tight_layout()
            plt.savefig(os.path.join(out_dir, "best_iou_gt_hist.png"), dpi=200)
            plt.close()
        except Exception as e:
            logger.warning(f"[UnifiedSegMetric] diagnostics plots skipped: {e}")

        logger.info(f"[UnifiedSegMetric] instance error diagnostics saved to: {out_dir}")

    def compute_metrics(self, results):
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        
        # `dataset_meta` 在运行时由 Runner 注入；若静态检查为 None，显式断言
        assert self.dataset_meta is not None, 'dataset_meta is None; ensure metric is built with dataset metainfo.'
        self.valid_class_ids = self.dataset_meta['seg_valid_class_ids']
        label2cat = self.metric_meta['label2cat']
        ignore_index = self.metric_meta['ignore_index']
        classes = self.metric_meta['classes']
        thing_classes = [classes[i] for i in self.thing_class_inds]
        stuff_classes = [classes[i] for i in self.stuff_class_inds]
        num_stuff_cls = len(stuff_classes)

        gt_semantic_masks_inst_task = []
        gt_instance_masks_inst_task = []
        pred_instance_masks_inst_task = []
        pred_instance_labels = []
        pred_instance_scores = []
        pred_instance_select_scores = []
        online_monitor_results = []

        gt_semantic_masks_sem_task = []
        pred_semantic_masks_sem_task = []

        gt_masks_pan = []
        pred_masks_pan = []

        for eval_ann, single_pred_results in results:
            # gt_masks_pan.append(eval_ann)
            # pred_masks_pan.append({
            #     'pts_instance_mask': \
            #         single_pred_results['pts_instance_mask'][1],
            #     'pts_semantic_mask': \
            #         single_pred_results['pts_semantic_mask'][1]
            # })

            # gt_semantic_masks_sem_task.append(eval_ann['pts_semantic_mask'])            
            # pred_semantic_masks_sem_task.append(
            #     single_pred_results['pts_semantic_mask'][0])

            sem_mask, inst_mask = self.map_inst_markup(
                eval_ann['pts_semantic_mask'].copy(), 
                eval_ann['pts_instance_mask'].copy(), 
                self.valid_class_ids[num_stuff_cls:],
                num_stuff_cls)

            gt_semantic_masks_inst_task.append(sem_mask)
            gt_instance_masks_inst_task.append(inst_mask)           
            pred_instance_masks_inst_task.append(
                torch.tensor(single_pred_results['pts_instance_mask'][0]))
            pred_instance_labels.append(
                torch.tensor(single_pred_results['instance_labels']))
            pred_instance_scores.append(
                torch.tensor(single_pred_results['instance_scores']))
            if 'instance_select_scores' in single_pred_results:
                pred_instance_select_scores.append(
                    torch.tensor(single_pred_results['instance_select_scores']))
            else:
                pred_instance_select_scores.append(
                    torch.tensor(single_pred_results['instance_scores']))
            if 'online_monitor' in single_pred_results:
                online_monitor_results.append(single_pred_results['online_monitor'])

        # ret_pan = panoptic_seg_eval(
        #     gt_masks_pan, pred_masks_pan, classes, thing_classes,
        #     stuff_classes, self.min_num_points, self.id_offset,
        #     label2cat, ignore_index, logger)

        # ret_sem = seg_eval(
        #     gt_semantic_masks_sem_task,
        #     pred_semantic_masks_sem_task,
        #     label2cat,
        #     ignore_index[0],
        #     logger=logger)
        # decide evaluation mode: auto / multi_class / cat_agnostic
        if self.eval_mode == 'cat_agnostic':
            use_cat_agnostic = True
        elif self.eval_mode == 'multi_class':
            use_cat_agnostic = False
        else:  # auto
            use_cat_agnostic = (pred_instance_labels[0].max() == 0)

        if use_cat_agnostic:
            ret_inst = instance_cat_agnostic_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger)
        else:
            # :-1 for unlabeled
            ret_inst = instance_seg_eval(
                gt_semantic_masks_inst_task,
                gt_instance_masks_inst_task,
                pred_instance_masks_inst_task,
                pred_instance_labels,
                pred_instance_scores,
                valid_class_ids=self.valid_class_ids[num_stuff_cls:],
                class_labels=classes[num_stuff_cls:-1],
                logger=logger)

        # Optional: instance error diagnostics (scatter/hist + heuristic bottleneck suggestion).
        try:
            diag_cfg = self.diagnostics or {}
            if bool(diag_cfg.get("enable", False)):
                out_dir = str(diag_cfg.get("out_dir", "instance_diagnostics"))
                iou_thr = float(diag_cfg.get("iou_thr", 0.5))
                purity_low_thr = float(diag_cfg.get("purity_low_thr", 0.85))
                purity_high_thr = float(diag_cfg.get("purity_high_thr", 0.90))
                iou_mid_lo = float(diag_cfg.get("iou_mid_lo", 0.6))
                iou_mid_hi = float(diag_cfg.get("iou_mid_hi", 0.8))
                max_pred_per_scene = diag_cfg.get("max_pred_per_scene", None)
                gt_size_thr = int(diag_cfg.get("gt_size_thr", 100))
                iou_lo_thr = float(diag_cfg.get("iou_lo_thr", 0.1))
                oracle_ks = diag_cfg.get("oracle_ks", (20, 30, 50, 100))
                rank_bins = diag_cfg.get("rank_bins", (10, 20, 50, 100))
                if max_pred_per_scene is not None:
                    max_pred_per_scene = int(max_pred_per_scene)

                # Convert lists into numpy arrays per scene.
                gt_inst_np = [np.asarray(x, dtype=np.int64) for x in gt_instance_masks_inst_task]
                pred_mask_np = [self._to_numpy_bool(x) for x in pred_instance_masks_inst_task]
                pred_score_np = None
                score_source = str(diag_cfg.get("score_source", "instance_scores"))
                if score_source == "instance_select_scores" and len(pred_instance_select_scores) == len(pred_mask_np):
                    pred_score_np = [self._to_numpy_float(s) for s in pred_instance_select_scores]
                elif len(pred_instance_scores) == len(pred_mask_np):
                    pred_score_np = [self._to_numpy_float(s) for s in pred_instance_scores]

                self._run_instance_error_diagnostics(
                    logger=logger,
                    gt_instance_masks=gt_inst_np,
                    pred_instance_masks=pred_mask_np,
                    pred_instance_scores=pred_score_np,
                    out_dir=out_dir,
                    iou_thr=iou_thr,
                    purity_low_thr=purity_low_thr,
                    purity_high_thr=purity_high_thr,
                    iou_mid_lo=iou_mid_lo,
                    iou_mid_hi=iou_mid_hi,
                    max_pred_per_scene=max_pred_per_scene,
                    gt_size_thr=gt_size_thr,
                    iou_lo_thr=iou_lo_thr,
                    oracle_ks=oracle_ks,
                    rank_bins=rank_bins,
                )
        except Exception as e:
            logger.warning(f"[UnifiedSegMetric] diagnostics failed: {e}")

        # Optional: online behavior monitoring dump (per-scene per-frame stats).
        try:
            mon_cfg = self.online_monitor or {}
            if bool(mon_cfg.get("enable", False)) and len(online_monitor_results) > 0:
                out_dir = str(mon_cfg.get("out_dir", "online_monitor"))
                out_dir = self._resolve_diagnostics_dir(logger, out_dir)
                Path(out_dir).mkdir(parents=True, exist_ok=True)

                with open(os.path.join(out_dir, "online_monitor.json"), "w", encoding="utf-8") as f:
                    json.dump(online_monitor_results, f, indent=2, ensure_ascii=False, default=str)

                summary = self._summarize_online_monitor(online_monitor_results)
                with open(os.path.join(out_dir, "online_monitor_summary.json"), "w", encoding="utf-8") as f:
                    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

                logger.info(f"[UnifiedSegMetric] online monitor saved to: {out_dir}")
        except Exception as e:
            logger.warning(f"[UnifiedSegMetric] online monitor dump failed: {e}")

        metrics = dict()
        # for ret, keys in zip((ret_sem, ret_inst, ret_pan), self.logger_keys):
        for ret, keys in zip((ret_inst,), self.logger_keys):
            for key in keys:
                metrics[key] = ret[key]
        return metrics

    def map_inst_markup(self,
                        pts_semantic_mask,
                        pts_instance_mask,
                        valid_class_ids,
                        num_stuff_cls):
        """Map gt instance and semantic classes back from panoptic annotations.

        Args:
            pts_semantic_mask (np.array): of shape (n_raw_points,)
            pts_instance_mask (np.array): of shape (n_raw_points.)
            valid_class_ids (Tuple): of len n_instance_classes
            num_stuff_cls (int): number of stuff classes
        
        Returns:
            Tuple:
                np.array: pts_semantic_mask of shape (n_raw_points,)
                np.array: pts_instance_mask of shape (n_raw_points,)
        """
        pts_instance_mask -= num_stuff_cls
        pts_instance_mask[pts_instance_mask < 0] = -1
        
        pts_semantic_mask -= num_stuff_cls
        pts_semantic_mask[pts_instance_mask == -1] = -1

        mapping = np.array(list(valid_class_ids) + [-1])
        pts_semantic_mask = mapping[pts_semantic_mask]
        
        return pts_semantic_mask, pts_instance_mask
