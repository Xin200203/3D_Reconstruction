import argparse
import glob
import os
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


def _as_float_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    arr = np.asarray(x, dtype=np.float32)
    if arr.size == 0:
        return None
    return arr


def _iter_records(diag_dir: str) -> Iterable[dict]:
    for fp in sorted(glob.glob(os.path.join(diag_dir, "*.npz"))):
        data = np.load(fp, allow_pickle=True)
        for r in data["records"]:
            if isinstance(r, dict):
                yield r


def _offdiag(arr: np.ndarray) -> np.ndarray:
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        return arr.reshape(-1)
    mask = ~np.eye(arr.shape[0], dtype=bool)
    return arr[mask]


def _require_same_shape(*arrays: Optional[np.ndarray]) -> Tuple[np.ndarray, ...]:
    arrays_ok = [a for a in arrays if a is not None]
    if not arrays_ok:
        raise ValueError("all arrays are None/empty")
    shape = arrays_ok[0].shape
    for a in arrays_ok[1:]:
        if a.shape != shape:
            raise ValueError(f"shape mismatch: {shape} vs {a.shape}")
    # Return with Nones filtered out in the same order.
    out: List[np.ndarray] = []
    for a in arrays:
        if a is None:
            raise ValueError("missing required matrix (None)")
        out.append(a)
    return tuple(out)


def _accumulate_pair_stats(
    pass_pos: np.ndarray,
    pass_neg: np.ndarray,
    stats: Dict[str, float],
    prefix: str,
) -> None:
    stats[f"{prefix}.pos_pairs"] += float(pass_pos.size)
    stats[f"{prefix}.neg_pairs"] += float(pass_neg.size)
    stats[f"{prefix}.pos_pass"] += float(pass_pos.sum())
    stats[f"{prefix}.neg_pass"] += float(pass_neg.sum())


def _accumulate_det_stats(
    pass_pos: np.ndarray,
    pass_neg: np.ndarray,
    stats: Dict[str, float],
    prefix: str,
) -> None:
    if pass_pos.ndim == 1:
        # Degenerate: 1 det x 1 track.
        pos_any = pass_pos.astype(bool)
    else:
        pos_any = pass_pos.any(axis=1)
    if pass_neg.ndim == 1:
        neg_any = pass_neg.astype(bool)
    else:
        neg_any = pass_neg.any(axis=1)
    n_det = float(pos_any.size)
    stats[f"{prefix}.det"] += n_det
    stats[f"{prefix}.det_pos_any"] += float(pos_any.sum())
    stats[f"{prefix}.det_neg_any"] += float(neg_any.sum())
    stats[f"{prefix}.det_safe_unique"] += float((pos_any & ~neg_any).sum())
    stats[f"{prefix}.det_danger_neg_any"] += float(neg_any.sum())


def _finalize(stats: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    def safe_div(a: float, b: float) -> float:
        return float(a / b) if b > 0 else 0.0

    # det-track
    for prefix in ("det_track", "track_track"):
        pos_pairs = stats.get(f"{prefix}.pos_pairs", 0.0)
        neg_pairs = stats.get(f"{prefix}.neg_pairs", 0.0)
        pos_pass = stats.get(f"{prefix}.pos_pass", 0.0)
        neg_pass = stats.get(f"{prefix}.neg_pass", 0.0)
        out[f"{prefix}.tpr_pair"] = safe_div(pos_pass, pos_pairs)
        out[f"{prefix}.fpr_pair"] = safe_div(neg_pass, neg_pairs)
        out[f"{prefix}.precision_pair"] = safe_div(pos_pass, pos_pass + neg_pass)

    # det-level
    for prefix in ("det_track",):
        n_det = stats.get(f"{prefix}.det", 0.0)
        out[f"{prefix}.pos_any_rate"] = safe_div(stats.get(f"{prefix}.det_pos_any", 0.0), n_det)
        out[f"{prefix}.neg_any_rate"] = safe_div(stats.get(f"{prefix}.det_neg_any", 0.0), n_det)
        out[f"{prefix}.safe_unique_rate"] = safe_div(stats.get(f"{prefix}.det_safe_unique", 0.0), n_det)
        out[f"{prefix}.danger_neg_any_rate"] = safe_div(stats.get(f"{prefix}.det_danger_neg_any", 0.0), n_det)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir", required=True, help="Path to track_geom_diag directory containing *.npz")
    ap.add_argument("--contain-thr", type=float, default=0.30)
    ap.add_argument(
        "--contain-dt-thr",
        type=float,
        default=None,
        help="Optional det-track contain threshold override (defaults to --contain-thr).",
    )
    ap.add_argument(
        "--contain-tt-thr",
        type=float,
        default=None,
        help="Optional track-track contain threshold override (defaults to --contain-thr).",
    )
    ap.add_argument("--dino-thr", type=float, default=0.75)
    ap.add_argument("--feat3d-thr", type=float, default=0.80)
    ap.add_argument(
        "--tt-geom",
        type=str,
        default="min",
        choices=("min", "max"),
        help=(
            "How to interpret track-track geometric overlap when records are asymmetric. "
            "min: min(A in B, B in A) (safe dedup). "
            "max: max(A in B, B in A) (absorb/fragment-friendly)."
        ),
    )
    ap.add_argument(
        "--require-dino",
        action="store_true",
        help="If set, skip records missing DINO matrices (M_dino_*).",
    )
    ap.add_argument(
        "--require-feat3d",
        action="store_true",
        help="If set, skip records missing 3D feature matrices (M_feat3d_*).",
    )
    args = ap.parse_args()

    contain_dt_thr = args.contain_thr if args.contain_dt_thr is None else float(args.contain_dt_thr)
    contain_tt_thr = args.contain_thr if args.contain_tt_thr is None else float(args.contain_tt_thr)

    stats: Dict[str, float] = {
        "det_track.pos_pairs": 0.0,
        "det_track.neg_pairs": 0.0,
        "det_track.pos_pass": 0.0,
        "det_track.neg_pass": 0.0,
        "det_track.det": 0.0,
        "det_track.det_pos_any": 0.0,
        "det_track.det_neg_any": 0.0,
        "det_track.det_safe_unique": 0.0,
        "det_track.det_danger_neg_any": 0.0,
        "track_track.pos_pairs": 0.0,
        "track_track.neg_pairs": 0.0,
        "track_track.pos_pass": 0.0,
        "track_track.neg_pass": 0.0,
    }

    def get_mat(rec: dict, *keys: str) -> Optional[np.ndarray]:
        for k in keys:
            v = rec.get(k, None)
            a = _as_float_array(v)
            if a is not None:
                return a
        return None

    skipped = 0
    used = 0
    skipped_missing = {"det_track": 0, "track_track": 0}
    for rec in _iter_records(args.diag_dir):
        try:
            # det-track matrices
            geom_pos = get_mat(rec, "M_pos")
            geom_neg = get_mat(rec, "M_neg")
            dino_pos = get_mat(rec, "M_dino_pos")
            dino_neg = get_mat(rec, "M_dino_neg")
            f3d_pos = get_mat(rec, "M_feat3d_pos")
            f3d_neg = get_mat(rec, "M_feat3d_neg")

            if geom_pos is None or geom_neg is None:
                skipped_missing["det_track"] += 1
                raise ValueError("missing det-track geom matrices")
            if args.require_dino and (dino_pos is None or dino_neg is None):
                skipped_missing["det_track"] += 1
                raise ValueError("missing det-track dino matrices")
            if args.require_feat3d and (f3d_pos is None or f3d_neg is None):
                skipped_missing["det_track"] += 1
                raise ValueError("missing det-track feat3d matrices")

            # If not required, treat missing modalities as pass-all.
            if dino_pos is None:
                dino_pos = np.ones_like(geom_pos, dtype=np.float32)
            if dino_neg is None:
                dino_neg = np.ones_like(geom_neg, dtype=np.float32)
            if f3d_pos is None:
                f3d_pos = np.ones_like(geom_pos, dtype=np.float32)
            if f3d_neg is None:
                f3d_neg = np.ones_like(geom_neg, dtype=np.float32)

            geom_pos, dino_pos, f3d_pos = _require_same_shape(geom_pos, dino_pos, f3d_pos)
            geom_neg, dino_neg, f3d_neg = _require_same_shape(geom_neg, dino_neg, f3d_neg)

            pass_pos = (geom_pos >= contain_dt_thr) & (dino_pos >= args.dino_thr) & (f3d_pos >= args.feat3d_thr)
            pass_neg = (geom_neg >= contain_dt_thr) & (dino_neg >= args.dino_thr) & (f3d_neg >= args.feat3d_thr)
            _accumulate_pair_stats(pass_pos, pass_neg, stats, "det_track")
            _accumulate_det_stats(pass_pos, pass_neg, stats, "det_track")

            # track-track geometry: legacy (M_tt_min/M_tt_neg) OR new asymmetric (M_tt_ij/M_tt_neg_ij+ji)
            tt_geom_legacy = get_mat(rec, "M_tt_min")
            tt_geom_asym = get_mat(rec, "M_tt_ij")
            tt_dino = get_mat(rec, "M_dino_tt")
            tt_f3d = get_mat(rec, "M_feat3d_tt")

            tt_neg_geom_legacy = get_mat(rec, "M_tt_neg")
            tt_neg_geom_ij = get_mat(rec, "M_tt_neg_ij")
            tt_neg_geom_ji = get_mat(rec, "M_tt_neg_ji")
            tt_neg_dino = get_mat(rec, "M_dino_tt_neg")
            tt_neg_f3d = get_mat(rec, "M_feat3d_tt_neg")

            if args.require_dino and tt_dino is None:
                skipped_missing["track_track"] += 1
                raise ValueError("missing track-track dino matrices")
            if args.require_feat3d and tt_f3d is None:
                skipped_missing["track_track"] += 1
                raise ValueError("missing track-track feat3d matrices")

            if tt_dino is None:
                # Pass-all if not required.
                if tt_geom_legacy is not None:
                    tt_dino = np.ones_like(tt_geom_legacy, dtype=np.float32)
                elif tt_geom_asym is not None:
                    tt_dino = np.ones_like(tt_geom_asym, dtype=np.float32)
            if tt_f3d is None:
                if tt_geom_legacy is not None:
                    tt_f3d = np.ones_like(tt_geom_legacy, dtype=np.float32)
                elif tt_geom_asym is not None:
                    tt_f3d = np.ones_like(tt_geom_asym, dtype=np.float32)

            # Positive TT pairs
            pass_tt_pos = np.zeros((0,), dtype=bool)
            if tt_geom_asym is not None and tt_dino is not None and tt_f3d is not None:
                g_asym = tt_geom_asym
                if g_asym.ndim == 2 and g_asym.shape[0] == g_asym.shape[1] and g_asym.shape[0] > 1:
                    if args.tt_geom == "min":
                        g = np.minimum(g_asym, g_asym.T)
                    else:
                        g = np.maximum(g_asym, g_asym.T)
                    g = _offdiag(g).reshape(-1)
                    d = _offdiag(tt_dino).reshape(-1)
                    f = _offdiag(tt_f3d).reshape(-1)
                    if not (g.shape == d.shape == f.shape):
                        raise ValueError(f"track-track pos shape mismatch {g.shape} {d.shape} {f.shape}")
                    pass_tt_pos = (g >= contain_tt_thr) & (d >= args.dino_thr) & (f >= args.feat3d_thr)
            elif tt_geom_legacy is not None and tt_dino is not None and tt_f3d is not None:
                g = _offdiag(tt_geom_legacy).reshape(-1)
                d = _offdiag(tt_dino).reshape(-1)
                f = _offdiag(tt_f3d).reshape(-1)
                if not (g.shape == d.shape == f.shape):
                    raise ValueError(f"track-track pos shape mismatch {g.shape} {d.shape} {f.shape}")
                pass_tt_pos = (g >= contain_tt_thr) & (d >= args.dino_thr) & (f >= args.feat3d_thr)

            # Negative TT pairs
            pass_tt_neg = np.zeros((0,), dtype=bool)
            if tt_neg_geom_ij is not None and tt_neg_geom_ji is not None and tt_neg_dino is not None and tt_neg_f3d is not None:
                # ij and ji have same shape (n_pos_tracks, n_neg_samples), not transpose.
                # ij[i,j] = A_i contained in neg_j, ji[i,j] = neg_j contained in A_i
                if tt_neg_geom_ij.shape != tt_neg_geom_ji.shape:
                    raise ValueError(f"track-track neg asym shape mismatch ij={tt_neg_geom_ij.shape} ji={tt_neg_geom_ji.shape}")
                if args.tt_geom == "min":
                    g = np.minimum(tt_neg_geom_ij, tt_neg_geom_ji)
                else:
                    g = np.maximum(tt_neg_geom_ij, tt_neg_geom_ji)
                g = g.reshape(-1)
                d = tt_neg_dino.reshape(-1)
                f = tt_neg_f3d.reshape(-1)
                if not (g.shape == d.shape == f.shape):
                    raise ValueError(f"track-track neg shape mismatch {g.shape} {d.shape} {f.shape}")
                pass_tt_neg = (g >= contain_tt_thr) & (d >= args.dino_thr) & (f >= args.feat3d_thr)
            elif tt_neg_geom_legacy is not None and tt_neg_dino is not None and tt_neg_f3d is not None:
                g = tt_neg_geom_legacy.reshape(-1)
                d = tt_neg_dino.reshape(-1)
                f = tt_neg_f3d.reshape(-1)
                if not (g.shape == d.shape == f.shape):
                    raise ValueError(f"track-track neg shape mismatch {g.shape} {d.shape} {f.shape}")
                pass_tt_neg = (g >= contain_tt_thr) & (d >= args.dino_thr) & (f >= args.feat3d_thr)

            _accumulate_pair_stats(pass_tt_pos, pass_tt_neg, stats, "track_track")
            used += 1
        except Exception as e:
            if skipped < 3:  # Print first 3 errors for debugging
                import traceback
                print(f"[DEBUG] Skipped record: {e}")
                traceback.print_exc()
            skipped += 1

    out = _finalize(stats)
    print("=== Hard gate evaluation (contain + dino + feat3d) ===")
    print(f"diag_dir: {args.diag_dir}")
    print(
        "thresholds: "
        f"contain_dt>={contain_dt_thr:.3f}, contain_tt>={contain_tt_thr:.3f}, "
        f"dino_cos>={args.dino_thr:.3f}, feat3d_cos>={args.feat3d_thr:.3f}"
    )
    print(f"track_track.geom_mode: {args.tt_geom}")
    print(f"records: used={used} skipped={skipped}")
    if any(v > 0 for v in skipped_missing.values()):
        print(f"skipped_missing: {skipped_missing}")
    print("")
    print("det-track:")
    print(f"  pair_tpr={out['det_track.tpr_pair']:.4f}  pair_fpr={out['det_track.fpr_pair']:.4f}  pair_precision={out['det_track.precision_pair']:.4f}")
    print(f"  det_pos_any_rate={out['det_track.pos_any_rate']:.4f}  det_neg_any_rate={out['det_track.neg_any_rate']:.4f}")
    print(f"  det_safe_unique_rate={out['det_track.safe_unique_rate']:.4f}  det_danger_neg_any_rate={out['det_track.danger_neg_any_rate']:.4f}")
    print("")
    print("track-track:")
    print(f"  pair_tpr={out['track_track.tpr_pair']:.4f}  pair_fpr={out['track_track.fpr_pair']:.4f}  pair_precision={out['track_track.precision_pair']:.4f}")


if __name__ == "__main__":
    main()

