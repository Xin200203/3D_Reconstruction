import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _as_float_array(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    # Some npz writers may store variable-shaped arrays as 0-d object arrays.
    # Unwrap those into a numeric ndarray.
    if isinstance(x, np.ndarray) and x.dtype == object:
        try:
            if x.shape == () or x.size == 1:
                x = x.item()
        except Exception:
            pass
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


def _get_first(rec: dict, keys: Sequence[str]) -> Optional[np.ndarray]:
    for k in keys:
        if k in rec:
            return _as_float_array(rec.get(k))
    return None


def _stack_or_none(chunks: List[np.ndarray]) -> Optional[np.ndarray]:
    if not chunks:
        return None
    return np.concatenate(chunks, axis=0)


@dataclass(frozen=True)
class GateThr:
    geom: float
    dino: float
    feat3d: float


@dataclass
class GateResult:
    thr: GateThr
    pos_rate: float
    neg_rate: float
    safe_unique_rate: float
    pos_pairs: int
    neg_pairs: int
    pos_any_rate: float
    neg_any_rate: float
    n_det: int
    pos_any_cnt: int
    neg_any_cnt: int
    safe_unique_cnt: int


def _eval_gate(
    geom_pos: np.ndarray,
    dino_pos: np.ndarray,
    feat_pos: np.ndarray,
    geom_neg: np.ndarray,
    dino_neg: np.ndarray,
    feat_neg: np.ndarray,
    thr: GateThr,
) -> GateResult:
    pos_pass = (geom_pos >= thr.geom) & (dino_pos >= thr.dino) & (feat_pos >= thr.feat3d)
    neg_pass = (geom_neg >= thr.geom) & (dino_neg >= thr.dino) & (feat_neg >= thr.feat3d)

    pos_pairs = int(pos_pass.size)
    neg_pairs = int(neg_pass.size)
    pair_pos_rate = float(pos_pass.mean()) if pos_pairs > 0 else 0.0
    pair_neg_rate = float(neg_pass.mean()) if neg_pairs > 0 else 0.0

    # det-level: any-positive / any-negative (this matches online "does this det have at least one viable track?")
    if pos_pass.ndim == 1:
        pos_any = pos_pass.astype(bool)
    else:
        pos_any = pos_pass.any(axis=1)
    if neg_pass.ndim == 1:
        neg_any = neg_pass.astype(bool)
    else:
        neg_any = neg_pass.any(axis=1)
    n_det = int(pos_any.size)
    pos_any_rate = float(pos_any.mean()) if n_det > 0 else 0.0
    neg_any_rate = float(neg_any.mean()) if n_det > 0 else 0.0
    safe_unique = (pos_any & ~neg_any)
    safe_unique_rate = float(safe_unique.mean()) if n_det > 0 else 0.0
    pos_any_cnt = int(pos_any.sum())
    neg_any_cnt = int(neg_any.sum())
    safe_unique_cnt = int(safe_unique.sum())

    return GateResult(
        thr=thr,
        pos_rate=pos_any_rate,
        neg_rate=neg_any_rate,
        safe_unique_rate=safe_unique_rate,
        pos_pairs=pos_pairs,
        neg_pairs=neg_pairs,
        pos_any_rate=pos_any_rate,
        neg_any_rate=neg_any_rate,
        n_det=n_det,
        pos_any_cnt=pos_any_cnt,
        neg_any_cnt=neg_any_cnt,
        safe_unique_cnt=safe_unique_cnt,
    )


def _format_pct(x: float) -> str:
    return f"{x * 100.0:.2f}%"


def _print_top(title: str, rows: List[GateResult], topk: int) -> None:
    print(f"\n{title}")
    for i, r in enumerate(rows[:topk], 1):
        print(
            f"{i:>2}. geom>={r.thr.geom:.02f} dino>={r.thr.dino:.02f} feat>={r.thr.feat3d:.02f}"
            f"  -> Pos {_format_pct(r.pos_rate)} | Neg {_format_pct(r.neg_rate)}"
            f" | SafeUnique {_format_pct(r.safe_unique_rate)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir", required=True, help="Path to track_geom_diag directory containing *.npz")
    ap.add_argument("--tt-geom", choices=("min", "max"), default="max")
    ap.add_argument(
        "--dt-metric",
        choices=("pair", "det_any"),
        default="pair",
        help=(
            "How to score det-track. "
            "pair: use pair-level TPR/FPR (matches earlier 'Pos xx | Neg yy' reports). "
            "det_any: use det-level any-positive/any-negative rates."
        ),
    )
    ap.add_argument("--geom-list", default="0.01,0.02,0.03,0.05,0.08")
    ap.add_argument("--dino-list", default="0.75,0.78,0.80,0.82,0.84")
    ap.add_argument("--feat3d-list", default="0.65,0.70,0.75,0.80")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--neg-max", type=float, default=0.02, help="Constraint: Neg <= this value")
    ap.add_argument("--pos-min", type=float, default=0.65, help="Constraint: Pos >= this value")
    args = ap.parse_args()

    geom_list = [float(x) for x in args.geom_list.split(",") if x.strip()]
    dino_list = [float(x) for x in args.dino_list.split(",") if x.strip()]
    feat_list = [float(x) for x in args.feat3d_list.split(",") if x.strip()]
    grid = [GateThr(g, d, f) for g in geom_list for d in dino_list for f in feat_list]

    dt_records: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    tt_records: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    used = 0
    skipped = 0
    for rec in _iter_records(args.diag_dir):
        try:
            # det-track (same GT vs different GT)
            g_pos = _get_first(rec, ("M_geom_pos", "M_pos"))
            g_neg = _get_first(rec, ("M_geom_neg", "M_neg"))
            d_pos = _get_first(rec, ("M_dino_pos",))
            d_neg = _get_first(rec, ("M_dino_neg",))
            f_pos = _get_first(rec, ("M_feat3d_pos",))
            f_neg = _get_first(rec, ("M_feat3d_neg",))
            if any(x is None for x in (g_pos, g_neg, d_pos, d_neg, f_pos, f_neg)):
                skipped += 1
                continue
            if not (g_pos.shape == d_pos.shape == f_pos.shape):
                skipped += 1
                continue
            if not (g_neg.shape == d_neg.shape == f_neg.shape):
                skipped += 1
                continue
            dt_records.append((g_pos, d_pos, f_pos, g_neg, d_neg, f_neg))

            # track-track (same GT track pairs vs neg sampled tracks)
            tt_ij = _get_first(rec, ("M_tt_ij", "M_geom_tt_asym", "M_geom_tt"))
            tt_ji = _get_first(rec, ("M_tt_ji",))
            tt_dino = _get_first(rec, ("M_dino_tt",))
            tt_f3d = _get_first(rec, ("M_feat3d_tt",))
            if tt_ij is None or tt_dino is None or tt_f3d is None:
                skipped += 1
                continue
            if tt_ji is None:
                if tt_ij.ndim == 2 and tt_ij.shape[0] == tt_ij.shape[1]:
                    tt_ji = tt_ij.T
                else:
                    tt_ji = tt_ij
            if tt_ij.shape != tt_ji.shape:
                skipped += 1
                continue
            if not (tt_dino.shape == tt_ij.shape == tt_f3d.shape):
                skipped += 1
                continue
            if tt_ij.ndim == 2 and tt_ij.shape[0] == tt_ij.shape[1] and tt_ij.shape[0] > 1:
                if args.tt_geom == "min":
                    tt_g = np.minimum(tt_ij, tt_ji)
                else:
                    tt_g = np.maximum(tt_ij, tt_ji)
                tt_g = _offdiag(tt_g)
                tt_d = _offdiag(tt_dino)
                tt_f = _offdiag(tt_f3d)
            else:
                tt_g = tt_ij.reshape(-1)
                tt_d = tt_dino.reshape(-1)
                tt_f = tt_f3d.reshape(-1)

            # TT negatives (sampled)
            tt_neg_ij = _get_first(rec, ("M_tt_neg_ij", "M_geom_tt_neg_ij", "M_geom_tt_neg"))
            tt_neg_ji = _get_first(rec, ("M_tt_neg_ji", "M_geom_tt_neg_ji"))
            tt_neg_d = _get_first(rec, ("M_dino_tt_neg",))
            tt_neg_f = _get_first(rec, ("M_feat3d_tt_neg",))
            if any(x is None for x in (tt_neg_ij, tt_neg_ji, tt_neg_d, tt_neg_f)):
                skipped += 1
                continue
            if tt_neg_ij.shape != tt_neg_ji.shape:
                skipped += 1
                continue
            if not (tt_neg_d.shape == tt_neg_f.shape == tt_neg_ij.shape):
                skipped += 1
                continue
            if args.tt_geom == "min":
                tt_neg_g = np.minimum(tt_neg_ij, tt_neg_ji).reshape(-1)
            else:
                tt_neg_g = np.maximum(tt_neg_ij, tt_neg_ji).reshape(-1)
            tt_neg_d = tt_neg_d.reshape(-1)
            tt_neg_f = tt_neg_f.reshape(-1)

            # Track-track is pair-level by definition here, use 1D arrays.
            tt_records.append((tt_g.reshape(-1), tt_d.reshape(-1), tt_f.reshape(-1), tt_neg_g, tt_neg_d, tt_neg_f))

            used += 1
        except Exception:
            skipped += 1

    if not dt_records:
        raise RuntimeError("det-track matrices are missing/empty; cannot evaluate")
    if not tt_records:
        raise RuntimeError("track-track matrices are missing/empty; cannot evaluate")

    print("=== Threshold grid search (geom + dino + feat3d, hard AND) ===")
    print(f"diag_dir: {args.diag_dir}")
    print(f"records used={used} skipped={skipped}")
    print(f"track-track geom_mode: {args.tt_geom}")
    print(f"grid size: {len(grid)}")

    # det-track results
    dt_results: List[GateResult] = []
    for thr in grid:
        pos_pass_pairs = 0
        neg_pass_pairs = 0
        pos_pairs = 0
        neg_pairs = 0
        pos_any_hit = 0
        neg_any_hit = 0
        safe_unique_hit = 0
        n_det = 0
        for g_pos, d_pos, f_pos, g_neg, d_neg, f_neg in dt_records:
            r = _eval_gate(g_pos, d_pos, f_pos, g_neg, d_neg, f_neg, thr)
            pos_pass_pairs += int(((g_pos >= thr.geom) & (d_pos >= thr.dino) & (f_pos >= thr.feat3d)).sum())
            neg_pass_pairs += int(((g_neg >= thr.geom) & (d_neg >= thr.dino) & (f_neg >= thr.feat3d)).sum())
            pos_pairs += int(r.pos_pairs)
            neg_pairs += int(r.neg_pairs)
            pos_any_hit += int(r.pos_any_cnt)
            neg_any_hit += int(r.neg_any_cnt)
            safe_unique_hit += int(r.safe_unique_cnt)
            n_det += int(r.n_det)

        pair_pos_rate = float(pos_pass_pairs / pos_pairs) if pos_pairs > 0 else 0.0
        pair_neg_rate = float(neg_pass_pairs / neg_pairs) if neg_pairs > 0 else 0.0
        det_pos_any_rate = float(pos_any_hit / n_det) if n_det > 0 else 0.0
        det_neg_any_rate = float(neg_any_hit / n_det) if n_det > 0 else 0.0
        safe_unique_rate = float(safe_unique_hit / n_det) if n_det > 0 else 0.0

        if args.dt_metric == "pair":
            pos_rate = pair_pos_rate
            neg_rate = pair_neg_rate
        else:
            pos_rate = det_pos_any_rate
            neg_rate = det_neg_any_rate

        dt_results.append(
            GateResult(
                thr=thr,
                pos_rate=pos_rate,
                neg_rate=neg_rate,
                safe_unique_rate=safe_unique_rate,
                pos_pairs=pos_pairs,
                neg_pairs=neg_pairs,
                pos_any_rate=det_pos_any_rate,
                neg_any_rate=det_neg_any_rate,
                n_det=n_det,
                pos_any_cnt=pos_any_hit,
                neg_any_cnt=neg_any_hit,
                safe_unique_cnt=safe_unique_hit,
            )
        )
    dt_neg_ok = sorted([r for r in dt_results if r.neg_rate <= args.neg_max], key=lambda r: (-r.pos_rate, r.neg_rate))
    dt_pos_ok = sorted([r for r in dt_results if r.pos_rate >= args.pos_min], key=lambda r: (r.neg_rate, -r.pos_rate))

    _print_top(f"\n[det-track] Best candidates satisfying Neg<={args.neg_max*100:.1f}%", dt_neg_ok, args.topk)
    _print_top(f"\n[det-track] Best candidates satisfying Pos>={args.pos_min*100:.1f}%", dt_pos_ok, args.topk)

    # track-track results
    tt_results: List[GateResult] = []
    for thr in grid:
        pos_pass = 0
        neg_pass = 0
        pos_pairs = 0
        neg_pairs = 0
        for g_pos, d_pos, f_pos, g_neg, d_neg, f_neg in tt_records:
            pos = (g_pos >= thr.geom) & (d_pos >= thr.dino) & (f_pos >= thr.feat3d)
            neg = (g_neg >= thr.geom) & (d_neg >= thr.dino) & (f_neg >= thr.feat3d)
            pos_pass += int(pos.sum())
            neg_pass += int(neg.sum())
            pos_pairs += int(pos.size)
            neg_pairs += int(neg.size)
        pos_rate = float(pos_pass / pos_pairs) if pos_pairs > 0 else 0.0
        neg_rate = float(neg_pass / neg_pairs) if neg_pairs > 0 else 0.0
        tt_results.append(
            GateResult(
                thr=thr,
                pos_rate=pos_rate,
                neg_rate=neg_rate,
                safe_unique_rate=0.0,
                pos_pairs=pos_pairs,
                neg_pairs=neg_pairs,
                pos_any_rate=0.0,
                neg_any_rate=0.0,
                n_det=0,
                pos_any_cnt=0,
                neg_any_cnt=0,
                safe_unique_cnt=0,
            )
        )
    tt_neg_ok = sorted([r for r in tt_results if r.neg_rate <= args.neg_max], key=lambda r: (-r.pos_rate, r.neg_rate))
    tt_pos_ok = sorted([r for r in tt_results if r.pos_rate >= args.pos_min], key=lambda r: (r.neg_rate, -r.pos_rate))

    _print_top(f"\n[track-track] Best candidates satisfying Neg<={args.neg_max*100:.1f}%", tt_neg_ok, args.topk)
    _print_top(f"\n[track-track] Best candidates satisfying Pos>={args.pos_min*100:.1f}%", tt_pos_ok, args.topk)


if __name__ == "__main__":
    main()
