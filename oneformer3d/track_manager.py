from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch


@dataclass
class BaseInstance:
    """A lightweight view of an instance (det or track).

    This class is intentionally minimal and torch-friendly. It can be used to
    attach multi-modal features (e.g. 2D DINO) in the future.
    """

    mask: Optional[torch.Tensor] = None  # (P,) bool or (T*P,) bool for tracks
    label: Optional[torch.Tensor] = None  # () long
    score: Optional[torch.Tensor] = None  # () float
    query: Optional[torch.Tensor] = None  # (C,)
    query_feat: Optional[torch.Tensor] = None  # (C,)
    sem_pred: Optional[torch.Tensor] = None  # (K,) or logits/prob vector
    xyz: Optional[torch.Tensor] = None  # (3,) or bbox (6,)
    track_id: Optional[int] = None

    @property
    def npoints(self) -> int:
        if self.mask is None:
            return 0
        try:
            return int(self.mask.sum().item())
        except Exception:
            return 0


class DetectionInstance(BaseInstance):
    """Per-frame detection/observation."""


class TrackInstance(BaseInstance):
    """Long-lived memory track."""


class TrackManager:
    """Maintain per-scene track state (masks, features, ids) for OnlineMerge.

    The goal is to separate *state maintenance* (padding/concatenation, EMA
    updates, track id bookkeeping, top-K output selection) from association
    logic (Hungarian / stage-2 gates), improving robustness and extensibility.
    """

    def __init__(self, merge_type: str = "count"):
        if merge_type not in ("count", "frame"):
            raise ValueError(f"merge_type must be 'count' or 'frame', got {merge_type!r}")
        self.merge_type = merge_type
        self.reset()

    def reset(self) -> None:
        self.masks: Optional[torch.Tensor] = None
        self.labels: Optional[torch.Tensor] = None
        self.scores: Optional[torch.Tensor] = None
        self.queries: Optional[torch.Tensor] = None
        self.query_feats: Optional[torch.Tensor] = None
        self.sem_preds: Optional[torch.Tensor] = None
        self.xyz: Optional[torch.Tensor] = None
        self.bboxes: Optional[torch.Tensor] = None
        self.merge_counts: Optional[torch.Tensor] = None
        self.fi: int = 0

        self.track_ids: Optional[torch.Tensor] = None
        self.next_track_id: int = 0
        self.output_track_ids: Optional[List[int]] = None
        # Optional extra modalities (e.g. DINO pooled per instance).
        self.dino_feats: Optional[torch.Tensor] = None  # (N_tracks, D)
        # Optional 3D feature memory (pooled from U-Net / superpoints).
        self.feat3d: Optional[torch.Tensor] = None  # (N_tracks, D)
        # Optional GT-aligned tracking diagnostics (votes are per-track).
        self.track_gt_votes: Optional[List[Dict[int, float]]] = None
        self.track_gt_ids: Optional[List[int]] = None
        self.track_gt_strength: Optional[List[float]] = None
        # Optional voxel occupancy cache per track (for geometric diagnostics).
        self.track_voxels: Optional[List[torch.Tensor]] = None
        # Optional tentative track confirmation state (delayed birth).
        # 0=confirmed, 1=tentative
        self.track_states: Optional[torch.Tensor] = None  # (N_tracks,) long
        self.tent_birth_fi: Optional[torch.Tensor] = None  # (N_tracks,) long
        self.tent_hits: Optional[torch.Tensor] = None  # (N_tracks,) long

    @property
    def initialized(self) -> bool:
        return self.masks is not None

    @property
    def num_tracks(self) -> int:
        if self.scores is None:
            return 0
        return int(self.scores.shape[0])

    def init_first_frame(
        self,
        masks: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        queries: torch.Tensor,
        query_feats: torch.Tensor,
        sem_preds: torch.Tensor,
        xyz: torch.Tensor,
        bboxes: Optional[torch.Tensor] = None,
        dino_feats: Optional[torch.Tensor] = None,
        feat3d: Optional[torch.Tensor] = None,
        det_voxels: Optional[List[torch.Tensor]] = None,
    ) -> None:
        self.masks = masks
        self.labels = labels
        self.scores = scores
        self.queries = queries
        self.query_feats = query_feats
        self.sem_preds = sem_preds
        self.xyz = xyz
        self.bboxes = bboxes
        self.merge_counts = torch.zeros_like(scores).long()
        self.fi = 0
        self.dino_feats = dino_feats
        self.feat3d = feat3d

        n = int(scores.shape[0])
        device = scores.device
        self.track_ids = torch.arange(self.next_track_id, self.next_track_id + n, device=device, dtype=torch.long)
        self.next_track_id += n
        self._init_track_gt_votes(n)
        self._init_track_voxels(det_voxels)
        # First-frame tracks are confirmed by default.
        self.track_states = torch.zeros((n,), device=device, dtype=torch.long)
        self.tent_birth_fi = torch.full((n,), fill_value=int(self.fi), device=device, dtype=torch.long)
        self.tent_hits = torch.zeros((n,), device=device, dtype=torch.long)

    def begin_next_frame(self) -> None:
        """Advance internal frame index (used for padding length)."""
        self.fi += 1

    def _init_track_gt_votes(self, n: int) -> None:
        self.track_gt_votes = [dict() for _ in range(int(n))]
        self.track_gt_ids = [-1 for _ in range(int(n))]
        self.track_gt_strength = [0.0 for _ in range(int(n))]

    def _append_track_gt_votes(self, n_new: int) -> None:
        if n_new <= 0:
            return
        if self.track_gt_votes is None:
            self._init_track_gt_votes(0)
        assert self.track_gt_votes is not None and self.track_gt_ids is not None and self.track_gt_strength is not None
        for _ in range(int(n_new)):
            self.track_gt_votes.append({})
            self.track_gt_ids.append(-1)
            self.track_gt_strength.append(0.0)
        # Extend tentative state tensors if present.
        try:
            if self.track_states is not None and self.tent_birth_fi is not None and self.tent_hits is not None:
                device = self.track_states.device
                z = torch.zeros((int(n_new),), device=device, dtype=self.track_states.dtype)
                self.track_states = torch.cat([self.track_states, z], dim=0)
                self.tent_birth_fi = torch.cat(
                    [self.tent_birth_fi, torch.full_like(z, fill_value=int(self.fi))], dim=0
                )
                self.tent_hits = torch.cat([self.tent_hits, torch.zeros_like(z)], dim=0)
        except Exception:
            pass

    def update_track_gt_votes(
        self,
        row_ind: torch.Tensor,
        col_ind: torch.Tensor,
        det_gt: Optional[torch.Tensor],
        det_gt_conf: Optional[torch.Tensor] = None,
        conf_thr: float = 0.5,
        weight_by_conf: bool = True,
    ) -> None:
        if det_gt is None or row_ind.numel() == 0:
            return
        if self.track_gt_votes is None or self.track_gt_ids is None or self.track_gt_strength is None:
            self._init_track_gt_votes(self.num_tracks)
        assert self.track_gt_votes is not None and self.track_gt_ids is not None and self.track_gt_strength is not None

        if not torch.is_tensor(det_gt):
            det_gt = torch.as_tensor(det_gt)
        if det_gt_conf is not None and not torch.is_tensor(det_gt_conf):
            det_gt_conf = torch.as_tensor(det_gt_conf)

        # Iterate matched pairs; row_ind and col_ind are memory/det indices.
        for mem_i, det_j in zip(row_ind.detach().cpu().tolist(), col_ind.detach().cpu().tolist()):
            if mem_i < 0 or mem_i >= len(self.track_gt_votes):
                continue
            if det_j < 0 or det_j >= int(det_gt.numel()):
                continue
            gid = int(det_gt[int(det_j)].item())
            if gid < 0:
                continue
            conf = 1.0
            if det_gt_conf is not None:
                try:
                    conf = float(det_gt_conf[int(det_j)].item())
                except Exception:
                    conf = 1.0
            if conf < conf_thr:
                continue
            w = conf if weight_by_conf else 1.0
            votes = self.track_gt_votes[mem_i]
            votes[gid] = votes.get(gid, 0.0) + w
            # Update cached best id/strength.
            best_id, best_val = -1, 0.0
            for gid, v in votes.items():
                if v > best_val:
                    best_id, best_val = gid, v
            self.track_gt_ids[mem_i] = int(best_id)
            self.track_gt_strength[mem_i] = float(best_val)

    def _init_track_voxels(self, det_voxels: Optional[List[torch.Tensor]]) -> None:
        if det_voxels is None:
            self.track_voxels = None
            return
        # Initialize with det_voxels per track (first frame).
        self.track_voxels = []
        for v in det_voxels:
            if torch.is_tensor(v):
                self.track_voxels.append(v.detach())
            else:
                self.track_voxels.append(torch.as_tensor(v))

    def update_track_voxels(
        self,
        row_ind: torch.Tensor,
        col_ind: torch.Tensor,
        det_voxels: Optional[List[torch.Tensor]],
        birth_det_idx: Optional[List[int]] = None,
        cap: int = 1024,
    ) -> None:
        if det_voxels is None:
            return
        if self.track_voxels is None:
            self.track_voxels = [torch.empty((0,), dtype=det_voxels[0].dtype, device=det_voxels[0].device)
                                 for _ in range(self.num_tracks)]
        assert self.track_voxels is not None
        # Matched updates.
        if row_ind.numel() > 0:
            for mem_i, det_j in zip(row_ind.detach().cpu().tolist(), col_ind.detach().cpu().tolist()):
                if mem_i < 0 or mem_i >= len(self.track_voxels):
                    continue
                if det_j < 0 or det_j >= len(det_voxels):
                    continue
                self.track_voxels[mem_i] = _cap_union_1d(self.track_voxels[mem_i], det_voxels[det_j], cap)
        # Birth updates (append).
        if birth_det_idx:
            for det_j in birth_det_idx:
                if det_j < 0 or det_j >= len(det_voxels):
                    continue
                self.track_voxels.append(det_voxels[det_j].detach())

    def update_track_voxels_pairs(
        self,
        *,
        mem_idx: List[int],
        det_idx: List[int],
        det_voxels: Optional[List[torch.Tensor]],
        cap: int = 1024,
    ) -> None:
        """Update voxel cache for arbitrary (track, det) pairs (e.g., rescue writes).

        This is intentionally separate from `update_track_voxels`, which only
        handles Hungarian matched pairs + births.
        """
        if det_voxels is None:
            return
        if self.track_voxels is None:
            # Initialize empty cache if needed.
            try:
                self.track_voxels = [torch.empty((0,), dtype=det_voxels[0].dtype, device=det_voxels[0].device)
                                     for _ in range(self.num_tracks)]
            except Exception:
                return
        assert self.track_voxels is not None
        if len(mem_idx) != len(det_idx):
            return
        for mi, dj in zip(mem_idx, det_idx):
            if mi < 0 or mi >= len(self.track_voxels):
                continue
            if dj < 0 or dj >= len(det_voxels):
                continue
            self.track_voxels[mi] = _cap_union_1d(self.track_voxels[mi], det_voxels[dj], cap)

    def apply_assignment_and_update(
        self,
        *,
        points_per_mask: int,
        row_ind: torch.Tensor,
        col_ind: torch.Tensor,
        temp_masks: torch.Tensor,
        next_masks: torch.Tensor,
        no_merge_masks: torch.Tensor,
        next_labels: torch.Tensor,
        next_scores: torch.Tensor,
        next_queries: torch.Tensor,
        next_query_feats: torch.Tensor,
        next_sem_preds: torch.Tensor,
        next_xyz: torch.Tensor,
        det_dino_feats: Optional[torch.Tensor] = None,
        dino_cfg: Optional[dict] = None,
        det_feat3d: Optional[torch.Tensor] = None,
        feat3d_cfg: Optional[dict] = None,
    ) -> Tuple[int, List[int], List[int]]:
        """Update track memory after association and optional stage-2 edits.

        Args:
            points_per_mask: points per frame mask (P).
            row_ind/col_ind: matched indices into (mem, det).
            temp_masks: (Nm, P) current-frame segment masks to append for existing tracks.
            next_*: per-det tensors for current frame.
            no_merge_masks: (Nc,) bool; True means this det will birth a new track.

        Returns:
            birth_cnt, birth_det_idx (list), birth_track_ids (list)
        """
        if not self.initialized:
            raise RuntimeError("TrackManager is not initialized")
        assert self.masks is not None and self.labels is not None and self.scores is not None
        assert self.queries is not None and self.query_feats is not None and self.sem_preds is not None
        assert self.xyz is not None and self.merge_counts is not None and self.track_ids is not None

        Nm = int(self.masks.shape[0])
        device = self.masks.device

        temp = torch.zeros((Nm,), dtype=torch.bool, device=device)
        if row_ind.numel() > 0:
            temp[row_ind] = True

        zeros_seg = torch.zeros((Nm, points_per_mask), dtype=torch.bool, device=device)
        next_masks_for_mem = torch.where(temp.unsqueeze(1), temp_masks.to(device=device), zeros_seg)
        self.masks = torch.cat((self.masks, next_masks_for_mem), dim=1)

        # Birth (new tracks): pad previous frames with zeros, then append current mask.
        birth_idx_t = torch.nonzero(no_merge_masks, as_tuple=False).reshape(-1)
        birth_cnt = int(birth_idx_t.numel())
        birth_det_idx = birth_idx_t.detach().cpu().tolist() if birth_cnt > 0 else []

        birth_track_ids: List[int] = []
        if birth_cnt > 0:
            former_padding = torch.zeros((birth_cnt, points_per_mask * self.fi), dtype=torch.bool, device=device)
            new_masks = torch.cat((former_padding, next_masks[birth_idx_t].to(device=device)), dim=1)
            self.masks = torch.cat((self.masks, new_masks), dim=0)

        # Merge counts and ids.
        if row_ind.numel() > 0:
            self.merge_counts[row_ind] += 1  # type: ignore[index]
        if birth_cnt > 0:
            self.merge_counts = torch.cat(
                (
                    self.merge_counts,
                    torch.zeros((birth_cnt,), dtype=torch.long, device=self.merge_counts.device),
                ),
                dim=0,
            )
            new_ids = torch.arange(
                self.next_track_id, self.next_track_id + birth_cnt, device=self.scores.device, dtype=torch.long
            )
            self.next_track_id += birth_cnt
            self.track_ids = torch.cat((self.track_ids, new_ids), dim=0)
            birth_track_ids = new_ids.detach().cpu().tolist()

        # Weighted update for matched tracks (EMA-like).
        if row_ind.numel() > 0:
            if self.merge_type == "count":
                count = self.merge_counts[row_ind]
            else:
                count = torch.full_like(self.merge_counts[row_ind], fill_value=int(self.fi))

            self.scores[row_ind] = (self.scores[row_ind] * count + next_scores[col_ind]) / (count + 1)

            count_vec = count.unsqueeze(-1)
            self.queries[row_ind] = (self.queries[row_ind] * count_vec + next_queries[col_ind]) / (count_vec + 1)
            self.query_feats[row_ind] = (self.query_feats[row_ind] * count_vec + next_query_feats[col_ind]) / (
                count_vec + 1
            )
            self.sem_preds[row_ind] = (self.sem_preds[row_ind] * count_vec + next_sem_preds[col_ind]) / (count_vec + 1)
            self.xyz[row_ind] = (self.xyz[row_ind] * count_vec + next_xyz[col_ind]) / (count_vec + 1)

        # Append births' features.
        if birth_cnt > 0:
            self.scores = torch.cat((self.scores, next_scores[birth_idx_t]), dim=0)
            self.labels = torch.cat((self.labels, next_labels[birth_idx_t]), dim=0)
            self.queries = torch.cat((self.queries, next_queries[birth_idx_t]), dim=0)
            self.query_feats = torch.cat((self.query_feats, next_query_feats[birth_idx_t]), dim=0)
            self.sem_preds = torch.cat((self.sem_preds, next_sem_preds[birth_idx_t]), dim=0)
            self.xyz = torch.cat((self.xyz, next_xyz[birth_idx_t]), dim=0)

        # Optional: update extra DINO modality (pooled per instance).
        # This is intentionally isolated so it does not affect existing behavior
        # when det_dino_feats is None.
        if det_dino_feats is not None:
            try:
                det_dino_feats = det_dino_feats.to(device=self.scores.device)
                if det_dino_feats.dim() == 1:
                    det_dino_feats = det_dino_feats.unsqueeze(1)
            except Exception:
                det_dino_feats = None

        if det_dino_feats is not None:
            cfg = dino_cfg if isinstance(dino_cfg, dict) else {}
            update = cfg.get("update", {}) if isinstance(cfg.get("update", {}), dict) else {}
            update_type = str(update.get("type", "ema")).lower()
            alpha = float(update.get("alpha", 0.2))
            normalize = bool(cfg.get("normalize", True))

            # Initialize if first time receiving DINO feats.
            if self.dino_feats is None:
                # At this stage we already have existing tracks. Initialize to zeros.
                D = int(det_dino_feats.shape[1])
                n_after = int(self.scores.shape[0])
                n_before = max(n_after - int(birth_cnt), 0)
                self.dino_feats = torch.zeros((int(n_before), D), device=self.scores.device, dtype=det_dino_feats.dtype)
            else:
                # Ensure dimensionality matches.
                if int(self.dino_feats.shape[1]) != int(det_dino_feats.shape[1]):
                    # Mismatch: skip update for robustness.
                    det_dino_feats = None

        if det_dino_feats is not None and self.dino_feats is not None:
            # Matched tracks: update.
            if row_ind.numel() > 0:
                try:
                    new = det_dino_feats[col_ind]
                    old = self.dino_feats[row_ind]
                    if update_type == "mean":
                        # Count-based mean consistent with merge_type.
                        count = self.merge_counts[row_ind].to(torch.float32).unsqueeze(1)
                        old = (old * count + new) / (count + 1.0)
                    else:
                        # EMA.
                        old = (1.0 - alpha) * old + alpha * new
                    if normalize:
                        old = torch.nn.functional.normalize(old, dim=1, eps=1e-6)
                    self.dino_feats[row_ind] = old
                except Exception:
                    pass

            # Birth tracks: append.
            if birth_cnt > 0:
                try:
                    birth_new = det_dino_feats[birth_idx_t]
                    if normalize:
                        birth_new = torch.nn.functional.normalize(birth_new, dim=1, eps=1e-6)
                    self.dino_feats = torch.cat((self.dino_feats, birth_new), dim=0)
                except Exception:
                    # Keep consistent length by padding zeros if append fails.
                    try:
                        D = int(self.dino_feats.shape[1])
                        z = torch.zeros((birth_cnt, D), device=self.dino_feats.device, dtype=self.dino_feats.dtype)
                        self.dino_feats = torch.cat((self.dino_feats, z), dim=0)
                    except Exception:
                        pass

        # Optional: update extra 3D pooled features (per instance).
        if det_feat3d is not None:
            try:
                det_feat3d = det_feat3d.to(device=self.scores.device)
                if det_feat3d.dim() == 1:
                    det_feat3d = det_feat3d.unsqueeze(1)
            except Exception:
                det_feat3d = None

        if det_feat3d is not None:
            cfg3d = feat3d_cfg if isinstance(feat3d_cfg, dict) else {}
            update3d = cfg3d.get("update", {}) if isinstance(cfg3d.get("update", {}), dict) else {}
            update_type3d = str(update3d.get("type", "ema")).lower()
            alpha3d = float(update3d.get("alpha", 0.2))
            normalize3d = bool(cfg3d.get("normalize", True))

            if self.feat3d is None:
                D = int(det_feat3d.shape[1])
                n_after = int(self.scores.shape[0])
                n_before = max(n_after - int(birth_cnt), 0)
                self.feat3d = torch.zeros((int(n_before), D), device=self.scores.device, dtype=det_feat3d.dtype)
            else:
                if int(self.feat3d.shape[1]) != int(det_feat3d.shape[1]):
                    det_feat3d = None

        if det_feat3d is not None and self.feat3d is not None:
            if row_ind.numel() > 0:
                try:
                    new = det_feat3d[col_ind]
                    old = self.feat3d[row_ind]
                    if update_type3d == "mean":
                        count = self.merge_counts[row_ind].to(torch.float32).unsqueeze(1)
                        old = (old * count + new) / (count + 1.0)
                    else:
                        old = (1.0 - alpha3d) * old + alpha3d * new
                    if normalize3d:
                        old = torch.nn.functional.normalize(old, dim=1, eps=1e-6)
                    self.feat3d[row_ind] = old
                except Exception:
                    pass

            if birth_cnt > 0:
                try:
                    birth_new = det_feat3d[birth_idx_t]
                    if normalize3d:
                        birth_new = torch.nn.functional.normalize(birth_new, dim=1, eps=1e-6)
                    self.feat3d = torch.cat((self.feat3d, birth_new), dim=0)
                except Exception:
                    try:
                        D = int(self.feat3d.shape[1])
                        z = torch.zeros((birth_cnt, D), device=self.feat3d.device, dtype=self.feat3d.dtype)
                        self.feat3d = torch.cat((self.feat3d, z), dim=0)
                    except Exception:
                        pass

        return birth_cnt, birth_det_idx, birth_track_ids

    def mark_last_births_tentative(self, *, birth_cnt: int) -> None:
        """Mark the last `birth_cnt` appended tracks as tentative (delayed birth)."""
        if birth_cnt <= 0:
            return
        if self.track_states is None or self.tent_birth_fi is None or self.tent_hits is None:
            return
        n = int(self.track_states.numel())
        start = max(n - int(birth_cnt), 0)
        try:
            self.track_states[start:n] = 1  # tentative
            self.tent_birth_fi[start:n] = int(self.fi)
            self.tent_hits[start:n] = 0
        except Exception:
            pass

    def update_tentative_hits(self, *, matched_mem_idx: torch.Tensor) -> None:
        """Increment hit counter for tentative tracks that are matched in this frame."""
        if self.track_states is None or self.tent_hits is None:
            return
        try:
            if matched_mem_idx.numel() == 0:
                return
            idx = matched_mem_idx.detach()
            tent = self.track_states[idx] == 1
            if tent.any():
                self.tent_hits[idx[tent]] += 1
        except Exception:
            pass

    def prune_tentatives(self, *, window: int, min_hits: int) -> Tuple[int, int]:
        """Confirm or delete tentative tracks whose window has elapsed.

        Args:
            window: number of frames in the confirmation window.
            min_hits: minimum matches within window to confirm.

        Returns:
            confirmed_cnt, deleted_cnt
        """
        if window <= 0:
            return 0, 0
        if self.track_states is None or self.tent_birth_fi is None or self.tent_hits is None:
            return 0, 0
        try:
            age = int(self.fi) - self.tent_birth_fi
            ready = (self.track_states == 1) & (age >= int(window - 1))
            if not bool(ready.any().item()):
                return 0, 0
            to_confirm = ready & (self.tent_hits >= int(min_hits))
            to_delete = ready & (~to_confirm)
            confirmed_cnt = int(to_confirm.sum().item())
            deleted_cnt = int(to_delete.sum().item())
            if confirmed_cnt > 0:
                self.track_states[to_confirm] = 0
            if deleted_cnt > 0:
                keep = ~to_delete
                self._prune_by_mask(keep)
            return confirmed_cnt, deleted_cnt
        except Exception:
            return 0, 0

    def _prune_by_mask(self, keep: torch.Tensor) -> None:
        """Prune internal track state by boolean mask."""
        if not self.initialized:
            return
        if keep.dtype != torch.bool:
            return
        try:
            if self.masks is not None:
                self.masks = self.masks[keep]
            if self.labels is not None:
                self.labels = self.labels[keep]
            if self.scores is not None:
                self.scores = self.scores[keep]
            if self.queries is not None:
                self.queries = self.queries[keep]
            if self.query_feats is not None:
                self.query_feats = self.query_feats[keep]
            if self.sem_preds is not None:
                self.sem_preds = self.sem_preds[keep]
            if self.xyz is not None:
                self.xyz = self.xyz[keep]
            if self.merge_counts is not None:
                self.merge_counts = self.merge_counts[keep]
            if self.track_ids is not None:
                self.track_ids = self.track_ids[keep]
            if self.dino_feats is not None:
                self.dino_feats = self.dino_feats[keep]
            if self.feat3d is not None:
                self.feat3d = self.feat3d[keep]
            if self.track_states is not None:
                self.track_states = self.track_states[keep]
            if self.tent_birth_fi is not None:
                self.tent_birth_fi = self.tent_birth_fi[keep]
            if self.tent_hits is not None:
                self.tent_hits = self.tent_hits[keep]

            # Python lists.
            k = keep.detach().cpu().tolist()
            if self.track_gt_votes is not None:
                self.track_gt_votes = [v for v, kk in zip(self.track_gt_votes, k) if kk]
            if self.track_gt_ids is not None:
                self.track_gt_ids = [v for v, kk in zip(self.track_gt_ids, k) if kk]
            if self.track_gt_strength is not None:
                self.track_gt_strength = [v for v, kk in zip(self.track_gt_strength, k) if kk]
            if self.track_voxels is not None:
                self.track_voxels = [v for v, kk in zip(self.track_voxels, k) if kk]
        except Exception:
            pass

    def get_confirmed_mask(self) -> Optional[torch.Tensor]:
        """Return boolean mask of confirmed tracks (0=confirmed, 1=tentative)."""
        if self.track_states is None:
            return None

    def merge_tracks_inplace(self, *, keep_idx: int, drop_idx: int, voxel_cap: int = 1024) -> None:
        """Merge `drop_idx` into `keep_idx` in-place (no pruning)."""
        if not self.initialized:
            return
        if keep_idx == drop_idx:
            return
        if self.masks is None or self.scores is None:
            return
        try:
            # Masks: union (bitwise OR).
            self.masks[keep_idx] = self.masks[keep_idx] | self.masks[drop_idx]
        except Exception:
            pass

        # Scores: keep max (more stable for ranking).
        try:
            self.scores[keep_idx] = torch.maximum(self.scores[keep_idx], self.scores[drop_idx])
        except Exception:
            pass

        # Merge counts: add (optional).
        try:
            if self.merge_counts is not None:
                self.merge_counts[keep_idx] = self.merge_counts[keep_idx] + self.merge_counts[drop_idx]
        except Exception:
            pass

        # Weighted average for continuous tensors (queries/query_feats/sem/xyz).
        def _wavg(a: Optional[torch.Tensor], b: Optional[torch.Tensor]) -> None:
            if a is None or b is None:
                return
            try:
                wa = float(self.scores[keep_idx].detach().cpu().item())
                wb = float(self.scores[drop_idx].detach().cpu().item())
                w = wa + wb + 1e-6
                a[keep_idx] = (a[keep_idx] * wa + a[drop_idx] * wb) / w
            except Exception:
                pass

        _wavg(self.queries, self.queries)
        _wavg(self.query_feats, self.query_feats)
        _wavg(self.sem_preds, self.sem_preds)
        _wavg(self.xyz, self.xyz)

        # Optional modalities.
        if self.dino_feats is not None:
            try:
                wa = float(self.scores[keep_idx].detach().cpu().item())
                wb = float(self.scores[drop_idx].detach().cpu().item())
                w = wa + wb + 1e-6
                v = (self.dino_feats[keep_idx] * wa + self.dino_feats[drop_idx] * wb) / w
                self.dino_feats[keep_idx] = torch.nn.functional.normalize(v, dim=0, eps=1e-6)
            except Exception:
                pass
        if self.feat3d is not None:
            try:
                wa = float(self.scores[keep_idx].detach().cpu().item())
                wb = float(self.scores[drop_idx].detach().cpu().item())
                w = wa + wb + 1e-6
                v = (self.feat3d[keep_idx] * wa + self.feat3d[drop_idx] * wb) / w
                self.feat3d[keep_idx] = torch.nn.functional.normalize(v, dim=0, eps=1e-6)
            except Exception:
                pass

        # Voxel cache union.
        if self.track_voxels is not None:
            try:
                if keep_idx < len(self.track_voxels) and drop_idx < len(self.track_voxels):
                    self.track_voxels[keep_idx] = _cap_union_1d(
                        self.track_voxels[keep_idx], self.track_voxels[drop_idx], voxel_cap
                    )
            except Exception:
                pass

        # Merge GT votes dicts.
        try:
            if self.track_gt_votes is not None and keep_idx < len(self.track_gt_votes) and drop_idx < len(self.track_gt_votes):
                a = self.track_gt_votes[keep_idx]
                b = self.track_gt_votes[drop_idx]
                for k, v in b.items():
                    a[k] = a.get(k, 0.0) + float(v)
                # Refresh cached best id/strength.
                best_id, best_val = -1, 0.0
                for gid, vv in a.items():
                    if vv > best_val:
                        best_id, best_val = int(gid), float(vv)
                if self.track_gt_ids is not None and keep_idx < len(self.track_gt_ids):
                    self.track_gt_ids[keep_idx] = int(best_id)
                if self.track_gt_strength is not None and keep_idx < len(self.track_gt_strength):
                    self.track_gt_strength[keep_idx] = float(best_val)
        except Exception:
            pass
        try:
            return self.track_states == 0
        except Exception:
            return None

    def select_topk_outputs(
        self, *, topk: int, use_bbox: bool, include_tentative: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Select top-K tracks for output (does not prune internal memory)."""
        if not self.initialized:
            raise RuntimeError("TrackManager is not initialized")
        assert self.masks is not None and self.labels is not None and self.scores is not None
        assert self.queries is not None and self.xyz is not None and self.track_ids is not None

        # Optionally exclude tentative tracks from output (delayed birth).
        cand_scores = self.scores
        cand_idx = torch.arange(self.scores.shape[0], device=self.scores.device)
        if not include_tentative:
            cm = self.get_confirmed_mask()
            if cm is not None and int(cm.numel()) == int(self.scores.shape[0]):
                cand_idx = cand_idx[cm]
                cand_scores = cand_scores[cm]

        if int(cand_scores.numel()) > int(topk):
            _, kept_local = cand_scores.topk(int(topk))
            kept = cand_idx[kept_local]
        else:
            kept = cand_idx

        self.output_track_ids = self.track_ids[kept].detach().cpu().tolist()

        out_masks = self.masks[kept]
        out_labels = self.labels[kept]
        out_scores = self.scores[kept]
        out_queries = self.queries[kept]
        out_bboxes = self.xyz[kept] if use_bbox else None
        return kept, out_masks, out_labels, out_scores, out_queries, out_bboxes

    def get_track(self, idx: int) -> TrackInstance:
        """Create a TrackInstance view for debugging (not used in hot path)."""
        if not self.initialized:
            raise RuntimeError("TrackManager is not initialized")
        assert self.masks is not None and self.labels is not None and self.scores is not None
        assert self.queries is not None and self.query_feats is not None and self.sem_preds is not None and self.xyz is not None
        assert self.track_ids is not None
        return TrackInstance(
            mask=self.masks[idx],
            label=self.labels[idx],
            score=self.scores[idx],
            query=self.queries[idx],
            query_feat=self.query_feats[idx],
            sem_pred=self.sem_preds[idx],
            xyz=self.xyz[idx],
            track_id=int(self.track_ids[idx].item()),
        )

    def get_detection(self, *, mask: torch.Tensor, label: torch.Tensor, score: torch.Tensor) -> DetectionInstance:
        """Create a DetectionInstance view for debugging (not used in hot path)."""
        return DetectionInstance(mask=mask, label=label, score=score)


def _cap_union_1d(a: torch.Tensor, b: torch.Tensor, cap: int) -> torch.Tensor:
    """Union two 1D tensors (voxel ids) with optional cap."""
    try:
        if a.numel() == 0:
            out = b
        elif b.numel() == 0:
            out = a
        else:
            out = torch.unique(torch.cat([a, b], dim=0))
        if cap > 0 and out.numel() > cap:
            out = out[:cap]
        return out
    except Exception:
        return a
