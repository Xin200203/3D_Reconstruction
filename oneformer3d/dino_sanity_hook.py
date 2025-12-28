"""Training-time sanity hook for online DINO injection.

This hook provides per-epoch statistics to ensure:
- 3D vertical BEV flip is actually happening (not silently disabled by sync_2d).
- DINO sparse feature injection is hitting backbone coordinates consistently.
"""

from __future__ import annotations

from typing import Any, Dict

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class DINOAlignmentSanityHook(Hook):
    """Log per-epoch pcd_vertical_flip rate and DINO hit ratios (s1..s16)."""

    def __init__(self, *, log_interval_epochs: int = 1) -> None:
        self.log_interval_epochs = int(log_interval_epochs)
        self._reset()

    def _reset(self) -> None:
        self._n_samples = 0
        self._n_vflip = 0
        self._hit_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._hit_cnt: Dict[str, int] = {k: 0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._n_dino_iters = 0
        self._n_dino_iters_missing = 0
        # DINO 融合尺度统计
        self._fuse_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._base_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._ratio_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._fuse_cnt: Dict[str, int] = {k: 0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._dino_mean_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._dino_std_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._base_mean_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        self._base_std_sum: Dict[str, float] = {k: 0.0 for k in ['s1', 's2', 's4', 's8', 's16']}
        # valid_rate 统计
        self._valid_sum = 0.0
        self._valid_cnt = 0
        self._valid_min = None
        self._valid_max = None

    @staticmethod
    def _unwrap_model(model: Any) -> Any:
        return getattr(model, 'module', model)

    def before_train_epoch(self, runner) -> None:
        self._reset()

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:
        # 1) Count vertical flip rate from data_samples meta.
        if isinstance(data_batch, dict):
            data_samples = data_batch.get('data_samples', None)
        else:
            data_samples = None

        if isinstance(data_samples, (list, tuple)):
            for ds in data_samples:
                meta = getattr(ds, 'img_metas', None)
                if not isinstance(meta, dict):
                    # fall back to Det3DDataSample.metainfo() if present
                    try:
                        meta = ds.metainfo
                    except Exception:
                        meta = {}
                vflip = bool(meta.get('pcd_vertical_flip', False))
                self._n_samples += 1
                self._n_vflip += int(vflip)

        # 2) Read latest DINO hit ratios from backbone (written during forward).
        model = self._unwrap_model(runner.model)
        backbone = getattr(model, 'backbone', None)
        hits = getattr(backbone, '_last_dino_hit', None) if backbone is not None else None

        if isinstance(hits, dict):
            self._n_dino_iters += 1
            if len(hits) == 0:
                self._n_dino_iters_missing += 1
            for k in ['s1', 's2', 's4', 's8', 's16']:
                v = hits.get(k, None)
                if isinstance(v, (int, float)):
                    self._hit_sum[k] += float(v)
                    self._hit_cnt[k] += 1

        # 3) Read latest DINO fuse stats from backbone (scale/ratio).
        fuse_stats = getattr(backbone, '_last_dino_fuse', None) if backbone is not None else None
        if isinstance(fuse_stats, dict):
            for k in ['s1', 's2', 's4', 's8', 's16']:
                s = fuse_stats.get(k, None)
                if isinstance(s, dict):
                    base_norm = s.get('base_norm', None)
                    fuse_norm = s.get('fuse_norm', None)
                    ratio = s.get('ratio', None)
                    dino_mean = s.get('dino_mean', None)
                    dino_std = s.get('dino_std', None)
                    base_mean = s.get('base_mean', None)
                    base_std = s.get('base_std', None)
                    if all(isinstance(x, (int, float)) for x in [base_norm, fuse_norm, ratio, dino_mean, dino_std, base_mean, base_std]):
                        self._base_sum[k] += float(base_norm)
                        self._fuse_sum[k] += float(fuse_norm)
                        self._ratio_sum[k] += float(ratio)
                        self._dino_mean_sum[k] += float(dino_mean)
                        self._dino_std_sum[k] += float(dino_std)
                        self._base_mean_sum[k] += float(base_mean)
                        self._base_std_sum[k] += float(base_std)
                        self._fuse_cnt[k] += 1

        # 4) Read latest valid_rate from detector (online projection).
        valid_rate = getattr(model, '_last_dino_valid_rate', None)
        if isinstance(valid_rate, (int, float)):
            self._valid_sum += float(valid_rate)
            self._valid_cnt += 1
            if self._valid_min is None or float(valid_rate) < self._valid_min:
                self._valid_min = float(valid_rate)
            if self._valid_max is None or float(valid_rate) > self._valid_max:
                self._valid_max = float(valid_rate)

    def after_train_epoch(self, runner) -> None:
        epoch = int(getattr(runner.train_loop, 'epoch', 0))
        if self.log_interval_epochs <= 0:
            return
        if (epoch + 1) % self.log_interval_epochs != 0:
            return

        vflip_rate = (self._n_vflip / self._n_samples) if self._n_samples > 0 else 0.0
        hit_means = {
            k: (self._hit_sum[k] / self._hit_cnt[k]) if self._hit_cnt[k] > 0 else float('nan')
            for k in ['s1', 's2', 's4', 's8', 's16']
        }
        ratio_means = {
            k: (self._ratio_sum[k] / self._fuse_cnt[k]) if self._fuse_cnt[k] > 0 else float('nan')
            for k in ['s1', 's2', 's4', 's8', 's16']
        }
        dino_mean_means = {
            k: (self._dino_mean_sum[k] / self._fuse_cnt[k]) if self._fuse_cnt[k] > 0 else float('nan')
            for k in ['s1', 's2', 's4', 's8', 's16']
        }
        dino_std_means = {
            k: (self._dino_std_sum[k] / self._fuse_cnt[k]) if self._fuse_cnt[k] > 0 else float('nan')
            for k in ['s1', 's2', 's4', 's8', 's16']
        }
        base_mean_means = {
            k: (self._base_mean_sum[k] / self._fuse_cnt[k]) if self._fuse_cnt[k] > 0 else float('nan')
            for k in ['s1', 's2', 's4', 's8', 's16']
        }
        base_std_means = {
            k: (self._base_std_sum[k] / self._fuse_cnt[k]) if self._fuse_cnt[k] > 0 else float('nan')
            for k in ['s1', 's2', 's4', 's8', 's16']
        }
        valid_mean = (self._valid_sum / self._valid_cnt) if self._valid_cnt > 0 else float('nan')

        logger = runner.logger
        logger.info(
            "[Sanity][Aug] pcd_vertical_flip_rate=%.4f (%d/%d)",
            vflip_rate, self._n_vflip, self._n_samples)
        logger.info(
            "[Sanity][DINO] hit_ratio_mean (s1,s2,s4,s8,s16) = "
            "(%.4f, %.4f, %.4f, %.4f, %.4f) | iters=%d missing_iters=%d",
            hit_means['s1'], hit_means['s2'], hit_means['s4'], hit_means['s8'], hit_means['s16'],
            self._n_dino_iters, self._n_dino_iters_missing)
        logger.info(
            "[Sanity][DINO] fuse_ratio_mean (s1,s2,s4,s8,s16) = "
            "(%.4f, %.4f, %.4f, %.4f, %.4f)",
            ratio_means['s1'], ratio_means['s2'], ratio_means['s4'], ratio_means['s8'], ratio_means['s16'])
        logger.info(
            "[Sanity][Scale] dino_mean/std (s1,s2,s4,s8,s16) = "
            "([%.3f,%.3f,%.3f,%.3f,%.3f], [%.3f,%.3f,%.3f,%.3f,%.3f])",
            dino_mean_means['s1'], dino_mean_means['s2'], dino_mean_means['s4'], dino_mean_means['s8'], dino_mean_means['s16'],
            dino_std_means['s1'], dino_std_means['s2'], dino_std_means['s4'], dino_std_means['s8'], dino_std_means['s16'])
        logger.info(
            "[Sanity][Scale] base_mean/std (s1,s2,s4,s8,s16) = "
            "([%.3f,%.3f,%.3f,%.3f,%.3f], [%.3f,%.3f,%.3f,%.3f,%.3f])",
            base_mean_means['s1'], base_mean_means['s2'], base_mean_means['s4'], base_mean_means['s8'], base_mean_means['s16'],
            base_std_means['s1'], base_std_means['s2'], base_std_means['s4'], base_std_means['s8'], base_std_means['s16'])
        logger.info(
            "[Sanity][Proj] valid_rate_mean=%.4f min=%.4f max=%.4f (n=%d)",
            valid_mean,
            self._valid_min if self._valid_min is not None else float('nan'),
            self._valid_max if self._valid_max is not None else float('nan'),
            self._valid_cnt)
