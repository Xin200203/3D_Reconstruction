from __future__ import annotations

from mmengine.hooks import Hook
from mmengine.registry import HOOKS


@HOOKS.register_module()
class MinkowskiClearCoordinateManagerHook(Hook):
    """Clear MinkowskiEngine global coordinate manager to avoid GPU memory growth.

    MinkowskiEngine may allocate/cached coordinate maps via its own CUDA
    allocator (not tracked by PyTorch). In long training runs, this can lead to
    gradually decreasing free GPU memory and eventual OOM.
    """

    def __init__(
        self,
        after_iter: bool = True,
        after_epoch: bool = False,
        interval: int = 1,
    ) -> None:
        self.after_iter = after_iter
        self.after_epoch = after_epoch
        self.interval = max(int(interval), 1)

    @staticmethod
    def _clear() -> None:
        try:
            import MinkowskiEngine as ME  # type: ignore

            ME.clear_global_coordinate_manager()
        except Exception:
            # MinkowskiEngine not available or API mismatch; do nothing.
            return

    def after_train_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:  # type: ignore[override]
        if not self.after_iter:
            return
        if batch_idx % self.interval != 0:
            return
        self._clear()

    def after_train_epoch(self, runner) -> None:  # type: ignore[override]
        if not self.after_epoch:
            return
        self._clear()

    def after_val_iter(self, runner, batch_idx: int, data_batch=None, outputs=None) -> None:  # type: ignore[override]
        if not self.after_iter:
            return
        if batch_idx % self.interval != 0:
            return
        self._clear()

