from __future__ import annotations

from typing import Any, Dict, Iterable, Tuple

from mmengine.logging import MMLogger
from mmdet3d.evaluation import seg_eval
from mmdet3d.evaluation.metrics import SegMetric
from mmdet3d.registry import METRICS


@METRICS.register_module()
class SemanticSegMetric(SegMetric):
    """Semantic-only metric wrapper around mmdet3d's seg_eval.

    Notes:
    - Expects each processed item to be (eval_ann, pred_dict).
    - pred_dict['pts_semantic_mask'] can be a numpy array or a list whose first
      element is the semantic prediction array.
    """

    def __init__(
        self,
        metric_meta: Dict[str, Any] | None = None,
        logger_keys: Tuple[str, ...] = ("miou", "macc", "aacc"),
        ignore_index: int | None = None,
        **kwargs,
    ) -> None:
        self.metric_meta = metric_meta
        self.logger_keys = logger_keys
        self.ignore_index = ignore_index
        super().__init__(**kwargs)

    def compute_metrics(self, results: Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]) -> Dict[str, float]:
        logger: MMLogger = MMLogger.get_current_instance()

        metric_meta = self.metric_meta or {}
        label2cat = metric_meta.get("label2cat", None)
        ignore_index = self.ignore_index if self.ignore_index is not None else metric_meta.get("ignore_index", None)
        if isinstance(ignore_index, (list, tuple)):
            ignore_index = ignore_index[0] if len(ignore_index) else None

        # Fallback to dataset_meta injected by runner.
        if label2cat is None:
            if self.dataset_meta is not None:
                classes = self.dataset_meta.get("classes", None)
                if classes is not None:
                    label2cat = {i: name for i, name in enumerate(classes)}
            if label2cat is None:
                raise AssertionError("SemanticSegMetric requires `metric_meta.label2cat` or `dataset_meta.classes`.")

        if ignore_index is None and self.dataset_meta is not None:
            ignore_index = self.dataset_meta.get("ignore_index", None)
            if isinstance(ignore_index, (list, tuple)):
                ignore_index = ignore_index[0] if len(ignore_index) else None
        if ignore_index is None:
            ignore_index = -1

        gt_semantic_masks = []
        pred_semantic_masks = []

        for eval_ann, single_pred_results in results:
            gt_semantic_masks.append(eval_ann["pts_semantic_mask"])
            pred = single_pred_results.get("pts_semantic_mask")
            if isinstance(pred, (list, tuple)):
                pred = pred[0] if len(pred) else pred
            pred_semantic_masks.append(pred)

        ret = seg_eval(
            gt_semantic_masks,
            pred_semantic_masks,
            label2cat,
            ignore_index,
            logger=logger,
        )

        metrics: Dict[str, float] = {}
        for key in self.logger_keys:
            if key in ret:
                metrics[key] = float(ret[key])
        if not metrics:
            # Fallback: keep any scalar float/int outputs.
            metrics = {k: float(v) for k, v in ret.items() if isinstance(v, (float, int))}
        return metrics
