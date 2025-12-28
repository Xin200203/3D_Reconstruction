import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class CrossFrameCriterion(nn.Module):
    """Compute cross-frame Slot-Transformer losses.

    Args:
        match_weight (float): weight of match (Hungarian / CE) loss.
        cons_weight  (float): weight of consistency (InfoNCE) loss.
        temperature  (float): temperature for InfoNCE.
    """
    def __init__(self, match_weight=1.0, cons_weight=0.5, temperature=0.07):
        super().__init__()
        self.match_weight = match_weight
        self.cons_weight = cons_weight
        self.temperature = temperature

    def _match_loss(self, attn, gt_affinity):
        """Binary cross-entropy on attn matrix vs GT affinity (0/1)."""
        return F.binary_cross_entropy(attn, gt_affinity.float())

    def _consistency_loss(self, queries):
        """InfoNCE across frames: positive = same instance id."""
        device = queries[0].device
        # flatten frames
        features = torch.cat(queries, 0)                     # (ΣNc, D)
        features = F.normalize(features, dim=-1)
        logits = features @ features.t() / self.temperature  # (N,N)
        labels = torch.arange(len(features), device=device)
        loss = F.cross_entropy(logits, labels)
        return loss

    def forward(self, attn, gt_affinity, queries=None):
        """Return dict of losses.

        Args:
            attn (Tensor): (Nc,Nm) softmax matrix from Transformer.
            gt_affinity (Tensor): same shape, 0/1 ground truth.
            queries (List[Tensor]|None): query features of multiple frames for consistency.
        """
        losses = {}
        losses['loss_match'] = self.match_weight * self._match_loss(attn, gt_affinity)
        if self.cons_weight > 0 and queries is not None:
            losses['loss_cons'] = self.cons_weight * self._consistency_loss(queries)
        return losses 