import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class ClipConsCriterion(nn.Module):
    """Cosine contrastive loss between fused point features and CLIP image features.

    Args:
        loss_weight (float): scaling factor applied to the loss.
    Notes:
        ‑ 假设输入 `feat_fusion` 为 List[Tensor]，每张场景 (N_p, C).
        ‑ `clip_feat_detach` 可以是 CLIP 图像全局特征 (C,) or per-pixel (N_p, C).
          如果是 (C,), 会 broadcast；若 shape 与 `feat_fusion[i]` 不同，将自动 broadcast last dim.
    """
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, feat_fusion, clip_feat_detach):
        # 支持输入 list 或 tensor（单 batch）
        if not isinstance(feat_fusion, (list, tuple)):
            feat_fusion = [feat_fusion]
        if not isinstance(clip_feat_detach, (list, tuple)):
            clip_feat_detach = [clip_feat_detach] * len(feat_fusion)

        total_loss, total_points = 0.0, 0
        for f_fuse, f_clip in zip(feat_fusion, clip_feat_detach):
            # detach clip features to avoid gradient back-prop into CLIP
            f_clip = f_clip.detach()
            if f_clip.dim() == 1:
                # global vector (C,) -> expand to (N,C)
                f_clip = f_clip.unsqueeze(0).expand(f_fuse.size(0), -1)
            # cosine similarity; add epsilon for numerical stability
            f_fuse_n = F.normalize(f_fuse, dim=-1)
            f_clip_n = F.normalize(f_clip, dim=-1)
            cos_sim = (f_fuse_n * f_clip_n).sum(dim=-1)
            loss_sample = (1 - cos_sim).mean()
            total_loss += loss_sample * f_fuse.shape[0]
            total_points += f_fuse.shape[0]
        loss = total_loss / max(total_points, 1)
        return self.loss_weight * loss 