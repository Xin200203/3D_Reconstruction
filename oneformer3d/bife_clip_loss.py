import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

@MODELS.register_module()
class ClipConsCriterion(nn.Module):
    """Enhanced CLIP consistency loss for 2D-3D feature alignment.
    
    This is the enhanced version that implements temperature scaling and gradient control
    as specified in the optimization document.

    Args:
        loss_weight (float): Scaling factor applied to the loss.
        temperature (float): Temperature parameter for contrastive learning.
        gradient_flow_ratio (float): Ratio of gradient allowed to flow back to CLIP features.
    """
    def __init__(self, 
                 loss_weight: float = 0.1, 
                 temperature: float = 0.07, 
                 gradient_flow_ratio: float = 0.05):
        super().__init__()
        self.loss_weight = loss_weight
        self.temperature = temperature
        self.gradient_flow_ratio = gradient_flow_ratio

    def forward(self, feat_fusion, clip_feat_detach, valid_projection_mask=None):
        """Enhanced CLIP alignment loss computation.
        
        Args:
            feat_fusion (List[Tensor]): Fused 3D features, each scene (N_i, 256)
            clip_feat_detach (List[Tensor]): CLIP features, each scene (N_i, 256) or (256,)
            valid_projection_mask (List[Tensor], optional): Valid projection masks (N_i,)
            
        Returns:
            Tensor: CLIP alignment loss
        """
        # Support single tensor input (convert to list)
        if not isinstance(feat_fusion, (list, tuple)):
            feat_fusion = [feat_fusion]
        if not isinstance(clip_feat_detach, (list, tuple)):
            clip_feat_detach = [clip_feat_detach] * len(feat_fusion)
        
        # Handle valid_projection_mask
        if valid_projection_mask is not None and not isinstance(valid_projection_mask, (list, tuple)):
            valid_projection_mask = [valid_projection_mask] * len(feat_fusion)

        total_loss = 0.0
        total_points = 0

        for i, (f_fuse, f_clip) in enumerate(zip(feat_fusion, clip_feat_detach)):
            # Enhanced gradient control: allow partial gradient flow to CLIP branch
            f_clip = (f_clip * self.gradient_flow_ratio + 
                     f_clip.detach() * (1 - self.gradient_flow_ratio))
            
            # Handle CLIP feature dimensions
            if f_clip.dim() == 1:  # Global feature (256,)
                f_clip = f_clip.unsqueeze(0).expand(f_fuse.size(0), -1)
            
            # Apply valid projection mask if provided
            if valid_projection_mask is not None:
                valid_mask = valid_projection_mask[i]
                if valid_mask.sum() == 0:  # No valid points
                    continue
                f_fuse = f_fuse[valid_mask]
                f_clip = f_clip[valid_mask]
            
            if f_fuse.size(0) == 0:  # Skip if no valid points
                continue
            
            # L2 normalization to unit sphere
            f_fuse_norm = F.normalize(f_fuse, dim=-1, p=2)
            f_clip_norm = F.normalize(f_clip, dim=-1, p=2)
            
            # Temperature-scaled cosine similarity
            cos_sim = torch.sum(f_fuse_norm * f_clip_norm, dim=-1)  # (N_valid,)
            scaled_sim = cos_sim / self.temperature
            
            # Contrastive loss: maximize similarity
            loss_i = -torch.log(torch.sigmoid(scaled_sim) + 1e-8).mean()
            
            total_loss += loss_i * f_fuse.size(0)
            total_points += f_fuse.size(0)
        
        if total_points == 0:
            return torch.tensor(0.0, device=feat_fusion[0].device, requires_grad=True)
        
        avg_loss = total_loss / total_points
        return self.loss_weight * avg_loss


# Keep the original ClipConsCriterion for backward compatibility
@MODELS.register_module()
class LegacyClipConsCriterion(nn.Module):
    """Legacy CLIP consistency loss for backward compatibility."""
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, feat_fusion, clip_feat_detach):
        # Original implementation
        if not isinstance(feat_fusion, (list, tuple)):
            feat_fusion = [feat_fusion]
        if not isinstance(clip_feat_detach, (list, tuple)):
            clip_feat_detach = [clip_feat_detach] * len(feat_fusion)

        total_loss, total_points = 0.0, 0
        for f_fuse, f_clip in zip(feat_fusion, clip_feat_detach):
            f_clip = f_clip.detach()
            if f_clip.dim() == 1:
                f_clip = f_clip.unsqueeze(0).expand(f_fuse.size(0), -1)
            f_fuse_n = F.normalize(f_fuse, dim=-1)
            f_clip_n = F.normalize(f_clip, dim=-1)
            cos_sim = (f_fuse_n * f_clip_n).sum(dim=-1)
            loss_sample = (1 - cos_sim).mean()
            total_loss += loss_sample * f_fuse.shape[0]
            total_points += f_fuse.shape[0]
        loss = total_loss / max(total_points, 1)
        return self.loss_weight * loss 