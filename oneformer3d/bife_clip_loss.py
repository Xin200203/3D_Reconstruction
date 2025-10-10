import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from typing import Optional
from .training_scheduler import ProgressScheduler, coverage_aware_weight

@MODELS.register_module()
class ClipConsCriterion(nn.Module):
    """Enhanced CLIP consistency loss for 2D-3D feature alignment.
    
    Optimized version based on v1 loss optimization guide:
    - Progress-based activation with warmup 
    - Coverage-aware weight adjustment
    - Improved numerical stability
    - Point-level cosine similarity only on valid projection points
    - Strong gradient protection for 2D/CLIP branch

    Args:
        loss_weight (float): Final scaling factor (will be gradually increased).
        temperature (float): Temperature parameter for contrastive learning.
        gradient_flow_ratio (float): Ratio of gradient allowed to flow back to CLIP features.
        warmup_end_progress (float): Training progress when warmup ends (0.0-1.0).
        ramp_duration_progress (float): Duration of weight ramp-up in progress units.
        coverage_threshold (float): Minimum valid ratio threshold for coverage gating.
        use_soft_coverage_gate (bool): Use soft coverage gating instead of hard threshold.
    """
    def __init__(self, 
                 loss_weight: float = 0.05,
                 temperature: float = 0.07, 
                 gradient_flow_ratio: float = 0.05,
                 warmup_end_progress: float = 0.10,
                 ramp_duration_progress: float = 0.10,
                 coverage_threshold: float = 0.6,
                 use_soft_coverage_gate: bool = True):
        super().__init__()
        self.final_loss_weight = loss_weight
        self.temperature = temperature
        self.gradient_flow_ratio = gradient_flow_ratio
        self.warmup_end_progress = warmup_end_progress
        self.ramp_duration_progress = ramp_duration_progress
        self.coverage_threshold = coverage_threshold
        self.use_soft_coverage_gate = use_soft_coverage_gate
        
        # Progress scheduler will be injected from outside
        self.progress_scheduler: Optional[ProgressScheduler] = None
        self.current_step = 0  # Fallback counter
        
        # For numerical stability
        self.eps = 1e-8
        self.logit_clamp = 20.0

    def get_current_weight(self) -> float:
        """Calculate current loss weight using progress scheduler."""
        if self.progress_scheduler is not None:
            # Use injected progress scheduler for modern progress-based scheduling
            base_weight = self.progress_scheduler.clip_weight_schedule(
                self.final_loss_weight,
                self.warmup_end_progress,
                self.ramp_duration_progress
            )
            return base_weight
        else:
            # Fallback to legacy step-based scheduling
            if self.current_step == 0:
                return 0.0
            # Simple linear ramp-up
            progress = min(1.0, self.current_step / 4000.0)  # 4000 steps = full weight
            return self.final_loss_weight * progress
    
    def step(self):
        """Call this after each training step to update internal counter."""
        self.current_step += 1

    def set_progress_scheduler(self, scheduler: 'ProgressScheduler'):
        """Inject progress scheduler for modern progress-based scheduling."""
        self.progress_scheduler = scheduler

    def forward(self, feat_fusion, clip_feat_detach, valid_projection_mask=None):
        """Enhanced CLIP alignment loss computation with v1 optimizations.
        
        Args:
            feat_fusion (List[Tensor]): Fused 3D features, each scene (N_i, 256)
            clip_feat_detach (List[Tensor]): CLIP features, each scene (N_i, 256) or (256,)
            valid_projection_mask (List[Tensor], optional): Valid projection masks (N_i,)
            
        Returns:
            Tensor: CLIP alignment loss
        """
        # Get current weight (warmup + scheduling)
        current_weight = self.get_current_weight()
        if current_weight == 0.0:
            return torch.tensor(0.0, device=feat_fusion[0].device if isinstance(feat_fusion, list) else feat_fusion.device, requires_grad=True)
        
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
        total_valid_ratio = 0.0

        for i, (f_fuse, f_clip) in enumerate(zip(feat_fusion, clip_feat_detach)):
            # Enhanced gradient control: strong protection for 2D/CLIP branch
            f_clip = (f_clip * self.gradient_flow_ratio + 
                     f_clip.detach() * (1 - self.gradient_flow_ratio))
            
            # Handle CLIP feature dimensions
            if f_clip.dim() == 1:  # Global feature (256,)
                f_clip = f_clip.unsqueeze(0).expand(f_fuse.size(0), -1)
            
            # Calculate valid ratio for coverage-aware weighting
            if valid_projection_mask is not None:
                valid_mask = valid_projection_mask[i]
                valid_ratio = valid_mask.float().mean().item()
                total_valid_ratio += valid_ratio
                
                if valid_mask.sum() == 0:  # No valid points
                    continue
                f_fuse = f_fuse[valid_mask]
                f_clip = f_clip[valid_mask]
            else:
                # If no mask provided, assume all points are valid
                valid_ratio = 1.0
                total_valid_ratio += valid_ratio
            
            if f_fuse.size(0) == 0:  # Skip if no valid points
                continue
            
            # L2 normalization to unit sphere (improved numerical stability)
            f_fuse_norm = F.normalize(f_fuse, dim=-1, p=2, eps=self.eps)
            f_clip_norm = F.normalize(f_clip, dim=-1, p=2, eps=self.eps)
            
            # Point-level cosine similarity (as specified in guide)
            cos_sim = torch.sum(f_fuse_norm * f_clip_norm, dim=-1)  # (N_valid,)
            
            # Point-level consistency loss: L_pix = mean(1 - cos(f2d, f3d))
            point_loss = (1.0 - cos_sim).mean()
            
            # Numerical stability: ensure loss is finite
            if torch.isnan(point_loss) or torch.isinf(point_loss):
                continue
                
            total_loss += point_loss * f_fuse.size(0)
            total_points += f_fuse.size(0)
        
        if total_points == 0:
            # Return zero loss if no valid points
            return torch.tensor(0.0, device=feat_fusion[0].device, requires_grad=True)
        
        avg_loss = total_loss / max(total_points, 1e-8)  # Prevent division by zero
        
        # Apply coverage-aware weighting
        num_batches = len(feat_fusion)
        avg_valid_ratio = total_valid_ratio / max(num_batches, 1e-8)
        coverage_weight = coverage_aware_weight(
            avg_valid_ratio, 
            self.coverage_threshold, 
            self.use_soft_coverage_gate
        )
        
        # Final weight combines schedule weight and coverage weight
        final_weight = current_weight * coverage_weight
        
        # Ensure loss is in FP32 for numerical stability  
        final_loss = torch.tensor(final_weight * avg_loss, dtype=torch.float32, 
                                 device=feat_fusion[0].device, requires_grad=True)
        return final_loss


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