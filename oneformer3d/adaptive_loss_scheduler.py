import torch
import torch.nn as nn
from typing import Dict


class AdaptiveLossScheduler:
    """Adaptive loss weight scheduler for multi-stage training.
    
    This scheduler implements the dynamic weight adjustment strategy
    as specified in the optimization document.
    """
    
    def __init__(self, total_epochs: int = 128):
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def get_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Get dynamic loss weights for current epoch.
        
        Args:
            epoch (int): Current training epoch
            
        Returns:
            Dict[str, float]: Loss weights for different components
        """
        self.current_epoch = epoch
        progress = epoch / self.total_epochs
        
        if progress < 0.3:  # Stage 0: Early stage (0-30%)
            return {
                'semantic': 0.3,     # Build basic semantic understanding
                'instance': 0.4,     # Learn instance boundaries
                'clip_align': 0.2,   # Start 2D-3D alignment
                'auxiliary': 0.1     # Light auxiliary supervision
            }
        elif progress < 0.7:  # Stage 1: Main training (30-70%)
            return {
                'semantic': 0.4,     # Strengthen semantic supervision
                'instance': 0.5,     # Focus on instance segmentation
                'clip_align': 0.08,  # Stable feature alignment
                'auxiliary': 0.02    # Reduce auxiliary interference
            }
        else:  # Stage 2: Fine-tuning (70-100%)
            return {
                'semantic': 0.45,    # Maintain semantic accuracy
                'instance': 0.5,     # Focus on instance quality
                'clip_align': 0.04,  # Minimize alignment loss
                'auxiliary': 0.01    # Minimal auxiliary supervision
            }
    
    def get_current_stage(self, epoch: int) -> str:
        """Get current training stage name."""
        progress = epoch / self.total_epochs
        if progress < 0.3:
            return "S0_Foundation"
        elif progress < 0.7:
            return "S1_MainTraining"
        else:
            return "S2_FineTuning"
