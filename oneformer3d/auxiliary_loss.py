import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS


@MODELS.register_module()
class SpatialConsistencyLoss(nn.Module):
    """Spatial consistency loss for neighboring points feature similarity.
    
    Args:
        loss_weight (float): Weight for this loss component.
        k_neighbors (int): Number of neighbors for KNN search.
    """
    
    def __init__(self, loss_weight: float = 0.02, k_neighbors: int = 8):
        super().__init__()
        self.loss_weight = loss_weight
        self.k_neighbors = k_neighbors
    
    def forward(self, feat_3d_list, spatial_coords_list):
        """Calculate spatial consistency loss.
        
        Args:
            feat_3d_list (List[Tensor]): 3D features, each scene (N_i, 256)
            spatial_coords_list (List[Tensor]): 3D coordinates, each scene (N_i, 3)
            
        Returns:
            Tensor: Spatial consistency loss
        """
        if not isinstance(feat_3d_list, (list, tuple)):
            feat_3d_list = [feat_3d_list]
        if not isinstance(spatial_coords_list, (list, tuple)):
            spatial_coords_list = [spatial_coords_list]
            
        total_loss = 0.0
        total_points = 0
        
        for feat_3d, spatial_coords in zip(feat_3d_list, spatial_coords_list):
            if feat_3d.size(0) <= self.k_neighbors:
                continue  # Skip if too few points
                
            # KNN search
            dist_matrix = torch.cdist(spatial_coords, spatial_coords)  # (N, N)
            _, knn_indices = torch.topk(dist_matrix, self.k_neighbors + 1, 
                                       dim=-1, largest=False)  # (N, k+1)
            knn_indices = knn_indices[:, 1:]  # Exclude self (N, k)
            
            # Neighbor features
            neighbor_feat = feat_3d[knn_indices.flatten()].view(
                feat_3d.size(0), self.k_neighbors, -1)  # (N, k, 256)
            
            # Calculate cosine similarity with neighbors
            feat_norm = F.normalize(feat_3d, dim=-1)  # (N, 256)
            neighbor_norm = F.normalize(neighbor_feat, dim=-1)  # (N, k, 256)
            
            cos_sim = torch.sum(
                feat_norm.unsqueeze(1) * neighbor_norm, dim=-1)  # (N, k)
            
            # Consistency loss: neighbors should be similar
            consistency_loss = (1 - cos_sim).mean()
            
            total_loss += consistency_loss * feat_3d.size(0)
            total_points += feat_3d.size(0)
        
        if total_points == 0:
            return torch.tensor(0.0, device=feat_3d_list[0].device, requires_grad=True)
            
        avg_loss = total_loss / total_points
        return self.loss_weight * avg_loss


@MODELS.register_module()
class NoViewSupervisionLoss(nn.Module):
    """Pseudo-supervision loss for points without view coverage.
    
    Args:
        loss_weight (float): Weight for this loss component.
        confidence_threshold (float): Confidence threshold for pseudo labels.
    """
    
    def __init__(self, loss_weight: float = 0.01, confidence_threshold: float = 0.8):
        super().__init__()
        self.loss_weight = loss_weight
        self.confidence_threshold = confidence_threshold
    
    def forward(self, feat_3d_list, valid_mask_list):
        """Calculate no-view supervision loss.
        
        Args:
            feat_3d_list (List[Tensor]): 3D features, each scene (N_i, 256)
            valid_mask_list (List[Tensor]): Valid projection masks, each scene (N_i,)
            
        Returns:
            Tensor: No-view supervision loss
        """
        if not isinstance(feat_3d_list, (list, tuple)):
            feat_3d_list = [feat_3d_list]
        if not isinstance(valid_mask_list, (list, tuple)):
            valid_mask_list = [valid_mask_list]
            
        total_loss = 0.0
        total_points = 0
        
        for feat_3d, valid_mask in zip(feat_3d_list, valid_mask_list):
            valid_feat = feat_3d[valid_mask]  # Points with view coverage
            invalid_feat = feat_3d[~valid_mask]  # Points without view coverage
            
            if invalid_feat.size(0) == 0 or valid_feat.size(0) == 0:
                continue
            
            # Calculate similarity between no-view and view points
            similarity = torch.mm(
                F.normalize(invalid_feat, dim=-1),
                F.normalize(valid_feat, dim=-1).t()
            )  # (N_invalid, N_valid)
            
            # Select high-confidence pseudo labels
            max_sim, _ = similarity.max(dim=-1)
            confident_mask = max_sim > self.confidence_threshold
            
            if confident_mask.sum() == 0:
                continue
            
            # Pseudo label loss: no-view points should align with most similar view points
            pseudo_loss = (1 - max_sim[confident_mask]).mean()
            
            total_loss += pseudo_loss * invalid_feat.size(0)
            total_points += invalid_feat.size(0)
        
        if total_points == 0:
            return torch.tensor(0.0, device=feat_3d_list[0].device, requires_grad=True)
            
        avg_loss = total_loss / total_points
        return self.loss_weight * avg_loss
