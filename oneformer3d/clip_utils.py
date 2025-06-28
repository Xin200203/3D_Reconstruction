import torch
import torch.nn.functional as F

__all__ = [
    'freeze_clip_except_last_blocks',
    'build_uv_index',
    'sample_img_feat',
    'build_uv_index_batch'
]

def freeze_clip_except_last_blocks(clip_visual, num_train_blocks: int = 2):
    """冻结除最后 *num_train_blocks* 之外的 CLIP Visual Blocks 参数。"""
    total = len(clip_visual.blocks)
    train_ids = {str(total - i - 1) for i in range(num_train_blocks)}
    for n, p in clip_visual.named_parameters():
        parts = n.split('.')
        requires = len(parts) > 1 and parts[0] == 'blocks' and parts[1] in train_ids
        p.requires_grad = requires


def build_uv_index(xyz_cam: torch.Tensor, intrinsics: torch.Tensor, img_shape):
    """将相机坐标系下点投影到像素平面，返回 valid mask 与 uv 坐标 (N,2)。

    Args:
        xyz_cam (Tensor): (N,3)
        intrinsics (Tensor): (4,) [fx,fy,cx,cy]
        img_shape (tuple[int,int]): (H,W)
    Returns:
        valid (BoolTensor): (N,)
        uv (Tensor): (N,2) 像素坐标 (float)
    """
    fx, fy, cx, cy = intrinsics
    x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    u = fx * x / z + cx
    v = fy * y / z + cy
    H, W = img_shape
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    return valid, torch.stack([u, v], dim=-1)


def sample_img_feat(feat_map: torch.Tensor, uv: torch.Tensor):
    """在特征图上双线性采样。支持梯度回传。

    Args:
        feat_map (Tensor): (1,C,H,W)
        uv (Tensor): (N,2) 像素坐标
    Returns:
        sampled (Tensor): (N,C)
    """
    H, W = feat_map.shape[-2:]
    uv_norm = uv.clone()
    uv_norm[:, 0] = uv[:, 0] / (W - 1) * 2 - 1
    uv_norm[:, 1] = uv[:, 1] / (H - 1) * 2 - 1
    grid = uv_norm.unsqueeze(0).unsqueeze(2)  # (1,N,1,2)
    sampled = F.grid_sample(feat_map, grid, align_corners=True).squeeze(3).squeeze(0).T  # (N,C)
    return sampled


def build_uv_index_batch(xyz_cam: torch.Tensor, intrinsics: torch.Tensor, img_shape):
    """批量版投影.
    Args:
        xyz_cam: (B,N,3) 或 (N,3) Tensor
        intrinsics: (B,4) 或 (4,) [fx,fy,cx,cy]
        img_shape: (H,W)
    Returns:
        valid: BoolTensor 同 xyz 前两维 (B,N)
        uv: (B,N,2) 浮点像素坐标
    """
    if xyz_cam.dim() == 2:
        xyz_cam = xyz_cam.unsqueeze(0)  # (1,N,3)
    B, N, _ = xyz_cam.shape
    if intrinsics.dim() == 1:
        intrinsics = intrinsics.unsqueeze(0).expand(B, 4)
    fx, fy, cx, cy = intrinsics.T  # each (B,)
    x, y, z = xyz_cam[..., 0], xyz_cam[..., 1], xyz_cam[..., 2]
    u = fx.view(B,1)*x/z + cx.view(B,1)
    v = fy.view(B,1)*y/z + cy.view(B,1)
    H, W = img_shape
    valid = (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z > 0)
    uv = torch.stack([u, v], dim=-1)  # (B,N,2)
    return valid, uv 