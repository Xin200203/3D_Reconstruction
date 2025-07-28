import torch
import torch.nn.functional as F

__all__ = [
    'freeze_clip_except_last_blocks',
    'build_uv_index',
    'sample_img_feat',
    'build_uv_index_batch',
    'build_world_xyz_batch'
]

def freeze_clip_except_last_blocks(clip_visual, num_train_blocks: int = 2):
    """冻结除最后 *num_train_blocks* 之外的 VisionTransformer 块参数。

    open_clip <=2.21: `clip_visual.blocks` 为 nn.Sequential。
    open_clip  >=2.22: 模型结构改为 `clip_visual.transformer.resblocks`。
    这里兼容两种情形；若未来再改，只需在 `block_prefixes` 中追加即可。
    """
    # 1. 获取 blocks 序列（nn.Sequential 或 list-like）
    blocks = None
    if hasattr(clip_visual, 'blocks'):
        blocks = getattr(clip_visual, 'blocks')
        block_attr_name = 'blocks'
    elif hasattr(clip_visual, 'transformer') and hasattr(clip_visual.transformer, 'resblocks'):
        blocks = getattr(clip_visual.transformer, 'resblocks')
        block_attr_name = 'transformer.resblocks'
    else:
        # fallback: 通过参数名推断
        blocks = None
        block_attr_name = None

    if blocks is not None:
        total = len(blocks)
        train_ids = {str(total - i - 1) for i in range(num_train_blocks)}
    else:
        # attempt to infer total blocks from parameter names (last resort)
        candidates = []
        for n, _ in clip_visual.named_parameters():
            if n.startswith('blocks.'):
                candidates.append(int(n.split('.')[1]))
            elif n.startswith('transformer.resblocks.'):
                candidates.append(int(n.split('.')[2]))
        if not candidates:
            raise AttributeError('Cannot locate transformer blocks in CLIP visual model')
        total = max(candidates) + 1
        train_ids = {str(total - i - 1) for i in range(num_train_blocks)}

    # 2. 设置 requires_grad
    for n, p in clip_visual.named_parameters():
        parts = n.split('.')
        requires = False
        if parts[0] == 'blocks' and len(parts) > 1 and parts[1] in train_ids:
            requires = True
        elif parts[0] == 'transformer' and len(parts) > 2 and parts[1] == 'resblocks' and parts[2] in train_ids:
            requires = True
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


def sample_img_feat(feat_map: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
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
    
    # 确保feat_map和grid有相同的数据类型，处理AMP混合精度
    if feat_map.dtype != grid.dtype:
        grid = grid.to(feat_map.dtype)
    
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


def build_world_xyz_batch(xyz_cam: torch.Tensor, extrinsics: torch.Tensor):
    """批量相机坐标 → 世界坐标
    Args:
        xyz_cam: (B,N,3) or (N,3) 相机系坐标
        extrinsics: (B,4,4) or (4,4) 变换矩阵 (cam→world)
    Returns:
        xyz_world: Tensor 同 shape, 世界坐标
    """
    if xyz_cam.dim() == 2:
        xyz_cam = xyz_cam.unsqueeze(0)
    B, N, _ = xyz_cam.shape
    if extrinsics.dim() == 2:
        extrinsics = extrinsics.unsqueeze(0).expand(B, 4, 4)
    # 齐次坐标
    pad = torch.ones((B, N, 1), dtype=xyz_cam.dtype, device=xyz_cam.device)
    xyz_h = torch.cat([xyz_cam, pad], dim=-1)  # (B,N,4)
    xyz_world = (extrinsics @ xyz_h.unsqueeze(-1)).squeeze(-1)[:, :, :3]  # (B,N,3)
    return xyz_world if xyz_cam.dim()==3 else xyz_world[0] 