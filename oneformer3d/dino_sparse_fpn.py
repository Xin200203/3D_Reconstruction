"""
基于 MinkowskiEngine 的 DINO 多尺度 FPN 构建工具。

用途：
- 已有点级 2D 特征（如投影得到的 DINO 特征）和对应坐标，可快速构建 stride=1/2/4/8/16 的稀疏金字塔。
- 不依赖具体模型，外部可选择合适的 1×1 投影层将各层通道对齐到 decoder 所需维度。

示例：
    # coords: (N, 1+3)，第一列为 batch 索引，后三列为网格坐标
    # feats:  (N, C_dino)
    fpn = build_sparse_fpn(coords, feats, tensor_stride=1, dimension=3)
    x_s1, x_s2, x_s4, x_s8, x_s16 = fpn
"""

from typing import List, Optional

import MinkowskiEngine as ME
import torch


def build_sparse_fpn(
    coords: torch.Tensor,
    feats: torch.Tensor,
    tensor_stride: int = 1,
    dimension: int = 3,
    coordinate_manager: Optional[ME.CoordinateManager] = None,
    pool: Optional[ME.MinkowskiMaxPooling] = None,
) -> List[ME.SparseTensor]:
    """
    根据输入坐标与特征构建稀疏 FPN（金字塔），返回 stride 1/2/4/8/16 五个尺度。

    Args:
        coords: (N, D+1) 整型坐标，首列为 batch 索引，后续为各维网格坐标。
        feats:  (N, C) 特征向量（如 DINO 点级特征）。
        tensor_stride: 输入的初始 stride，一般为 1。
        dimension: 维度，默认 3。
        coordinate_manager: 可选，共享的 CoordinateManager。
        pool: 可选，自定义 MinkowskiMaxPooling，否则默认 kernel=2, stride=2。

    Returns:
        List[ME.SparseTensor]: [s1, s2, s4, s8, s16]
    """
    if pool is None:
        pool = ME.MinkowskiMaxPooling(kernel_size=2, stride=2, dimension=dimension)

    # MinkowskiEngine 的池化目前仅在 float32/double 上有完整实现，
    # 为避免 half 精度导致的 'local_pooling_forward_gpu' 报错，这里统一转为 float32。
    feats = feats.float()
    coords = coords.to(dtype=torch.int32)

    x_s1 = ME.SparseTensor(
        features=feats,
        coordinates=coords,
        tensor_stride=tensor_stride,
        coordinate_manager=coordinate_manager,
    )
    x_s2 = pool(x_s1)
    x_s4 = pool(x_s2)
    x_s8 = pool(x_s4)
    x_s16 = pool(x_s8)
    return [x_s1, x_s2, x_s4, x_s8, x_s16]
