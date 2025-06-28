import torch
import pytest

from oneformer3d.bi_fusion_encoder import BiFusionEncoder


def _gen_dummy_batch(batch_size=4, n_points=2048, img_size_pair=(128, 160), same_hw=True, device='cpu'):
    """生成一批随机点云 / 图像 / 相机内参数据。

    Args:
        batch_size (int): 批大小。
        n_points (int): 每帧点数。
        img_size_pair (Tuple[int,int]): 两种分辨率，用于测试异尺寸路径。
        same_hw (bool): 若 True 则所有图像尺寸相同；否则交替尺寸。
        device (str): cpu / cuda
    Returns:
        points_list, imgs, cam_info
    """
    size_a, size_b = img_size_pair
    points_list, imgs, cam_info = [], [], []
    for i in range(batch_size):
        size = size_a if (same_hw or i % 2 == 0) else size_b
        xyz = torch.rand(n_points, 3, device=device) * 5 + 0.5
        color = torch.rand(n_points, 3, device=device)
        points_list.append(torch.cat([xyz, color], dim=-1))

        imgs.append(torch.rand(3, size, size, device=device))
        intr = torch.tensor([size, size, size / 2, size / 2], device=device)
        cam_info.append({'intrinsics': intr, 'extrinsics': torch.eye(4, device=device)})
    return points_list, imgs, cam_info


@pytest.mark.parametrize('use_tiny_sa', [False, True])
@pytest.mark.parametrize('same_hw', [True, False])
@pytest.mark.parametrize('device', ['cuda' if torch.cuda.is_available() else 'cpu'])
def test_bifusion_encoder_batch(use_tiny_sa, same_hw, device):
    """验证 BiFusionEncoder 在批量 CLIP 路径及 fallback 路径下均工作正常。"""
    torch.manual_seed(0)
    pts, imgs, cam_info = _gen_dummy_batch(same_hw=same_hw, device=device)

    encoder = BiFusionEncoder(
        clip_pretrained='',  # empty string -> 随机初始化，避免下载权重
        freeze_blocks=0,
        use_tiny_sa_2d=use_tiny_sa
    ).to(device)

    out = encoder(pts, imgs, cam_info)

    # === 基本输出检查 ===
    B = len(pts)
    for fused, conf, pe in zip(out['feat_fusion'], out['conf_2d'], out['pe_xyz']):
        assert fused.shape[0] == pts[0].shape[0], '点数不一致'
        assert fused.shape[1] == 96, '融合维度应为 96'
        assert conf.shape == (pts[0].shape[0], 1)
        assert pe.shape[1] == 32
        assert not torch.isnan(fused).any(), '检测到 NaN'

    # === 梯度检查（随机取一层参数） ===
    loss = sum(f.sum() for f in out['feat_fusion'])
    loss.backward()
    grad_param = encoder.lin3d[0].weight.grad
    assert grad_param is not None and torch.any(grad_param != 0), '梯度未正确回传' 