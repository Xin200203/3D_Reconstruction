import torch
import pytest

from oneformer3d.bi_fusion_encoder import BiFusionEncoder


def gen_dummy_data(batch_size=2, n_points=1024, img_size=128):
    points_list, imgs, cam_info = [], [], []
    for _ in range(batch_size):
        # xyz in [0.5, 5] meters, color random
        xyz = torch.rand(n_points, 3) * 5 + 0.5
        color = torch.rand(n_points, 3)
        points = torch.cat([xyz, color], dim=-1)
        points_list.append(points)

        img = torch.rand(3, img_size, img_size)
        imgs.append(img)

        intr = torch.tensor([img_size, img_size, img_size/2, img_size/2])
        extr = torch.eye(4)
        cam_info.append({'intrinsics': intr, 'extrinsics': extr})
    return points_list, imgs, cam_info


def test_bifusion_encoder_forward():
    encoder = BiFusionEncoder(clip_pretrained='', freeze_blocks=0)  # no pretrain to save time
    points, imgs, cam_info = gen_dummy_data()
    out = encoder(points, imgs, cam_info)
    assert len(out['feat_fusion']) == 2
    for fused, conf, pe in zip(out['feat_fusion'], out['conf_2d'], out['pe_xyz']):
        assert not torch.isnan(fused).any(), 'NaN in fused features'
        assert fused.shape[1] == 96, 'Fused feature dim mismatch'
        assert conf.shape[-1] == 1
        assert pe.shape[1] == 32 