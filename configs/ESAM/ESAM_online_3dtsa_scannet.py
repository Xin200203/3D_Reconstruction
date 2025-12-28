# 3D + Tiny-SA Online（ScanNet200-MV）
# 在纯 3D Online baseline 上添加 TinySA neck。

_base_ = ['./ESAM_online_3d_scannet.py']

custom_imports = dict(imports=['oneformer3d'])

model = dict(
    neck=dict(
        type='TinySANeck',
        dim=96,
        num_heads=4,
        radius=0.3,
        max_k=32,
        sample_ratio=0.25,
        num_layers=2,
    )
)

optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(lr_mult=0.1),
        }
    )
) 