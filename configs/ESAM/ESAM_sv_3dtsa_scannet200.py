# 3D + Tiny-SA（ScanNet200-SV）

_base_ = ['./ESAM_sv_3d_scannet200.py']

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
        custom_keys={'pos_embed': dict(lr_mult=0.1)}
    )
) 