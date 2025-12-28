# 3D + Tiny-SA（ScanNet200-SV）
# 在纯 3D baseline 基础上添加两层 TinySA neck。

_base_ = ['./ESAM_sv_3d_scannet.py']

custom_imports = dict(imports=['oneformer3d'])

model = dict(
    neck=dict(
        type='TinySANeck',
        dim=96,             # 与 backbone 输出一致
        num_heads=4,
        radius=0.3,
        max_k=32,
        sample_ratio=0.25,
        num_layers=2,
    )
)

# 排除 Tiny-SA 中的 pos_embed 参数放缩：
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(lr_mult=0.1),
        }
    )
) 