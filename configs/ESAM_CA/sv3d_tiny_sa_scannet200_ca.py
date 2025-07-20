"""Category-Agnostic TinySA (ScanNet200) – second stage

继承上一阶段纯 3D（sv3d_scannet200_ca.py），仅新增 TinySA 颈部；
所有类别仍视为 1（object）。
"""

_base_ = ['./sv3d_scannet200_ca.py']

custom_imports = dict(imports=['oneformer3d'])

# —— 模型新增 TinySANeck ————————————————
model = dict(
    neck=dict(
        type='TinySANeck',
        dim=96,
        num_heads=4,
        max_k=32,
        sample_ratio=0.05,
        radius=0.2,
        num_layers=2,
    )
)

# —— 优化器对 pos_embed 等设定小学习率 ——
optim_wrapper = dict(
    paramwise_cfg=dict(
        custom_keys={'pos_embed': dict(lr_mult=0.1)}
    )
)

# 继续训练
load_from = 'work_dirs/sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth' 