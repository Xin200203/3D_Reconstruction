# 纯 3D baseline（ScanNet200-SV）
# 仅保留 3D Backbone，无 Tiny-SA / Bi-Fusion。
# 继承官方 CA 配置，方便与原实验对齐。

_base_ = ['../ESAM_CA/ESAM_sv_scannet200_CA.py']

# 无额外修改；如需调整 batch_size / work_dir 等，请在命令行指定。

# === 预训练权重（仅加载 backbone 和 decoder） ===
model = dict(
    backbone=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/tmp/ESAM-E_online_epoch_128.pth',  # 请替换为实际路径
            prefix='backbone'
        )
    ),
    decoder=dict(
        init_cfg=dict(
            type='Pretrained',
            checkpoint='work_dirs/tmp/ESAM-E_online_epoch_128.pth',  # 同上
            prefix='decoder'
        )
    )
)

# 当维度不匹配时跳过
load_pretrained_strict = False 