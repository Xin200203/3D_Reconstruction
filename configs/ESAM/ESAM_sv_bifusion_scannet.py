"""ScanNet single-view baseline using BiFusionEncoder backbone.
Inherits ESAM_sv_scannet and only overrides the backbone and data_root.
"""
_base_ = ['ESAM_sv_scannet.py']

# import oneformer3d modules
custom_imports = dict(imports=['oneformer3d', 'oneformer3d.partial_load_hook'])

# -----------------------------------------------------------------------------
# 1. Backbone override – replace Res16UNet with BiFusionEncoder (ViT frozen)
# -----------------------------------------------------------------------------
model = dict(
    # 移除原 backbone，改用 bi_encoder
    backbone=dict(_delete_=True),
    bi_encoder=dict(
        type='BiFusionEncoder',
        clip_pretrained='',
        freeze_blocks=0,
        use_tiny_sa_2d=False,
        voxel_size=0.02),
    criterion=dict(
        inst_criterion=dict(
            bbox_loss=dict(_delete_=True)
        )
    )
)

# -----------------------------------------------------------------------------
# 2. Dataset root修改
# -----------------------------------------------------------------------------
train_dataloader = dict(dataset=dict(data_root='/home/nebula/xxy/ESAM/data/scannet200-sv', ann_file='scannet200_sv_oneformer3d_infos_train.pkl'))
val_dataloader = dict(dataset=dict(data_root='/home/nebula/xxy/ESAM/data/scannet200-sv', ann_file='scannet200_sv_oneformer3d_infos_val.pkl'))
test_dataloader = val_dataloader

# -----------------------------------------------------------------------------
# 3. 训练设置 (AMP 已在父配置中启用 AdamW/AMP，可按需覆写)
# -----------------------------------------------------------------------------
optim_wrapper = dict(type='AmpOptimWrapper')

# 输出目录
work_dir = './work_dirs/ESAM_sv_bifusion_scannet'

# 使用 Mask3D 预训练 3D U-Net 权重，仅加载 backbone3d
load_from = None  # 禁止父配置加载整个模型

custom_hooks = [
    dict(type='PartialLoadHook',
         pretrained='work_dirs/tmp/mask3d_scannet200.pth',
         submodule='bi_encoder.backbone3d'
    )
] 