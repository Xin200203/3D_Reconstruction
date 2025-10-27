# Enhanced Bi-Fusion（Category-Agnostic，ScanNet200-SV）- 3DMV（Phase B-S2：联合融合）

_base_ = ['./sv_bifusion_scannet200_3dmv_phaseB.py']

# 训练从 S1 的最佳模型继续，建议使用 --resume 传入 S1 的 best.pth
resume = False

# 调整 BiFusion 细节：适度正则，保持 128 接口
model = dict(
    bi_encoder=dict(
        conv3d_dropout=0.05  # S2：适度 dropout 稳定训练
    ),
    # S2：提升 2D 权重并将 alpha 拉满（带 warmup）
    two_d_losses=dict(
        alpha_max=1.0,
        alpha_warmup=3000,
        w_recon=0.01,
        w_seg=0.01,
        w_align=0.02,
        recon_warmup=3000,
        seg_warmup=3000,
        align_warmup=3000,
    ),
)

# S2：解冻 decoder 与 3D 分支融合层；backbone3d 用小 LR 微调
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=8e-05, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # backbone3d 以更小学习率微调（其余模块使用默认 LR）
            'bi_encoder.backbone3d': dict(lr_mult=0.2, decay_mult=1.0),
        }
    ),
)

