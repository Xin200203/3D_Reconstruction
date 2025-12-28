# Enhanced Bi-Fusion（Category-Agnostic，ScanNet200-SV）- 3DMV（Phase B-S2：联合融合）

_base_ = ['./sv_bifusion_scannet200_3dmv_phaseB.py']

# 训练从 S1 的最佳模型继续，建议使用 --resume 传入 S1 的 best.pth
resume = False

# 调整 BiFusion 细节：适度正则，保持 128 接口
model = dict(
    bi_encoder=dict(
        conv3d_dropout=0.05  # S2：适度 dropout 稳定训练
    ),
    # S2（温和版）：放缓接口迁移，扩大 2D 有效监督覆盖
    two_d_losses=dict(
        # 接口门控
        alpha_max=0.8,          # 先到 0.8，稳定后再拉满到 1.0
        alpha_warmup=5000,
        # 权重（对齐项先保持 0.01，稳定后再提到 0.02）
        w_recon=0.01,
        w_seg=0.01,
        w_align=0.01,
        recon_warmup=5000,
        seg_warmup=5000,
        align_warmup=5000,
        # 覆盖阈值（扩大监督像素）
        recon_tau=0.3,
    ),
)

# S2：解冻 decoder 与 3D 分支融合层；backbone3d 用小 LR 微调
optim_wrapper = dict(
    # 临时降 LR 让适配更平滑；2～3 次验证后可回到 8e-5
    optimizer=dict(type='AdamW', lr=6e-05, weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            # backbone3d 小学习率微调（解冻后 0.2×）
            'bi_encoder.backbone3d': dict(lr_mult=0.2, decay_mult=1.0),
            # decoder 加速适配接口分布（恢复到 1.0）
            'decoder': dict(lr_mult=1.0, decay_mult=1.0),
            # 融合适配头再降一些，降低接口继续被 2D 推动速度
            'bi_encoder.conv3d_fusion.features_fusion': dict(lr_mult=0.6, decay_mult=1.0),
        }
    ),
)
