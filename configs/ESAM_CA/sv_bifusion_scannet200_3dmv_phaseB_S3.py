# Enhanced Bi-Fusion（Category-Agnostic，ScanNet200-SV）- 3DMV（Phase B-S3：目标微调）

_base_ = ['./sv_bifusion_scannet200_3dmv_phaseB.py']

# 训练从 S2 的最佳模型继续，建议使用 --resume 传入 S2 的 best.pth
resume = False

# S3：回归 3D 目标为主，弱化 2D 约束；维持 128 接口
model = dict(
    bi_encoder=dict(
        conv3d_dropout=0.0  # S3：去除 dropout，便于精细收敛
    ),
    two_d_losses=dict(
        alpha_max=1.0,     # 可根据验证情况固定在 0.8~1.0
        alpha_warmup=3000,
        w_recon=0.005,
        w_seg=0.005,
        w_align=0.005,
        recon_warmup=3000,
        seg_warmup=3000,
        align_warmup=3000,
    ),
)

# S3：小学习率精调，全部解冻（去掉 paramwise_cfg 即可）
optim_wrapper = dict(
    optimizer=dict(type='AdamW', lr=5e-05, weight_decay=0.05),
)

