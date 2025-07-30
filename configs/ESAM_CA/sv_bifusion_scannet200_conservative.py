# 保守的BiFusion配置 - 大幅减少CLIP可训练参数
# 基于内存压力分析，只训练CLIP最后一层

# 继承原配置
_base_ = ['./sv_bifusion_scannet200.py']

# 更激进的CLIP冻结策略
model = dict(
    bi_encoder=dict(
        # 只使用前4层CLIP (而不是6层)
        clip_num_layers=4,
        
        # 更激进的冻结策略
        freeze_clip_conv1=True,         # 冻结conv1，进一步减少参数
        freeze_clip_early_layers=True,  # 冻结前3层
        
        # 这意味着只有layer 3是可训练的
        # 从21M可训练参数 → 7M可训练参数
        # AdamW状态内存从0.26GB → 0.085GB
    ),
    
    # 进一步降低CLIP损失权重，因为可训练参数少了
    clip_criterion=dict(
        type='ClipConsCriterion',
        loss_weight=0.002,  # 从0.005进一步降低
    ),
)

# 恢复稍高的batch_size，因为内存压力减少了
train_dataloader = dict(
    batch_size=6,           # 可以提升到6
    num_workers=12,
    prefetch_factor=4,
)

# 降低累积步数
optim_wrapper = dict(
    accumulative_counts=3,  # 6×3=18等效batch_size
)

# 配置说明
"""
保守策略的内存节省:
- CLIP可训练参数: 21M → 7M (-67%)
- AdamW状态内存: 0.26GB → 0.085GB (-67%)
- 总内存节省: ~0.2GB

预期效果:
- 内存压力大幅减少
- 训练更稳定，不容易OOM
- 性能可能略有下降，但仍应超过基线
- 适合硬件资源有限的情况

使用建议:
- 如果立即方案仍有OOM，使用此配置
- 训练稳定后可以考虑逐步解冻更多层
""" 