# 优化的BiFusion训练策略配置
# 基于训练日志分析和多模态学习理论

# 继承原配置
_base_ = ['./sv_bifusion_scannet200.py']

# ======== 优化的训练配置 ========

# 1. 分阶段优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', 
        lr=0.0001,  # 基础学习率恢复到0.0001
        weight_decay=0.05,
        # 分组学习率 - BiFusion的核心优化
        paramwise_cfg=dict(
            custom_keys={
                # CLIP分支：较低学习率，避免破坏预训练特征
                'bi_encoder.enhanced_clip.clip_visual': dict(lr_mult=0.1),
                'bi_encoder.enhanced_clip.spatial_proj': dict(lr_mult=0.5),
                
                # 3D分支：正常学习率，充分利用预训练权重
                'bi_encoder.backbone3d': dict(lr_mult=1.0),
                'bi_encoder.backbone_adapter': dict(lr_mult=1.0),
                
                # 融合层：更高学习率，快速学习2D-3D对齐
                'bi_encoder.fusion_gate': dict(lr_mult=2.0),
                'bi_encoder.lin2d_final': dict(lr_mult=1.5),
                'bi_encoder.lin3d_final': dict(lr_mult=1.5),
                
                # 几何编码：适中学习率
                'bi_encoder.pe_mlp': dict(lr_mult=1.0),
            }
        )
    ),
    clip_grad=dict(max_norm=20, norm_type=2),  # 提高梯度剪裁阈值，应对BiFusion的大梯度
    accumulative_counts=2  # 降低累积步数，增加更新频率
)

# 2. 更精细的损失权重调整
model = dict(
    criterion=dict(
        # 语义损失权重稍微降低
        sem_criterion=dict(loss_weight=0.4),  # 从0.5降到0.4
        
        # 实例损失保持，但调整内部权重
        inst_criterion=dict(
            # [cls, bce, dice, score, bbox]
            loss_weight=[0.3, 1.0, 1.2, 0.4, 0.3],  # 增强dice损失，降低分类损失
            non_object_weight=0.05,  # 降低背景权重，突出前景
        )
    ),
    
    # 3. 渐进式CLIP损失策略
    clip_criterion=dict(
        type='ClipConsCriterion',
        loss_weight=0.005,  # 初期进一步降低，让3D分支充分学习
    ),
)

# 4. 更平滑的学习率调度
param_scheduler = [
    # Warmup阶段：让各分支逐步对齐
    dict(
        type='LinearLR',
        start_factor=0.1,
        by_epoch=False,
        begin=0,
        end=1000,  # 前1000步warmup
    ),
    # 主训练阶段：余弦退火
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=1,
        end=128,
        T_max=127,
        eta_min=1e-6,
    )
]

# 5. 数据加载优化
train_dataloader = dict(
    batch_size=8,     # 恢复到8，平衡显存和收敛速度
    num_workers=10,   # 减少worker，降低内存压力
    persistent_workers=True,
    prefetch_factor=3,  # 减少预取，避免内存爆炸
)

# 6. 训练监控和早停
default_hooks = dict(
    checkpoint=dict(
        interval=5,  # 更频繁保存，防止训练中断
        max_keep_ckpts=3,
        save_best=['all_ap_50%'],
        rule='greater'
    ),
    logger=dict(interval=25),  # 更频繁的日志记录，便于调试
)

# 7. 添加自定义hooks用于动态调整
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    # 保持原有的权重加载
    dict(
        type='PartialLoadHook',
        pretrained='/home/nebula/xxy/ESAM/work_dirs/sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth',
        submodule='bi_encoder.backbone3d',
        prefix_replace=('backbone.', ''),
        strict=False
    ),
    # 添加损失监控hook（如果有的话）
    # dict(
    #     type='LossWeightScheduleHook',
    #     clip_weight_schedule={0: 0.005, 20: 0.01, 50: 0.02}
    # )
]

# ======== 训练建议 ========
"""
预期训练表现：
1. 前10个epoch：损失较高（8-12），主要学习3D-2D对齐
2. 10-30个epoch：损失稳定下降（6-8），特征融合逐步优化  
3. 30+个epoch：损失平稳（4-6），接近基线性能

关键监控指标：
- grad_norm应该在10-20之间波动，不应超过25
- inst_loss和seg_loss比例应该保持在4:1左右
- loss_clip应该在0.005-0.02之间

如果出现问题：
- grad_norm过大(>30)：降低学习率或增加梯度剪裁
- 损失不下降：检查数据loading和权重加载
- 显存OOM：降低batch_size或prefetch_factor
""" 