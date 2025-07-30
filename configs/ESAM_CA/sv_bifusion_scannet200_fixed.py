# 修复BiFusion架构的根本问题
# 优化坐标变换、特征融合、训练策略

_base_ = ['./sv_bifusion_scannet200.py']

# ============ 核心架构修复 ============

model = dict(
    bi_encoder=dict(
        # 1. 减少坐标变换误差
        use_direct_projection=True,      # 直接投影，避免双重变换
        coordinate_jitter=0.01,          # 添加坐标抖动，提高鲁棒性
        
        # 2. 改进特征采样策略
        sampling_strategy='adaptive',     # 自适应采样
        invalid_fill_strategy='nearest',  # 用最近邻填充而非零填充
        valid_threshold=0.3,             # 降低有效点阈值
        
        # 3. 优化融合机制
        use_enhanced_gate=True,
        gate_type='attention',           # 使用注意力机制而非简单gate
        fusion_strategy='adaptive',      # 自适应融合权重
        geometric_consistency=True,      # 增加几何一致性约束
        
        # 4. 降低CLIP依赖
        clip_guidance_weight=0.001,      # 进一步降低CLIP权重
        progressive_clip_weight=True,    # 渐进式CLIP权重衰减
        
        # 5. 稳定性增强
        feature_normalization=True,      # 特征标准化
        gradient_checkpoint=True,        # 梯度检查点
    ),
    
    # 修复损失函数配置
    clip_criterion=dict(
        type='ClipConsCriterion',
        loss_weight=0.001,               # 大幅降低
        progressive_decay=True,          # 训练过程中逐步衰减
        consistency_threshold=0.8,       # 只有高一致性才计算损失
    ),
)

# ============ 训练策略修复 ============

# 分组学习率：解决收敛速度不匹配
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        # 关键：分组学习率
        paramwise_cfg=dict(
            custom_keys={
                # 2D分支：预训练权重，需要小学习率
                'bi_encoder.enhanced_clip': dict(lr_mult=0.1),
                
                # 3D分支：需要正常学习率
                'bi_encoder.backbone3d': dict(lr_mult=1.0),
                'bi_encoder.backbone_adapter': dict(lr_mult=1.0),
                
                # 融合层：需要高学习率快速学习对齐
                'bi_encoder.fusion_gate': dict(lr_mult=2.0),
                'bi_encoder.lin2d_final': dict(lr_mult=1.5),
                'bi_encoder.lin3d_final': dict(lr_mult=1.5),
                
                # 坐标变换：需要稳定学习
                'bi_encoder.pe_mlp': dict(lr_mult=0.8),
            }
        )
    ),
    clip_grad=dict(max_norm=15, norm_type=2),  # 更严格的梯度剪裁
    accumulative_counts=2,
)

# 动态损失权重调度
custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    
    # 保持3D预训练权重加载
    dict(
        type='PartialLoadHook',
        pretrained='/home/nebula/xxy/ESAM/work_dirs/sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth',
        submodule='bi_encoder.backbone3d',
        prefix_replace=('backbone.', ''),
        strict=False
    ),
    
    # 新增：动态损失权重调度
    dict(
        type='DynamicLossWeightHook',
        clip_weight_schedule={
            0: 0.001,      # 初期很小，让3D充分学习
            20: 0.0005,    # 逐步减少
            40: 0.0001,    # 后期接近0
        },
        adjust_frequency=5,  # 每5个epoch调整一次
    ),
]

# 更保守的数据增强
train_pipeline = [
    # ... 保持原有pipeline，但降低ElasticTransform概率
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=0.02,
        p=0.1),  # 从0.2进一步降到0.1，减少噪声
    # ...
]

# ============ 监控与调试 ============

default_hooks = dict(
    logger=dict(
        type='LoggerHook', 
        interval=25,
        # 新增：详细监控
        log_metric_by_epoch=True,
        log_code_filename=True,
    ),
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=5,
        save_best=['all_ap_50%', 'all_ap_25%'],  # 监控多个指标
        rule='greater'
    ),
)

# ============ 验证策略 ============
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        # ... 与train相同的数据集配置，但test_mode=True
        test_mode=True
    )
)

# 每10个epoch验证一次，及时发现问题
train_cfg = dict(
    type='EpochBasedTrainLoop', 
    max_epochs=128, 
    val_interval=10  # 从128改为10，更频繁验证
)

"""
预期效果：
1. 坐标变换误差：-5-10% → -1-2%
2. 特征采样丢失：-10-15% → -3-5%  
3. 融合策略次优：-5-10% → -1-3%
4. CLIP损失干扰：-5-8% → -1-2%
5. 训练策略不当：-3-5% → 0%

总体预期：从-28-48%提升到-6-12%
目标性能：AP_0.5: 0.65-0.75 (vs 基线0.81)
"""
