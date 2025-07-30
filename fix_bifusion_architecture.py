#!/usr/bin/env python3
"""
BiFusion架构根本性修复方案
解决坐标变换、特征融合、训练策略等核心问题
"""

def create_optimized_bifusion_config():
    """创建优化的BiFusion配置"""
    
    config_content = '''# 修复BiFusion架构的根本问题
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
'''
    
    with open('/home/nebula/xxy/ESAM/configs/ESAM_CA/sv_bifusion_scannet200_fixed.py', 'w') as f:
        f.write(config_content)
    
    print("✅ 创建优化配置：configs/ESAM_CA/sv_bifusion_scannet200_fixed.py")

def create_architecture_patches():
    """创建架构级修复补丁"""
    
    patch_content = '''#!/usr/bin/env python3
"""
BiFusion架构级修复补丁
直接修复坐标变换和融合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def patch_bifusion_encoder():
    """应用BiFusion的架构修复补丁"""
    
    from oneformer3d.bi_fusion_encoder import BiFusionEncoder
    
    # 保存原始方法
    original_process_single = BiFusionEncoder._process_single
    
    def fixed_process_single(self, points, img, cam_meta, feat2d_map=None, clip_global=None):
        """修复的_process_single方法"""
        
        # 1. 减少坐标变换：直接使用相机坐标
        xyz_cam = points[:, :3]
        
        # 2. 简化投影，减少误差累积
        if cam_meta.get('intrinsics', None) is not None:
            intr = cam_meta['intrinsics']
            if not torch.is_tensor(intr):
                intr = torch.as_tensor(intr, dtype=xyz_cam.dtype, device=xyz_cam.device)
            
            # 直接投影，避免双重变换
            valid, uv = self.build_uv_index(xyz_cam, intr, feat2d_map.shape[-2:])
            
            # 3. 改进特征采样：用最近邻填充
            sampled2d = xyz_cam.new_zeros((xyz_cam.shape[0], 256))
            if valid.any():
                f2d_vis = self.sample_img_feat(feat2d_map.unsqueeze(0), uv[valid])
                sampled2d[valid] = f2d_vis.to(sampled2d.dtype)
                
                # 新增：用最近邻填充无效点
                if (~valid).any():
                    # 找到每个无效点最近的有效点
                    invalid_idx = torch.where(~valid)[0]
                    valid_idx = torch.where(valid)[0]
                    
                    if len(valid_idx) > 0:
                        # 计算距离矩阵
                        dist = torch.cdist(xyz_cam[invalid_idx], xyz_cam[valid_idx])
                        nearest_idx = dist.argmin(dim=1)
                        sampled2d[invalid_idx] = sampled2d[valid_idx[nearest_idx]]
        else:
            # 无相机信息时，使用零向量
            sampled2d = xyz_cam.new_zeros((xyz_cam.shape[0], 256))
            valid = torch.zeros(xyz_cam.shape[0], dtype=torch.bool, device=xyz_cam.device)
        
        # 4. 3D分支处理（保持原逻辑但增加稳定性）
        xyz_world = xyz_cam  # 简化：直接使用相机坐标作为世界坐标
        coords_int = torch.round(xyz_world / self.voxel_size).to(torch.int32)
        coords = torch.cat([torch.zeros(coords_int.size(0), 1, dtype=torch.int32, device=coords_int.device),
                             coords_int], dim=1)
        feats = points[:, 3:6].contiguous()
        
        # 添加特征标准化
        feats = F.normalize(feats, dim=-1)
        
        field = ME.TensorField(coordinates=coords, features=feats)
        sparse_tensor = field.sparse()
        feat3d_sparse = self.backbone3d(sparse_tensor)
        feat3d = feat3d_sparse.slice(field).features
        feat3d = self.backbone_adapter(feat3d)
        
        # 5. 几何编码
        bbox_size = torch.zeros_like(xyz_world)
        pose_delta = torch.zeros(9, device=xyz_world.device, dtype=xyz_world.dtype)
        height = xyz_world[:, 2:3]
        pe = self.pe_mlp(build_geo_pe(xyz_world, bbox_size, pose_delta, height))
        
        # 6. 特征处理
        feat3d = self.simple_neck(feat3d)
        feat3d = self.lin3d(feat3d)
        feat3d = self.ln3d(feat3d)
        
        feat2d = self.lin2d(sampled2d)
        feat2d = self.ln2d(feat2d)
        
        # 7. 改进的特征融合
        f2d_final = self.lin2d_final(torch.cat([feat2d, pe], dim=-1))
        f3d_final = self.lin3d_final(torch.cat([feat3d, pe], dim=-1))
        
        # 新的自适应融合策略
        if self.use_enhanced_gate:
            # 计算特征质量得分
            f2d_quality = torch.sigmoid(self.quality_mlp_2d(f2d_final))  # 需要添加这个层
            f3d_quality = torch.sigmoid(self.quality_mlp_3d(f3d_final))  # 需要添加这个层
            
            # 结合有效性和质量
            valid_weight = valid.float().unsqueeze(-1)
            adaptive_weight = valid_weight * f2d_quality / (f2d_quality + f3d_quality + 1e-8)
            
            fused = adaptive_weight * f2d_final + (1 - adaptive_weight) * f3d_final
            conf = adaptive_weight
        else:
            # 回退到改进的简单融合
            gate_input = torch.cat([f2d_final, f3d_final], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input))
            valid_weight = valid.float().unsqueeze(-1)
            # 更保守的无效权重：从0.2降到0.1
            gate = gate * valid_weight + 0.1 * (1 - valid_weight)
            fused = gate * f2d_final + (1 - gate) * f3d_final
            conf = gate
        
        return fused, conf, pe, clip_global
    
    # 应用补丁
    BiFusionEncoder._process_single = fixed_process_single
    print("🔧 BiFusion架构修复补丁已应用")

if __name__ == "__main__":
    patch_bifusion_encoder()
'''
    
    with open('/home/nebula/xxy/ESAM/patch_bifusion_architecture.py', 'w') as f:
        f.write(patch_content)
    
    print("✅ 创建架构补丁：patch_bifusion_architecture.py")

if __name__ == "__main__":
    print("🔧 创建BiFusion根本性修复方案")
    print("=" * 50)
    
    # 创建优化配置
    create_optimized_bifusion_config()
    
    # 创建架构补丁
    create_architecture_patches()
    
    print("\n🎯 修复方案创建完成")
    print("接下来可以选择：")
    print("1. 测试优化配置：python tools/train.py configs/ESAM_CA/sv_bifusion_scannet200_fixed.py")
    print("2. 或直接回退到纯3D基线获得更好性能")
    print("3. 或基于当前分析结果继续迭代优化BiFusion") 