# Enhanced Bi-Fusion（Category-Agnostic，ScanNet200-SV）
# 完全独立的BiFusion配置，无TinySA依赖，维度匹配优化

_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]

custom_imports = dict(imports=[
    'oneformer3d', 
    'oneformer3d.partial_load_hook',
    'oneformer3d.detailed_loss_hook',
    'oneformer3d.enhanced_training_hook'  # 🔥 导入增强训练监控Hook
])

# ======== 类别和维度设置 ========
num_instance_classes = 1
num_semantic_classes = 200
num_instance_classes_eval = 1

# ======== 数据路径配置 ========
DATA_ROOT = '/home/nebula/xxy/ESAM/data/scannet200-sv/'

# ======== 颜色归一化参数 ========
color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

# ======== 类别名称 ========
class_names = [
    'wall', 'floor', 'chair', 'table', 'door', 'couch', 'cabinet', 'shelf',
    'desk', 'office chair', 'bed', 'pillow', 'sink', 'picture', 'window',
    'toilet', 'bookshelf', 'monitor', 'curtain', 'book', 'armchair',
    'coffee table', 'box', 'refrigerator', 'lamp', 'kitchen cabinet', 'towel',
    'clothes', 'tv', 'nightstand', 'counter', 'dresser', 'stool', 'cushion',
    'plant', 'ceiling', 'bathtub', 'end table', 'dining table', 'keyboard',
    'bag', 'backpack', 'toilet paper', 'printer', 'tv stand', 'whiteboard',
    'blanket', 'shower curtain', 'trash can', 'closet', 'stairs', 'microwave',
    'stove', 'shoe', 'computer tower', 'bottle', 'bin', 'ottoman', 'bench',
    'board', 'washing machine', 'mirror', 'copier', 'basket', 'sofa chair',
    'file cabinet', 'fan', 'laptop', 'shower', 'paper', 'person',
    'paper towel dispenser', 'oven', 'blinds', 'rack', 'plate', 'blackboard',
    'piano', 'suitcase', 'rail', 'radiator', 'recycling bin', 'container',
    'wardrobe', 'soap dispenser', 'telephone', 'bucket', 'clock', 'stand',
    'light', 'laundry basket', 'pipe', 'clothes dryer', 'guitar',
    'toilet paper holder', 'seat', 'speaker', 'column', 'bicycle', 'ladder',
    'bathroom stall', 'shower wall', 'cup', 'jacket', 'storage bin',
    'coffee maker', 'dishwasher', 'paper towel roll', 'machine', 'mat',
    'windowsill', 'bar', 'toaster', 'bulletin board', 'ironing board',
    'fireplace', 'soap dish', 'kitchen counter', 'doorframe',
    'toilet paper dispenser', 'mini fridge', 'fire extinguisher', 'ball',
    'hat', 'shower curtain rod', 'water cooler', 'paper cutter', 'tray',
    'shower door', 'pillar', 'ledge', 'toaster oven', 'mouse',
    'toilet seat cover dispenser', 'furniture', 'cart', 'storage container',
    'scale', 'tissue box', 'light switch', 'crate', 'power outlet',
    'decoration', 'sign', 'projector', 'closet door', 'vacuum cleaner',
    'candle', 'plunger', 'stuffed animal', 'headphones', 'dish rack',
    'broom', 'guitar case', 'range hood', 'dustpan', 'hair dryer',
    'water bottle', 'handicap bar', 'purse', 'vent', 'shower floor',
    'water pitcher', 'mailbox', 'bowl', 'paper bag', 'alarm clock',
    'music stand', 'projector screen', 'divider', 'laundry detergent',
    'bathroom counter', 'object', 'bathroom vanity', 'closet wall',
    'laundry hamper', 'bathroom stall door', 'ceiling light', 'trash bin',
    'dumbbell', 'stair rail', 'tube', 'bathroom cabinet', 'cd case',
    'closet rod', 'coffee kettle', 'structure', 'shower head',
    'keyboard piano', 'case of water bottles', 'coat rack',
    'storage organizer', 'folded chair', 'fire alarm', 'power strip',
    'calendar', 'poster', 'potted plant', 'luggage', 'mattress'
]

# ======== 数据管道 ========
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadClipFeature', data_root=DATA_ROOT),
    dict(type='LoadSingleImageFromFile'),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='SwapChairAndFloor'),
    dict(type='PointSegClassMapping'),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-3.14, 3.14],
        scale_ratio_range=[0.8, 1.2],
        translation_std=[0.1, 0.1, 0.1],
        shift_height=False),
    dict(
        type='NormalizePointsColor_',
        color_mean=color_mean,
        color_std=color_std,
        clamp_range=[-3.0, 3.0]),
    dict(
        type='AddSuperPointAnnotations',
        num_classes=num_semantic_classes,
        stuff_classes=[0, 1],
        merge_non_stuff_cls=False),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=0.02,
        p=0.2),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'imgs', 'cam_info', 'clip_pix', 'clip_global', 
            'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords'
        ])
]

test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='LoadClipFeature', data_root=DATA_ROOT),
    dict(type='LoadSingleImageFromFile'),
    dict(
        type='LoadAnnotations3D_',
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True),
    dict(type='SwapChairAndFloor'),
    dict(type='PointSegClassMapping'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='NormalizePointsColor_',
                color_mean=color_mean,
                color_std=color_std,
                clamp_range=[-3.0, 3.0]),
            dict(
                type='AddSuperPointAnnotations',
                num_classes=num_semantic_classes,
                stuff_classes=[0, 1],
                merge_non_stuff_cls=False),
        ]),
    dict(
        type='Pack3DDetInputs_', 
        keys=['points', 'imgs', 'cam_info', 'clip_pix', 'clip_global', 'sp_pts_mask'])
]

# ======== 模型配置 ========
model = dict(
    type='ScanNet200MixFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    voxel_size=0.02,
    num_classes=num_instance_classes_eval,
    query_thr=0.5,
    
    # 传统backbone配置（为了满足模型初始化要求，实际不使用）
    backbone=dict(
        type='Res16UNet34C',
        in_channels=3,
        out_channels=96,
        config=dict(
            dilations=[1, 1, 1, 1],
            conv1_kernel_size=5,
            bn_momentum=0.02)),
    
    # 使用BiFusionEncoder替代传统backbone+neck组合
    bi_encoder=dict(
        type='BiFusionEncoder',
        clip_pretrained='/home/nebula/xxy/ESAM/data/open_clip_pytorch_model.bin',
        voxel_size=0.02,
        
        # Enhanced CLIP配置
        clip_num_layers=6,
        freeze_clip_conv1=False,
        freeze_clip_early_layers=True,
        
        # Enhanced Gate配置 - 使用LiteFusionGate 
        use_enhanced_gate=False,
        use_spatial_attention=False,
        spatial_k=16,
        
        # 关键：移除TinySA依赖
        use_tiny_sa_2d=False,
        use_tiny_sa_3d=False,
        
        # 其他配置
        use_amp=True,
        freeze_blocks=0,
        
        # 调试输出控制
        debug=False,  # 设置为False以关闭详细调试输出
    ),
    
    # 几何感知池化
    pool=dict(type='GeoAwarePooling', channel_proj=256),
    
    # 查询解码器
    decoder=dict(
        type='ScanNetMixQueryDecoder',
        num_layers=3,
        share_attn_mlp=False, 
        share_mask_mlp=False,
        cross_attn_mode=["", "SP", "SP", "SP"], 
        mask_pred_mode=["SP", "SP", "P", "P"],
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=256,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False),
    
    # Enhanced损失函数配置
    criterion=dict(
        type='ScanNetMixedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.4),
        inst_criterion=dict(
            type='MixedInstanceCriterion',
            matcher=dict(
                type='SparseMatcher',
                costs=[
                    dict(type='QueryClassificationCost', weight=0.5),
                    dict(type='MaskBCECost', weight=1.0),
                    dict(type='MaskDiceCost', weight=1.0)],
                topk=1),
            bbox_loss=dict(type='AxisAlignedIoULoss'),
            loss_weight=[1.0, 0.8, 0.6, 0.3, 0.0],
            num_classes=num_instance_classes,
            non_object_weight=0.05,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),
    
    # CLIP一致性损失
    clip_criterion=dict(
        type='ClipConsCriterion',
        loss_weight=0.1,
        temperature=0.07,
        gradient_flow_ratio=0.05,
    ),
    
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=100,
        inst_score_thr=0.0,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1]))

# ======== 数据集配置 ========
dataset_type = 'ScanNet200SegDataset_'
data_prefix = dict(
    pts='points',
    pts_instance_mask='instance_mask',
    pts_semantic_mask='semantic_mask',
    sp_pts_mask='super_points')

train_dataloader = dict(
    batch_size=6,
    num_workers=8,
    persistent_workers=True,
    prefetch_factor=2,
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        ann_file='scannet200_sv_oneformer3d_infos_train_clip.pkl',
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline,
        ignore_index=num_semantic_classes,
        scene_idxs=None,
        test_mode=False))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        data_root=DATA_ROOT,
        ann_file='scannet200_sv_oneformer3d_infos_val_clip.pkl',
        data_prefix=data_prefix,
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))

test_dataloader = val_dataloader

# ======== 评估配置 ========
label2cat = {i: name for i, name in enumerate(class_names + ['unlabeled'])}
metric_meta = dict(
    label2cat=label2cat,
    ignore_index=[num_semantic_classes],
    classes=class_names + ['unlabeled'])

sem_mapping = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23,
    24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90,
    93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112,
    115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139,
    140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163, 165, 166, 168, 169,
    170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230,
    232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331,
    342, 356, 370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581, 609,
    748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185,
    1186, 1187, 1188, 1189, 1190, 1191
]
inst_mapping = sem_mapping[2:]

val_evaluator = dict(
    type='UnifiedSegMetric',
    stuff_class_inds=[0, 1], 
    thing_class_inds=list(range(2, num_semantic_classes)),
    min_num_points=1, 
    id_offset=2**16,
    sem_mapping=sem_mapping,
    inst_mapping=inst_mapping,
    metric_meta=metric_meta)
test_evaluator = val_evaluator

# ======== 训练配置 ========
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.01),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    accumulative_counts=4)

param_scheduler = dict(type='PolyLR', begin=0, end=128, power=0.9)

custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    
    # 增强训练监控Hook
    dict(
        type='EnhancedTrainingHook',
        log_interval=10,
        grad_monitor_interval=50,
        detailed_stats=True
    ),
    
    # 原有详细损失监控Hook
    dict(
        type='DetailedLossMonitorHook',
        log_interval=20,
        collect_grad_norm=True,
        collect_clip_stats=True
    ),
    
    # NaN检测Hook
    dict(
        type='NaNDetectionHook',
        check_interval=5
    ),
    
    # 加载3D预训练权重
    dict(
        type='PartialLoadHook',
        pretrained='/home/nebula/xxy/ESAM/work_dirs/sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth',
        submodule='bi_encoder.backbone3d',
        prefix_replace=('backbone\\.', 'bi_encoder.backbone3d.'),
        strict=False
    )
]

default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best=['all_ap_50%'],
        rule='greater'),
    logger=dict(
        type='LoggerHook', 
        interval=10,
        log_metric_by_epoch=False,
        out_suffix='.log'
    ))

# ======== 训练调度 ========
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=128, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# ======== 日志配置 - 禁用TensorBoard解决Python 3.8兼容性问题 ========
log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)

# 禁用TensorBoard相关组件
vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
