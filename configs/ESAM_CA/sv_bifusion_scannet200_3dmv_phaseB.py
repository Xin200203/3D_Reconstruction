# Enhanced Bi-Fusion（Category-Agnostic，ScanNet200-SV）- 3DMV架构版本（Phase B：接入2D）
# 目的：
# - 在Phase A稳住3D表征的基础上，接入2D语义（带warmup）
# - 仅保留与位姿一致的刚体增强，避免投影错配

_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]

custom_imports = dict(imports=[
    'oneformer3d', 
    'oneformer3d.bi_fusion_encoder_3dmv',
    'oneformer3d.partial_load_hook',
    'oneformer3d.detailed_loss_hook',
    'oneformer3d.enhanced_training_hook'
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

# ======== 数据管道（Phase B：仅保留刚体WithPose） ========
train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5]),
    dict(type='InitRigidTransform'),
    dict(
        type='RandomRotateAroundZWithPose',
        angle_range=[-3.14, 3.14],
        prob=1.0),
    dict(
        type='RandomFlipWithPose',
        flip_ratio_horizontal=0.5,
        flip_ratio_vertical=0.5),
    dict(type='LoadClipFeature', data_root=DATA_ROOT),
    dict(type='LoadSingleImageFromFile'),
    dict(type='ApplyRigidTransformToPose'),
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
        type='NormalizePointsColor_',
        color_mean=color_mean,
        color_std=color_std),
    dict(
        type='AddSuperPointAnnotations',
        num_classes=num_semantic_classes,
        stuff_classes=[0, 1],
        merge_non_stuff_cls=False),
    dict(
        type='Pack3DDetInputs_',
        keys=[
            'points', 'imgs', 'cam_info', 'clip_pix', 'clip_global', 
            'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks'
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
                color_std=color_std),
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

# ======== 模型配置（与Phase A通道保持一致） ========
model = dict(
    type='ScanNet200MixFormer3D',
    data_preprocessor=dict(type='Det3DDataPreprocessor_'),
    voxel_size=0.02,
    num_classes=num_instance_classes_eval,
    query_thr=0.5,

    backbone=dict(
        type='Res16UNet34C',
        in_channels=3,
        out_channels=96,
        config=dict(
            dilations=[1, 1, 1, 1],
            conv1_kernel_size=5,
            bn_momentum=0.02)),

    bi_encoder=dict(
        type='BiFusionEncoder3DMV',
        voxel_size=0.02,
        conv3d_output_dim=128,
        conv3d_dropout=0.1,  # Phase B：适度正则
        feat_space='precomp_60x80',
        use_precomp_2d=True,
        use_amp=True,
        debug=False,
    ),

    pool=dict(type='GeoAwarePooling', channel_proj=128),

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
        in_channels=128,
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False),

    # Phase B：启用2D监督（带warmup）
    two_d_losses=dict(
        enable_recon=True,
        enable_seg=True,
        w_recon=0.01,
        w_seg=0.01,
        w_align=0.02,
        recon_tau=1.0,
        seg_conf=1.0,
        depth_tol=0.05,
        grid_hw=(60, 80),
        alpha_max=1.0,
        alpha_warmup=1000,
        recon_warmup=1000,
        seg_warmup=1000,
        align_warmup=1000,
    ),

    criterion=dict(
        type='ScanNetMixedCriterion',
        num_semantic_classes=num_semantic_classes,
        sem_criterion=dict(
            type='ScanNetSemanticCriterion',
            ignore_index=num_semantic_classes,
            loss_weight=0.5),
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
            loss_weight=[0.5, 1.0, 1.0, 0.5, 0.5],
            num_classes=num_instance_classes,
            non_object_weight=0.1,
            fix_dice_loss_weight=True,
            iter_matcher=True,
            fix_mean_loss=True)),

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
    batch_size=16,
    num_workers=6,
    persistent_workers=True,
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

# ======== 训练配置（Phase B） ========
optim_wrapper = dict(
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(type='AdamW', lr=0.00005, weight_decay=0.05),
    type='OptimWrapper')

param_scheduler = dict(
    type='CosineAnnealingLR',
    T_max=128,
    eta_min=1e-6)

custom_hooks = [
    dict(type='EmptyCacheHook', after_iter=True),
    dict(
        type='EnhancedTrainingHook',
        log_interval=50,
        grad_monitor_interval=50,
        detailed_stats=True
    ),
    dict(
        type='DetailedLossMonitorHook',
        log_interval=50,
        collect_grad_norm=True,
        collect_clip_stats=True
    ),
    dict(
        type='NaNDetectionHook',
        check_interval=50
    ),
    dict(
        type='PartialLoadHook',
        pretrained='/home/nebula/xxy/ESAM/work_dirs/tmp/mask3d_scannet200.pth',
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
        interval=50,
        log_metric_by_epoch=False,
        out_suffix='.log'
    ))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=128, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

log_processor = dict(type='LogProcessor', window_size=1, by_epoch=True)

vis_backends = [
    dict(type='LocalVisBackend'),
]

visualizer = dict(
    type='Det3DLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')
