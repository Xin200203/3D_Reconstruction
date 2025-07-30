_base_ = [
    'mmdet3d::_base_/default_runtime.py',
    'mmdet3d::_base_/datasets/scannet-seg.py'
]
custom_imports = dict(imports=['oneformer3d'])

num_instance_classes = 1
num_semantic_classes = 200
num_instance_classes_eval = 1
use_bbox = True

model = dict(
    type='ScanNet200MixFormer3D_Online',
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
    
    # 🆕 使用BiFusionEncoder替代传统backbone+neck组合
    bi_encoder=dict(
        type='BiFusionEncoder',
        clip_pretrained='/home/nebula/xxy/ESAM/data/open_clip_pytorch_model.bin',
        voxel_size=0.02,
        clip_num_layers=6,
        freeze_clip_conv1=False,
        freeze_clip_early_layers=True,
        use_enhanced_gate=True,
        use_spatial_attention=True,
        spatial_k=16,
        use_tiny_sa_2d=False,
        use_tiny_sa_3d=False,
        use_amp=True,
        freeze_blocks=0,
    ),
    
    # Memory模块配置
    memory=dict(
        type='MultilevelMemory', 
        in_channels=[32, 64, 128, 256], 
        queue=-1, 
        vmp_layer=(0,1,2,3)
    ),
    
    # 更新pool配置以匹配BiFusionEncoder输出
    pool=dict(type='GeoAwarePooling', channel_proj=256),
    
    decoder=dict(
        type='ScanNetMixQueryDecoder',
        num_layers=3,
        share_attn_mlp=False, 
        share_mask_mlp=False,
        temporal_attn=False,
        cross_attn_mode=["", "SP", "SP", "SP"], 
        mask_pred_mode=["SP", "SP", "P", "P"],
        num_instance_queries=0,
        num_semantic_queries=0,
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
        num_semantic_linears=1,
        in_channels=256,  # 与BiFusionEncoder输出维度匹配
        d_model=256,
        num_heads=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='gelu',
        iter_pred=True,
        attn_mask=True,
        fix_attention=True,
        objectness_flag=False,
        bbox_flag=use_bbox),
    
    merge_head=dict(type='MergeHead', in_channels=256, out_channels=256),
    merge_criterion=dict(type='ScanNetMergeCriterion_Fast', tmp=True, p2s=False),
    
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
    
    # 🆕 添加CLIP对比损失
    clip_criterion=dict(
        type='ClipConsCriterion',
        loss_weight=0.1,
    ),
    
    train_cfg=dict(),
    test_cfg=dict(
        topk_insts=20,
        inscat_topk_insts=100,
        inst_score_thr=0.25,
        pan_score_thr=0.5,
        npoint_thr=100,
        obj_normalization=True,
        sp_score_thr=0.4,
        nms=True,
        matrix_nms_kernel='linear',
        stuff_classes=[0, 1],
        # 🆕 配置OnlineMerge和TimeDividedTransformer
        merge_type='online_tdt',  # 新的合并策略
        online_merge_cfg=dict(
            type='OnlineMerge',
            inscat_topk_insts=200,
            use_bbox=True,  # 使用3D bbox IoU计算
            merge_type='count',
            iou_thr=0.15,  # IoU预剪枝阈值
            ema_alpha=0.9,  # EMA更新系数
            # Time Divided Transformer配置
            tformer_cfg=dict(
                type='TimeDividedTransformer',
                d_model=256,  # 与decoder输出维度一致
                nhead=8,
                num_layers=3,
                dropout=0.1
            )
        )))

# ======== 数据集配置 ========
dataset_type = 'ScanNet200SegMVDataset_'
data_root = 'data/scannet200-mv/'

# ScanNet200类别名称（200类）
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

color_mean = (
    0.47793125906962 * 255,
    0.4303257521323044 * 255,
    0.3749598901421883 * 255)
color_std = (
    0.2834475483823543 * 255,
    0.27566157565723015 * 255,
    0.27018971370874995 * 255)

# 数据处理pipeline
train_pipeline = [
    dict(
        type='LoadAdjacentDataFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        num_frames=8,
        num_sample=20000,
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True,
        with_rec=use_bbox, 
        cat_rec=use_bbox,
        dataset_type='scannet200'),
    # 🆕 添加CLIP特征加载
    dict(type='LoadClipFeature', 
         clip_path='data/scannet200-mv/clip_features'),
    # 🆕 添加图像加载
    dict(type='LoadSingleImageFromFile'),
    dict(type='SwapChairAndFloorWithRec' if use_bbox else 'SwapChairAndFloor'),
    dict(type='PointSegClassMappingWithRec' if use_bbox else 'PointSegClassMapping'),
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
        type='AddSuperPointAnnotations_Online',
        num_classes=num_semantic_classes,
        stuff_classes=[0, 1],
        merge_non_stuff_cls=False,
        with_rec=use_bbox),
    dict(
        type='ElasticTransfrom',
        gran=[6, 20],
        mag=[40, 160],
        voxel_size=0.02,
        p=0.5,
        with_rec=use_bbox),
    dict(type='BboxCalculation' if use_bbox else 'NoOperation', voxel_size=0.02),
    dict(
        type='Pack3DDetInputs_Online',
        keys=[
            'points', 'gt_labels_3d', 'pts_semantic_mask', 'pts_instance_mask',
            'sp_pts_mask', 'gt_sp_masks', 'elastic_coords',
            'imgs', 'cam_info', 'clip_pix', 'clip_global'  # 🆕 添加图像和CLIP特征
        ] + (['gt_bboxes_3d'] if use_bbox else []))
]

test_pipeline = [
    dict(
        type='LoadAdjacentDataFromFile',
        coord_type='DEPTH',
        shift_height=False,
        use_color=True,
        load_dim=6,
        use_dim=[0, 1, 2, 3, 4, 5],
        num_frames=-1,
        num_sample=20000,
        with_bbox_3d=False,
        with_label_3d=False,
        with_mask_3d=True,
        with_seg_3d=True,
        with_sp_mask_3d=True,
        with_rec=True,
        dataset_type='scannet200'),
    dict(type='LoadClipFeature', 
         clip_path='data/scannet200-mv/clip_features'),
    dict(type='LoadSingleImageFromFile'),
    dict(type='SwapChairAndFloorWithRec'),
    dict(type='PointSegClassMappingWithRec'),
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
                type='AddSuperPointAnnotations_Online',
                num_classes=num_semantic_classes,
                stuff_classes=[0, 1],
                merge_non_stuff_cls=False,
                with_rec=True),
        ]),
    dict(type='Pack3DDetInputs_Online', 
         keys=['points', 'sp_pts_mask', 'imgs', 'cam_info', 'clip_pix', 'clip_global'])
]

# DataLoader配置
train_dataloader = dict(
    batch_size=4,  # 降低batch_size因为BiFusion需要更多显存
    num_workers=8,
    dataset=dict(
        type=dataset_type,
        ann_file='scannet200_mv_oneformer3d_infos_train.pkl',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        pipeline=train_pipeline,
        ignore_index=num_semantic_classes,
        scene_idxs=None,
        test_mode=False))

val_dataloader = dict(
    dataset=dict(
        type=dataset_type,
        ann_file='scannet200_mv_oneformer3d_infos_val.pkl',
        data_root=data_root,
        metainfo=dict(classes=class_names),
        pipeline=test_pipeline,
        ignore_index=num_semantic_classes,
        test_mode=True))

test_dataloader = val_dataloader

# 评估配置
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

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
    clip_grad=dict(max_norm=10, norm_type=2))

# 学习率调度器
param_scheduler = dict(type='PolyLR', begin=0, end=128, power=0.9)

# 自定义hooks
custom_hooks = [dict(type='EmptyCacheHook', after_iter=True)]
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best=['all_ap_50%'],
        rule='greater'))

# 🆕 从单帧训练好的模型加载权重
load_from = 'work_dirs/sv_bifusion_scannet200/epoch_128.pth'

# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=128, val_interval=128)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 日志处理器
log_processor = dict(type='LogProcessor', window_size=50, by_epoch=True)
