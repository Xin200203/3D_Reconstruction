"""
ScanNet200-SV 基线（无 DINO）+ 类别敏感(thing-class)实例分类版本。

对齐 ESAM 的 panoptic 设定：
- stuff 类：wall/floor => 2 类（索引 0,1）
- thing 类：其余 198 类（索引 0..197），在后处理时会 +2 映射回语义类 id

该配置用于对照实验：在不引入 2D 特征的情况下，将原本 class-agnostic(1类)实例
分类改为 thing-class aware(198类)，观察对 instance AP / panoptic 的影响。
"""

_base_ = ['../ESAM_CA/ESAM_sv_scannet200_CA.py']

num_semantic_classes = 200
num_stuff_classes = 2
num_instance_classes = num_semantic_classes - num_stuff_classes  # 198
num_instance_classes_eval = num_instance_classes

model = dict(
    num_classes=num_instance_classes_eval,
    decoder=dict(num_instance_classes=num_instance_classes),
    criterion=dict(inst_criterion=dict(num_classes=num_instance_classes)),
)

# 训练建议：从 Mask3D ScanNet200 预训练初始化（与 DINO 实验对齐），避免加载 CA 的 1-class checkpoint
load_from = '/home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/mask3d_scannet200.pth'
