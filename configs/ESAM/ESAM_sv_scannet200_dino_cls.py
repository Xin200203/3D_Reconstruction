"""
ScanNet200-SV + Online DINOv2 + 类别感知(thing-class)实例分类版本。

基于已跑通的在线 DINO 配置（ESAM_CA/ESAM_sv_scannet200_CA_dino.py）：
- 保持 2D 增强/Resize/内参更新/2D-3D 对齐链路不变
- 将 instance 分支从 class-agnostic(1类)切换为 thing-class aware（198 类）

说明：
ScanNet200 的 stuff 类为 [wall, floor] => num_stuff=2
thing 类数 = 200 - 2 = 198，对应 predict_by_feat_panoptic 中的 `labels + n_stuff_classes` 逻辑。
"""

_base_ = ['../ESAM_CA/ESAM_sv_scannet200_CA_dino.py']

# --- 类别设置：由 class-agnostic -> thing-class aware ---
num_semantic_classes = 200
num_stuff_classes = 2
num_instance_classes = num_semantic_classes - num_stuff_classes  # 198
num_instance_classes_eval = num_instance_classes

# 覆盖模型中的 class 数配置（decoder/criterion/test-time panoptic 映射依赖该设定）
model = dict(
    num_classes=num_instance_classes_eval,
    decoder=dict(
        num_instance_classes=num_instance_classes,
        num_semantic_classes=num_semantic_classes,
    ),
    criterion=dict(
        num_semantic_classes=num_semantic_classes,
        inst_criterion=dict(num_classes=num_instance_classes),
    ),
)

