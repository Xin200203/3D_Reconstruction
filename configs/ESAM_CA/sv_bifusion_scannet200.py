# 删除旧内容并替换为新的 CA-BiFusion 配置
# Bi-Fusion（Category-Agnostic，ScanNet200-SV）

# 继承上一阶段 TinySA CA 配置
_base_ = ['./sv3d_tiny_sa_scannet200_ca.py']

custom_imports = dict(imports=['oneformer3d'])

# 仅在本文件内部使用 Config 类，改名 _Cfg，后续删除避免写入
from copy import deepcopy
from mmengine.config import Config as _Cfg
import inspect, os
# 直接定位 TinySA CA 配置文件
_cfg_dir = os.path.dirname(inspect.getfile(inspect.currentframe()))
_base_cfg = _Cfg.fromfile(os.path.join(_cfg_dir, 'sv3d_tiny_sa_scannet200_ca.py'))
# 转为普通 list 以避免 ConfigList 的懒索引
train_pipeline = [item for item in _base_cfg.train_pipeline]
DATA_ROOT = '/home/nebula/xxy/ESAM/data/scannet200-sv/'

# 替换 ClipFeature 路径为绝对
train_pipeline.insert(1, dict(type='LoadClipFeature', data_root=DATA_ROOT))
for item in train_pipeline:
    if item.get('type') in ['Pack3DDetInputs_', 'Pack3DDetInputs_Online']:
        item['keys'] = ['points', 'clip_pix', 'clip_global'] + [k for k in item['keys'] if k not in ['clip_pix', 'clip_global']]

# test / val pipeline 同理
test_pipeline = [item for item in _base_cfg.test_pipeline]
for item in test_pipeline:
    if item.get('type') in ['Pack3DDetInputs_', 'Pack3DDetInputs_Online']:
        item['keys'] = ['points', 'clip_pix', 'clip_global'] + [k for k in item['keys'] if k not in ['clip_pix', 'clip_global']]

# 删除 Config 对象，防止 mmengine.pretty_text 序列化出错
del _base_cfg

# 移除 _Cfg 以防泄漏
del _Cfg

# —— 模型部分：添加 BiFusionEncoder，保持 TinySA-3D 分支参数不变 ————
model = dict(
    # 保留父配置 backbone / neck，但新增 bi_encoder；在 MixFormer3D 类内部会优先使用 bi_encoder
    bi_encoder=dict(
        type='BiFusionEncoder',
        clip_pretrained='/home/nebula/xxy/ESAM/data/open_clip_pytorch_model.bin',  # 本地离线权重
        voxel_size=0.02,
        freeze_blocks=0,           # 完全冻结 2D ViT 权重，如需解冻可调整此值
        use_tiny_sa_2d=False,     # 默认关闭 2D Tiny-SA，避免显存过高
    ),
)

# —— 继续训练：加载上一阶段 TinySA CA 权重（仅回填 3D 分支） ————
load_from = '/home/nebula/xxy/ESAM/work_dirs/sv3d_tiny_sa_scannet200_ca/best_all_ap_50%_epoch_128.pth'

# ——— 数据集路径与 ann_file 覆写 ——————————————————————————
data_root = DATA_ROOT  # 覆写父级变量

train_dataloader = dict(dataset=dict(
    data_root=DATA_ROOT,
    ann_file='scannet200_sv_oneformer3d_infos_train_clip.pkl'))

val_dataloader = dict(dataset=dict(
    data_root=DATA_ROOT,
    ann_file='scannet200_sv_oneformer3d_infos_val_clip.pkl'))

test_dataloader = val_dataloader 