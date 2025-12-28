# Bi-Fusion（ScanNet200-SV）
# 基于 3D+TinySA 配置，加入 2D-3D 融合 Encoder 与离线 CLIP 特征加载。

_base_ = ['./ESAM_sv_3dtsa_scannet.py']

custom_imports = dict(imports=['oneformer3d'])

# --- 数据管线修改：插入 LoadClipFeature 并补 keys ---
from copy import deepcopy
from mmengine.config import Config

_base_cfg = Config.fromfile(_base_[0])
train_pipeline = deepcopy(_base_cfg.train_pipeline)
# 插入 LoadClipFeature 紧跟 Points loader 之后（索引1）
train_pipeline.insert(1, dict(type='LoadClipFeature', data_root='data/scannet200-sv'))
# 更新 Pack3DDetInputs_ keys
for item in train_pipeline:
    if item.get('type') == 'Pack3DDetInputs_':
        item['keys'] = ['points', 'clip_pix', 'clip_global'] + [k for k in item['keys'] if k not in ['clip_pix','clip_global']]

# test_pipeline 同理
test_pipeline = deepcopy(_base_cfg.test_pipeline)
for item in test_pipeline:
    if item.get('type') == 'Pack3DDetInputs_':
        item['keys'] = ['points', 'clip_pix', 'clip_global'] + [k for k in item['keys'] if k not in ['clip_pix','clip_global']]

# --- 模型改动 ---
model = dict(
    _delete_=False,
    bi_encoder=dict(
        type='BiFusionEncoder',
        clip_pretrained='openai',
        voxel_size=0.02,
        freeze_blocks=0,
        use_tiny_sa_2d=False,
    ),
) 