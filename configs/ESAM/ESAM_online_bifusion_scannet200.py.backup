# Bi-Fusion Online（ScanNet200-MV）

_base_ = ['./ESAM_online_3dtsa_scannet200.py']

custom_imports = dict(imports=['oneformer3d'])

from copy import deepcopy
from mmengine.config import Config

_base_cfg = Config.fromfile(_base_[0])
train_pipeline = deepcopy(_base_cfg.train_pipeline)
train_pipeline.insert(1, dict(type='LoadClipFeature', data_root='data/scannet200-mv'))
for item in train_pipeline:
    if item.get('type') == 'Pack3DDetInputs_Online':
        item['keys'] = ['points', 'clip_pix', 'clip_global'] + [k for k in item['keys'] if k not in ['clip_pix','clip_global']]

test_pipeline = deepcopy(_base_cfg.test_pipeline)
for item in test_pipeline:
    if item.get('type') == 'Pack3DDetInputs_Online':
        item['keys'] = ['points', 'clip_pix', 'clip_global'] + [k for k in item['keys'] if k not in ['clip_pix','clip_global']]

model = dict(
    bi_encoder=dict(
        type='BiFusionEncoder',
        clip_pretrained='openai',
        voxel_size=0.02,
        freeze_blocks=0,
        use_tiny_sa_2d=False,
    ),
    memory=dict(queue=-1),
)

train_dataloader = dict(batch_size=4) 