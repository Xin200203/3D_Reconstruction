_base_ = ['./ESAM_online_scannet200_CA.py']

# Match `ESAM-E_CA_online_epoch_128.pth` training config (see checkpoint meta['cfg']):
# - MergeHead uses LayerNorm (not BatchNorm), otherwise checkpoint will miss BN running stats
#   and online merging quality will collapse.
# - Instance score threshold aligns with the checkpoint config.

model = dict(
    merge_head=dict(norm='layer'),
    test_cfg=dict(inst_score_thr=0.3),
)

