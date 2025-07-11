# Multi-class TinySA config – inherits original
_base_ = './ESAM_sv_3dtsa_scannet200.py'

load_from = 'work_dirs/sv3d_scannet200/latest.pth' 

radius=0.15,
max_k=16,
sample_ratio=0.02, 