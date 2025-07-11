# TinySA 版本（CA），继承旧 ESAM_sv_3dtsa_scannet200.py + CA
_base_ = '../ESAM/ESAM_sv_3dtsa_scannet200.py'

# CA 模式只需在 test_cfg 或数据集部分改 label mapping，这里复用旧文件
load_from = 'work_dirs/sv3d_scannet200_ca/latest.pth' 