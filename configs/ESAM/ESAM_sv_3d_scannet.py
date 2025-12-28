# 纯 3D baseline（ScanNet200-SV）
# 仅保留 3D Backbone，无 Tiny-SA / Bi-Fusion。
# 继承官方 CA 配置，方便与原实验对齐。

_base_ = ['../ESAM_CA/ESAM_sv_scannet200_CA.py']
 
# 无额外修改；如需调整 batch_size / work_dir 等，请在命令行指定。 