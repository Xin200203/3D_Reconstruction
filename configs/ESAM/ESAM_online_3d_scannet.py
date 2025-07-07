# 纯 3D baseline Online（ScanNet200-MV）
# 继承 CA Online 配置，无 Tiny-SA / Bi-Fusion。

_base_ = ['../ESAM_CA/ESAM_online_scannet200_CA.py']
 
# 无额外修改；如需调整 batch_size / work_dir / num_frames 等，请在命令行或自定义 config 中覆盖。 