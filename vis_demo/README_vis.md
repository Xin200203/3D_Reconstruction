# 离线可视化（单场景 DataConverter 路径）

本工具用于将 3D 分割与 2D/3D 特征在点云和图像平面上的分布进行离线可视化导出，便于对齐质量与融合效果的分析。

- 3D 载体：
  - 分割上色点云（实例颜色）；
  - （可选）融合特征上色点云（PCA→RGB）；
  - （可选）投影权重 α 与投影有效性 valid 上色点云（灰度）。
- 2D 载体：
  - 分割反投叠图（z-buffer 选择前景实例）；
  - 特征反投热力图（按点特征栅格化）；
  - 覆盖度 cover（热力图，可诊断视角/几何一致性）；
  - （可选）原始 clip 特征图（PCA→RGB），用于对比。

该实现复用现有投影/采样工具：`oneformer3d/projection_utils.py`（`project_points_to_uv`、`splat_to_grid`、`sample_img_feat`），并以 `vis_demo/online_demo.py:DataConverter` 构造单场景样本。

---

## 快速开始

示例（BiFusion 配置）

```
python 3D_Reconstruction/vis_demo/offline_visualize.py \
  --config 3D_Reconstruction/configs/ESAM_CA/sv_bifusion_scannet200_3dmv.py \
  --checkpoint /path/to/your_ckpt.pth \
  --scene_idx scene0011_00 \
  --save_dir 3D_Reconstruction/work_dirs/vis \
  --device cuda:0 \
  --m3d all \
  --m2d all \
  --grid_hw 60 80 \
  --points_sample_ratio 0.5 \
  --frame_ids 0 1 2 3
```

- `--m3d`：3D 导出类别，`seg|feat|alpha|valid|all`
- `--m2d`：2D 导出类别，`seg|feat|cover|all|none`
- `--grid_hw`：2D 反投网格大小（H W），默认从配置 `two_d_losses.grid_hw` 读取，缺省回退 `60x80`
- `--points_sample_ratio`：PLY 导出点下采样比例（0,1]，仅影响文件大小与渲染速度
- `--frame_ids`：仅导出指定帧（索引从 0 开始）

> 注意：若样本中 `cam_info/pose` 缺失，则跳过 2D 反投相关导出；若 `clip_pix` 缺失，则跳过使用 clip 特征的 3D/2D 导出项。

---

## 输出目录结构

```
work_dirs/vis/<scene_idx>/
  ├─ 3d_seg.ply                # 分割上色点云（必有）
  ├─ 3d_feat_fused.ply         # 融合特征上色点云（BiFusion 可用）
  ├─ 3d_feat_2d.ply            # 2D 特征上色点云（若 clip_pix 存在）
  ├─ 3d_alpha.ply              # 融合权重 α 灰度点云（BiFusion 可用）
  ├─ 3d_valid.ply              # 投影有效性 valid 灰度点云（BiFusion 可用）
  ├─ 2d_000_seg_overlay.png    # 分割反投叠图
  ├─ 2d_000_feat.png           # 特征反投热力图（按点特征栅格化）
  ├─ 2d_000_cover.png          # 覆盖度热力图
  ├─ 2d_000_clip.png           # 原始 clip 特征图（PCA→RGB，可选）
  ├─ images.npy                # 原始图像数组（H,W,3），便于复盘
  └─ meta.json                 # 元信息（grid_hw、实例数、图像路径等）
```

---

## 关键设计与一致性

- 投影与采样：
  - 3D→2D 投影使用 `project_points_to_uv`，内参按特征尺寸缩放；
  - 点→像素栅格化使用 `splat_to_grid`：
    - 分割使用 `mode='zbuf'`，优先最近深度实例；
    - 特征使用 `mode='bilinear'`，得到平滑热力图；
  - 采样 2D 特征（如 clip_pix）使用 `sample_img_feat`。
- 尺度与对齐：
  - 默认 2D 反投网格取 `grid_hw`（配置可覆盖），并上采样到原图尺寸；
  - 每帧优先使用该帧 `cam_info['pose'|'extrinsics']`，避免视差误差；
  - 统一最小深度与最大深度（默认 `MIN_DEPTH=0.3`，`max_depth=20.0`）。
- 色彩映射：
  - 特征统一使用 SVD-PCA 压缩到 RGB，并在 [0,255] 归一；
  - α/valid 以灰度显示；
  - 实例调色板使用固定随机种子保证复现。

---

## 常见问题（FAQ）

- Q：为什么没有导出 `3d_feat_3d.ply`？
  - A：当前 `encoder_out` 未显式提供 3D 分支点特征（f3d）。若后续模型输出包含该项，可直接按 `3d_feat_fused` 的方式导出。

- Q：2D 反投与 clip 特征图的区别？
  - A：2D 反投是将点特征栅格化到图像平面（几何一致）；clip 特征图是直接展示模型输入的 2D 特征本身（语义/纹理侧的可视化），两者互补。

- Q：为何部分帧没有 2D 导出？
  - A：该帧可能缺失 `pose/intrinsics` 或点在当前视角不可见（覆盖度为 0）。脚本会自动跳过，保证流程不中断。

---

## 后续增强（建议）

- 导出 3D `feat2d/feat3d` 的分离版本（当模型输出或可计算时）；
- CLI：支持 `--export_clip_map` 显式开关；
- 记录关键统计（cover_mean、supervised_pixel_ratio 等）到 JSON；
- 支持多场景批处理与自动生成汇总报告。

---

## 依赖与环境

- 依赖：`torch`、`numpy`、`Pillow`、`mmengine`、`mmdet3d`、`open3d`（若仅保存 PLY/PNG，可不使用 open3d）。
- 数据：需正确准备 `cfg.data_root` 下的点云/图像/clip 特征与 `pose/intrinsics`（由配置的 pipeline 加载）。

如需进一步集成或展示更复杂的融合统计，我可以继续扩展脚本与接口。
