# 可视化 TODO（单场景 DataConverter 离线路径）

目标：基于 DataConverter 的单场景离线脚本，导出两类载体的可视化产物。
- 3D：分割上色点云；（可选）3D/2D/融合特征上色；（可选）融合权重 α、投影有效性 valid 上色。
- 2D：分割结果反投叠图；特征反投热力图；覆盖度 cover 热力图。

参考代码位置：
- 投影/采样与栅格化：`oneformer3d/projection_utils.py`（`project_points_to_uv`、`splat_to_grid`）
- 单场景数据构造：`vis_demo/online_demo.py: DataConverter`
- 流式演示与上色策略：`vis_demo/stream_demo.py`

---

## A. 修复/清理（必须，体量小）
- [ ] `vis_demo/utils/vis_utils.py`：在 `vis_pointcloud.__init__` 里补 `self.online_vis = online_vis`
- [ ] `vis_demo/online_demo.py`：导入改为 `from vis_demo.utils.vis_utils import vis_pointcloud, Vis_color`

验收：在线/离线 demo 脚本可正常创建可视化器，无属性异常。

## B. 新建离线脚本（最小可用）
- [ ] 新建 `vis_demo/offline_visualize.py`，解析 CLI：
  - `--config --checkpoint --scene_idx --save_dir --device`
  - `--grid_hw 60 80`（默认与配置一致，可覆盖）
  - `--m3d [seg,3d,2d,fused,alpha,valid,all]`，`--m2d [seg,feat,cover,all]`
- [ ] 初始化模型（参考 `vis_demo/stream_demo.py:init_model`）并设置 `model.map_to_rec_pcd=False`
- [ ] 复用 `online_demo.DataConverter`：构造单场景 `info` → `Compose(cfg.test_pipeline)` → `pseudo_collate` → `model.test_step`
- [ ] 读取并保存原图 `images.npy`（便于 2D 叠图）

验收：可对单个 `scene_idx` 跑通并拿到 `result, data, img_paths, images.npy`。

## C. 工具函数（脚本内先实现）
- [ ] 实例掩码→点标签：按 `instance_scores` 降序覆盖，得到每点实例 id（参考 `stream_demo.mask_to_color`）
- [ ] PCA 上色：`features_to_rgb(feat, fit_once=True)`（第一次拟合并缓存）
- [ ] 标量上色：`scalar_to_cmap(x, cmap='viridis')`
- [ ] 保存 PLY：`save_ply(points, colors, path)`（颜色范围 [0,255]）

验收：随机特征/标量能稳定映射到 RGB，PLY 打开正确。

## D. 3D 导出（离线）
- [ ] 分割上色点云：
  - 输入：`points = data['inputs']['points'][:,:,:3]`，`pred = result.pred_pts_seg`
  - 处理：掩码→点标签→调色板→保存 `3d_seg.ply`
- [ ] （可选，BiFusion）融合/2D/3D特征上色：
  - 来源：`model._encoder_out['feat_fusion'][0]`（若有）；如能获取单独 f3d/f2d 也分别导出
  - 上色：PCA→`3d_feat_fused.ply`、`3d_feat_2d.ply`、`3d_feat_3d.ply`
- [ ] （可选，BiFusion）α/valid 上色：
  - 来源：`_encoder_out['conf_2d'][0]`、`_encoder_out['valid_projection_mask'][0]`
  - 上色：colormap→`3d_alpha.ply`、`3d_valid.ply`

验收：对应 PLY 文件生成且色彩/渐变合理；无 NaN/Inf。

## E. 2D 反投与导出
- [ ] 相机参数：
  - 外参 `pose`（C2W）按帧读取，`W2C = inverse(pose)`
  - 内参 `intrinsic`：从数据根目录或 `cfg.data_root` 中读取，或由 DataConverter/数据集提供
- [ ] 点到相机：`xyz_cam = [xyz,1] @ W2C^T`
- [ ] 投影：`uv, valid = project_points_to_uv(xyz_cam, (H,W), max_depth, intrinsics)`
- [ ] 分割反投（zbuf）：
  - 点标签 → one-hot
  - `label_map = argmax(splat_to_grid(uv, z, one_hot, valid, H, W, mode='zbuf'))`
  - 叠图到原图，保存 `2d_<k>_seg_overlay.png`
- [ ] 特征反投（bilinear）：
  - 选择点特征（优先：融合特征；否则：3D特征或简易 RGB）
  - `F2D, cover = splat_to_grid(...)`
  - `F2D` → PCA/能量热力图，保存 `2d_<k>_feat.png`
  - `cover` → 热力图，保存 `2d_<k>_cover.png`

验收：分割叠图位置合理；特征/cover 热力图视觉连贯；边界无明显半像素偏移。

## F. 一致性与参数
- [ ] `grid_hw` 默认取 `cfg.model.two_d_losses.grid_hw`，可被 CLI 覆盖
- [ ] 深度阈值与 `projection_utils` 一致（`MIN_DEPTH` 与 `max_depth`）
- [ ] 内参缩放与 `(H,W)` 一致（`_scale_intrinsics` 已处理）

验收：换不同 `(H,W)` 不出现越界或极端失真。

## G. 运行验证与文档
- [ ] 选择一个 ScanNet 场景（如 `scene0011_00`）完整跑通导出
- [ ] 在 `work_dirs/vis/<scene>/` 下检查 PLY/PNG/Numpy 文件
- [ ] 新建 `README_vis.md`：说明参数、输出样例、注意事项

验收：文档清晰、可复现；两次运行颜色/投影一致（PCA 固定随机种子）。

## H. 可选增强（后续）
- [ ] CLI：`--frame_ids`、`--points_sample_ratio`、`--export_full`
- [ ] 导出关键统计（如 cover_mean、supervised_pixel_ratio）到 JSON
- [ ] 可视化“2D/3D特征夹角相似度” 上色（3D/2D 诊断）

