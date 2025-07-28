## Bi-Fusion Encoder 开发日志

### Step-1  skeleton 实现  (commit <hash>)
- 新增 `oneformer3d/bi_fusion_encoder.py`
  * CLIP ViT-B/16 视觉分支加载 (`open_clip`)，冻结除 blocks.10/11
  * 2D PixelShuffle×2 + Conv1×1 → 256d → Linear→128
  * 3D Res16UNet34C (无 Memory) + TinySA ×2 → 128d
  * Geo-PE 64d → MLP→32d
  * Linear 对齐 + `FusionGate` 输出 96d `feat_fusion`
  * 工具：`build_uv_index` / `sample_img_feat` 保持可微
  * 输出 dict: `feat_fusion`, `conf_2d`, `pe_xyz`

### Step-2  功能补强  (commit <hash>)
- ✅ 支持 **Batch 推理/训练**：`forward` 改为循环遍历 batch，并返回 List 结构。
- ✅ 接入 **Camera Extrinsics**：若 `cam_info` 含 `extrinsics`，自动从 cam→world 逆变换。

### TODO / Step-3
- ✅ **TinySA Neck 邻域采样**：实现 ball-query (r=0.3 m, k≤32) 下采样 + Attention，再使用 nearest-interp upsample。
- ✅ **单元测试**：新增 `tests/test_bi_encoder.py` 检查 batch≥2、外参输入、nan 值。

### Step-3  集成与对齐  (commit <hash>)
- ✅ **ClipConsCriterion** 实现并注册；
- ✅ **MixFormer3D 集成**（`bi_encoder` & `clip_criterion`），`extract_feat` & `loss()` 完整对接。

### 2025-07-23  坐标一致化修复
- 将 3D 体素化 `coords_int` 计算从 **xyz_cam** 改为 **xyz_world**，保持三分支坐标系一致。
  - 文件：`oneformer3d/bi_fusion_encoder.py`
  - 行：`coords_int = torch.round(xyz_world / voxel_size)`
- 保留 `xyz_cam` 仅用于 2D 投影；PE 分支无需改。
- 预期效果：valid 投影命中率提升，Gate 对齐更稳定。

### 2025-07-23  FusionGate 重构
- 双分支加入 `LayerNorm` 对齐方差；删去 128→96 压缩。
- 新增 `lin2d128/lin3d128` (160→128) 保留信息量。
- Gate 改为 `Linear(256,256)+Tanh`，并应用温度缩放 τ=5：`gate=0.5*(1+tanh/τ)`。
- 投影无效点 (`valid=False`) 强制 `gate=0`，避免 2D 噪声注入。
- 融合公式：`fused = gate*f2d + (1-gate)*f3d`；`conf` 返回 gate 均值。
- 预期：Gate 均值≈0.4–0.6，梯度不再饱和，mAP ↑≈0.5–1pt。

### 2025-07-23  配置切换到纯 3D 基线
- 将 BiFusion 配置 `_base_` 改为 `sv3d_scannet200_ca.py`，彻底去除 Tiny-SA 依赖。
- 更新 `load_from` 路径：`sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth`。

### 2025-07-23  Gate 监控日志
- 在 `forward()` 末尾调用 `runner.message_hub.update_scalar('gate_mean', value)`，配合默认 `LoggerHook` 即可在训练日志中看到 `gate_mean=...`。
- 触发条件 `self.training`，推理阶段不记录。

> 更新日期：2025-06-21
