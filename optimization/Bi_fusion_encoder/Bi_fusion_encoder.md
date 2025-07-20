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

> 更新日期：2025-06-21
