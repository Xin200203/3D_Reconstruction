# CLIP 特征处理与调试计划记录

> 本文档用于跟踪 **Bi-Fusion Encoder** 中 2D 语义分支（CLIP ViT-B/16）特征生成与采样的实现细节、调试步骤以及后续改进 TODO。

---

## 1. 总体目标
1. 基于 **open_clip** 加载预训练 `ViT-B/16`（ImageNet + LAION400M），提取 **224×224** 输入分辨率下的 `grid feat`。
2. 通过 `PixelShuffle ×2` → `Conv1×1(256d)` 将 `14×14` patch token展开至 **56×56**, 方便高分辨率采样。
3. 与 3D 点云配准：利用相机 **intrinsic / extrinsic** 将 3D 点投影到 2D 像素，得到 `(u,v)` → **UV Index**。
4. 用 `sample_img_feat()` 在对齐网格上双线性插值采样得到 **N_pts × 256** 的 2D 语义特征。
5. 对特征通过 `Linear(256→128)` 完成降维并返回给 **Fusion Gate**。
6. 模块需支持 **batch > 1** 与多 GPU；先在 Singular 场景（batch=1）打通流水，再扩展。

---

## 2. 处理流程拆解
| 步骤 | I/O | 关键实现 | 备注 |
|----|----|----|----|
| 2.1 | RGB 图像 → `torch.Tensor[B,3,H,W]` | `load_image(path).resize(224).to(device)` | 归一化到 CLIP 均值/方差 |
| 2.2 | 3D 点云 + 相机参数 → `uv_idx (N,2)` | `build_uv_index(points, intrins, extrins)` | 深度裁剪 & FOV 掩码 |
| 2.3 | Tensor → CLIP → `f_map [B,C,14,14]` | `open_clip.create_model()` | `with torch.no_grad()` & `half()` |
| 2.4 | `f_map` → `upsampled [B,256,56,56]` | `PixelShuffle(scale=2)` | 对应 196→224 像素 |
| 2.5 | `upsampled + uv_idx` → `pts_feat [N,256]` | `sample_img_feat()` | `grid_sample` 双线性 |
| 2.6 | `pts_feat` → `Linear128` | `nn.Linear(256,128)` | **FP32** 保存 |

---

## 3. 调试用小数据集 (Mini-Test)
- 文件：`data/scannet200-mv/meta_data/scannetv2_train_debug.txt`（共 30 个场景）。
- 场景示例：`scene0000_00`, `scene0001_01`, …
- 目标：
  1. 对每个场景随机选取 **前 5 帧** 执行完整流程；
  2. 输出以下产物到 `outputs/clip_feat_debug/`：
     - `scene_id/frame_{idx}.npz`：包含 `pts_feat (N,128)`、`uv_idx (N,2)`、`conf_2d (N,)`；
     - `vis_{idx}.png`：特征 TSNE 投影或伪彩热图。
  3. 记录处理耗时 / 显存 / 文件大小，验证缓存策略。

### 3.1 调试脚本
```bash
# Example (single-GPU)
python tools/extract_clip_feat.py \
    --scenes_file data/scannet200-mv/meta_data/scannetv2_train_debug.txt \
    --frames_per_scene 5 \
    --output_dir outputs/clip_feat_debug \
    --vis 1 --dtype fp16
```

### 3.2 自动化单元测试
- `tests/test_clip_pipeline.py`
  - 检查所有输出文件可读且维度匹配。
  - 随机抽取点验证 `(x,y,z)` 投影回像素误差 < **1px**。

---

## 4. 代码结构草案
```
oneformer3d/
  ├─ bi_fusion_encoder.py        # 主模块实现 (已有)
  ├─ clip_utils.py               # build_uv_index / sample_img_feat
  └─ ...

tools/
  ├─ build_uv_index.py           # 线下批量生成 uv 索引 (任选)
  ├─ extract_clip_feat.py        # 图像 → 特征 (*.npz / *.h5)
  └─ vis_clip_feat.py            # 可视化/TSNE

tests/
  └─ test_clip_pipeline.py       # pytest 单元测试
configs/
  └─ clip_feat.yaml              # CLI 参数聚合 (hydra)
```

---

## 5. 关键实现细节
1. **open_clip 加载**：
   ```python
   model, _, _ = open_clip.create_model_and_transforms(
       "ViT-B-16", pretrained="laion2b_s34b_b88k", device=device
   )
   # 【更新】训练阶段完全冻结 CLIP Visual Encoder，防止过拟合
   for p in model.visual.parameters():
       p.requires_grad = False
   ```
2. **UV Index 计算**：
   - 利用深度图反投影得到 camera 坐标，再乘 `extrins` → 世界坐标；
   - 反向时使用 `intrins @ extrins[:3]` 投影到像素；对 `z<0`、`x∉[0,W-1]`、`y∉[0,H-1]` 点过滤。
3. **采样精度**：双线性插值默认在 ‑1~1 归一化网格，注意 **(+0.5)/H -1** 的偏移。
4. **缓存策略**：按 `scene/frame` 为键，将 `upsampled feat` 保存在 LMDB，以减少重复前向；若 GPU RAM < 8GB 可启用 **CPU Offload**。
5. **多 GPU/DataLoader**：后续用 `torch.distributed` 切分场景级 batch，通过 `torch.save` 聚合结果。
6. **与 Bi-Fusion Encoder 集成**：若 `offline_clip_feat=True`，直接从磁盘读取，否则在线缓存。
7. **通用函数抽离**：`clip_utils.py` 提供 freeze_clip/UV 投影/采样；新增 `build_uv_index_batch` 支持 (B,N,3) 批量投影，并将在后续批量化 Encoder 中使用。
8. **混合精度**：在 `BiFusionEncoder` 新增 `use_amp` 开关（默认 True），2D 分支推理包裹 `torch.cuda.amp.autocast`，减少显存。

---

## 6. 性能基线 (Debug 模式)
| 场景数 | 帧数/场景 | 分辨率 | GPU (RTX 3090) | 耗时 | 显存峰值 |
|------|---------|---------|----------------|------|-----------|
| 30   | 5       | 224²    | 单卡 fp16     | ≈ 2 min | < 4 GB |

> 以上为预估值，实际执行后更新。

---

## 7. TODO & 后续计划
- [ ] **extrinsic 批次化**：支持同场景多视角合并投影。
- [ ] **Clip-Cons 正则**：实现 `ClipConsCriterion` 计算跨图像一致性约束。
- [ ] **Tiny-SA Neck ×2**：在 2D 分支引入轻量自注意力颈，评估增益。
- [ ] **混合精度回传**：`torch.cuda.amp.grad_scaler` 管理。
- [ ] **写入 Composer**：将 `extract_clip_feat.py` 参数写入 *composer.yaml*，支持一键生成。
- [ ] **性能评测**：与直接 14×14 patch 采样对比。
- [ ] **Mask-Aware 投影过滤**：融合外部 2D 实例掩码，进一步过滤无效投影视素。
- [ ] **Encoder→OnlineMerge 适配**：批量化输出后，调整 OnlineMerge / Memory Bank 接口。

---

## 8. 优化记录（2024-06-25）
- ✅ 补全核心 util：`_freeze_clip_except_last_blocks`、`build_geo_pe`、`build_uv_index`、`sample_img_feat` 已在 `oneformer3d/bi_fusion_encoder.py`/`tiny_sa.py` 实现。
- ✅ 集成 `TinySAModule`，点云分支支持球查询 + 自注意力两层。
- ✅ `BiFusionEncoder.forward()` 实装单批流水，包含 Geo-PE、FusionGate、`clip_global` 输出。
- ✅ 新增依赖 `open_clip_torch>=2.22`（requirements.txt），以及 `.lmdb`/`outputs/` 路径写入 `.gitignore`。
- ✅ **Batch 支持(v1)**：forward 现接受 batched Tensor (B,N,6)/(B,3,H,W) 或原 List，内部自动拆分循环，后续将继续张量化 3D 分支。
- ⏳ **Clip-Cons Criterion**：接口文件 `oneformer3d/bife_clip_loss.py` 规划中。
- ✅ `tools/extract_clip_feat.py` 初版实现：输出 **PixelShuffle 后 192-d 特征**（不含 Conv1×1），Conv1×1 保留在线可训练；依赖 `clip_utils` 完成 UV 投影与采样。
- ✅ 抽离通用函数至 `oneformer3d/clip_utils.py` 并在 Encoder 内引用，方便后续脚本共享。
- ✅ CLIP Visual Encoder 默认 **全冻结**（`freeze_blocks=0`），训练时不更新权重。
- ✅ **Tiny-SA Neck ×2 (2D)**：`use_tiny_sa_2d` 参数引入，TinySA2D 模块添加至 2D 上采样特征。

---

### TODO 清单更新
- [ ] **extrinsic 批次化**：支持同场景多视角合并投影。
- [ ] **Clip-Cons 正则**：实现 `ClipConsCriterion` 计算跨图像一致性约束。
- [ ] **Batch 支持**：`BiFusionEncoder` forward 扩展至 B>1。
- [x] **Tiny-SA Neck ×2**：在 3D 分支加入 TinySA，两层完成。
- [ ] **混合精度回传**：`torch.cuda.amp.grad_scaler` 管理。
- [ ] **写入 Composer**：将 `extract_clip_feat.py` 参数写入 *composer.yaml*，支持一键生成。
- [ ] **性能评测**：与直接 14×14 patch 采样对比。
- [ ] **Mask-Aware 投影过滤**：融合外部 2D 实例掩码，进一步过滤无效投影视素。
- [ ] **Encoder→OnlineMerge 适配**：批量化输出后，调整 OnlineMerge / Memory Bank 接口。

---

**更新日志**
- *2024-06-25*：创建文档，补充 Mini-Test 方案与代码结构。（by ChatGPT-assistant）
- *2024-06-25*：实现核心 util、TinySA、BiFusionEncoder 单批推理；同步更新依赖及 `.gitignore`。（by ChatGPT-assistant）
