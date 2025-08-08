# EmbodiedSAM 代码流程梳理

## 0. 入口脚本
| 文件 | 作用 | 主函数 |
| ---- | ---- | ------ |
| `tools/train.py` | 训练入口，构造 Runner 并启动 `runner.train()` | `main()` |
| `tools/test.py`  | 推理 / 评测入口，加载模型权重后 `runner.test()` | `main()` |
| `demo/scannet200_sv_demo.py` | 单场景可视化 Demo，调用 `init_model → inference_detector` | `main()` |

---

## 1. 2D Mask 生成
⚠️ 差异：本仓库未包含 **SAM** 前向代码。训练 & 推理阶段直接读取离线计算好的「super-point index」。

- **输入**：RGB (B, 3, H, W) + 深度 & 相机位姿（离线）
- **输出**：`sp_pts_mask` (B, N) – 点云中每点所属 super-point
- **关键函数 / 读写**  
  | 文件 | 函数 | 说明 |
  | ---- | ---- | ---- |
  | `oneformer3d/loading.py` | `LoadAdjacentDataFromFile.transform()` | 读取 RGB-D、深度投影点云、`super_pts_mask` |
  | `oneformer3d/transforms_3d.py` | `AddSuperPointAnnotations_Online.transform()` | 将 `sp_pts_mask` 与实例 / 语义标签对齐，生成 GT |

- **数据流**  
  离线 `SAM ➜ 2D masks ↘`  
  `project_depth` ↘ → `super_pts_mask.bin` → Pipeline 读取

---

## 2. Sparse 3D Encoder（带 Memory Adapter）
- **输入**：点云 `P_t` (N, 6)  (xyz + rgb / feats)
- **输出**：点特征 `F_P` (batch 列表，每帧 N_i × C)  
  坐标残差 / SparseTensor 继续传下游
- **关键函数**
  | 文件 | 函数 | 说明 |
  | ---- | ---- | ---- |
  | `oneformer3d/mink_unet.py` | `Res16UNet34C.forward()` | 3D Sparse U-Net 主干 |
  | `oneformer3d/multilevel_memory.py` | `MultilevelMemory.forward()` | 时序特征累积（VMP + 缓存 queue） |
  | `oneformer3d/mixformer3d.py` | `ScanNet200MixFormer3D*_Online.extract_feat()` | 组装 TensorField → 调 Backbone → 返回 `point_features` |
  
  ```480:540:oneformer3d/mink_unet.py
  # ... forward passes (含 memory 注入位置) ...
  ```

- **数据流**  
  `P_t → TensorField → Res16UNet34C`  
  ├─ 若启用 Memory：`MultilevelMemory` 注入多层特征  
  └─ `x.slice(field)` 得到稀疏点特征列表 `point_features`

---

## 3. Geometric-aware Query Lifting
- **输入**：`F_P`、`sp_idx` (N)，点坐标 `xyz`
- **输出**：super-point 特征 `F_S` (M, C)、权重 `all_xyz_w`；后续随机采样得到初始 Query `Q0`
- **关键函数**
  | 文件 | 函数 | 说明 |
  | ---- | ---- | ---- |
  | `oneformer3d/geo_aware_pool.py` | `GeoAwarePooling.forward()` | 归一化坐标 → MLP → Sigmoid 权重池化 |
  | `oneformer3d/mixformer3d.py` | `extract_feat()` 内调用 `self.pool()` | 将 `F_P` 聚合至 super-point |
  
  ```25:40:oneformer3d/geo_aware_pool.py
  def forward(self, x, sp_idx, all_xyz, with_xyz=False):
      # 几何归一化 + 权重计算 + scatter_mean
      ...
  ```

- **数据流**  
  `F_P + sp_idx → GeoAwarePooling ➜ F_S`  
  `F_S` 经 `_select_queries` 随机选(0.5~1)×M 行 → `Q0`

---

## 4. Dual-level Query Decoder (迭代 L=3)
- **输入**：`Q_l`、`F_S`、点特征 `F_P`
- **输出**：精细点级 mask `M_l` (N)、最终 Query `Q3`
- **关键函数**
  | 文件 | 函数 | 说明 |
  | ---- | ---- | ---- |
  | `oneformer3d/query_decoder.py` | `ScanNetMixQueryDecoder.forward()` | 按 `cross_attn_mode / mask_pred_mode` 迭代 3 层 |
  | same | `CrossAttentionLayer / SelfAttentionLayer / FFN` | Transformer 子层 |
  
  ```140:190:oneformer3d/query_decoder.py
  self.cross_attn_layers.append(CrossAttentionLayer(...))
  ...
  def forward_iter_pred(...):
      # 迭代生成 cls_preds / masks / scores
  ```

- **数据流（单层）**  
  `Q_l` × `F_S` → Masked Cross-Attention → Self-Attn+FFN → `Q_{l+1}`  
  `Q_{l+1}` 与 `F_P` 做 `einsum` → 点级 `M_l` → super-point 投票更新下一轮 Attn mask

---

## 5. Query Heads & 在线实例合并
- **输入**：最终 Query `Q3`
- **输出**：向量表征 `B`(bbox)、`f`(对比特征)、`S`(语义分布)；合并后全局 Memory
- **关键函数**
  | 文件 | 函数 | 说明 |
  | ---- | ---- | ---- |
  | `oneformer3d/merge_head.py` | `MergeHead.forward()` | Linear → L2 Norm 得到对比特征 f |
  | `oneformer3d/query_decoder.py` | `_forward_head()` | 额外 `bbox_pred / sem_pred` 线性层 |
  | `oneformer3d/instance_merge.py` | `OnlineMerge.merge()` | IoU + cos sim + label 匈牙利匹配 |
  
  ```200:240:oneformer3d/instance_merge.py
  def merge(...):
      mix_scores = query_feat * xyz_scores
      row_ind, col_ind = linear_sum_assignment(-mix_scores)
      # 更新 cur_masks / cur_queries / cur_scores ...
  ```

- **数据流**  
  `Q3 → MergeHead ➜ f`  
  `Q3 → Linear ➜ bbox_pred (B)` / sem_pred (S)  
  当前帧 (B,f,S) 与 `Memory_{t-1}` → 计算相似度 C → Hungarian → 更新 `Memory_t`

---

## 6. 训练入口 / 损失拼装
- **入口脚本**：`tools/train.py` → cfg (`configs/ESAM/ESAM_online_scannet.py`)  
- **主要 Criterion**
  | 文件 | 类 | 负责 loss |
  | ---- | --- | --------- |
  | `oneformer3d/instance_criterion.py` | `MixedInstanceCriterion` | `L_cls`, `L_bce`, `L_dice`, `L_iou`, `L_cont` |
  | `oneformer3d/semantic_criterion.py` | `ScanNetSemanticCriterion` | 语义 CE |
  | `oneformer3d/merge_criterion.py` | `ScanNetMergeCriterion_Fast` | 跨帧对比约束 |
- `configs/ESAM/ESAM_online_scannet.py` 中 `loss_weight`、`topk_insts`、`query_thr` 等超参与论文一致。

---

## A. 重要常量 & 超参
- `configs/ESAM/ESAM_online_scannet.py`
  - `voxel_size = 0.02`
  - `num_layers(decoder) = 3`
  - `test_cfg.topk_insts = 20`, `inscat_topk_insts = 100`
  - `merge_type = 'learnable_online'`, `queue = -1` (Memory 关闭 FIFO)

---

## B. 数据结构一览
| 名称 | 定义位置 | 关键字段 |
| ---- | -------- | -------- |
| `InstanceData` (mmengine) | 三维 GT / 预测容器 | `labels_3d`, `sp_masks`, `p_masks`, `bboxes_3d` |
| `PointData` (mmengine) | 推理输出 | `pts_semantic_mask`, `pts_instance_mask`, `instance_scores` |
| `OnlineMerge` 内缓存 | `oneformer3d/instance_merge.py` | `cur_masks, cur_labels, cur_scores, cur_queries, merge_counts` |

---

## C. 调用栈示例（在线推理 ScanNet-MV）
```text
tools/test.py::main
  └─ Runner.test
     └─ ScanNet200MixFormer3D_Online.predict
        ├─ extract_feat  (Res16UNet → MultilevelMemory)
        ├─ GeoAwarePooling
        ├─ decoder.forward_iter_pred  (×3)
        ├─ MergeHead.forward
        └─ OnlineMerge.merge   (跨帧)
```

---