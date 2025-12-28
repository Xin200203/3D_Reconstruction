## [0.2.0] - 2025-06-21
### Added
- 跨帧匹配模块 `TimeDividedTransformer` 与 `_GeomBiasAttnLayer`，实现几何偏置 Cross-Attention、GRU 更新、Self-Attention 与 FFN。
- 匹配与对比损失 `oneformer3d/tdt_loss.py` (`CrossFrameCriterion`)。
- 2D-3D `Bi-Fusion Encoder` (`oneformer3d/bi_fusion_encoder.py`)，包含 CLIP ViT-B/16 2D 分支、Res16UNet34C + TinySA 3D 分支、Geo-PE、FusionGate 等。
- 文档 `time_divided_transformer.md`、`Bi_fusion_encoder/Bi_fusion_encoder.md` 与 `composer.yaml`。
- 依赖 `open_clip_torch>=2.22` 加入 `requirements.txt`。

### Changed
- `instance_merge.OnlineMerge` 增加 `tformer_cfg` 并接入跨帧匹配；保留旧逻辑作 fallback。
- `.gitignore` 递归忽略 `__pycache__`。

### Upcoming
- 完成 TinySA 邻域采样、Batch 支持、ClipConsCriterion、单元测试与 MixFormer3D 整合。

---

## Changes
Apart from hyperparameter tuning, we also make several modifications on the design.
#### Mask refine strategy：
In the decoder, the mask refinement process is no longer fixed at a specific level but adopts a coarse-to-fine refinement strategy. Compared to the original approach, where the initial mask and three subsequent refinements were all at the point level (P), the new strategy generates an initial mask at the superpoint (SP) level and performs the first refinement at this level. It is then converted to the point level for the following two refinements. This approach can further improve the quality of the masks.

Specifically, we added the corresponding code and modified the `mask_pred_mode` parameter in the decoder's configuration from `["P", "P", "P", "P"]` to `["SP", "SP", "P", "P"]`.

#### Merging metric:
In the original method, our merging metric was defined as a weighted sum of geometric similarity, contrastive similarity, and semantic similarity, with the threshold determined through experimental tuning. Now, we have adjusted the merging metric to be the product of geometric similarity and contrastive similarity, with a threshold set to 0 for class-agnostic setting. For class-aware setting, we additionally set the similarity score between two masks to 0 if their semantic categories are inconsistent. The new metric better measures the similarity between masks and utilizes semantic predictions in a more explicit way.

## [0.2.1] - Unreleased
### Added
- `ClipConsCriterion` (oneformer3d/bife_clip_loss.py) 注册到 mmdet3d.REGISTRY，提供跨模态余弦对比损失。
- 在 `mixformer3d` 新增可选 `bi_encoder`、`clip_criterion` 构造参数。

### Changed
- `BiFusionEncoder` 现支持 Batch 输入、支持 Extrinsics，并输出 `clip_global` 向量；TinySA 邻域采样(球半径0.3m, k≤32) 集成。
- `mixformer3d.extract_feat()` 当 `bi_encoder` 启用时走融合分支，返回 superpoint 特征；`loss()` 中叠加 `loss_clip`。
- 文档 `Bi_fusion_encoder.md` 更新完成项与后续计划。
- 新增 `tests/test_bi_encoder.py` 自动化检查。

---
