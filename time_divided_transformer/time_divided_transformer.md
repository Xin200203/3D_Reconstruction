记录time_divided_transformer进行的优化操作

### Step-1  创建 `oneformer3d/time_divided_transformer.py`
1. 实现 `_GeomBiasAttnLayer` ：几何偏置交叉注意力 + GRU + Self-Attn + FFN。
2. 实现 `TimeDividedTransformer` 支持可配置层数与多头。
3. 在 `oneformer3d/__init__.py` 中导出，注册到 `mmdet3d.registry.MODELS`。

> commit: step-1 scaffold 完成，可通过随机张量 quick-test。

### Step-3  集成到 `instance_merge.OnlineMerge`
1. `OnlineMerge.__init__` 新增 `tformer_cfg`，可由外部 cfg 构建 `TimeDividedTransformer`。
2. 在 `merge()` 内，当 `self.tformer` 不为空时：
   • 构造 9-维几何向量 `[xyz, sin(xyz), size]`
   • 调用 `self.tformer.forward` 生成 `attn_mat` 作为匹配分数。
   • 保留旧逻辑作为 fallback。
3. 记录 `self.cur_bboxes` 以便构造 Memory 几何向量。

> commit: step-3 完成，OnlineMerge 已可选择使用跨帧 Transformer 进行匹配。