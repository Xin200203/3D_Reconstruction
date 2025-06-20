记录time_divided_transformer进行的优化操作

### Step-1  创建 `oneformer3d/time_divided_transformer.py`
1. 实现 `_GeomBiasAttnLayer` ：几何偏置交叉注意力 + GRU + Self-Attn + FFN。
2. 实现 `TimeDividedTransformer` 支持可配置层数与多头。
3. 在 `oneformer3d/__init__.py` 中导出，注册到 `mmdet3d.registry.MODELS`。

> commit: step-1 scaffold 完成，可通过随机张量 quick-test。