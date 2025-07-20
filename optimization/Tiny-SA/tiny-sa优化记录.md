# Tiny-SA 模块优化记录

> 更新时间：7-20

---

## 1. 当前问题分析

**1.1 随机中心采样噪声**  
- 训练与推理阶段每次前向都会重新随机挑选中心点 (见 `oneformer3d/tiny_sa.py:L39-L43`)，导致输出分布在验证/测试时不稳定，表现为指标高方差甚至低于 baseline。  
- 对小 batch / 小 epoch 任务尤为敏感，随机性无法被统计平均。

**1.2 LayerNorm 复用引起的统计失衡**  
- 同一个 `LayerNorm` 同时作用于 MHSA 与 FFN 的输出 (`L91-L92`)，梯度来源混杂。  
- 导致 `γ/β` 参数难以兼容两种分布，训练不稳定。

**1.3 “硬复制”上采样导致细节丢失**  
- 现实现采用最近中心硬复制：每个点只继承单个中心特征，边界处出现阶梯跳变。  
- 对小物体、复杂几何或稠密区域产生过平滑和类别混淆。

---

## 2. 整体优化计划

| 阶段 | 目标 | 关键思路 | 预期收益 |
|------|------|----------|----------|
| **0. 错误修正** | 去除显著 Bug，恢复 baseline | 1) 随机中心采样改为 **SFP/FPS+缓存**；<br/>2) `LayerNorm` 拆分为 `norm1` + `norm2`（或 Pre-LN） | 消除方差 & 爆梯，指标回升到原基线 |
| **短期** | 引入连续可导插值 | 上采样改为 **kNN + 距离反比权重 (α=2, k=8)** | 指标≥baseline 且可见小幅提升 |
| **中期** | 细化插值结果 | 在 kNN 插值后叠加 `Linear+ReLU` 残差微调 | 额外提升 mIoU≈1‒2 pt |
| **长期探索** | 深度融合局部与全局 | 1) 两级 SA (0.25→0.5→1.0) 逐层上采；<br/>2) WeightNet 可学习核；<br/>3) Intr/Extr 误差建模 & Gaussian/GNN 传播 | 极限性能、细粒度边界保持 |

---

## 3. 具体实现规划

### 3.1 采样修正（错误修正阶段）
- **策略：SFP / FPS + 缓存**  
  1. 训练阶段：保留随机采样或改用 **Stochastic Farthest Point (SFP)**，增加多样性。  
  2. `eval()` 阶段：首次前向执行 Farthest Point Sampling (FPS) 并将 `idx_center` 缓存到 `self.register_buffer('fps_idx', ...)`，之后复用，保证推理稳定。  
  3. 选项：若需对比，亦可直接设定固定随机种子 `torch.manual_seed(seed)`。

### 3.2 LayerNorm 拆分（错误修正阶段）
- **实现**  
  ```python
  # __init__
  self.norm1 = nn.LayerNorm(dim)
  self.norm2 = nn.LayerNorm(dim)
  # forward
  center_feat = center_feat + self.norm1(updated_center)
  center_feat = center_feat + self.norm2(self.ffn(center_feat))
  ```
- **兼容**：保留旧 `self.norm` 以加载历史权重，再转移到新参数或初始化。

### 3.3 kNN + 反距离加权插值（短期阶段）
1. 在 forward 末尾替换：
   ```python
   # 原有
   nearest_center_idx = dist2.argmin(dim=0)
   output_feats = center_feat[nearest_center_idx]
   
   # 新实现
   knn_dist, knn_idx = dist2.topk(k, largest=False)  # k=8
   weight = 1 / (knn_dist + 1e-6) ** 2              # α=2
   weight = weight / weight.sum(-1, keepdim=True)
   output_feats = (center_feat[knn_idx] * weight.unsqueeze(-1)).sum(dim=1)
   ```
2. 复杂度：O(kN)，GPU 端 `gather` + `einsum` 即可。
3. k 与 α 作为可配置超参（默认 k=8, α=2）。

### 3.4 Residual Refinement（中期阶段）
- 在 `TinySAModule.__init__` 末尾新增：
  ```python
  self.post_mlp = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
  ```
- 前向完成插值后：
  ```python
  output_feats = output_feats + self.post_mlp(output_feats)
  ```
- 可选 bottleneck：dim→dim/2→dim。

### 3.5 评估与消融
1. **实验矩阵**  
   - Baseline、+固定采样、+kNN、+LayerNorm 拆分、+Residual。  
2. **指标**  
   - ScanNet200 mIoU、训练/验证方差、显存 & 时间开销。  
3. **可视化**  
   - 点云特征 TSNE、误分类边界可视化对比。

---

> 注：以上实现步骤按 *短期 → 中期* 依序进行，每一步都需单独验证并记录日志。长期探索方向待短中期方案稳定后开启。

---

## 4. 代码修改记录

| 日期 | 文件 | 行号/位置 | 变动概述 |
|------|------|-----------|----------|
| 7-20 | `oneformer3d/tiny_sa.py` | `__init__` | 拆分 `LayerNorm` → `norm1` / `norm2`；注册 `_fps_idx_cache` 缓存 |
| 7-20 | `oneformer3d/tiny_sa.py` | `forward` | 替换最近中心复制→kNN(8)反距离(α=2)加权插值，连续可导 |
| 7-20 | `configs/ESAM_CA/sv3d_tiny_sa_scannet200_ca.py` | `model.neck` | sample_ratio→0.05, radius→0.2, max_k→32 |
| 7-20 | `oneformer3d/tiny_sa.py` | `forward` | 添加邻居缺失保护逻辑；默认sample_ratio改0.05 |
| 7-20 | `oneformer3d/tiny_sa.py` | `_load_from_state_dict` | 自动将旧 `norm` 权重复制到 `norm1/2`，无需手工迁移 |
| 7-20 | `oneformer3d/tiny_sa.py` | `__init__` + `forward` | 新增 `post_mlp` 残差细化并在上采样后应用 |

> 提交分支：`tiny-sa`；commit msg：`fix: split LayerNorm & add FPS sampling cache`
