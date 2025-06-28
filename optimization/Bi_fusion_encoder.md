## 单帧 RGB-D → 点云实例分割完整流程（含 Transformer Decoder）

> **管线总览**
> `RGB-D 帧  →  2D-VFM 分支  ➜  F²ᴰ`
> `                 ↘ 透视投影             ↘`
> `                                 F²ᴰ_pts  ┐`
> `点云  → Tiny-SA U-Net  →  F³ᴰ ───────┼─ Geo-PE 拼接 → MLP 压缩`
> `                                          └─ Fusion-Gate → F̄_fusion`
> `F̄_fusion + mask → grouping → 初始实例  → Transformer Decoder → 细化掩码 / 类别 / 属性`

下文对 **每一个箭头** 交代输入、计算、输出与实现要点。

---

### 1. 2D 语义分支（VFM → F²ᴰ）

1. **特征提取**

   * ViT-B/16 backbone（冻结）输出 patch grid `F_patch ∈ ℝ^{H/16×W/16×512}`。
   * 去 prompt、去 CAVP。
2. **上采样**

   * `PixelShuffle ×2` 得 `H/8×W/8×512`，再 `Conv1×1(512→256)` → `F²ᴰ_up`。
3. **线性投影**

   * 对每个像素 (u,v) 以双线性采样方式读出其最近的上采样特征 → `F²ᴰ_px(u,v)`。

---

### 2. 深度投影与点索引

```python
x_c = (u-Kcx)*d/Kfx;   y_c = (v-Kcy)*d/Kfy;   z_c = d
p_w = Tcw · [x_c, y_c, z_c, 1]^T
idx(u,v) ← i
```

得到点云坐标 `points ∈ ℝ^{N_p×3}` 与像素-点索引表。

---

### 3. 3D 几何分支（Tiny-SA U-Net → F³ᴰ）

1. **稀疏体素化**：voxel = 4 cm，`MinkowskiEngine` 建立稀疏张量。
2. **U-Net**：4 level 编码/解码，输出体素特征 128d。
3. **Voxel → Point**：三线性插值到每点 → `F³ᴰ ∈ ℝ^{N_p×128}`。
4. **Tiny-SA×2**：8-head window attention (32 邻点上限)，Residual+FFN，保持 128d，产出 `F³ᴰ_SA`。

---

### 4. **Geo-PE 构造（修订）**

当前实现位于 `oneformer3d/bi_fusion_encoder.py::build_geo_pe`，维度拆解如下

| 组成 | 维度 | 说明 |
|------|------|------|
| xyz 世界坐标 | 3 | \(x,y,z\)（米）|
| sin/cos 周期编码 | 48 | 对每轴取 *8* 个频率 \(2^k\pi, k=0..7)\)，共 \(3×2×8=48\) |
| 物体 bbox size | 3 | w,h,l，单帧默认为 0，留接口 |
| Δpose (R6+t3) | 9 | 相对于上一帧的位姿差；单帧推理时为 0 |
| height | 1 | z 方向高度 |
| **总计** | **64** | 与源码一致 |

随后经 `MLP_pe: 64→32`（ReLU）压缩，供 2D/3D 分支拼接。

---

### 5. **特征拼接与压缩**

```python
F2D_ = Linear_cat2d(torch.cat([F2D_px[idx], PE], -1))   # (N_p, 96)
F3D_ = Linear_cat3d(torch.cat([F3D_SA, PE],   -1))      # (N_p, 96)
```

---

### 6. **Fusion-Gate-Lite**

```python
g = sigmoid(MLP_g(torch.cat([F2D_,F3D_], -1)))   # (N_p, 96)
F̄ = g * F2D_ + (1-g) * F3D_                     # (N_p, 96)
```

* `MLP_g`: 192→64→96，ReLU。
* 无 CAVP，gate 仍学习到由深度有效性、纹理清晰度驱动的自适应权重。

---

### 7. **2D *Depth-Aware* 点筛选（修订）**

文档原称"2D 掩码-Aware"，但在当前代码路径中**并未**依赖外部 2D 实例分割模型；真正做的是：

1. 通过 `build_uv_index()` 投影每个三维点到像素 (u,v)，并检查
   * z>0（位于相机前方）；
   * 0≤u<W, 0≤v<H（在视场内）。
2. 若投影有效 (`valid=True`) 即执行双线性采样，否则用零向量占位。

因此该步骤更准确的称呼应为 **"Depth/Frustum Mask 过滤"**，而非依赖 2D Mask-RCNN 结果。

> 后续若要融合 2D 检测掩码，可在本过滤基础上再叠加像素级 mask。

---

### 8. **Transformer Decoder（与源码对齐）**

对应文件：`oneformer3d/query_decoder.py`

| 层次 | 主要组件 |
|------|----------|
| L 层循环（默认 6）| Cross-Attn(Queries←Points) → Self-Attn(Queries) → FFN |
| 头部 | `num_heads=8`，多头缩放 √d |
| Query 初始化 | 训练时由 `_select_queries()` 在 superpoint 特征中随机采样 / 全选；推理时 Query=全点特征 |
| Mask 预测 | `pred_mask = Q ⋅ mask_feats`（点线性）；sigmoid 后可做 attn mask |
| 分类头 | Instance cls（+可选 semantic cls）|

与 ESAM 原文相比：
* **Self-Attn/FFN 顺序一致**；
* 额外支持 *iterative prediction* 与 *attention mask* 收敛策略；
* 未显式采用 point–query 距离 bias，而是直接点乘 + learnable proj。

---

### 9. **Grouping & Instance Init（修订）**

在 `mixformer3d._select_queries()` 与后续 Loss 函数中，实例初始化流程为：

1. 将点聚合为 **superpoint**（离线预计算的语义一致 + 空间连通单元）；
2. 每个 superpoint 作为候选 Query；训练时以 `query_thr` 随机下采样；
3. Decoder 输出的 `pred_mask` 用于比对 GT superpoint mask 计算 BCE + Dice；
4. 若启用 OnlineMerge（推理流），解码器输出 + 置信度通过 `OnlineMerge.merge()` 随时间融合。

因此文档中关于 "DBSCAN 聚类" 的描述与现实现不符，已删除，并明确 **superpoint 驱动** 的初始化逻辑。

---

### 10. **数据流小结**

1. **RGB-D →** VFM 特征 $F^{2D}_{px}$
2. **→** 投影 → 点索引 $idx$
3. **→** Tiny-SA U-Net → $F^{3D}_{SA}$
4. **+ Geo-PE →** 拼接 & 压缩 → $F2D_,F3D_$
5. **→ Fusion-Gate →** 融合点特征 $F̄$
6. **+ 2D mask →** 点抽样 → 初始实例 Query
7. **→ Transformer Decoder →** 语义 & 掩码 & pose
8. **→ Grouping →** 细化实例 & Memory 写入

这样，一个独立 RGB-D 帧即可完成**高质量点云实例分割 + 鲁棒特征存储**，为后续跨帧 Slot-Transformer 匹配奠定统一 feature 空间。
