下面给出 **EmbodiedSAM（ESAM）** 在 *在线、流式* RGB-D 场景上的完整 **pipeline** 以及数据在各模块之间的流动路径。为了方便他人阅读代码并复现，这里把“输入→中间表示→输出”逐级展开，并标注每一步所用的特征 / 张量及其去向。

---

### 0. 时刻 t 的输入

| 数据                           | 维度        | 备注           |
| ---------------------------- | --------- | ------------ |
| 彩色图像 **I<sub>t</sub>**       | 3 × H × W | RGB          |
| 深度图 / 投影点云 **P<sub>t</sub>** | N × 3     | 由深度 + 相机位姿投影 |

---

### 1. **SAM 2D 掩码生成**

1. **Segment-Anything Model** 在 **I<sub>t</sub>** 上一次性产生 *M* 个 2D instance masks **M<sub>t</sub><sup>2d</sup>**。
2. 每个 2D mask 通过像素-深度对应，映射到点云 P<sub>t</sub>，得到 **super-point 索引 S (N × 1)** —— 也就是“哪一个点属于哪一个 2D mask”。

数据流 → *2D mask* → *super-point 标签*。

---

### 2. **时序 3D 特征提取**

1. 把点云 **P<sub>t</sub>** voxelize 后送入 **Temporal-aware 3D Sparse U-Net**（带 Memory Adapter）。
2. 输出 **点特征 F<sub>P</sub> (N × C)**；历史帧特征通过 Adapter 注入，实现跨帧时序一致性。

数据流 → *P<sub>t</sub>* → *F<sub>P</sub>*。

---

### 3. **几何感知 Query Lifting**

> **目标：** 将每个 2D mask “抬升”成 **3D Query**，同时保留几何形状信息。

1. **几何感知 pooling**

   * 对每个 super-point P<sup>i</sup>，用归一化坐标 p<sub>rj</sub> 计算

     * `z_local = MLP(p_rj)`
     * `z_global = max_pool(z_local)`
   * 用 **Sigmoid-MLP** 预测点权 w<sub>j</sub>，再作加权平均：
     `F_S^i = mean(F_P^i * w) + z_global`
     → 得到 **super-point 特征 F<sub>S</sub> (M × C)**。
2. 从 F<sub>S</sub> 随机抽取(0.5\~1)×M 个作为 **初始 Query Q<sub>0</sub>**。

数据流：F<sub>P</sub> + S → F<sub>S</sub> → Q<sub>0</sub>。

---

### 4. **Dual-level Query Decoder（迭代 L = 3 次）**

| 步骤                             | 输入 / 输出                                     | 说明                                                                             |
| ------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------ |
| 4.1 **Masked Cross-Attention** | Q<sub>l</sub>, F<sub>S</sub>, A<sub>l</sub> | 先用 super-point 特征做交叉注意；A<sub>l</sub> 来自上一轮预测 mask；未被 mask 的 token 权重正常，其他置 −∞。 |
| 4.2 **Self-Attention + FFN**   | —                                           | 得到 Q̂<sub>l</sub> → Q<sub>l+1</sub>                                            |
| 4.3 **Point-mask 生成**          | `M_l = σ(Linear(Q_l) · F_P^T) > φ`          | 用 **点特征 F<sub>P</sub>** 生成细粒度 **M<sub>l</sub><sup>cur</sup> (N × 1)**          |
| 4.4 **Mask 投票**                | point → super-point                         | 将 M<sub>l</sub> pool 回 super-point，产生下一轮 A<sub>l+1</sub>                       |

循环 3 次得到 **精细 3D masks M<sub>t</sub><sup>cur</sup>** 与 **最终 Query Q<sub>3</sub>**。随后做 NMS 去重。

---

### 5. **Query-based 高效在线合并**

> **核心：** 用 **固定长度向量** 而非逐点距离，快速判定“当前 mask 是否与历史实例同属一物”。

1. **从 Query 生成三类表征**

   * **几何盒 B**：以 super-point 中心 c<sub>i</sub> 为基准，MLP 回归轴对齐 3D box；用于 IoU<sub>box</sub>。
   * **对比特征 f**：MLP→L2 归一化；跨帧正负对比学习。
   * **语义分布 S**：MLP 输出 K-类 soft label（可用 Semantic-SAM 时直接继承）。
2. **构建相似度矩阵**

   $$
   C = \text{IoU}(B^{pre}, B^{cur}) + f^{pre}\!\cdot\!f^{curT} + S^{pre}\!\cdot\!S^{curT}
   $$

   小于阈值 ε 的元素置 −∞。
3. **匈牙利匹配 + 合并**

   * **匹配成功**：将新 mask 并到历史实例，掩码并集；B/f/S 做加权平均 (n,n+1)。
   * **匹配失败**：注册为新实例。
4. **更新 Memory：** 得到 **M<sub>t</sub><sup>pre</sup>** 供下一帧使用。

整个合并过程完全矩阵化，实现 **80 ms/帧** 的 3D 部分推理（FastSAM 时全流程≈10 FPS）。

---

### 6. 训练信号与数据流

| Loss                                   | 监督目标          | 依赖张量                           | 作用         |
| -------------------------------------- | ------------- | ------------------------------ | ---------- |
| **L<sub>cls</sub>**                    | 前景 / 背景       | Q                              | 构建有效 Query |
| **L<sub>bce</sub> + L<sub>dice</sub>** | 3D point-mask | M<sub>l</sub>                  | 掩码精度       |
| **L<sub>iou</sub>**                    | 3D box        | B                              | 强化几何       |
| **L<sub>sem</sub>**                    | 语义分类          | S                              | 语义一致       |
| **L<sub>cont</sub>**                   | 对比约束          | f<sub>t</sub>, f<sub>t±1</sub> | 提升实例判别     |

多任务总损 L = L<sub>camera</sub> + L<sub>depth</sub> + L<sub>pmap</sub> + λ L<sub>track</sub>（纸中 λ=0.05）。训练时对 RGB-D **流** 随机采样 2-24 帧，每批总帧数 48；A100×64 训练 9 天。

---

## **端到端数据流总览（简明版）**

```
RGB-D Frame ─┐
             ├─ SAM → 2D Masks ─┐
             │                  │    (pixel ↔ depth)
             │                  ├─ super-point index S
             │                  │
             │                  └─┐
Sparse 3D U-Net → Point feats FP ─┼─ Geometric-aware Pooling → F_S → Q0
                                  └─ (FP, S)

Q0 → Dual-level Decoder (×3) → M_t^cur , Q3
                                    │
          (B, f, S) ← Heads ←───────┘
                                    │
Memory M_{t-1} ── similarity C ── bipartite match/merge ──► M_t   (更新)
```

---

### 结论

EmbodiedSAM 通过“**2D mask ⇨ 3D Query ⇨ 迭代细化 ⇨ 向量化合并**”这一流水线，实现了：

* **在线**：逐帧更新，不依赖未来帧。
* **实时**：矩阵化合并将原本 >1 s 的几何匹配降到 \~80 ms。
* **细粒度**：点级掩码由点特征直接预测，多轮 refinement 消除 2D-→3D 投影误差。
* **通用性**：Query 表征（B, f, S）抽象，可与任何下游 3D 任务或 BA 优化衔接。
