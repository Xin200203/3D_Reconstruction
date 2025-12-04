# 基于 DITR 的 DINOv2 特征融合整体规划（ESAM / 3D_Reconstruction）

> 目标：在 **单帧在线场景重建** 场景下，将 DITR 的核心思想（DINOv2 2D 特征 → 3D U‑Net 多尺度注入）引入到当前 ESAM 框架中，并且尽量复用现有 RN50‑BiFusion 管线。  
> 本文只做规划，不改代码；硬件假设为单卡 RTX 4090（24G）。

当前基线：  
- 模型：`configs/ESAM_CA/ESAM_sv_scannet200_CA.py`，骨干 `Res16UNet34C`，仅使用 3D 点云；  
- 任务：ScanNet200‑SV 单帧 **instance segmentation**；  
- 预训练：已从 `mask3d_scannet200.pth` 加载 3D 主干权重；  
- 评测日志：`work_dirs/ESAM_sv_scannet200_CA/20251014_194611/20251014_194611.log`。

> 单帧在线设置下，为保证 2D–3D 一一对应，很多几何增强不可用，模型容易欠拟合。**因此后续所有 DINO 方案都默认「在现有单帧预训练模型上微调」，不从头训练。**

---

## 阶段 0：基线确认 & 训练策略约定

**目标**：确认当前基线行为，明确后续实验一律从预训练权重出发。

- 使用 `ESAM_sv_scannet200_CA.py` 与现有数据，在单卡 4090 上：
  - 只启动 **一次完整训练+验证或近期已有的 best checkpoint 评测**，确认当前 `all_ap_50%` 数值范围；  
  - 记录训练 / 推理单 iter 显存占用与时间。  
- 训练策略约定：
  - 后续所有 DINO 实验均复用本配置的 3D 主干初始化（`load_from`），只在新的 2D 分支 / 注入层上随机初始化；  
  - 由于单帧数据增强受限，**不再进行从头训练** 的大规模实验，只做基于该基线的微调对比。

> 阶段测试：确认当前配置可稳定跑通，指标与历史 log 一致即可。

---

## 阶段 1：DINOv2 2D 特征离线提取与打包

**目标**：用 DINOv2 替代 RN50，建立与现有 `clip_pix` / `clip_feat_path` 兼容的 DINO 特征离线存储与打包流程，保持几何对齐。

### 1.1 模型与输入规范

- 模型：优先使用 **DINOv2 ViT‑L**；4090 24G 足够离线提取，batch size 可控制在 4–8。  
- 输入分辨率：
  - 第一版直接使用 **480×640** 原始 RGB（与深度一致），只做 ToTensor + DINO 归一化；  
  - 保证不裁剪、不拉伸，只允许**等比例整体缩放**；如后续需要接近 518×518 的 token 数，可改为等比例缩放到稍大分辨率，并在投影时使用实际 `(H_in, W_in)`。  
- 输出：最后一层 patch 特征图 `F ∈ ℝ^{C×H_p×W_p}`（如 C=768，`H_p×W_p` 由 ViT patch size 与输入尺寸决定）。

### 1.2 特征提取脚本（参考 `tools/extract_clip_rn50_features.py`）

规划新脚本 `tools/extract_dino_v2_features.py`（结构比照 RN50 版本）：

- 遍历 `data/scannet200-sv/2D/<scene>/color/*.jpg`；  
- 根据帧号采样策略与 `_add_clip_paths_to_pkl` 一致：  
  - SV：每 200 帧采样一帧；  
  - MV（如后续需要）：每 40 帧采样一帧。  
- 对每帧图像：
  - 读原图（480×640）；  
  - 预处理后送入 DINOv2；  
  - 取最后一层 patch 特征 → reshape 为 `Tensor(C, H_p, W_p)`。

**保存格式与路径约定**

- 输出路径完全复用 CLIP 约定，便于 `create_data.py` 与 `LoadClipFeature`：  
  - `2D/.../XXX.jpg` → `clip_feat/.../XXX.pt`；  
- 每个 `.pt` 文件直接保存一个 `Tensor(C, H_p, W_p)`（建议 fp16）；  
- `oneformer3d/loading.LoadClipFeature._load_single` 已支持「直接 tensor」形式，会将其作为 `clip_pix` 使用。

### 1.3 使用 `tools/create_data.py` 重新打包 info PKL

- 运行 `tools/create_data.py`，保持现有 dataset / root / out_dir 配置，并加上 `--pack-clip`：  
  - `_add_clip_paths_to_pkl` 会按 SV / MV 自动插入 `clip_feat_path` / `clip_feat_paths`；  
  - 由于路径结构与 RN50 一致，无需修改该脚本。  
- 获得新的 `*_oneformer3d_infos_{train,val}_clip.pkl`，后续训练 / 验证统一使用带 `_clip` 的版本。

**阶段 1 测试（轻量）**

- 随机检查若干场景：  
  - `.pt` 文件形状一致（如 `(768, H_p, W_p)`）；  
  - 对应 PKL 里的 `clip_feat_path(_s)` 确实存在；  
  - 单独跑一次 `LoadClipFeature`，确认 `results['clip_pix']` 是 Tensor，shape 与保存内容一致。

---

## 阶段 2：基于 DINOv2 的点级 2D–3D 映射（单尺度，前向验证为主）

**目标**：复用现有 BiFusion 2D‑3D 投影路径，用 DINO 特征代替 RN50，在不做长时间训练的前提下，确认几何投影与数值稳定性。

### 2.1 数据→模型链路复用

已存在的 CLIP 流程：

1. `LoadClipFeature`：从 `clip_feat_path` 读 `.pt` → `clip_pix`；  
2. `Det3DDataPreprocessor_`：在 `cam_info` 中附加 `clip_pix`；  
3. `BiFusionEncoder`：  
   - 使用 `_extract_pose_matrix` + `_transform_coordinates` 得到相机系点云 `xyz_cam`；  
   - 用 `unified_projection_and_sample` 将点投影到 `clip_pix` 特征图上，得到点级 `feat2d_raw (N×C_dino)` 和 `valid_mask`；  
   - 通过 `_ensure_precomp_adapter` 做 `C_dino→256`，再经 `Head` + `LiteFusionGate` 融合 2D/3D。

对于 DINO，只需：

- 将 `feat_space` 标记为 DINO 版本，确保 `_ensure_precomp_adapter` 的 `c_in` 使用 DINO 的通道数；  
- 保持 `auto_pose=False`，统一使用「`pose_centered` 是 C2W → 显式取逆为 W2C」的确定性投影路径（这在 9.2 文档与当前代码中已经约定）。

### 2.2 验证方式（避免耗时训练）

考虑到你只有单卡 4090，且不希望为检查 NaN/Inf 再跑一轮训练，这里只做**前向级别**的轻量验证：

- 从 `*_clip.pkl` 中选若干（如 10）样本构造一个小 dataloader；  
- 对每个样本：
  - 前向通过 `BiFusionEncoder`（或带 BiFusion 的完整模型）一次，记录：  
    - `valid_projection_mask` 的平均值（投影有效点比例）；  
    - `fusion_2d_ratio / fusion_3d_ratio` 等统计（若已实现）；  
  - 检查：  
    - 中间特征是否存在 NaN/Inf；  
    - 投影统计中 `depth_ok / boundary_ok` 是否在合理范围；  
    - 单次前向时间与显存是否在可接受范围内。

> 不进行「少量 epoch 训练观察 loss 曲线」这种耗时测试，只做静态前向检查与简单统计。

---

## 阶段 3：构建 DINO 2D FPN（金字塔）并与 Minkowski 网格对齐

**目标**：在 3D 体素网格上构建多尺度 DINO 特征金字塔，既保证**空间分辨率（stride）匹配**，也保证**通道维度的一致性**，为后续解码注入做准备。

### 3.1 点 → voxel 聚合（stride=1 层）

- 在 ESAM 的 3D 主干（`Res16UNet34C`）前端，体素化得到：  
  - `coords (N_voxel, 4)`，`features_3d (N_voxel, C3D)`；  
- 利用同一批点的 DINO 点级特征（来自阶段 2 的投影），按 voxel 归属进行聚合：  
  - 同一 voxel 内的多点 DINO 特征可用 **max‑pool** 或 **平均**；  
  - 得到 `features_2d_voxel (N_voxel, C_dino)`；  
- 构造 `x2d_s1 = ME.SparseTensor(features=features_2d_voxel, coordinates=coords, tensor_stride=1, coordinate_manager=cm)`。

> 这一层上，2D / 3D 已经在「空间坐标」上严格 1:1 对齐，但**通道数不同**（`C3D=96`，`C_dino≈768`），后续需要通过 1×1 投影对齐。

### 3.2 多尺度 DINO FPN（stride 匹配）

- 使用 Minkowski 的 max‑pool 构建多尺度 DINO 特征：  
  - `x2d_s2  = pool(x2d_s1)`（stride=2）  
  - `x2d_s4  = pool(x2d_s2)`（stride=4）  
  - `x2d_s8  = pool(x2d_s4)`（stride=8）  
  - `x2d_s16 = pool(x2d_s8)`（stride=16）  
- 使用与 3D 主干相同的 `coordinate_manager` 和 pool 参数，使得：  
  - `x3d_s2 / s4 / s8 / s16` 与 `x2d_s2 / s4 / s8 / s16` 在空间上严格对齐；  
  - `tensor_stride` 分别为 2/4/8/16。

### 3.3 通道维度的一致性检查

多尺度 FPN 不仅要保证 stride 对齐，还要保证后续注入时**通道可对齐**，规划如下：

- 每个尺度的 DINO FPN 层在注入前都会经过 1×1 投影到对应 decoder 通道数（在阶段 4 具体说明）；  
- 在本阶段测试时，需要检查：  
  - 原始 `x2d_s*` 的通道数统一为 `C_dino`；  
  - 规划的 1×1 投影输出通道与 `PLANES[4..7]` 一致（例如 256/128/96/96）。

**阶段 3 测试（前向检查）**

- 在一个样本上构造 `x2d_s1/2/4/8/16`，对比：  
  - 每层 `tensor_stride` 与对应 3D 层是否一致；  
  - 每层的 `coordinates` 数量与分布是否合理；  
  - `features_2d_voxel` / `x2d_s*` 是否无 NaN/Inf。  
- 简单检查 1×1 投影后的特征形状，确保「(N_voxel, C_proj)」与对应 decoder 输入通道匹配。

---

## 阶段 4：在 ESAM MinkUNet 解码器中做 DITR 风格多尺度注入

**目标**：在 `Res16UNet34C` 的 decoder 中，实现 DITR 风格的多尺度 DINO 注入，保持原有 concat skip 结构不变，并严格满足「stride 与通道」双重一致性。

### 4.1 注入位置与通道设计

以 `mink_unet.py` 中的 16→8 层为例：

```python
out = self.convtr4p16s2(out)
out = self.bntr4(out)
out = self.relu(out)          # 上采样分支: stride=8, C = PLANES[4]
out = me.cat(out, out_b3p8)   # 与 encoder skip 拼接
out = self.block5(out)
```

DITR 风格注入规划：

- 在 `convtr+BN+ReLU` 之后、`me.cat` 之前，对上采样分支 `out` 做：
  - `out = out + proj_2d_8(x2d_s8)`；  
  - 其中 `proj_2d_8` 是 1×1 Minkowski 卷积，`C_dino → PLANES[4]`；  
- 之后仍然 `me.cat(out, out_b3p8)`，`block5` 的输入通道数不变；  
- 其它尺度同理：  
  - 8→4：`proj_2d_4: C_dino→PLANES[5]`，用 `x2d_s4` 注入；  
  - 4→2：`proj_2d_2: C_dino→PLANES[6]`，用 `x2d_s2`；  
  - 2→1：`proj_2d_1: C_dino→PLANES[7]`，用 `x2d_s1`。

这样既保证：

- 空间上：使用相同 pooling / stride → `x2d_s*` 与上采样输出的坐标一一对应；  
- 通道上：1×1 投影后与上采样分支通道完全一致，可以直接加法。

### 4.2 实现要点（后续编码阶段遵循）

- 在 `Res16UNetBase` 或子类中增加 4 组投影层 `dino_proj_8/4/2/1`，通道设置见上；  
- forward 增加一个可选参数 `dino_feats`，包含 `x2d_s1/2/4/8/16`；  
- 在解码各层上采样输出后调用统一的 `inject_dino` 函数：  
  - 使用 `features_at_coordinates` 将对应 `x2d_s*` 采样到当前 `out.coordinates`；  
  - 通过 1×1 投影后直接加到 `out.features` 上；  
  - 对于无 DINO 或投影无效的 voxel，用 0 特征回退，不改变 3D 主干行为。

### 4.3 阶段 4 测试（前向+少量评测）

考虑时间成本，这一阶段仍然以前向检查为主，只在最终方案上跑一次完整评测：

1. **前向检查（不开训练）**  
   - 使用少量样本（如 10 帧）做前向：  
     - 检查每个 decoder 层注入前后通道数是否一致；  
     - 检查有无 NaN/Inf；  
     - 确认 `features_at_coordinates` 不会大量返回 0（除非本来无 DINO）。  
2. **完整评测（对比基线）**  
   - 选定一个「最终配置」（例如 DINO BiFusion + FPN 注入），在加载原基线预训练权重的基础上，进行**短周期微调**（epochs 数远小于 128，学习率相应调小），并在 ScanNet200‑SV 验证集上跑一次完整评测；  
   - 只与「原 ESAM_sv_scannet200_CA 基线」对比 instance AP，不再与 RN50 BiFusion 等历史优化做系统对比。

---

## 阶段 5：系统对比与简化消融（仅对比基线）

**目标**：在尽量少的实验次数下，验证 DINO 注入设计是否相比当前 3D‑only 基线带来提升。

建议的最小对比配置：

1. **Baseline**：  
   - 仅 3D，`ESAM_sv_scannet200_CA.py` 原配置（当前基线）。  

2. **DINO‑BiFusion（单尺度）**：  
   - 仅使用阶段 2 的 DINO BiFusion（替换 RN50）；  
   - 不在 MinkUNet decoder 中注入，只在 BiFusion 路径融合 2D/3D。  

3. **DINO‑FPN‑Inject（多尺度）**：  
   - 在 BiFusion 提供的点级 DINO 基础上，构建阶段 3 的 FPN，并在阶段 4 位置注入；  
   - 这是完整 DITR 风格的版本。

每个配置：

- 统一从相同的 3D 预训练权重出发；  
- 使用相同的训练 schedule（可以是基线 schedule 的缩短版，例如 128→64 或更少），方便对比；  
- 评测统一使用 ScanNet200‑SV 的 instance AP 指标。

> 最终结论重点回答两个问题：  
> 1）在单帧在线场景且增强受限的前提下，**只用 3D 预训练 + DINO 注入** 能否显著优于「仅 3D」？  
> 2）多尺度 FPN 注入是否明显优于仅 BiFusion 的单尺度方案。  

至此，「基于 DITR 特征融合」的规划路径为：  
**基线确认 → DINO 特征离线提取与打包 → 单尺度 BiFusion 前向验证 → 3D FPN 金字塔构建（stride+通道一致） → MinkUNet 解码多尺度注入 → 与基线的简化对比实验**。  
每一阶段都可单独验证、逐步落地，且在单卡 4090 条件下控制了计算成本。 
