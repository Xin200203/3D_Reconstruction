# ESAM 进阶训练指南 – Bi-Fusion Encoder & Time-Divided Transformer

> 本文档面向希望在 **EmbodiedSAM / ESAM** 框架中启用 _2D-3D Bi-Fusion 特征编码器_ 及 _跨帧 Time-Divided Transformer_ 的使用者，记录所需的最小修改与可选拓展，保持与原版配置完全兼容（即 **可按需开关**）。

---

## 1. 环境前提
1. 已按 `docs/installation.md` 配置 Python≥3.8、PyTorch、MinkowskiEngine、MMDetection3D 等依赖。
2. 追加依赖（requirements.txt 已列出，仅提醒）：
   ```bash
   pip install open_clip_torch==2.22
   ```
3. 数据集目录与 `.pkl` 索引已按 `docs/dataset_preparation.md` 准备完毕。

---

## 2. 配置文件层级回顾
ESAM 采用 **mmdet3d** 的 Config 体系：以 `configs/<variant>/<file>.py` 为入口，整体结构如下
```python
model = dict(
    voxel_size=0.02,
    backbone=...,           # 3D Sparse UNet
    pool=...,
    decoder=...,            # Query Decoder
    criterion=...,          # 损失
    # 新增两处可选模块 ↓↓↓
    bi_encoder=None,        # 👉 2D-3D 特征融合 (Bi-Fusion)
    clip_criterion=None,    # 👉 融合一致性正则 (可选)
)

train_pipeline = [...]

test_cfg = dict(
    merge_type='concat',    # Online / Offline 区分
    inscat_topk_insts=100,
    # 新增👇 跨帧 Transformer 的构造参数
    tformer_cfg=None,
)
```
若 **bi_encoder / tformer_cfg** 设为 `None` 即退化为原始 ESAM。

---

## 3. 启用 Bi-Fusion Encoder
### 3.1 修改 `model` 字典
```python
bi_encoder = dict(
    type='BiFusionEncoder',         # 已在 oneformer3d/__init__.py 注册
    clip_pretrained='openai',       # ViT-B/16 权重来源，可换
    voxel_size=0.02,               # 与全局一致即可
    freeze_blocks=2,               # 仅微调后 2 个 Transformer block
)

# 可选：CLIP 一致性正则
clip_criterion = dict(
    type='ClipConsCriterion',      # oneformer3d/bife_clip_loss.py
    loss_weight=0.05,              # λ_clip
)
```
⚠️ 同时移除 `img_backbone`（Bi-Fusion 内部已带 CLIP-ViT），否则会冲突。

### 3.2 修改数据管线
Bi-Fusion 需要 RGB 及相机内、外参：
```python
train_pipeline[  # and test_pipeline
    ...,
    dict(type='LoadAdjacentDataFromFile',
         with_img=True,
         with_cam=True,
         # 其余参数保持不变
    ),
    ...
]
```
管线会在 `batch_inputs_dict` 中填充：
```
points  : List[(N_i,6)]
imgs    : List[Tensor 3×H×W]
cam_info: List[dict {intrinsics, extrinsics}]
```
`mixformer3d.py` 已根据 `bi_encoder is not None` 自动切换到融合分支，无需更多改动。

---

## 4. 启用 Time-Divided Transformer（跨帧匹配）
仅适用于 **MV / Online** 推理配置；训练阶段可选是否联合监督。

### 4.1 `test_cfg` 中开启
```python
test_cfg = dict(
    merge_type='learnable_online',   # 确保调用 OnlineMerge
    inscat_topk_insts=100,
    tformer_cfg=dict(
        type='TimeDividedTransformer',
        d_model=256,
        nhead=8,
        num_layers=3,
        dropout=0.0,
    ),
)
```
`OnlineMerge` 检测到 `tformer_cfg` 后会实例化 Transformer 替代原有混合分数矩阵。

### 4.2 训练监督（可选）
若希望对跨帧注意力矩阵加入监督（`CrossFrameCriterion`）:
1. 确认 `oneformer3d/instance_criterion.py` 中已集成该 loss；
2. 在 `model.criterion` 对应 loss 权重里加入 `L_match`, `L_cons`。

---

## 5. 示例配置集
| 任务 | cfg 文件 | 关键改动 |
| ---- | -------- | -------- |
| ScanNet200-SV (Bi-Fusion) | `configs/ESAM/ESAM_sv_scannet200_bifusion.py` | Section 3.1 & 3.2 |
| ScanNet200-MV (Bi-Fusion + T-former Online) | `configs/ESAM_CA/ESAM_online_scannet200_CA_bifusion_tf.py` | Section 3 + 4 |

复制原 cfg 并按章节替换字段即可。

---

## 6. 训练 / 推理指令模板
```bash
# 单 GPU 训练
CUDA_VISIBLE_DEVICES=0 \
python tools/train.py \
    <cfg_path>.py \
    --work-dir work_dirs/$(basename <cfg_path>)

# 推理 / 评测
CUDA_VISIBLE_DEVICES=0 \
python tools/test.py \
    <cfg_path>.py \
    work_dirs/$(basename <cfg_path>)/epoch_*.pth \
    --work-dir work_dirs/$(basename <cfg_path>)
```
> 若使用多 GPU，可加 `--launcher pytorch` 或改用 slurm 分布式脚本，均与 MMEngine 保持一致。

---

## 7. 故障排查 Checklist
1. **内存不足**：降低批量、增大 `freeze_blocks`、或启用混合精度 (`--amp`)。
2. **找不到 `imgs` 键**：检查 `with_img=True`。确保数据集目录下有 RGB png 并在 meta 中记录路径。
3. **tformer 无效果**：确认 `merge_type='learnable_online'` 且数据加载为 **MV**（同一序列多帧）。
4. **CLIP 权重下载失败**：提前手动放到 `~/.cache/clip` 或设置环境变量 `OPENAI_CLIP_CACHE_PATH`。

---

> 更新于 2025-06-24
