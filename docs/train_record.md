# ESAM-ScanNet200 训练记录

> 更新时间：2025-07-07

## 1. 任务与数据

| 任务类型 | 类别设置 | 数据源 | 备注 |
|-----------|----------|--------|------|
| Category-Agnostic (CA) 实例分割 | 1 类 (object) | ScanNet200 | 不区分细类，只预测 instance mask |
| Multi-Class (MC) 实例分割 | 198 类 | ScanNet200 | 与官方 ScanNet200 类别一致 |

采用官方 `scannet200_infos_*.pkl` 及超体素 / CLIP 特征等 8 个辅助 pkl（见《数据迁移清单》）。

## 2. 配置文件层级与承接关系

```
CA-Pipeline
 ├─ sv3d_scannet200_ca.py          # 纯 3D
 ├─ sv3d_tiny_sa_scannet200_ca.py  # 纯 3D + TinySA neck，`load_from = sv3d_scannet200_ca`
 ├─ sv_bifusion_scannet200_ca.py   # BiFusion，`load_from = sv3d_tiny_sa_scannet200_ca`
 └─ mv_scannet200_ca.py            # Multi-view 合并，`load_from = sv_bifusion_scannet200_ca`

MC-Pipeline
 ├─ sv3d_scannet200.py             # 纯 3D
 ├─ sv3d_tiny_sa_scannet200.py     # 纯 3D + TinySA，`load_from = sv3d_scannet200`
 ├─ sv_bifusion_scannet200.py      # BiFusion，`load_from = sv3d_tiny_sa_scannet200`
 └─ mv_scannet200.py               # Multi-view 合并，`load_from = sv_bifusion_scannet200`
```

### 文件命名规则
```
<stage>_<dataset>[_ca].py

stage:
  sv3d      ‑ 单帧纯 3D（sparse voxel）
  sv3d_tiny_sa ‑ 带 TinySA neck 的纯 3D
  sv_bifusion  ‑ BiFusion（点云+图像）
  mv           ‑ Multi-view Online/Offline 合并
```

### work_dir 约定
```
./work_dirs/
  sv3d_scannet200_ca/
  sv3d_tiny_sa_scannet200_ca/
  sv_bifusion_scannet200_ca/
  mv_scannet200_ca/
  sv3d_scannet200/
  sv3d_tiny_sa_scannet200/
  sv_bifusion_scannet200/
  mv_scannet200/
```

> 训练、评测脚本示例参见 `docs/run.sh`（自行编写）。

## 3. 训练流程

1. **CA-Pipeline**
   ```bash
   # 1) 纯 3D
   python tools/train.py configs/sv3d_scannet200_ca.py --work-dir work_dirs/sv3d_scannet200_ca

   # 2) TinySA（加载上一步）
   python tools/train.py configs/sv3d_tiny_sa_scannet200_ca.py --resume-from work_dirs/sv3d_scannet200_ca/latest.pth --work-dir work_dirs/sv3d_tiny_sa_scannet200_ca

   # 3) BiFusion（加载 TinySA）
   python tools/train.py configs/sv_bifusion_scannet200_ca.py --resume-from work_dirs/sv3d_tiny_sa_scannet200_ca/latest.pth --work-dir work_dirs/sv_bifusion_scannet200_ca

   # 4) Multi-view 合并（离线 / learnable_online）
   python tools/test.py configs/mv_scannet200_ca.py work_dirs/sv_bifusion_scannet200_ca/best.pth --work-dir work_dirs/mv_scannet200_ca
   ```

2. **MC-Pipeline** 同理，将 `*_ca` 去掉即可。

## 4. 数据迁移清单（服务器）
```
scannet200_infos_train.pkl
scannet200_infos_val.pkl
scannet200_infos_test.pkl
scannet200_labels.pkl
superpoints_train.pkl
superpoints_val.pkl
clip_features_train.pkl
clip_features_val.pkl
esam_pretrain_scannet200.pth
```

## 5. Git 版本管理

```
git init
# 按 .gitignore 过滤大文件
# ... push to GitHub 详见上一回复 ...
```

---
如需添加新实验，请按照以上命名-继承规范新建 config 并补充 `load_from` 路径及 work_dir 名称，然后在本文件追加记录。
