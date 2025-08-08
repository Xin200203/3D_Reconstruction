# ESAM ScanNet数据流完整分析报告

## 1. ScanNet数据在传入BiFusion编码器时的形状和坐标系

### 1.1 ESAM数据预处理完整流程（基于官方文档）

#### 阶段1: 原始ScanNet数据结构
```
scannet200-sv/
├── 2D/           # RGB图像 (640×480) + 相机内参/外参
│   └── scenexxxx_xx/
│       ├── color/     # RGB图像序列
│       ├── depth/     # 深度图序列  
│       ├── intrinsic/ # 相机内参
│       └── pose/      # 相机外参(姿态)
├── 3D/           # 原始3D扫描数据
│   └── scenexxxx_xx/
│       └── scenexxxx_xx_vh_clean_2.ply  # 点云网格
└── meta_data/    # 训练/验证split文件
```

#### 阶段2: load_scannet_sv_data_v2.py数据处理
**关键处理步骤：**
1. **点云提取：** 从.ply文件提取点云，转换为DEPTH坐标系
2. **SAM实例分割：** 对RGB图像运行SAM生成2D实例掩码
3. **坐标对齐：** 应用axis_align_matrix进行坐标轴对齐
4. **多帧融合：** 整合多视角信息生成单一点云表示

**生成的数据结构：**
```
scannet200-sv/
├── points/                    # 点云数据 (.bin格式, DEPTH坐标系)
│   └── scenexxxx_xx_xx.bin   # 每帧点云 [N, 6] (xyz+rgb)
├── instance_mask/             # SAM生成的实例分割掩码
├── semantic_mask/             # 语义分割标注
├── super_points/              # 超点聚类结果
├── pose_centered/             # 相机姿态数据
│   └── scenexxxx_xx/
│       └── xx.npy            # 4×4变换矩阵
└── scannet_sv_instance_data/  # SAM处理的实例数据
    ├── scenexxxx_xx_xx_vert.npy           # 顶点坐标
    ├── scenexxxx_xx_xx_ins_label.npy      # 实例标签
    ├── scenexxxx_xx_xx_sem_label.npy      # 语义标签
    └── scenexxxx_xx_xx_axis_align_matrix.npy  # 轴对齐矩阵
```

#### 阶段3: create_data.py生成训练数据索引
```python
# tools/create_data.py scannet200_sv
# 生成最终的训练数据索引
└── scannet200_sv_oneformer3d_infos_train.pkl
```

### 1.2 传入BiFusion时的数据形状分析

现在让我检查实际的数据文件：
