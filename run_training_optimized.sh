#!/bin/bash

# 进入项目目录
cd /home/nebula/xxy/ESAM

# 激活环境
conda activate ESAM

# 设置显存优化环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128
export CUDA_LAUNCH_BLOCKING=0
export OMP_NUM_THREADS=12

# 清理GPU显存
python -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print('GPU缓存已清理')
    print(f'可用显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"

# 启动训练（带显存监控）
echo "开始训练..."
python tools/train.py configs/ESAM_CA/sv_bifusion_scannet200.py

echo "训练完成或中断" 