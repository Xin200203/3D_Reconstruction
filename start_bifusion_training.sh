#!/bin/bash
# 🚀 ESAM BiFusion Stage 2 优化训练启动脚本
# 
# 功能特性:
# ✅ Stage 1: CLIP损失(0.1) + 空间一致性(0.02) + 5轮评估
# ✅ Stage 2: Enhanced监控Hook + BiFusion统计 + 梯度健康分析
# ✅ 完整的TensorBoard可视化支持
# ✅ 自动故障恢复和检查点管理

set -e  # 遇到错误立即退出

# 📍 环境配置
export WORKSPACE_ROOT="/home/nebula/xxy/ESAM"
export CONDA_ENV="ESAM"
export CONFIG_FILE="configs/ESAM_CA/sv_bifusion_scannet200.py"
export WORK_DIR="work_dirs/bifusion_stage2_optimized"

echo "🚀 ESAM BiFusion Stage 2 优化训练启动"
echo "================================================================================"
echo "📍 工作空间: $WORKSPACE_ROOT"
echo "🐍 Conda环境: $CONDA_ENV" 
echo "⚙️  配置文件: $CONFIG_FILE"
echo "💾 工作目录: $WORK_DIR"
echo "================================================================================"

# 🔧 环境准备
cd $WORKSPACE_ROOT
echo "✅ 切换到工作目录: $(pwd)"

# 激活conda环境
source /home/nebula/anaconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV
echo "✅ 激活Conda环境: $CONDA_ENV"

# 设置Python路径
export PYTHONPATH=$PYTHONPATH:$(pwd)
echo "✅ 设置PYTHONPATH: $PYTHONPATH"

# 🧪 预训练验证 (可选 - 注释掉跳过)
echo ""
echo "🧪 执行预训练验证..."
python stage2_validation.py
echo "✅ 预训练验证完成"
echo ""

# 🎯 训练配置优化
echo "🎯 应用Stage 2优化配置..."

# 创建工作目录
mkdir -p $WORK_DIR
echo "✅ 创建工作目录: $WORK_DIR"

# 🔥 核心训练命令
echo "🔥 开始BiFusion优化训练..."
echo ""

# 方式1: 标准训练 (推荐)
python tools/train.py \
    $CONFIG_FILE \
    --work-dir $WORK_DIR \
    --cfg-options \
        default_hooks.logger.interval=50 \
        default_hooks.checkpoint.interval=10 \

# 如果需要从检查点恢复，取消注释下面的命令并注释上面的命令
# python tools/train.py \
#     $CONFIG_FILE \
#     --work-dir $WORK_DIR \
#     --resume-from $WORK_DIR/latest.pth \
#     --cfg-options \
#         train_cfg.max_epochs=128 \
#         train_cfg.val_interval=5 \
#         model.bi_encoder._collect_fusion_stats=True

echo ""
echo "🎉 训练启动完成!"
echo "📊 监控信息:"
echo "   - 训练日志: $WORK_DIR/$(date +%Y%m%d_%H%M%S).log"
echo "   - TensorBoard: tensorboard --logdir $WORK_DIR"
echo "   - 检查点: $WORK_DIR/*.pth"
echo ""
echo "🔍 关键监控指标:"
echo "   - BiFusion gate statistics (融合门统计)"
echo "   - Valid projection rate (投影有效率)"
echo "   - Gradient health monitor (梯度健康度)"
echo "   - Loss components breakdown (损失分解)"
echo ""
echo "⚠️  训练过程中请关注:"
echo "   1. 梯度范数保持在合理范围 (总范数 < 10.0)"
echo "   2. 投影有效率维持在 80%+ "
echo "   3. 融合门权重分配合理 (2D:3D 约 6:4)"
echo "   4. 损失收敛稳定，无NaN异常"
