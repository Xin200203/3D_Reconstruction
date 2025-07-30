#!/bin/bash

# 测试BiFusion模型 - epoch_49.pth - CA模式
# 自动设置环境并执行测试

echo "🚀 开始测试BiFusion模型 (CA模式)"
echo "=" * 50

# 设置环境变量
export PYTHONPATH=/home/nebula/xxy/ESAM:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# 配置文件和模型路径
CONFIG="configs/ESAM_CA/sv_bifusion_scannet200.py"
CHECKPOINT="/home/nebula/xxy/ESAM/work_dirs/enhanced_bifusion_debug/epoch_49.pth"
WORK_DIR="/home/nebula/xxy/ESAM/work_dirs/test_epoch49_ca"

echo "📂 配置文件: $CONFIG"
echo "🎯 模型文件: $CHECKPOINT"  
echo "💾 输出目录: $WORK_DIR"
echo "🔧 测试模式: Category-Agnostic (CA)"

# 检查文件是否存在
if [ ! -f "$CONFIG" ]; then
    echo "❌ 错误: 配置文件不存在 $CONFIG"
    exit 1
fi

if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ 错误: 模型文件不存在 $CHECKPOINT"
    exit 1
fi

# 创建输出目录
mkdir -p "$WORK_DIR"

echo ""
echo "🔍 开始测试..."

# 执行测试 - CA模式
python tools/test.py \
    "$CONFIG" \
    "$CHECKPOINT" \
    --cat-agnostic \
    --work-dir "$WORK_DIR" \
    --cfg-options test_evaluator.format_only=False

echo ""
echo "✅ 测试完成！"
echo "📊 结果保存在: $WORK_DIR"
echo "📝 查看详细结果:"
echo "   cat $WORK_DIR/*/scalars.json" 