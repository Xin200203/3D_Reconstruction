#!/usr/bin/env python3
"""
快速配置测试脚本
验证修复后的配置是否能正常初始化
"""

import os
import sys

# 设置环境
os.chdir('/home/nebula/xxy/ESAM')
sys.path.insert(0, os.getcwd())

def test_config_loading():
    """测试配置文件加载"""
    print("🧪 测试配置文件加载...")
    
    try:
        from mmengine.config import Config
        
        config_path = 'configs/ESAM_CA/sv_bifusion_scannet200.py'
        print(f"📁 加载配置文件: {config_path}")
        
        cfg = Config.fromfile(config_path)
        print("✅ 配置文件加载成功")
        
        # 检查关键配置
        print("🔍 检查关键配置:")
        print(f"   模型类型: {cfg.model.type}")
        print(f"   BiFusion编码器: {cfg.model.bi_encoder.type}")
        print(f"   训练epochs: {cfg.train_cfg.max_epochs}")
        print(f"   评估间隔: {cfg.train_cfg.val_interval}")
        
        # 检查可视化后端
        print(f"   可视化后端: {[backend.type for backend in cfg.vis_backends]}")
        
        # 检查自定义Hook
        custom_hooks = [hook.type for hook in cfg.custom_hooks]
        print(f"   自定义Hook: {custom_hooks}")
        
        return True
        
    except Exception as e:
        print(f"❌ 配置文件测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_import():
    """测试模型相关导入"""
    print("\n🧪 测试模型导入...")
    
    try:
        # 测试核心组件导入
        from oneformer3d.enhanced_training_hook import EnhancedTrainingHook
        print("✅ EnhancedTrainingHook 导入成功")
        
        from oneformer3d.auxiliary_loss import SpatialConsistencyLoss
        print("✅ SpatialConsistencyLoss 导入成功")
        
        return True
        
    except Exception as e:
        print(f"❌ 模型导入测试失败: {str(e)}")
        return False

def main():
    """主测试流程"""
    print("🚀 ESAM BiFusion配置修复验证")
    print("=" * 50)
    
    # 检查环境
    print(f"📍 当前目录: {os.getcwd()}")
    print(f"🐍 Python版本: {sys.version}")
    
    # 执行测试
    config_ok = test_config_loading()
    import_ok = test_model_import()
    
    print("\n" + "=" * 50)
    if config_ok and import_ok:
        print("✅ 所有测试通过！配置修复成功")
        print("🎯 现在可以开始训练了")
        print("\n💡 启动训练:")
        print("   ./start_bifusion_training.sh")
        print("\n📝 注意: TensorBoard已禁用以解决Python 3.8兼容性问题")
        print("   训练日志将保存在文本文件中，Enhanced Hook仍会输出详细统计")
    else:
        print("❌ 测试失败，需要进一步排查问题")

if __name__ == "__main__":
    main()
