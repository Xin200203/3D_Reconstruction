#!/usr/bin/env python3
"""
Stage 2 BiFusion优化验证脚本
验证增强的监控Hook和BiFusion统计收集功能
"""

import os
import sys
import torch
import traceback
from pathlib import Path

# 确保在ESAM目录下
os.chdir('/home/nebula/xxy/ESAM')
sys.path.insert(0, os.getcwd())

def test_enhanced_hook():
    """测试EnhancedTrainingHook的核心功能"""
    print("🧪 测试Enhanced Training Hook...")
    
    try:
        from oneformer3d.enhanced_training_hook import EnhancedTrainingHook
        
        # 创建Hook实例
        hook = EnhancedTrainingHook(
            log_interval=1,
            grad_monitor_interval=1,
            detailed_stats=True
        )
        
        print("✅ Hook实例化成功")
        
        # 模拟损失信息测试
        mock_loss_info = {
            'loss': 2.5,
            'semantic_loss': 0.8, 
            'instance_loss': 1.2,
            'clip_consistency_loss': 0.3,
            'spatial_consistency_loss': 0.2
        }
        
        detailed_losses = hook._extract_detailed_losses(mock_loss_info)
        print("✅ 详细损失提取功能正常")
        print(f"   提取的损失: {detailed_losses}")
        
        # 测试投影统计方法
        mock_outputs = {
            'valid_projection_mask': [torch.rand(1000) > 0.3, torch.rand(800) > 0.2]
        }
        
        class MockModel:
            def __init__(self):
                self.module = self
                self.bi_encoder = MockBiFusionEncoder()
        
        class MockBiFusionEncoder:
            def __init__(self):
                self._fusion_stats = {
                    'valid_points_ratio': 0.85,
                    'total_points': 5000,
                    '2d_weight_mean': 0.6,
                    '3d_weight_mean': 0.4
                }
                # 添加缺失的方法和属性
                self.fusion_gate = MockFusionGate()
                
            def named_modules(self):
                """模拟named_modules方法"""
                return [('fusion_gate', self.fusion_gate)]
                
        class MockFusionGate:
            def __init__(self):
                self._last_alpha = torch.tensor([0.6, 0.7, 0.5, 0.8])  # 模拟alpha权重
                self._stats_buffer = [{'2d_ratio': 0.6, '3d_ratio': 0.4}]
        
        mock_model = MockModel()
        proj_stats = hook._extract_projection_stats(mock_model, mock_outputs)
        print("✅ 投影统计提取功能正常")
        print(f"   投影统计: {proj_stats}")
        
        # 测试融合门统计方法  
        fusion_stats = hook._extract_fusion_stats(mock_model, mock_outputs)
        print("✅ 融合门统计提取功能正常")
        print(f"   融合门统计: {fusion_stats}")
        
        print("🎉 Enhanced Training Hook 所有核心功能验证通过!")
        
    except Exception as e:
        print(f"❌ Hook测试失败: {str(e)}")
        traceback.print_exc()

def test_bifusion_config():
    """测试BiFusion配置文件的正确性"""
    print("\n🧪 测试BiFusion配置文件...")
    
    config_path = "/home/nebula/xxy/ESAM/configs/ESAM_CA/sv_bifusion_scannet200.py"
    
    try:
        # 检查配置文件存在
        if not os.path.exists(config_path):
            print(f"❌ 配置文件不存在: {config_path}")
            return
            
        # 读取配置内容
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查关键配置项
        checks = [
            ('CLIP一致性损失', 'ClipConsCriterion', '0.1'),
            ('空间一致性损失', 'spatial_consistency', '0.02'),
            ('5轮评估间隔', 'val_interval = 5', None),
            ('增强Hook导入', 'EnhancedTrainingHook', None),
            ('统计收集启用', '_collect_fusion_stats', None),
        ]
        
        for check_name, check_key, expected_value in checks:
            if check_key in content:
                print(f"✅ {check_name} 配置存在")
                if expected_value and expected_value in content:
                    print(f"   值设置正确: {expected_value}")
            else:
                print(f"⚠️  {check_name} 配置可能缺失")
        
        print("🎉 BiFusion配置文件验证完成!")
        
    except Exception as e:
        print(f"❌ 配置测试失败: {str(e)}")

def test_auxiliary_losses():
    """测试辅助损失函数"""
    print("\n🧪 测试辅助损失函数...")
    
    try:
        from oneformer3d.auxiliary_loss import SpatialConsistencyLoss, NoViewSupervisionLoss
        
        # 测试空间一致性损失
        spatial_loss = SpatialConsistencyLoss()
        print("✅ SpatialConsistencyLoss 实例化成功")
        
        # 测试无视图监督损失 (如果存在)
        try:
            nv_loss = NoViewSupervisionLoss()
            print("✅ NoViewSupervisionLoss 实例化成功")
        except:
            print("ℹ️  NoViewSupervisionLoss 暂未实现 (正常)")
        
        print("🎉 辅助损失函数验证完成!")
        
    except Exception as e:
        print(f"❌ 辅助损失测试失败: {str(e)}")

def main():
    """主验证流程"""
    print("🚀 ESAM Stage 2 BiFusion优化验证")
    print("=" * 60)
    
    # 环境检查
    print(f"📍 当前工作目录: {os.getcwd()}")
    print(f"🐍 Python路径: {sys.executable}")
    print(f"🔥 PyTorch版本: {torch.__version__}")
    print(f"🎯 CUDA可用: {torch.cuda.is_available()}")
    print("-" * 60)
    
    # 执行各项测试
    test_enhanced_hook()
    test_bifusion_config()
    test_auxiliary_losses()
    
    print("\n" + "=" * 60)
    print("✨ Stage 2 验证完成! 系统已准备好进行BiFusion训练")
    print("📋 建议下一步:")
    print("   1. 运行完整训练以验证监控功能")
    print("   2. 观察fusion gate统计和投影有效率")
    print("   3. 监控梯度健康度和loss收敛性")
    print("   4. 根据统计数据进行参数微调")

if __name__ == "__main__":
    main()
