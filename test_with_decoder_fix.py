#!/usr/bin/env python3
"""
修复QueryDecoder权重后的测试脚本
"""

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, '/home/nebula/xxy/ESAM')

from mmengine.config import Config
from mmengine.runner import Runner

def patch_query_decoder():
    """给QueryDecoder打补丁，正确加载input_proj权重"""
    
    # 加载修复配置
    if os.path.exists('/home/nebula/xxy/ESAM/decoder_fix_config.pth'):
        fix_config = torch.load('/home/nebula/xxy/ESAM/decoder_fix_config.pth', map_location='cpu')
        print(f"🔧 加载修复配置: {fix_config['expected_in_channels']} -> {fix_config['d_model']}")
        
        # 获取原始QueryDecoder类
        from oneformer3d.query_decoder import QueryDecoder
        original_forward_iter_pred = QueryDecoder.forward_iter_pred
        
        def patched_forward_iter_pred(self, sp_feats, p_feats, queries, super_points, prev_queries=None):
            """修复后的forward方法"""
            
            # 如果input_proj未初始化，使用预训练权重初始化
            if self.input_proj is None and sp_feats and "SP" in self.cross_attn_mode:
                print(f"🔧 使用预训练权重初始化input_proj")
                
                # 创建与训练时相同的结构
                self.input_proj = nn.Sequential(
                    nn.Linear(fix_config['expected_in_channels'], fix_config['d_model']),
                    nn.LayerNorm(fix_config['d_model']),
                    nn.ReLU()
                ).to(sp_feats[0].device)
                
                # 加载预训练权重
                with torch.no_grad():
                    self.input_proj[0].weight.copy_(fix_config['linear_weight'])
                    self.input_proj[0].bias.copy_(fix_config['linear_bias'])
                    if fix_config['norm_weight'] is not None:
                        self.input_proj[1].weight.copy_(fix_config['norm_weight'])
                        self.input_proj[1].bias.copy_(fix_config['norm_bias'])
                
                print(f"✅ input_proj权重加载完成")
            
            # 调用原始方法
            return original_forward_iter_pred(self, sp_feats, p_feats, queries, super_points, prev_queries)
        
        # 应用补丁
        QueryDecoder.forward_iter_pred = patched_forward_iter_pred
        print("🔧 QueryDecoder补丁已应用")
    else:
        print("⚠️  未找到修复配置，跳过权重修复")

def main():
    """主测试函数"""
    
    print("🚀 开始修复后的测试")
    
    # 应用补丁
    patch_query_decoder()
    
    # 设置环境
    os.environ['PYTHONPATH'] = '/home/nebula/xxy/ESAM:' + os.environ.get('PYTHONPATH', '')
    
    # 加载配置
    config_path = 'configs/ESAM_CA/sv_bifusion_scannet200.py'
    checkpoint_path = '/home/nebula/xxy/ESAM/work_dirs/enhanced_bifusion_debug/epoch_49.pth'
    work_dir = 'work_dirs/test_epoch49_ca_fixed'
    
    print(f"📂 配置文件: {config_path}")
    print(f"🎯 模型文件: {checkpoint_path}")
    print(f"💾 输出目录: {work_dir}")
    
    # 加载配置
    cfg = Config.fromfile(config_path)
    
    # 设置CA模式
    cfg.test_evaluator.eval_mode = 'cat_agnostic'
    cfg.work_dir = work_dir
    cfg.load_from = checkpoint_path
    
    # 创建runner并测试
    runner = Runner.from_cfg(cfg)
    runner.test()
    
    print("✅ 修复后的测试完成")

if __name__ == "__main__":
    main()
