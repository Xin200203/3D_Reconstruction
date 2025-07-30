#!/usr/bin/env python3

import sys
import os
sys.path.append('/home/nebula/xxy/ESAM')
os.chdir('/home/nebula/xxy/ESAM')

import torch
import subprocess
import time
from mmengine import Config

def monitor_model_memory():
    """分析模型显存使用的增长模式"""
    
    print("=== BiFusion模型显存增长分析 ===")
    
    # 清空显存
    torch.cuda.empty_cache()
    initial_memory = torch.cuda.memory_allocated() / 1024**3
    print(f"初始显存使用: {initial_memory:.2f} GB")
    
    try:
        # 加载配置
        cfg = Config.fromfile('configs/ESAM_CA/sv_bifusion_scannet200.py')
        print("✓ 配置加载成功")
        
        # 模拟不同参数组的显存占用
        param_groups = {
            'CLIP_conv1': 590592,          # CLIP conv1层参数
            'CLIP_transformer': 87084032,  # CLIP transformer参数 
            'backbone3d': 16777216,        # 3D backbone参数
            'decoder': 33554432,           # 解码器参数
        }
        
        print("\n模拟优化器状态创建过程:")
        cumulative_memory = initial_memory
        
        for epoch in range(1, 12):
            print(f"\nEpoch {epoch}:")
            
            # 模拟不同epoch下参数的激活
            active_groups = []
            if epoch >= 1:
                active_groups.extend(['backbone3d', 'decoder'])
            if epoch >= 3:  # CLIP可能在第3个epoch后开始解冻
                active_groups.append('CLIP_conv1')
            if epoch >= 6:  # 更多CLIP层解冻
                active_groups.append('CLIP_transformer')
            
            # 计算当前epoch的优化器状态内存需求
            optimizer_memory = 0
            for group in active_groups:
                param_count = param_groups[group]
                # AdamW需要2个状态：momentum + variance
                state_memory = param_count * 4 * 2 / 1024**3  # float32 * 2状态
                optimizer_memory += state_memory
                
            total_memory = initial_memory + optimizer_memory
            print(f"  激活参数组: {active_groups}")
            print(f"  优化器状态显存: {optimizer_memory:.2f} GB")
            print(f"  总显存预估: {total_memory:.2f} GB")
            
            if total_memory > 22.0:  # RTX 4090约22GB可用
                print(f"  ⚠️ 显存不足！可能在Epoch {epoch}触发OOM")
                break
                
    except Exception as e:
        print(f"分析失败: {e}")

def check_checkpoint_compatibility():
    """检查预训练模型的兼容性"""
    print("\n=== 预训练模型兼容性检查 ===")
    
    checkpoint_path = '/home/nebula/xxy/ESAM/work_dirs/sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth'
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️ 预训练模型不存在: {checkpoint_path}")
        return
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
            
            # 分析预训练模型的参数
            clip_params = [k for k in state_dict.keys() if 'clip' in k.lower() or 'enhanced_clip' in k]
            fusion_params = [k for k in state_dict.keys() if 'bi_encoder' in k or 'fusion' in k]
            
            print(f"预训练模型参数统计:")
            print(f"  总参数: {len(state_dict)}")
            print(f"  CLIP相关参数: {len(clip_params)}")
            print(f"  Fusion相关参数: {len(fusion_params)}")
            
            if len(clip_params) == 0:
                print("  ⚠️ 预训练模型缺少CLIP参数 - 这解释了为什么后期才OOM！")
                print("  💡 CLIP参数需要新初始化，优化器状态会在后期创建")
            
    except Exception as e:
        print(f"检查失败: {e}")

if __name__ == "__main__":
    monitor_model_memory()
    check_checkpoint_compatibility() 