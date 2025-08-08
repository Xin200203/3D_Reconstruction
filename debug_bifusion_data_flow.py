#!/usr/bin/env python3
"""
BiFusion 数据流调试脚本
分析pose数据在训练过程中的传递情况
"""

import torch
import sys
import os
sys.path.append('/home/nebula/xxy/ESAM')

def debug_batch_data():
    """模拟训练时的batch数据结构"""
    print("=== BiFusion 数据流调试 ===\n")
    
    print("1. 模拟training数据结构 (batch_size=6):")
    print("   batch_inputs_dict包含6个场景的数据:")
    
    # 模拟数据结构
    batch_inputs_dict = {
        'points': [f"scene_{i}_points" for i in range(6)],
        'imgs': [f"scene_{i}_img" for i in range(6)],
        'cam_info': [f"scene_{i}_cam_info" for i in range(6)]
    }
    
    for key, val in batch_inputs_dict.items():
        print(f"   {key}: {val}")
    
    print(f"\n2. BiFusion.forward()接收到的数据:")
    print(f"   points_list: {len(batch_inputs_dict['points'])} 个场景")
    print(f"   imgs: {len(batch_inputs_dict['imgs'])} 个场景") 
    print(f"   cam_info: {len(batch_inputs_dict['cam_info'])} 个场景")
    
    print(f"\n3. _process_single()处理流程:")
    for i in range(6):
        print(f"   场景 {i}: pts={batch_inputs_dict['points'][i]}, img={batch_inputs_dict['imgs'][i]}, meta={batch_inputs_dict['cam_info'][i]}")
    
    print(f"\n4. 问题分析:")
    print(f"   ❌ 当前错误: _process_single收到pose形状为[6,4,4]")
    print(f"   ✅ 正确应该: 每个场景的meta包含单独的[4,4] pose")
    print(f"   🔍 原因推测: 数据加载时pose没有被正确分解")
    
    print(f"\n5. 修复方向:")
    print(f"   - 检查数据加载器中cam_info的构建")
    print(f"   - 确认pose数据是否正确对应到单个场景")
    print(f"   - 验证dataloader输出的cam_info格式")

if __name__ == "__main__":
    debug_batch_data()
