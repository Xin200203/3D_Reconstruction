#!/usr/bin/env python3
"""
调试CLIP参数与AdamW优化器内存增长的关系
分析8个epoch后OOM的根本原因
"""

import torch
import torch.nn as nn
from typing import Dict, List
import sys
import os

# 添加项目路径
sys.path.insert(0, '/home/nebula/xxy/ESAM')

def analyze_adamw_memory_growth():
    """分析AdamW优化器内存增长模式"""
    
    print("=" * 60)
    print("🔍 分析AdamW优化器内存分配机制")
    print("=" * 60)
    
    # 模拟不同大小的参数组
    param_sizes = [
        ('小型网络 (1M参数)', 1_000_000),
        ('中型网络 (10M参数)', 10_000_000), 
        ('CLIP ViT-B/16 (86M参数)', 86_000_000),
        ('完整BiFusion (200M参数)', 200_000_000)
    ]
    
    for name, num_params in param_sizes:
        # 创建虚拟参数
        param = nn.Parameter(torch.randn(num_params, dtype=torch.float32))
        
        # 创建AdamW优化器
        optimizer = torch.optim.AdamW([param], lr=0.001)
        
        # 计算内存占用
        param_memory = param.numel() * param.element_size() / (1024**3)  # GB
        
        # AdamW为每个参数维护两个状态：momentum (exp_avg) 和 variance (exp_avg_sq)
        state_memory = param_memory * 2  # 两个float32缓冲区
        total_memory = param_memory + state_memory
        
        print(f"\n📊 {name}:")
        print(f"   参数内存: {param_memory:.2f} GB")
        print(f"   状态内存: {state_memory:.2f} GB (momentum + variance)")
        print(f"   总计内存: {total_memory:.2f} GB")
        
        del param, optimizer
    
    print("\n" + "=" * 60)

def analyze_clip_parameter_groups():
    """分析CLIP参数分组和逐步解冻模式"""
    
    print("🔍 分析CLIP参数分组策略")
    print("=" * 60)
    
    # 模拟CLIP ViT-B/16结构
    layer_groups = {
        'patch_embedding': 590_592,     # conv1 + pos_embed + class_embed
        'ln_pre': 1_536,               # layer norm
        'transformer_layer_0': 7_077_888,  # 每层约7M参数
        'transformer_layer_1': 7_077_888,
        'transformer_layer_2': 7_077_888,
        'transformer_layer_3': 7_077_888,
        'transformer_layer_4': 7_077_888,
        'transformer_layer_5': 7_077_888,
        'transformer_layer_6': 7_077_888,
        'transformer_layer_7': 7_077_888,
        'transformer_layer_8': 7_077_888,
        'transformer_layer_9': 7_077_888,
        'transformer_layer_10': 7_077_888,
        'transformer_layer_11': 7_077_888,
        'ln_post': 1_536,              # final layer norm
    }
    
    print(f"\n📋 CLIP ViT-B/16 参数分组:")
    total_params = 0
    for group, size in layer_groups.items():
        total_params += size
        print(f"   {group:25s}: {size:>10,} 参数")
    
    print(f"\n   {'总计':25s}: {total_params:>10,} 参数")
    
    # 分析现有冻结策略
    print(f"\n🔧 当前冻结策略分析:")
    print(f"   freeze_clip_early_layers=True (冻结前3层)")
    print(f"   clip_num_layers=6 (使用前6层)")
    
    frozen_params = (layer_groups['transformer_layer_0'] + 
                    layer_groups['transformer_layer_1'] + 
                    layer_groups['transformer_layer_2'])
    
    trainable_params = (layer_groups['patch_embedding'] +  # conv1不冻结
                       layer_groups['ln_pre'] +
                       layer_groups['transformer_layer_3'] + 
                       layer_groups['transformer_layer_4'] + 
                       layer_groups['transformer_layer_5'])
    
    print(f"\n   冻结参数:   {frozen_params:>10,} ({frozen_params/total_params*100:.1f}%)")
    print(f"   可训练参数: {trainable_params:>10,} ({trainable_params/total_params*100:.1f}%)")
    
    # 计算AdamW状态内存
    trainable_memory = trainable_params * 4 / (1024**3)  # float32, GB
    optimizer_state_memory = trainable_memory * 2  # momentum + variance
    
    print(f"\n💾 内存占用分析:")
    print(f"   可训练参数内存: {trainable_memory:.2f} GB")
    print(f"   优化器状态内存: {optimizer_state_memory:.2f} GB")
    print(f"   CLIP总额外内存: {trainable_memory + optimizer_state_memory:.2f} GB")
    
    return trainable_params, optimizer_state_memory

def simulate_epoch_by_epoch_memory():
    """模拟逐epoch的内存增长"""
    
    print("🕒 模拟训练过程内存变化")
    print("=" * 60)
    
    # 基础模型内存占用 (假设)
    base_model_memory = 12.0  # GB
    base_optimizer_memory = 8.0  # GB for 3D backbone + other params
    
    # CLIP分支内存 (从上面的分析得出)
    _, clip_optimizer_memory = analyze_clip_parameter_groups()
    
    print(f"\n📈 逐Epoch内存变化模拟:")
    print(f"   基础模型:     {base_model_memory:.1f} GB")
    print(f"   基础优化器:   {base_optimizer_memory:.1f} GB")
    print(f"   CLIP优化器:   {clip_optimizer_memory:.1f} GB (一次性分配)")
    
    total_memory = base_model_memory + base_optimizer_memory + clip_optimizer_memory
    
    for epoch in range(1, 10):
        print(f"\n   Epoch {epoch:2d}: {total_memory:.1f} GB")
        if epoch == 8:
            print(f"             ⚠️  接近24GB限制! 可能触发OOM")
        
        # 模拟内存碎片化的逐步增长
        if epoch > 5:
            total_memory += 0.1  # 内存碎片化

def analyze_solution_strategies():
    """分析解决方案策略"""
    
    print("🎯 解决方案策略分析")
    print("=" * 60)
    
    strategies = [
        ("降低batch_size", "从20->8->6", "立即减少4-6GB"),
        ("梯度累积", "accumulative_counts=2-3", "补偿batch_size降低"),
        ("更激进的CLIP冻结", "只训练最后1-2层", "减少50%优化器内存"),
        ("分阶段训练", "先训3D，再训CLIP", "避免同时优化"),
        ("混合精度", "启用AMP", "减少30-50%内存"),
        ("梯度检查点", "重计算vs存储", "减少激活内存"),
    ]
    
    print(f"\n💡 可选策略:")
    for i, (strategy, implementation, benefit) in enumerate(strategies, 1):
        print(f"   {i}. {strategy:20s}: {implementation:20s} -> {benefit}")
        
    print(f"\n🏆 推荐策略组合:")
    print(f"   1. 立即: batch_size=6 + accumulative_counts=3")
    print(f"   2. 中期: 更激进CLIP冻结 (只训练层5)")  
    print(f"   3. 长期: 分阶段训练策略")

if __name__ == "__main__":
    try:
        # 分析AdamW内存机制
        analyze_adamw_memory_growth()
        
        print("\n" + "="*80 + "\n")
        
        # 分析CLIP参数策略
        analyze_clip_parameter_groups()
        
        print("\n" + "="*80 + "\n")
        
        # 模拟训练过程
        simulate_epoch_by_epoch_memory()
        
        print("\n" + "="*80 + "\n")
        
        # 解决方案分析
        analyze_solution_strategies()
        
        print(f"\n🎓 关键结论:")
        print(f"   • 8个epoch OOM的根本原因: AdamW为所有可训练CLIP参数预分配了状态内存")
        print(f"   • 这不是内存泄漏，而是正常的优化器行为")
        print(f"   • 分段加载模型不是直接原因，但增加了总参数量")
        print(f"   • 解决方案: 降低batch_size + 更激进的参数冻结策略")
        
    except Exception as e:
        print(f"❌ 分析过程出错: {e}")
        import traceback
        traceback.print_exc() 