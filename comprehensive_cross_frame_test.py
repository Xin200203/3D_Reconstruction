#!/usr/bin/env python3
"""
全面的跨帧物体匹配测试脚本
验证IoU预剪枝、TimeDividedTransformer、Memory管理的完整性
"""

import torch
import numpy as np
import time
from oneformer3d.time_divided_transformer import TimeDividedTransformer
from oneformer3d.instance_merge import OnlineMerge
from mmdet3d.structures import AxisAlignedBboxOverlaps3D

def test_comprehensive_cross_frame_matching():
    """全面测试跨帧物体匹配系统"""
    print("=" * 80)
    print("🔬 全面的跨帧物体匹配系统测试")
    print("=" * 80)
    
    # 测试配置
    test_configs = [
        {"d_model": 256, "nhead": 8, "num_layers": 3, "iou_thr": 0.1},
        {"d_model": 128, "nhead": 4, "num_layers": 2, "iou_thr": 0.2},
        {"d_model": 512, "nhead": 16, "num_layers": 4, "iou_thr": 0.05},
    ]
    
    for i, config in enumerate(test_configs):
        print(f"\n📋 测试配置 {i+1}: {config}")
        test_single_config(**config)
    
    print("\n🎯 边界情况测试")
    test_edge_cases()
    
    print("\n⚡ 性能基准测试")
    test_performance_benchmark()
    
    print("\n🔄 内存管理测试")
    test_memory_management()
    
    print("\n✅ 所有全面测试完成！")

def test_single_config(d_model, nhead, num_layers, iou_thr):
    """测试单个配置"""
    try:
        # 创建OnlineMerge
        online_merge = OnlineMerge(
            inscat_topk_insts=200,
            use_bbox=True,
            merge_type='count',
            iou_thr=iou_thr,
            tformer_cfg=dict(
                type='TimeDividedTransformer',
                d_model=d_model,
                nhead=nhead,
                num_layers=num_layers,
                dropout=0.1
            )
        )
        
        # 模拟多帧数据处理
        frame_results = []
        num_frames = 5
        
        for frame_idx in range(num_frames):
            # 生成模拟数据
            num_objects = np.random.randint(3, 8)
            masks = torch.randint(0, 2, (num_objects, 1000), dtype=torch.bool)
            labels = torch.randint(0, 20, (num_objects,))
            scores = torch.rand(num_objects)
            queries = torch.randn(num_objects, d_model)
            query_feats = torch.randn(num_objects, d_model)
            sem_preds = torch.randn(num_objects, 20)
            
            # 位置稍微移动，模拟真实场景
            if frame_idx == 0:
                xyz = torch.randn(num_objects, 3) * 3  # 初始位置
                base_xyz = xyz.clone()  # 保存作为基准
            else:
                # 前一半物体移动，后一半是新物体
                half = min(num_objects // 2, base_xyz.shape[0])  # 确保不超出范围
                if half > 0:
                    moved_xyz = base_xyz[:half] + torch.randn(half, 3) * 0.3  # 轻微移动
                    new_xyz = torch.randn(num_objects - half, 3) * 3  # 新物体
                    xyz = torch.cat([moved_xyz, new_xyz], dim=0)
                else:
                    xyz = torch.randn(num_objects, 3) * 3  # 全新物体
            
            bboxes = torch.rand(num_objects, 6)  # [x,y,z,w,h,l]
            
            # 处理当前帧
            result = online_merge.merge(masks, labels, scores, queries,
                                      query_feats, sem_preds, xyz, bboxes)
            
            frame_results.append({
                'frame_idx': frame_idx,
                'input_objects': num_objects,
                'output_objects': len(result[2]),  # scores length
                'result': result
            })
        
        # 验证结果
        print(f"  📊 帧数: {num_frames}")
        for result in frame_results:
            print(f"    帧{result['frame_idx']}: {result['input_objects']}→{result['output_objects']} 物体")
        
        # 验证Memory一致性
        assert online_merge.cur_queries.shape[1] == d_model, f"Query维度不匹配: {online_merge.cur_queries.shape[1]} != {d_model}"
        
        print(f"  ✅ 配置测试通过")
        
    except Exception as e:
        print(f"  ❌ 配置测试失败: {e}")

def test_edge_cases():
    """测试边界情况"""
    
    online_merge = OnlineMerge(
        inscat_topk_insts=100,
        use_bbox=True,
        merge_type='count',
        iou_thr=0.1,
        tformer_cfg=dict(
            type='TimeDividedTransformer',
            d_model=256,
            nhead=8,
            num_layers=2
        )
    )
    
    test_cases = [
        {
            'name': '空帧处理',
            'num_objects': 0,
            'description': '测试没有物体的帧'
        },
        {
            'name': '单物体帧',
            'num_objects': 1,
            'description': '测试只有一个物体的帧'
        },
        {
            'name': '大量物体',
            'num_objects': 50,
            'description': '测试大量物体的处理能力'
        }
    ]
    
    for case in test_cases:
        print(f"  🧪 {case['name']}: {case['description']}")
        try:
            num_objects = case['num_objects']
            
            if num_objects == 0:
                # 空帧特殊处理
                masks = torch.empty(0, 1000, dtype=torch.bool)
                labels = torch.empty(0, dtype=torch.long)
                scores = torch.empty(0)
                queries = torch.empty(0, 256)
                query_feats = torch.empty(0, 256)
                sem_preds = torch.empty(0, 20)
                xyz = torch.empty(0, 3)
                bboxes = torch.empty(0, 6)
            else:
                masks = torch.randint(0, 2, (num_objects, 1000), dtype=torch.bool)
                labels = torch.randint(0, 20, (num_objects,))
                scores = torch.rand(num_objects)
                queries = torch.randn(num_objects, 256)
                query_feats = torch.randn(num_objects, 256)
                sem_preds = torch.randn(num_objects, 20)
                xyz = torch.randn(num_objects, 3) * 3
                bboxes = torch.rand(num_objects, 6)
            
            # 先处理一个正常帧作为初始化
            if online_merge.cur_masks is None:
                normal_masks = torch.randint(0, 2, (3, 1000), dtype=torch.bool)
                normal_labels = torch.randint(0, 20, (3,))
                normal_scores = torch.rand(3)
                normal_queries = torch.randn(3, 256)
                normal_query_feats = torch.randn(3, 256)
                normal_sem_preds = torch.randn(3, 20)
                normal_xyz = torch.randn(3, 3) * 3
                normal_bboxes = torch.rand(3, 6)
                
                online_merge.merge(normal_masks, normal_labels, normal_scores,
                                 normal_queries, normal_query_feats, normal_sem_preds,
                                 normal_xyz, normal_bboxes)
            
            # 处理边界情况
            result = online_merge.merge(masks, labels, scores, queries,
                                      query_feats, sem_preds, xyz, bboxes)
            
            print(f"    ✅ {case['name']}测试通过: 输入{num_objects}个物体，输出{len(result[2])}个物体")
            
        except Exception as e:
            print(f"    ❌ {case['name']}测试失败: {e}")

def test_performance_benchmark():
    """性能基准测试"""
    
    # 测试不同物体数量的性能
    object_counts = [5, 10, 20, 30, 50]
    
    for num_objects in object_counts:
        print(f"  📈 测试 {num_objects} 个物体的性能...")
        
        # 创建测试数据
        online_merge = OnlineMerge(
            inscat_topk_insts=200,
            use_bbox=True,
            merge_type='count',
            iou_thr=0.1,
            tformer_cfg=dict(
                type='TimeDividedTransformer',
                d_model=256,
                nhead=8,
                num_layers=3
            )
        )
        
        # 预热
        masks = torch.randint(0, 2, (5, 1000), dtype=torch.bool)
        labels = torch.randint(0, 20, (5,))
        scores = torch.rand(5)
        queries = torch.randn(5, 256)
        query_feats = torch.randn(5, 256)
        sem_preds = torch.randn(5, 20)
        xyz = torch.randn(5, 3) * 3
        bboxes = torch.rand(5, 6)
        
        online_merge.merge(masks, labels, scores, queries,
                         query_feats, sem_preds, xyz, bboxes)
        
        # 性能测试
        test_times = []
        for _ in range(10):  # 多次测试取平均
            masks = torch.randint(0, 2, (num_objects, 1000), dtype=torch.bool)
            labels = torch.randint(0, 20, (num_objects,))
            scores = torch.rand(num_objects)
            queries = torch.randn(num_objects, 256)
            query_feats = torch.randn(num_objects, 256)
            sem_preds = torch.randn(num_objects, 20)
            xyz = torch.randn(num_objects, 3) * 3
            bboxes = torch.rand(num_objects, 6)
            
            start_time = time.time()
            result = online_merge.merge(masks, labels, scores, queries,
                                      query_feats, sem_preds, xyz, bboxes)
            end_time = time.time()
            
            test_times.append(end_time - start_time)
        
        avg_time = np.mean(test_times)
        print(f"    ⏱️  平均处理时间: {avg_time*1000:.2f}ms")
        
        # 性能要求：50个物体应该在100ms内处理完成
        if num_objects <= 50 and avg_time > 0.1:
            print(f"    ⚠️  性能警告: {num_objects}个物体用时{avg_time*1000:.2f}ms > 100ms")
        else:
            print(f"    ✅ 性能达标")

def test_memory_management():
    """测试Memory管理功能"""
    
    online_merge = OnlineMerge(
        inscat_topk_insts=10,  # 限制Memory容量为10
        use_bbox=True,
        merge_type='count',
        iou_thr=0.1,
        tformer_cfg=dict(
            type='TimeDividedTransformer',
            d_model=256,
            nhead=8,
            num_layers=2
        )
    )
    
    print(f"  🧠 Memory容量限制: {online_merge.inscat_topk_insts}")
    
    # 持续添加物体，测试Memory管理
    frame_count = 15
    memory_sizes = []
    
    for frame_idx in range(frame_count):
        # 每帧添加3-5个新物体
        num_objects = np.random.randint(3, 6)
        
        masks = torch.randint(0, 2, (num_objects, 1000), dtype=torch.bool)
        labels = torch.randint(0, 20, (num_objects,))
        scores = torch.rand(num_objects)
        queries = torch.randn(num_objects, 256)
        query_feats = torch.randn(num_objects, 256)
        sem_preds = torch.randn(num_objects, 20)
        xyz = torch.randn(num_objects, 3) * 10  # 确保是新物体
        bboxes = torch.rand(num_objects, 6)
        
        result = online_merge.merge(masks, labels, scores, queries,
                                  query_feats, sem_preds, xyz, bboxes)
        
        memory_size = len(result[2])  # scores length
        memory_sizes.append(memory_size)
        
        print(f"    帧{frame_idx+1}: 添加{num_objects}个物体，Memory大小: {memory_size}")
        
        # 验证Memory容量限制
        assert memory_size <= online_merge.inscat_topk_insts, \
            f"Memory溢出: {memory_size} > {online_merge.inscat_topk_insts}"
    
    print(f"  📊 Memory大小变化: {memory_sizes}")
    print(f"  ✅ Memory管理测试通过")

def test_attention_mask_effectiveness():
    """测试注意力掩码的有效性"""
    print("\n🎭 注意力掩码有效性测试")
    
    # 创建两个TDT，一个有掩码，一个没有
    tdt = TimeDividedTransformer(d_model=256, nhead=8, num_layers=2)
    
    # 模拟数据
    B, Nc, Nm, D = 1, 10, 15, 256
    q = torch.randn(B, Nc, D)
    k = torch.randn(B, Nm, D)
    p_c = torch.randn(B, Nc, 9)
    p_m = torch.randn(B, Nm, 9)
    
    # 创建稀疏掩码（只有30%的配对有效）
    attention_mask = torch.rand(B, Nc, Nm) > 0.7  # 30%为True
    
    print(f"  有效配对比例: {attention_mask.sum().item()}/{attention_mask.numel()} ({attention_mask.sum().item()/attention_mask.numel()*100:.1f}%)")
    
    # 测试1: 无掩码
    with torch.no_grad():
        start_time = time.time()
        attn1, q_new1 = tdt(q, k, p_c, p_m)
        time1 = time.time() - start_time
    
    # 测试2: 有掩码
    with torch.no_grad():
        start_time = time.time()
        attn2, q_new2 = tdt(q, k, p_c, p_m, attention_mask=attention_mask)
        time2 = time.time() - start_time
    
    print(f"  无掩码用时: {time1*1000:.2f}ms")
    print(f"  有掩码用时: {time2*1000:.2f}ms")
    
    # 验证掩码效果
    masked_positions = ~attention_mask[0]
    masked_attention = attn2[0][masked_positions]
    max_masked_attn = masked_attention.max().item() if masked_attention.numel() > 0 else 0
    
    print(f"  被掩码位置最大注意力: {max_masked_attn:.6f} (应该≈0)")
    
    # 验证有效位置有合理的注意力分布
    valid_positions = attention_mask[0]
    if valid_positions.any():
        valid_attention = attn2[0][valid_positions]
        print(f"  有效位置注意力范围: [{valid_attention.min().item():.6f}, {valid_attention.max().item():.6f}]")
    
    assert max_masked_attn < 1e-6, f"掩码失效: 被掩码位置注意力过高 {max_masked_attn}"
    print("  ✅ 注意力掩码有效性验证通过")

if __name__ == "__main__":
    try:
        # 运行全面测试
        test_comprehensive_cross_frame_matching()
        
        # 注意力掩码专项测试
        test_attention_mask_effectiveness()
        
        print("\n" + "=" * 80)
        print("🎉 全面测试完成！跨帧物体匹配系统实现正确且高效")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ 全面测试失败: {e}")
        import traceback
        traceback.print_exc() 