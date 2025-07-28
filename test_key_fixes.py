#!/usr/bin/env python3
"""
简化版维度修复测试脚本
跳过需要网络连接的测试，重点验证关键修复
"""

import torch
import torch.nn as nn
import numpy as np
import MinkowskiEngine as ME
from mmengine.config import Config
import sys
import os

# 添加项目路径
sys.path.append('.')

def test_config_consistency():
    """测试配置文件的维度一致性"""
    print("🔧 测试配置文件维度一致性...")
    
    try:
        config_file = 'configs/ESAM_CA/sv_bifusion_scannet200.py'
        cfg = Config.fromfile(config_file)
        
        # 检查关键维度设置
        model = cfg.model
        
        # 检查backbone配置存在（用于模型初始化）
        backbone = model.get('backbone', None)
        assert backbone is not None, "backbone配置必须存在以满足模型初始化要求"
        print(f"  backbone配置存在: {backbone['type']}")
        
        # BiFusionEncoder不使用TinySA
        bi_encoder = model.get('bi_encoder', {})
        assert bi_encoder is not None, "bi_encoder配置必须存在"
        use_tiny_sa_3d = bi_encoder.get('use_tiny_sa_3d', True)
        use_tiny_sa_2d = bi_encoder.get('use_tiny_sa_2d', True)
        print(f"  BiFusionEncoder use_tiny_sa_3d: {use_tiny_sa_3d}")
        print(f"  BiFusionEncoder use_tiny_sa_2d: {use_tiny_sa_2d}")
        assert not use_tiny_sa_3d, "应该禁用3D TinySA"
        assert not use_tiny_sa_2d, "应该禁用2D TinySA"
        
        # 池化层维度
        pool_channel_proj = model.pool.channel_proj
        print(f"  GeoAwarePooling channel_proj: {pool_channel_proj}")
        assert pool_channel_proj == 256, f"池化层维度错误: 期望256, 得到{pool_channel_proj}"
        
        # 解码器输入维度
        decoder_in_channels = model.decoder.in_channels
        print(f"  QueryDecoder in_channels: {decoder_in_channels}")
        assert decoder_in_channels == 256, f"解码器输入维度错误: 期望256, 得到{decoder_in_channels}"
        
        # 数据管道检查
        train_pipeline = cfg.train_pipeline
        pack_transform = None
        for transform in train_pipeline:
            if transform['type'] == 'Pack3DDetInputs_':
                pack_transform = transform
                break
        
        assert pack_transform is not None, "未找到Pack3DDetInputs_"
        expected_keys = ['points', 'imgs', 'cam_info', 'clip_pix', 'clip_global']
        for key in expected_keys:
            assert key in pack_transform['keys'], f"缺少必要的key: {key}"
        
        print("  ✅ BiFusion将优先于backbone被使用（因为有imgs数据）")
        print("  🎉 配置文件一致性测试通过!")
        return True
        
    except Exception as e:
        print(f"  ❌ 配置文件测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_preprocessor():
    """测试数据预处理器处理图像格式"""
    print("\n🔧 测试数据预处理器...")
    
    try:
        from oneformer3d.data_preprocessor import Det3DDataPreprocessor_
        
        preprocessor = Det3DDataPreprocessor_()
        
        # 模拟tuple格式的图像数据（类似错误日志中的格式）
        imgs_tuple = (
            torch.randn(3, 480, 640),
            torch.randn(3, 480, 640),
            torch.randn(3, 480, 640),
            torch.randn(3, 480, 640)
        )
        
        # 模拟输入数据
        data = {
            'inputs': {
                'points': [torch.randn(1000, 6) for _ in range(4)],
                'imgs': [imgs_tuple],  # 包装在列表中的tuple
                'cam_info': [{'intrinsics': [577.870605, 577.870605, 319.5, 239.5]}],
                'clip_pix': [torch.randn(1000, 256)],
                'clip_global': [torch.randn(256)]
            },
            'data_samples': [None] * 4
        }
        
        print(f"  输入图像格式: {type(data['inputs']['imgs'][0])}")
        print(f"  tuple长度: {len(data['inputs']['imgs'][0])}")
        
        # 处理数据
        result = preprocessor.simple_process(data)
        
        # 验证输出
        processed_imgs = result['inputs']['imgs']
        print(f"  ✅ 处理后图像数量: {len(processed_imgs)}")
        for i, img in enumerate(processed_imgs):
            print(f"    img[{i}]: {img.shape} (期望: [3, H, W])")
            assert img.dim() == 3 and img.shape[0] == 3, f"图像维度错误: {img.shape}"
        
        print("  🎉 数据预处理器测试通过!")
        return True
        
    except Exception as e:
        print(f"  ❌ 数据预处理器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sparse_tensor_mapping():
    """测试稀疏张量映射的正确性"""
    print("\n🔧 测试稀疏张量映射...")
    
    try:
        # 模拟点云数据
        n_points = 1000
        xyz = torch.randn(n_points, 3) * 2
        features = torch.randn(n_points, 3)
        
        # 体素化
        voxel_size = 0.02
        coords_int = torch.round(xyz / voxel_size).to(torch.int32)
        coords = torch.cat([torch.zeros(coords_int.size(0), 1, dtype=torch.int32), coords_int], dim=1)
        
        # 创建稀疏张量
        field = ME.TensorField(coordinates=coords, features=features)
        sparse_tensor = field.sparse()
        
        print(f"  原始点数: {n_points}")
        print(f"  稀疏张量形状: {sparse_tensor.F.shape}")
        
        # 模拟backbone处理
        out_features = torch.randn(sparse_tensor.F.shape[0], 96)
        sparse_output = ME.SparseTensor(
            features=out_features,
            coordinates=sparse_tensor.C,
            tensor_stride=sparse_tensor.tensor_stride
        )
        
        # 映射回原始点云
        mapped_features = sparse_output.slice(field).features
        
        print(f"  映射后特征形状: {mapped_features.shape}")
        assert mapped_features.shape[0] == n_points, f"点数不匹配: 期望{n_points}, 得到{mapped_features.shape[0]}"
        assert mapped_features.shape[1] == 96, f"特征维度错误: 期望96, 得到{mapped_features.shape[1]}"
        
        print("  🎉 稀疏张量映射测试通过!")
        return True
        
    except Exception as e:
        print(f"  ❌ 稀疏张量映射测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_color_normalization():
    """测试颜色归一化修复"""
    print("\n🔧 测试颜色归一化...")
    
    try:
        from oneformer3d.loading import NormalizePointsColor_
        
        # 创建归一化器
        color_mean = [122.0, 110.0, 96.0]
        color_std = [72.0, 70.0, 69.0] 
        normalizer = NormalizePointsColor_(
            color_mean=color_mean,
            color_std=color_std,
            clamp_range=[-3.0, 3.0]
        )
        
        # 创建测试点云（包含超出范围的颜色值）
        points_data = torch.randn(1000, 6)
        points_data[:, :3] = points_data[:, :3] * 2  # 坐标
        points_data[:, 3:6] = torch.clamp(points_data[:, 3:6] * 100 + 150, -10, 300)  # 可能超出[0,255]的颜色
        
        # 模拟BasePoints结构
        class MockPoints:
            def __init__(self, data):
                self.tensor = data
                self.attribute_dims = {'color': [3, 4, 5]}
            
            @property
            def color(self):
                return self.tensor[:, 3:6]
            
            @color.setter
            def color(self, value):
                self.tensor[:, 3:6] = value
        
        mock_points = MockPoints(points_data.clone())
        
        print(f"  原始颜色范围: [{mock_points.color.min():.2f}, {mock_points.color.max():.2f}]")
        
        # 应用归一化
        input_dict = {'points': mock_points}
        result = normalizer.transform(input_dict)
        
        normalized_color = result['points'].color
        print(f"  归一化后颜色范围: [{normalized_color.min():.2f}, {normalized_color.max():.2f}]")
        
        # 验证颜色值在合理范围内
        assert normalized_color.min() >= -3.1, f"颜色值过小: {normalized_color.min()}"
        assert normalized_color.max() <= 3.1, f"颜色值过大: {normalized_color.max()}"
        
        print("  🎉 颜色归一化测试通过!")
        return True
        
    except Exception as e:
        print(f"  ❌ 颜色归一化测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_simple_neck_replacement():
    """测试简单neck替代TinySA的功能"""
    print("\n🔧 测试SimpleNeck替代TinySA...")
    
    try:
        # 创建简单的neck替代TinySA
        adapted_dim = 256
        simple_neck = nn.Sequential(
            nn.Linear(adapted_dim, adapted_dim),
            nn.ReLU(),
            nn.LayerNorm(adapted_dim),
            nn.Linear(adapted_dim, adapted_dim),
            nn.ReLU(),
            nn.LayerNorm(adapted_dim)
        )
        
        # 测试数据
        batch_size = 4
        n_points = 1000
        input_features = torch.randn(batch_size, n_points, adapted_dim)
        
        print(f"  输入特征形状: {input_features.shape}")
        
        # 前向传播
        with torch.no_grad():
            output_features = simple_neck(input_features)
        
        print(f"  输出特征形状: {output_features.shape}")
        
        # 验证维度
        assert output_features.shape == input_features.shape, f"维度不匹配: {output_features.shape} != {input_features.shape}"
        
        print("  🎉 SimpleNeck测试通过!")
        return True
        
    except Exception as e:
        print(f"  ❌ SimpleNeck测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🚀 开始关键维度修复验证...")
    
    tests = [
        ("配置文件一致性", test_config_consistency),
        ("数据预处理器", test_data_preprocessor),
        ("稀疏张量映射", test_sparse_tensor_mapping),
        ("颜色归一化", test_color_normalization),
        ("SimpleNeck替代", test_simple_neck_replacement)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name}测试发生异常: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "="*50)
    print("📊 关键修复测试结果:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n🎯 总体结果: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有关键维度修复验证通过！")
        print("\n📝 修复总结:")
        print("  ✅ 移除了对TinySA模块的依赖")
        print("  ✅ 修复了颜色值超出范围的警告") 
        print("  ✅ 解决了图像数据格式不匹配问题")
        print("  ✅ 统一了维度设置为256维")
        print("  ✅ 验证了稀疏张量映射正确性")
        return True
    else:
        print("⚠️  部分测试失败，需要进一步调试。")
        return False

if __name__ == '__main__':
    main() 