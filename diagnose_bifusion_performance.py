#!/usr/bin/env python3

import os
import sys
import torch
import subprocess
sys.path.append('/home/nebula/xxy/ESAM')

def run_test(config_path, checkpoint_path, work_dir, description):
    """运行测试并返回结果"""
    print(f"\n=== {description} ===")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint不存在: {checkpoint_path}")
        return None
    
    # 设置环境变量
    env = os.environ.copy()
    env['PYTHONPATH'] = '/home/nebula/xxy/ESAM:' + env.get('PYTHONPATH', '')
    env['CUDA_VISIBLE_DEVICES'] = '0'
    
    cmd = [
        'python', 'tools/test.py',
        config_path,
        checkpoint_path,
        '--cat-agnostic',
        '--work-dir', work_dir
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd='/home/nebula/xxy/ESAM',
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1小时超时
        )
        
        if result.returncode == 0:
            print("✅ 测试成功完成")
            # 提取关键指标
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'all_ap' in line and '{' in line:
                    print(f"📊 结果: {line}")
                    break
            return result.stdout
        else:
            print(f"❌ 测试失败 (返回码: {result.returncode})")
            print(f"错误信息: {result.stderr}")
            return None
            
    except subprocess.TimeoutExpired:
        print("❌ 测试超时")
        return None
    except Exception as e:
        print(f"❌ 运行出错: {e}")
        return None

def main():
    print("=== BiFusion性能诊断 ===")
    
    # 测试1：基线3D模型
    baseline_result = run_test(
        'configs/ESAM_CA/sv3d_scannet200_ca.py',
        '/home/nebula/xxy/ESAM/work_dirs/sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth',
        'work_dirs/diagnosis_baseline',
        '基线3D模型 (参考性能)'
    )
    
    # 测试2：修正后的BiFusion模型
    bifusion_result = run_test(
        'configs/ESAM_CA/sv_bifusion_scannet200.py',
        '/home/nebula/xxy/ESAM/work_dirs/sv3d_scannet200_ca/best_all_ap_50%_epoch_128.pth',
        'work_dirs/diagnosis_bifusion',
        '修正BiFusion模型 (使用3D预训练权重)'
    )
    
    # 分析结果
    print("\n" + "="*50)
    print("性能对比分析:")
    
    if baseline_result and bifusion_result:
        print("✅ 两个模型都测试成功")
        print("💡 建议: 如果BiFusion性能明显低于基线，考虑:")
        print("   1. 进一步降低CLIP损失权重到0.005")
        print("   2. 冻结更多CLIP层")
        print("   3. 检查数据增强配置")
    elif baseline_result:
        print("⚠️ 基线模型正常，BiFusion模型有问题")
        print("💡 建议: 检查BiFusion模型架构和配置")
    else:
        print("❌ 基线模型也有问题，检查环境配置")

if __name__ == "__main__":
    main() 