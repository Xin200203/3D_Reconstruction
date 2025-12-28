#!/usr/bin/env python3
"""
创建CLIP特征的软链接
- SV阶段：每200帧采样一次（从40间隔特征中每5个取1个）
- MV阶段：每40帧采样一次（使用所有40间隔特征）
"""

import os
import glob
from pathlib import Path

def create_clip_symlinks():
    # 源目录和目标目录
    source_base = "/home/nebula/xxy/dataset/scannet/clip_feat_RN50_layer2"
    sv_target = "/home/nebula/xxy/ESAM/data/scannet200-sv/clip_feat"
    mv_target = "/home/nebula/xxy/ESAM/data/scannet200-mv/clip_feat"
    
    # 获取所有场景
    scene_dirs = glob.glob(os.path.join(source_base, "scene*"))
    scene_dirs.sort()
    
    print(f"找到 {len(scene_dirs)} 个场景")
    
    sv_count = 0
    mv_count = 0
    
    for scene_dir in scene_dirs:
        scene_name = os.path.basename(scene_dir)
        print(f"处理场景: {scene_name}")
        
        # 获取该场景的所有特征文件
        feature_files = glob.glob(os.path.join(scene_dir, "*.pt"))
        
        # 提取帧号并排序
        frame_numbers = []
        for f in feature_files:
            frame_num = int(os.path.splitext(os.path.basename(f))[0])
            frame_numbers.append(frame_num)
        frame_numbers.sort()
        
        print(f"  场景 {scene_name} 有 {len(frame_numbers)} 帧")
        
        # 创建场景目录
        sv_scene_dir = os.path.join(sv_target, scene_name)
        mv_scene_dir = os.path.join(mv_target, scene_name)
        os.makedirs(sv_scene_dir, exist_ok=True)
        os.makedirs(mv_scene_dir, exist_ok=True)
        
        # 为MV创建链接（所有40间隔的帧）
        for frame_num in frame_numbers:
            source_file = os.path.join(scene_dir, f"{frame_num}.pt")
            mv_link = os.path.join(mv_scene_dir, f"{frame_num}.pt")
            
            # 删除已存在的链接
            if os.path.exists(mv_link) or os.path.islink(mv_link):
                os.unlink(mv_link)
            
            os.symlink(source_file, mv_link)
            mv_count += 1
        
        # 为SV创建链接（每200帧，即每5个40间隔帧取1个）
        # 200 = 40 * 5，所以从40间隔序列中每5个取1个
        sv_frames = []
        for frame_num in frame_numbers:
            # 检查是否是200的倍数
            if frame_num % 200 == 0:
                sv_frames.append(frame_num)
        
        print(f"  SV选择的帧: {sv_frames}")
        
        for frame_num in sv_frames:
            source_file = os.path.join(scene_dir, f"{frame_num}.pt")
            sv_link = os.path.join(sv_scene_dir, f"{frame_num}.pt")
            
            # 删除已存在的链接
            if os.path.exists(sv_link) or os.path.islink(sv_link):
                os.unlink(sv_link)
            
            if os.path.exists(source_file):
                os.symlink(source_file, sv_link)
                sv_count += 1
            else:
                print(f"  警告: 源文件不存在 {source_file}")
    
    print(f"\n链接创建完成:")
    print(f"SV链接数: {sv_count}")
    print(f"MV链接数: {mv_count}")
    
    # 验证创建的链接
    print(f"\nSV目录验证:")
    sv_scenes = len(glob.glob(os.path.join(sv_target, "scene*")))
    print(f"  场景数: {sv_scenes}")
    
    print(f"\nMV目录验证:")
    mv_scenes = len(glob.glob(os.path.join(mv_target, "scene*")))
    print(f"  场景数: {mv_scenes}")

if __name__ == "__main__":
    create_clip_symlinks()
