#!/usr/bin/env python3
"""
RN50 CLIP Layer2 特征提取工具（无裁剪版本）

根据几何一致性原则提取 OpenAI CLIP RN50 的 layer2 特征：
- 输入：RGB 图像 (480×640) - 保持原始分辨率，不进行中心裁剪
- 预处理：仅 ToTensor + CLIP归一化，避免几何变形
- 输出：512×60×80 特征图
- 存储格式：clip_pix (fp16)，每个场景40帧采样

修改要点：
- 替换 CLIP 官方预处理，避免 Resize(224) + CenterCrop(224×224) 
- 保持 480×640 → layer2特征 ~30×40 → 插值到 60×80
- 确保 3D-2D 投影的几何一致性，提高 valid ratio
"""

import argparse
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import logging
import time
import json
import glob
from typing import Union, Any

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# CLIP 官方归一化参数
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

def preprocess_no_crop(image_pil, expect_size=(480, 640), strict=True):
    """
    仅 ToTensor + Normalize(CLIP)，不做 Resize/Crop/Pad。
    保持原始 480×640 分辨率，避免中心裁剪导致的几何变形。
    
    Args:
        image_pil: PIL Image 对象
        expect_size: 期望的图像尺寸 (H, W)
        strict: 是否严格检查尺寸
        
    Returns:
        torch.Tensor: 预处理后的张量 [C, H, W]
    """
    img = image_pil.convert("RGB")
    if strict and (img.size != (expect_size[1], expect_size[0])):
        # 这里用 warning 更安全；真正统一尺寸请在数据准备阶段完成
        logger.warning(f"输入尺寸 {img.size} != 640x480（W×H），请确保数据已统一。")
    
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])(img)


class CLIPRN50FeatureExtractor:
    """RN50 CLIP 特征提取器，提取 layer2 输出"""
    
    def __init__(self, device='cuda', dtype=torch.float32):  # 默认使用 float32
        self.device = device
        self.dtype = dtype
        self.model: Any = None
        self.preprocess: Any = None
        self._load_clip_model()
        
    def _load_clip_model(self):
        """加载 OpenAI 原版 CLIP RN50 模型"""
        try:
            import clip
            logger.info("正在加载 OpenAI CLIP RN50 模型...")
            model, preprocess = clip.load("RN50", device=self.device)
            self.model = model.float()  # 强制使用 float32
            # 不再使用官方 preprocess（它会 Resize+CenterCrop 到 224）
            self.preprocess = None
            self.model.eval()
            
            # 确认是 RN50 架构
            if not hasattr(self.model.visual, 'layer2'):
                raise ValueError("模型必须是 ResNet 架构")
            logger.info("✓ CLIP RN50 模型加载成功")
            
        except ImportError:
            logger.error("❌ 未找到 CLIP 库，请安装: pip install git+https://github.com/openai/CLIP.git")
            raise
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise
    
    def extract_layer2_features(self, image_path):
        """
        提取 RN50 layer2 特征
        
        Args:
            image_path: 图像路径
            
        Returns:
            torch.Tensor: 特征图 (512, 60, 80)
        """
        if self.model is None:
            raise RuntimeError("模型未正确加载")
            
        # 加载并预处理图像
        image = Image.open(image_path).convert('RGB')
        
        # 使用「不裁剪」的 CLIP 归一化预处理，保留 480x640
        image_tensor = preprocess_no_crop(image).unsqueeze(0).to(self.device)
        
        # 与模型权重 dtype 对齐（CLIP 通常 float32）
        image_tensor = image_tensor.type(self.model.visual.conv1.weight.dtype)
        
        with torch.no_grad():
            # 按照 ModifiedResNet 的结构进行前向传播
            
            # stem 部分
            x = self.model.visual.relu1(self.model.visual.bn1(self.model.visual.conv1(image_tensor)))
            x = self.model.visual.relu2(self.model.visual.bn2(self.model.visual.conv2(x)))
            x = self.model.visual.relu3(self.model.visual.bn3(self.model.visual.conv3(x)))
            x = self.model.visual.avgpool(x)  # 注意：这一步使得后续 layer2 的总 stride 为 16
            
            # layer1 和 layer2
            x = self.model.visual.layer1(x)
            layer2_output = self.model.visual.layer2(x)  # [B, 512, ~30, ~40] 对 480×640 输入
            
            # 统一到 60×80（方便与 3D 端对齐）
            features = torch.nn.functional.interpolate(
                layer2_output, 
                size=(60, 80), 
                mode='bilinear', 
                align_corners=False
            )
            
            # 移除 batch 维度
            features = features.squeeze(0)  # (512, 60, 80)
            
        return features
    
    def save_features(self, features, save_path):
        """保存特征到文件"""
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 根据设置的数据类型保存
        if self.dtype == torch.float16:
            features_to_save = features.half().cpu()
        else:
            features_to_save = features.cpu()
            
        torch.save(features_to_save, save_path)


def get_scene_image_paths(scene_dir, frame_interval=40):
    """
    获取场景的图像路径，按帧号间隔采样
    
    Args:
        scene_dir: 场景目录路径
        frame_interval: 帧号采样间隔（默认40，即每40帧采样一次）
        
    Returns:
        list: 采样后的图像路径列表
    """
    color_dir = scene_dir / 'color'
    if not color_dir.exists():
        return []
    
    # 获取所有jpg文件并按帧号排序
    image_files = sorted(color_dir.glob('*.jpg'), key=lambda x: int(x.stem))
    
    if len(image_files) == 0:
        return []
    
    # 按帧号间隔采样
    sampled_files = []
    
    # 获取所有帧号
    frame_numbers = [int(f.stem) for f in image_files]
    min_frame = min(frame_numbers)
    max_frame = max(frame_numbers)
    
    # 从最小帧号开始，按间隔采样
    current_frame = min_frame
    while current_frame <= max_frame:
        # 找到最接近当前目标帧号的实际帧
        closest_frame = min(frame_numbers, key=lambda x: abs(x - current_frame))
        
        # 找到对应的文件
        for f in image_files:
            if int(f.stem) == closest_frame:
                if f not in sampled_files:  # 避免重复
                    sampled_files.append(f)
                break
        
        current_frame += frame_interval
    
    return sampled_files
    
    return sampled_files


def process_scene(scene_name, data_root, output_root, extractor, frame_interval=40):
    """
    处理单个场景
    
    Args:
        scene_name: 场景名称 (如 scene0000_00)
        data_root: 数据根目录
        output_root: 输出根目录  
        extractor: 特征提取器
        frame_interval: 帧间隔（每N帧采样一次）
        
    Returns:
        int: 处理的帧数
    """
    scene_dir = data_root / scene_name
    output_scene_dir = output_root / scene_name
    
    if not scene_dir.exists():
        logger.warning(f"场景目录不存在: {scene_dir}")
        return 0
    
    # 获取采样的图像路径
    image_paths = get_scene_image_paths(scene_dir, frame_interval)
    
    if len(image_paths) == 0:
        logger.warning(f"场景 {scene_name} 没有找到图像文件")
        return 0
    
    logger.info(f"处理场景 {scene_name}: {len(image_paths)} 帧")
    
    processed_count = 0
    for img_path in tqdm(image_paths, desc=f"处理 {scene_name}", leave=False):
        try:
            # 检查输出文件是否已存在
            output_path = output_scene_dir / f"{img_path.stem}.pt"
            if output_path.exists():
                processed_count += 1
                continue
            
            # 提取特征
            features = extractor.extract_layer2_features(img_path)
            
            # 保存特征
            extractor.save_features(features, output_path)
            processed_count += 1
            
        except Exception as e:
            logger.error(f"处理图像 {img_path} 时出错: {e}")
            continue
    
    return processed_count


def main():
    parser = argparse.ArgumentParser(description='提取 RN50 CLIP Layer2 特征')
    parser.add_argument('--data-root', 
                       default='/home/nebula/xxy/ESAM/data/scannet200-sv/2D',
                       help='ScanNet 2D 数据根目录')
    parser.add_argument('--output-root',
                       default='/home/nebula/xxy/dataset/scannet/clip_feat_RN50_layer2',
                       help='输出目录')
    parser.add_argument('--scenes', nargs='+', default=None,
                       help='指定要处理的场景列表，如不指定则处理所有场景')
    parser.add_argument('--frame-interval', type=int, default=40,
                       help='帧采样间隔（每N帧采样一次）')
    parser.add_argument('--device', default='cuda',
                       help='计算设备')
    parser.add_argument('--dtype', choices=['fp16', 'fp32'], default='fp16',
                       help='数据类型')
    parser.add_argument('--resume', action='store_true',
                       help='跳过已存在的输出文件')
    
    args = parser.parse_args()
    
    # 设置路径
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    
    if not data_root.exists():
        logger.error(f"数据根目录不存在: {data_root}")
        return
    
    # 创建输出目录
    output_root.mkdir(parents=True, exist_ok=True)
    
    # 设置设备和数据类型
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    
    logger.info(f"使用设备: {device}")
    logger.info(f"数据类型: {dtype}")
    logger.info(f"帧采样间隔: {args.frame_interval}")
    
    # 初始化特征提取器
    try:
        extractor = CLIPRN50FeatureExtractor(device=str(device), dtype=dtype)
    except Exception as e:
        logger.error(f"初始化特征提取器失败: {e}")
        return
    
    # 获取要处理的场景列表
    if args.scenes:
        scene_names = args.scenes
    else:
        # 自动发现所有场景
        scene_dirs = [d for d in data_root.iterdir() 
                     if d.is_dir() and d.name.startswith('scene')]
        scene_names = [d.name for d in sorted(scene_dirs)]
    
    logger.info(f"待处理场景数量: {len(scene_names)}")
    
    # 处理统计
    total_scenes = len(scene_names)
    total_frames = 0
    processed_scenes = 0
    start_time = time.time()
    
    # 处理每个场景
    for i, scene_name in enumerate(scene_names):
        logger.info(f"[{i+1}/{total_scenes}] 开始处理场景: {scene_name}")
        
        try:
            frame_count = process_scene(
                scene_name, data_root, output_root, extractor, args.frame_interval
            )
            total_frames += frame_count
            processed_scenes += 1
            
            # 打印进度
            elapsed = time.time() - start_time
            avg_time_per_scene = elapsed / (i + 1)
            eta = avg_time_per_scene * (total_scenes - i - 1)
            
            logger.info(f"✓ 场景 {scene_name} 完成: {frame_count} 帧")
            logger.info(f"进度: {i+1}/{total_scenes}, 预计剩余时间: {eta/60:.1f} 分钟")
            
        except Exception as e:
            logger.error(f"❌ 处理场景 {scene_name} 失败: {e}")
            continue
    
    # 最终统计
    elapsed_total = time.time() - start_time
    logger.info(f"\n=== 处理完成 ===")
    logger.info(f"处理场景数: {processed_scenes}/{total_scenes}")
    logger.info(f"总处理帧数: {total_frames}")
    logger.info(f"总用时: {elapsed_total/60:.1f} 分钟")
    logger.info(f"平均每帧: {elapsed_total/total_frames:.2f} 秒" if total_frames > 0 else "")
    logger.info(f"输出目录: {output_root}")
    
    # 保存处理信息
    info_file = output_root / "extraction_info.json"
    info = {
        "processed_scenes": processed_scenes,
        "total_scenes": total_scenes, 
        "total_frames": total_frames,
        "frame_interval": args.frame_interval,
        "processing_time": elapsed_total,
        "feature_shape": [512, 60, 80],
        "dtype": args.dtype,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(info_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    logger.info(f"处理信息已保存到: {info_file}")


if __name__ == "__main__":
    main()
