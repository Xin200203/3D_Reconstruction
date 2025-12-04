#!/usr/bin/env python3
"""
离线提取 DINOv2 ViT 特征（保持几何一致性，不裁剪、不拉伸）。

要点：
1) 输入：默认使用 480×640 原图（与深度一致），仅做 ToTensor + DINOv2 归一化。
2) 模型：torch.hub 加载 facebookresearch/dinov2 的 vitl14（可改配置）。
3) 输出：最后一层 patch token 特征，reshape 为 (C, H_p, W_p)，保存到 clip_feat 目录。
4) 路径/命名：保持与 RN50 脚本一致，便于 create_data.py 的 _add_clip_paths_to_pkl 复用。

使用示例：
python tools/extract_dinov2_features.py \
  --data-root data/scannet200-sv/2D \
  --output-root data/scannet200-sv/clip_feat_dino \
  --frame-interval 200 \
  --device cuda --dtype fp16
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("dinov2_extract")

# DINOv2 预训练的归一化参数（ImageNet 标准）
DINO_MEAN = (0.485, 0.456, 0.406)
DINO_STD = (0.229, 0.224, 0.225)


def preprocess_image(image_pil: Image.Image,
                     expect_size: Tuple[int, int] = (480, 640),
                     strict: bool = False,
                     patch_size: int = 14) -> torch.Tensor:
    """预处理：RGB -> Tensor -> Normalize，并在需要时 pad 到 patch_size 的倍数。

    - 不做拉伸/裁剪，保持几何一致性。
    - 若 H 或 W 不是 patch_size 的倍数，则在右/下做零填充到最近的倍数。
    """
    img = image_pil.convert("RGB")
    if strict and img.size != (expect_size[1], expect_size[0]):
        logger.warning(f"输入尺寸 {img.size} 与期望 {expect_size[1]}x{expect_size[0]} 不一致，请确认数据预处理。")
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=DINO_MEAN, std=DINO_STD),
    ])
    tensor = transform(img)  # (3, H, W)
    _, h, w = tensor.shape
    # pad 到 patch_size 的倍数（只在下/右填充，避免几何失真）
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return tensor


class DINOv2FeatureExtractor:
    """DINOv2 提取器：加载模型 -> 前向 -> 取 patch token 特征 -> reshape 保存。"""

    def __init__(self,
                 arch: str = "dinov2_vitl14",
                 device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 checkpoint: Optional[str] = None):
        self.arch = arch
        self.device = device
        self.dtype = dtype
        self.checkpoint = checkpoint
        self.model = None
        self._load_model()

    def _load_model(self):
        """加载 DINOv2；若提供 checkpoint 则优先使用本地权重。"""
        valid_arch = {
            "dinov2_vits14", "dinov2_vits14_reg",
            "dinov2_vitb14", "dinov2_vitb14_reg",
            "dinov2_vitl14", "dinov2_vitl14_reg",
            "dinov2_vitg14", "dinov2_vitg14_reg",
        }
        if self.arch not in valid_arch:
            raise RuntimeError(f"arch={self.arch} 不在合法列表 {valid_arch}")
        try:
            # 若提供 checkpoint，关闭预训练下载
            self.model = torch.hub.load(
                "facebookresearch/dinov2", self.arch, pretrained=self.checkpoint is None
            )
            if self.checkpoint:
                ckpt = torch.load(self.checkpoint, map_location=self.device)
                # 兼容多种保存格式
                if isinstance(ckpt, dict):
                    if "model" in ckpt:
                        state_dict = ckpt["model"]
                    elif "state_dict" in ckpt:
                        state_dict = ckpt["state_dict"]
                    else:
                        state_dict = ckpt
                else:
                    state_dict = ckpt
                missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"load_state_dict 缺少参数: {missing[:5]}{' ...' if len(missing)>5 else ''}")
                if unexpected:
                    logger.warning(f"load_state_dict 存在未使用参数: {unexpected[:5]}{' ...' if len(unexpected)>5 else ''}")
                logger.info(f"✓ 从本地权重加载: {self.checkpoint}")
            self.model.eval()
            self.model.to(self.device)
            logger.info(f"✓ 成功加载 {self.arch}")
        except Exception as e:
            logger.error(f"加载 DINOv2 模型失败: {e}")
            raise

    @torch.inference_mode()
    def extract_features(self, image_path: Path) -> torch.Tensor:
        """对单张图像提取特征，返回 (C, H_p, W_p)。"""
        img = Image.open(image_path).convert("RGB")
        img_tensor = preprocess_image(img)  # (3, H_pad, W_pad), float32
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        # 对齐模型权重 dtype（通常 float32）
        img_tensor = img_tensor.to(dtype=self.model.patch_embed.proj.weight.dtype)

        # forward_features 输出包含 patch tokens，需要 reshape
        feats = self.model.forward_features(img_tensor)
        # 兼容不同返回格式，避免对 Tensor 取布尔值
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                tokens = feats["x_norm_patchtokens"]
            elif "x_prenorm" in feats:
                tokens = feats["x_prenorm"]
            else:
                tokens = None
            h, w = feats.get("h"), feats.get("w")
            if isinstance(h, torch.Tensor):
                h = int(h.item())
            if isinstance(w, torch.Tensor):
                w = int(w.item())
        else:
            # 退化情形，不应出现；留作兜底
            tokens, h, w = feats, None, None

        if tokens is None:
            raise RuntimeError("未能从 forward_features 得到 patch tokens。")
        # tokens: (B, L, C)
        B, L, C = tokens.shape
        if h is None or w is None:
            # 若未提供 h, w，则尝试平方根推断
            hw = int(np.sqrt(L))
            h = w = hw
        tokens = tokens[0, :h * w, :]  # (L, C)
        feat_map = tokens.transpose(0, 1).reshape(C, h, w).contiguous()  # (C, H_p, W_p)

        # 若需要节省存储，可保存 fp16
        if self.dtype == torch.float16:
            feat_map = feat_map.half()
        return feat_map.cpu()

    def save(self, tensor: torch.Tensor, save_path: Path):
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, save_path)


def get_scene_image_paths(scene_dir: Path, frame_interval: int) -> List[Path]:
    """按帧号间隔采样场景中的 color/*.jpg。"""
    color_dir = scene_dir / "color"
    if not color_dir.exists():
        return []
    image_files = sorted(color_dir.glob("*.jpg"), key=lambda x: int(x.stem))
    if not image_files:
        return []

    sampled = []
    frame_numbers = [int(f.stem) for f in image_files]
    min_frame, max_frame = min(frame_numbers), max(frame_numbers)
    current = min_frame
    while current <= max_frame:
        # 选最近的帧号
        closest = min(frame_numbers, key=lambda x: abs(x - current))
        for f in image_files:
            if int(f.stem) == closest:
                if f not in sampled:
                    sampled.append(f)
                break
        current += frame_interval
    return sampled


def process_scene(scene_name: str,
                  data_root: Path,
                  output_root: Path,
                  extractor: DINOv2FeatureExtractor,
                  frame_interval: int,
                  strict_size: bool = False) -> int:
    """处理单个场景，返回成功处理的帧数。"""
    scene_dir = data_root / scene_name
    output_scene_dir = output_root / scene_name
    if not scene_dir.exists():
        logger.warning(f"场景目录不存在: {scene_dir}")
        return 0

    image_paths = get_scene_image_paths(scene_dir, frame_interval)
    if not image_paths:
        logger.warning(f"场景 {scene_name} 未找到图像")
        return 0

    logger.info(f"处理场景 {scene_name}: {len(image_paths)} 帧")
    count = 0
    for img_path in tqdm(image_paths, desc=f"{scene_name}", leave=False):
        out_path = output_scene_dir / f"{img_path.stem}.pt"
        if out_path.exists():
            count += 1
            continue
        try:
            # 预处理严格尺寸检查（可选）
            _ = preprocess_image(Image.open(img_path), strict=strict_size)
            feat = extractor.extract_features(img_path)
            extractor.save(feat, out_path)
            count += 1
        except Exception as e:
            logger.error(f"处理 {img_path} 失败: {e}")
            continue
    return count


def main():
    parser = argparse.ArgumentParser(description="离线提取 DINOv2 特征并保存为 clip_feat/*.pt")
    parser.add_argument("--data-root", default="data/scannet200-sv/2D", help="2D 图像根目录")
    parser.add_argument("--output-root", default="data/scannet200-sv/clip_feat_dino", help="特征输出目录")
    parser.add_argument("--scenes", nargs="+", default=None, help="指定场景列表（默认自动遍历 scene*）")
    parser.add_argument("--frame-interval", type=int, default=200, help="帧采样间隔（SV=200，MV=40）")
    parser.add_argument("--device", default="cuda", help="设备（cuda/cpu）")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="保存精度")
    parser.add_argument("--arch", default="dinov2_vitl14", help="DINOv2 模型架构标识（hubconf 名称，如 dinov2_vitl14_reg）")
    parser.add_argument("--checkpoint", type=str, default=None, help="本地权重路径，提供则不自动下载")
    parser.add_argument("--strict-size", action="store_true", help="严格检查输入分辨率与期望尺寸一致")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    if not data_root.exists():
        logger.error(f"数据根目录不存在: {data_root}")
        return
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    logger.info(f"设备: {device}, dtype: {dtype}, 帧采样间隔: {args.frame_interval}")

    # 初始化提取器
    extractor = DINOv2FeatureExtractor(
        arch=args.arch, device=str(device), dtype=dtype, checkpoint=args.checkpoint
    )

    # 场景列表
    if args.scenes:
        scene_names = args.scenes
    else:
        scene_dirs = [d for d in data_root.iterdir() if d.is_dir() and d.name.startswith("scene")]
        scene_names = [d.name for d in sorted(scene_dirs)]
    logger.info(f"待处理场景: {len(scene_names)}")

    total_frames = 0
    start = time.time()
    for idx, scene in enumerate(scene_names):
        logger.info(f"[{idx+1}/{len(scene_names)}] 处理 {scene}")
        n = process_scene(scene, data_root, output_root, extractor, args.frame_interval, strict_size=args.strict_size)
        total_frames += n
    elapsed = time.time() - start
    logger.info(f"完成。场景数 {len(scene_names)}，帧数 {total_frames}，总用时 {elapsed/60:.1f} 分钟")

    info = {
        "scenes": len(scene_names),
        "frames": total_frames,
        "frame_interval": args.frame_interval,
        "arch": args.arch,
        "dtype": args.dtype,
        "output_root": str(output_root),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(output_root / "extraction_info.json", "w") as f:
        json.dump(info, f, indent=2)
    logger.info(f"信息已保存到 {output_root/'extraction_info.json'}")


if __name__ == "__main__":
    main()
