import argparse
import os
import time
import json
from pathlib import Path

import numpy as np
import torch
import open_clip
from tqdm import tqdm

from oneformer3d.clip_utils import build_uv_index, sample_img_feat


@torch.inference_mode()
def extract_single(scene_id: str, frame_idx: int, rgb_path: Path, pcd_path: Path,
                   intr: torch.Tensor, extr: torch.Tensor, device, dtype, output_dir: Path):
    """提取单帧 CLIP 特征并保存 .npz & 可视化占位图。"""
    # 1. load image
    from PIL import Image
    img = Image.open(rgb_path).convert('RGB').resize((224, 224))
    img = torch.tensor(np.array(img)).permute(2, 0, 1).float() / 255.
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
    img = (img - mean) / std
    img = img.to(device, dtype)

    # 2. load points (N,6) xyzrgb
    pts = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 6)
    xyz_cam = torch.from_numpy(pts[:, :3]).to(device, dtype)

    # 3. CLIP visual encoder
    model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained='openai', device=device)
    model.visual.eval().requires_grad_(False)
    with torch.cuda.amp.autocast(dtype=torch.float16 if dtype==torch.float16 else torch.float32):
        feat_map = model.visual(img.unsqueeze(0))  # 1,C,14,14
    # PixelShuffle ×2（无可训练参数）
    feat_map = torch.nn.functional.pixel_shuffle(feat_map, 2)  # 1,192,28,28
    #   若需再次上采样至 56×56，可再调用一次 PixelShuffle
    #   此处为了示例保持 28×28；Conv1×1 留待在线流程训练。

    # 4. projection & sampling
    valid, uv = build_uv_index(xyz_cam, intr.to(device, dtype), feat_map.shape[-2:])
    sampled = xyz_cam.new_zeros((xyz_cam.shape[0], 256))
    if valid.any():
        sampled[valid] = sample_img_feat(feat_map, uv[valid])

    # 5. save
    save_dir = output_dir / scene_id
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(save_dir / f'frame_{frame_idx}.npz',
                        pts_feat=sampled.cpu().numpy().astype(np.float16),
                        uv_idx=uv.cpu().numpy(),
                        conf_2d=valid.cpu().numpy().astype(np.uint8))

    # optional: 返回统计信息
    return sampled.shape[0]


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP visual features for point cloud.')
    parser.add_argument('--scenes_file', required=True, help='txt of scene ids')
    parser.add_argument('--frames_per_scene', type=int, default=5)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--dtype', choices=['fp16', 'fp32'], default='fp16')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if args.dtype == 'fp16' else torch.float32
    output_dir = Path(args.output_dir)

    # dummy intr/extr placeholder (实际应读取文件)
    intr = torch.tensor([577.5, 577.5, 319.5, 239.5])  # fx fy cx cy
    extr = torch.eye(4)

    with open(args.scenes_file) as f:
        scenes = [l.strip() for l in f if l.strip()]

    t0 = time.time()
    total_pts = 0
    for sid in tqdm(scenes):
        for idx in range(args.frames_per_scene):
            rgb_path = Path(f'data/{sid}/frame_{idx:04d}.png')
            pcd_path = Path(f'data/{sid}/frame_{idx:04d}.bin')  # xyzrgb float32
            if not rgb_path.exists() or not pcd_path.exists():
                continue
            total_pts += extract_single(sid, idx, rgb_path, pcd_path, intr, extr, device, dtype, output_dir)

    print(f'Finished. total points={total_pts}, time={time.time()-t0:.1f}s')


if __name__ == '__main__':
    main() 