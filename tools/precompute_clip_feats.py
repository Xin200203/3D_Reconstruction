import argparse
import os
from pathlib import Path
import mmengine
import torch
import torch.nn.functional as F
import open_clip
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from functools import partial
from PIL import Image
from torchvision.transforms import ToTensor


def parse_args():
    parser = argparse.ArgumentParser(
        description='Precompute CLIP conv1 features for ScanNet datasets')
    parser.add_argument('--info-pkl', required=True,
                        help='Path to *_infos_train.pkl or *_infos_val.pkl')
    parser.add_argument('--data-root', required=True,
                        help='Root directory that contains points/ 2D/ 等')
    parser.add_argument('--output-root', default=None,
                        help='Root path to save clip_feat (default: data_root)')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--dtype', default='fp16', choices=['fp16', 'fp32'])
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--clip-ckpt', default=None,
                        help='Local path to CLIP checkpoint (e.g. clip_pytorch_model.bin). If not provided, will try to download pretrained="openai".')
    return parser.parse_args()


def load_image(path: Path):
    """读取一张 RGB jpg 并转换为 float32 Tensor(C,H,W)，像素值 0~255"""
    pil_img = Image.open(path).convert('RGB')  # PIL.Image
    img_t = ToTensor()(pil_img) * 255.0        # float32, (0,255)
    return img_t


def process_single(record, data_root: Path, output_root: Path, clip_conv1, device, dtype):
    """Process one record in infos: could be SV (img_path) or MV (img_paths)."""
    paths = record.get('img_paths', None)
    if paths is None:
        paths = [record['img_path']]
    saved = 0
    for rel_path in paths:
        img_path = data_root / rel_path
        # build save path
        save_path = output_root / rel_path.replace('2D', 'clip_feat').replace('.jpg', '.pt')
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if save_path.exists():
            continue
        # load image
        img = load_image(img_path).to(device)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(dtype == 'fp16')):
            feat1 = clip_conv1(img.unsqueeze(0))  # 1,768,H/16,W/16
            feat2 = F.pixel_shuffle(feat1, 2).squeeze(0)  # 192,H/8,W/8
            global_feat = feat1.mean(dim=[2, 3]).squeeze(0)
        data = {
            'pix': feat2.half().cpu() if dtype == 'fp16' else feat2.cpu(),
            'global': global_feat.half().cpu() if dtype == 'fp16' else global_feat.cpu()
        }
        torch.save(data, save_path)
        saved += 1
    return saved


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root) if args.output_root else data_root

    infos = mmengine.load(args.info_pkl)
    # *_infos.pkl may be list or dict{'data_list':list}
    if isinstance(infos, dict) and 'data_list' in infos:
        infos = infos['data_list']

    # 根据是否提供本地权重来选择加载方式
    pretrained_source = args.clip_ckpt if args.clip_ckpt is not None else 'openai'

    # 如果用户给了本地权重但路径不存在，直接报错，避免 fallback 到在线下载
    if args.clip_ckpt is not None and not Path(args.clip_ckpt).exists():
        raise FileNotFoundError(f"CLIP checkpoint not found: {args.clip_ckpt}")

    # load clip visual conv1 once（open_clip 会自动判定本地文件 or 预设权重名）
    clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-16', pretrained=pretrained_source)
    clip_conv1 = clip_model.visual.conv1.to(args.device)
    clip_conv1.eval()

    # 兼容 SV (img_path) 与 MV (img_paths) 两种格式，避免提前评估导致的 KeyError
    def _count_imgs(rec):
        if 'img_paths' in rec:
            return len(rec['img_paths'])
        elif 'img_path' in rec:
            return 1
        else:
            return 0

    total_imgs = sum(_count_imgs(r) for r in infos)

    process = partial(process_single, data_root=data_root, output_root=output_root,
                      clip_conv1=clip_conv1, device=args.device,
                      dtype=args.dtype)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(tqdm(pool.map(process, infos), total=len(infos), desc='scenes'))

    print('Completed.')


if __name__ == '__main__':
    main() 