# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from indoor_converter import create_indoor_info_file
from update_infos_to_v2 import update_pkl_infos
import mmengine
from pathlib import Path


def scannet_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scannet dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_train_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_train.pkl')
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    # info_test_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_test.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_train_path)
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)
    # update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_test_path)

def scenenn_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scenenn dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)

def ThreeRScan_data_prep(root_path, info_prefix, out_dir, workers):
    """Prepare the info file for scenenn dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
    """
    create_indoor_info_file(
        root_path, info_prefix, out_dir, workers=workers)
    info_val_path = osp.join(out_dir, f'{info_prefix}_oneformer3d_infos_val.pkl')
    update_pkl_infos(info_prefix, out_dir=out_dir, pkl_path=info_val_path)

def _add_clip_paths_to_pkl(pkl_path: str, data_root: str, suffix: str = '_clip'):
    """在 pkl 的每条记录中加入 clip_feat 路径字段。

    对 SV 记录新增 ``clip_feat_path`` (str)，对 MV 记录新增 ``clip_feat_paths`` (list[str])。
    路径按照 ``2D -> clip_feat``、``.jpg -> .pt`` 规则转换；若对应 .pt 不存在则跳过该条，保持原 pkl 内容。
    """
    if not Path(pkl_path).exists():
        print(f"[add_clip_paths] pkl not found: {pkl_path}")
        return

    infos = mmengine.load(pkl_path)
    # 支持 list 以及 dict{data_list: list}
    if isinstance(infos, dict) and 'data_list' in infos:
        data_list = infos['data_list']
    else:
        data_list = infos

    updated = 0
    for rec in data_list:
        # MV
        if 'img_paths' in rec:
            if 'clip_feat_paths' in rec:
                continue  # 已处理
            feat_paths = [p.replace('2D', 'clip_feat').replace('.jpg', '.pt') for p in rec['img_paths']]
            # 可选存在性检查
            abs_paths = [Path(data_root)/fp for fp in feat_paths]
            if not all(ap.exists() for ap in abs_paths):
                # 若有缺失则保留推断路径，后续加载时自行回退
                print(f"[add_clip_paths] warning: some .pt missing for record, still writing paths")
            rec['clip_feat_paths'] = feat_paths
            updated += 1
        # SV
        elif 'img_path' in rec:
            if 'clip_feat_path' in rec:
                continue
            feat_path = rec['img_path'].replace('2D', 'clip_feat').replace('.jpg', '.pt')
            rec['clip_feat_path'] = feat_path
            updated += 1

    if updated:
        # 生成新文件名：插入 suffix（含前导 _）到 .pkl 之前
        if suffix:
            new_pkl = str(Path(pkl_path).with_suffix('')) + f"{suffix}.pkl"
        else:
            new_pkl = pkl_path  # 若 suffix 为空，则覆盖原文件

        mmengine.dump(infos, new_pkl)
        print(f"[add_clip_paths] {updated} records written to {new_pkl}")
    else:
        print(f"[add_clip_paths] no record updated for {pkl_path} (maybe already done)")

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='kitti', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/kitti',
    help='specify the root path of dataset')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/kitti',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='kitti')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
parser.add_argument('--pack-clip', action='store_true', help='Generate extra *_clip.pkl with clip_feat paths')
parser.add_argument('--clip-suffix', type=str, default='_clip', help='Suffix to append before .pkl when --pack-clip is on')
args = parser.parse_args()

if __name__ == '__main__':
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    if args.dataset in ('scannet', 'scannet_sv', 'scannet_mv', 'scannet200', 'scannet200_sv', 'scannet200_mv'):
        scannet_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
        if args.pack_clip:
            # 定位生成的 pkl 并补丁 clip 路径
            for split in ['train', 'val']:
                pkl_path = osp.join(args.out_dir, f"{args.extra_tag}_oneformer3d_infos_{split}.pkl")
                _add_clip_paths_to_pkl(pkl_path, args.root_path, suffix=args.clip_suffix)
    elif args.dataset in ('scenenn','scenenn_mv'):
        scenenn_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
        if args.pack_clip:
            for split in ['val']:
                pkl_path = osp.join(args.out_dir, f"{args.extra_tag}_oneformer3d_infos_{split}.pkl")
                _add_clip_paths_to_pkl(pkl_path, args.root_path, suffix=args.clip_suffix)
    elif args.dataset in ('3rscan', '3rscan_mv'):
        ThreeRScan_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            out_dir=args.out_dir,
            workers=args.workers)
        if args.pack_clip:
            for split in ['val']:
                pkl_path = osp.join(args.out_dir, f"{args.extra_tag}_oneformer3d_infos_{split}.pkl")
                _add_clip_paths_to_pkl(pkl_path, args.root_path, suffix=args.clip_suffix)
    else:
        raise NotImplementedError(f'Don\'t support {args.dataset} dataset.')
