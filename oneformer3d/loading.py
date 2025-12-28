# Adapted from mmdet3d/datasets/transforms/loading.py
import mmengine
import numpy as np
from typing import List, Optional, Union, Tuple
import os, pdb, json

from mmdet3d.datasets.transforms import LoadAnnotations3D
from mmdet3d.datasets.transforms.loading import get
from mmdet3d.datasets.transforms.loading import NormalizePointsColor
from mmdet3d.datasets.transforms.transforms_3d import RandomFlip3D
from mmdet.datasets.transforms import RandomFlip as MMDetRandomFlip
from mmcv.transforms.base import BaseTransform
from mmcv.transforms import Compose, LoadImageFromFile
import torchvision.transforms as T

from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures.bbox_3d import get_box_type
from mmdet3d.structures.points import BasePoints, get_points_type
import torch


@TRANSFORMS.register_module()
class LoadAnnotations3D_(LoadAnnotations3D):
    """Just add super point mask loading.
    
    Args:
        with_sp_mask_3d (bool): Whether to load super point maks. 
    """

    def __init__(self, with_sp_mask_3d, **kwargs):
        self.with_sp_mask_3d = with_sp_mask_3d
        super().__init__(**kwargs)

    def _load_sp_pts_3d(self, results):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        sp_pts_mask_path = results['super_pts_path']
        try:
            mask_bytes = get(
                sp_pts_mask_path, backend_args=self.backend_args)
            # add .copy() to fix read-only bug
            sp_pts_mask = np.frombuffer(
                mask_bytes, dtype=np.int64).copy()
        except ConnectionError:
            mmengine.check_file_exist(sp_pts_mask_path)
            sp_pts_mask = np.fromfile(
                sp_pts_mask_path, dtype=np.int64)
        results['sp_pts_mask'] = sp_pts_mask

        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['sp_pts_mask'] = sp_pts_mask
            results['eval_ann_info']['lidar_idx'] = \
                sp_pts_mask_path.split("/")[-1][:-4]
        return results

    def transform(self, results: dict) -> dict:
        """Function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().transform(results)
        if self.with_sp_mask_3d:
            results = self._load_sp_pts_3d(results)
        return results


@TRANSFORMS.register_module()
class NormalizeCamInfo(BaseTransform):
    """Canonicalize cam_info to a single dict with tensor fields.

    This prevents default_collate from producing nested list-of-tensor(B)
    structures when cam_info is a list, and guarantees stable downstream
    parsing (intrinsics/img_size_dino shapes).
    """

    def __init__(self, *, strict: bool = False) -> None:
        super().__init__()
        self.strict = strict

    @staticmethod
    def _to_tensor_1d(x, *, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.reshape(-1).to(dtype=dtype)
        else:
            t = torch.tensor(list(x), dtype=dtype).reshape(-1)
        return t

    @staticmethod
    def _to_tensor_2d(x, *, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(x):
            t = x.to(dtype=dtype)
        else:
            t = torch.tensor(x, dtype=dtype)
        return t.reshape(4, 4)

    def _normalize_single(self, cam_meta: dict) -> dict:
        intr = cam_meta.get('intrinsics', None)
        if intr is not None:
            intr_t = self._to_tensor_1d(intr, dtype=torch.float32)
            if intr_t.numel() != 4 and self.strict:
                raise RuntimeError(f"[NormalizeCamInfo] invalid intrinsics shape: {intr_t.shape}")
            cam_meta['intrinsics'] = intr_t[:4].to(torch.float32)

        img_size = cam_meta.get('img_size_dino', None)
        if img_size is not None:
            img_t = self._to_tensor_1d(img_size, dtype=torch.int64)
            if img_t.numel() != 2 and self.strict:
                raise RuntimeError(f"[NormalizeCamInfo] invalid img_size_dino shape: {img_t.shape}")
            cam_meta['img_size_dino'] = img_t[:2].to(torch.int64)

        for k in ('pose', 'extrinsics'):
            mat = cam_meta.get(k, None)
            if mat is None:
                continue
            mat_t = self._to_tensor_2d(mat, dtype=torch.float32)
            if mat_t.shape != (4, 4) and self.strict:
                raise RuntimeError(f"[NormalizeCamInfo] invalid {k} shape: {mat_t.shape}")
            cam_meta[k] = mat_t

        if 'img_valid' in cam_meta:
            cam_meta['img_valid'] = bool(cam_meta['img_valid'])

        return cam_meta

    def transform(self, results: dict) -> dict:
        cam_info = results.get('cam_info', None)
        if cam_info is None:
            return results

        if isinstance(cam_info, list):
            if len(cam_info) == 1 and isinstance(cam_info[0], dict):
                cam_info = cam_info[0]
            elif self.strict:
                raise RuntimeError(f"[NormalizeCamInfo] unexpected cam_info list len={len(cam_info)}")
            else:
                cam_info = [self._normalize_single(m) for m in cam_info if isinstance(m, dict)]
                results['cam_info'] = cam_info
                return results

        if isinstance(cam_info, dict):
            results['cam_info'] = self._normalize_single(cam_info)
            return results

        if self.strict:
            raise RuntimeError(f"[NormalizeCamInfo] unsupported cam_info type: {type(cam_info)}")

        return results

@TRANSFORMS.register_module()
class RandomFlip3D_Sync2DWithVF(RandomFlip3D):
    """RandomFlip3D with sync_2d horizontal flip but keeps vertical BEV flip.

    Upstream mmdet3d `RandomFlip3D(sync_2d=True)` forces `pcd_vertical_flip=False`
    when `img` is present. In our online-DINO pipeline we include `img`, so the
    default behavior silently disables vertical BEV augmentation compared to the
    baseline (image-free) pipeline.

    This transform:
    - Keeps 2D flip (and meta keys `flip`, `flip_direction`) consistent with
      horizontal 3D flip when `sync_2d=True`.
    - Still samples and applies vertical 3D flip by `flip_ratio_bev_vertical`
      (vertical flip has no 2D counterpart here, so we do NOT flip the image).
    """

    def transform(self, input_dict: dict) -> dict:
        # 仅打印一次，避免训练日志噪声
        if not hasattr(self, '_vf_logged'):
            try:
                from torch.utils.data import get_worker_info
                wi = get_worker_info()
                should_print = (wi is None) or (getattr(wi, 'id', 0) == 0)
            except Exception:
                should_print = True
            if should_print and self.sync_2d and float(self.flip_ratio_bev_vertical) > 0:
                print(
                    "[RandomFlip3D_Sync2DWithVF] sync_2d=True (image horizontal flip synced) "
                    f"+ pcd_vertical_flip enabled (p={float(self.flip_ratio_bev_vertical):.2f})."
                )
            self._vf_logged = True

        # 1) Flip 2D image (and set `flip` / `flip_direction`) if present.
        if 'img' in input_dict:
            MMDetRandomFlip.transform(self, input_dict)

        # 2) Decide 3D flips.
        if self.sync_2d and 'img' in input_dict:
            input_dict['pcd_horizontal_flip'] = bool(input_dict.get('flip', False))
            if 'pcd_vertical_flip' not in input_dict:
                input_dict['pcd_vertical_flip'] = bool(
                    np.random.rand() < float(self.flip_ratio_bev_vertical))
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                input_dict['pcd_horizontal_flip'] = bool(
                    np.random.rand() < float(self.flip_ratio_bev_horizontal))
            if 'pcd_vertical_flip' not in input_dict:
                input_dict['pcd_vertical_flip'] = bool(
                    np.random.rand() < float(self.flip_ratio_bev_vertical))

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        # 3) Apply 3D flip(s) and record the flow for later reverse mapping.
        if input_dict.get('pcd_horizontal_flip', False):
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])

        if input_dict.get('pcd_vertical_flip', False):
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        return input_dict


@TRANSFORMS.register_module()
class NormalizePointsColor_(NormalizePointsColor):
    """Just add color_std parameter.

    Args:
        color_mean (list[float]): Mean color of the point cloud.
        color_std (list[float]): Std color of the point cloud.
            Default value is from SPFormer preprocessing.
    """

    def __init__(self, color_mean, color_std=127.5, clamp_range=None):
        self.color_mean = color_mean
        self.color_std = color_std
        # 添加颜色值钳制范围，默认为[-3, 3]以允许合理的标准化范围
        self.clamp_range = clamp_range or [-3.0, 3.0]

    def transform(self, input_dict):
        """Call function to normalize color of points.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the normalized points.
            Updated key and value are described below.
                - points (:obj:`BasePoints`): Points after color normalization.
        """
        points = input_dict['points']
        assert points.attribute_dims is not None and \
               'color' in points.attribute_dims.keys(), \
               'Expect points have color attribute'
        
        # 记录原始颜色值范围用于调试
        orig_min = points.color.min().item()
        orig_max = points.color.max().item()
        
        if self.color_mean is not None:
            points.color = points.color - \
                           points.color.new_tensor(self.color_mean)
        if self.color_std is not None:
            points.color = points.color / \
                points.color.new_tensor(self.color_std)
        
        # 钳制颜色值到合理范围
        if self.clamp_range is not None:
            points.color = torch.clamp(points.color, 
                                     min=self.clamp_range[0], 
                                     max=self.clamp_range[1])
        
        # 检查是否有异常值（静默处理）
        if orig_min < 0 or orig_max > 255:
            pass  # 颜色值超出正常范围，已通过clamp处理
        
        input_dict['points'] = points
        return input_dict


@TRANSFORMS.register_module()
class LoadAdjacentDataFromFile(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 num_frames: int = 8,
                 num_sample: int = 20000,
                 max_frames: int = -1,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 with_bbox_3d=False,
                 with_label_3d=False,
                 with_mask_3d=True,
                 with_seg_3d=True,
                 with_sp_mask_3d=True,
                 with_rec=False,
                 cat_rec=False,
                 use_FF=False,
                 rec_data_root: Optional[str] = None,
                 backend_args: Optional[dict] = None,
                 dataset_type = 'scannet200') -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']
        assert dataset_type in ['scannet', 'scannet200', 'scenenn', '3RScan']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.num_frames = num_frames
        self.num_sample = num_sample
        self.max_frames = max_frames
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.with_bbox_3d = with_bbox_3d
        self.with_label_3d = with_label_3d
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.with_sp_mask_3d = with_sp_mask_3d
        self.with_rec = with_rec
        self.cat_rec = cat_rec
        self.use_FF = use_FF
        self.rec_data_root = rec_data_root
        self.backend_args = backend_args
        self.dataset_type = dataset_type
        
        self.loader = Compose([dict(type='LoadImageFromFile')])
        
        self.rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ])
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3,:3] = self.rotation_matrix

    def _load_points(self, pts_filenames):
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:        
            points = [np.frombuffer(get(pts_filename, backend_args=self.backend_args), dtype=np.float32)
                for pts_filename in pts_filenames]
        except ConnectionError:
            if pts_filenames[0].endswith('.npy'):
                points = [np.load(pts_filename) for pts_filename in pts_filenames]
            else:
                points = [np.fromfile(pts_filename, dtype=np.float32) for pts_filename in pts_filenames]
        points = np.concatenate(points, axis=0)

        return points
    
    def _load_masks_3d(self, results, pts_instance_mask_paths):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        try:
            pts_instance_mask = [np.frombuffer(get(pts_instance_mask_path, backend_args=self.backend_args),
                 dtype=np.int64) for pts_instance_mask_path in pts_instance_mask_paths]
        except ConnectionError:
            pts_instance_mask = [np.fromfile(pts_instance_mask_path, dtype=np.int64)
                 for pts_instance_mask_path in pts_instance_mask_paths]
        pts_instance_mask = np.concatenate(pts_instance_mask, axis=0)

        results['pts_instance_mask'] = pts_instance_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_instance_mask'] = pts_instance_mask
        return results

    def _load_semantic_seg_3d(self, results, pts_semantic_mask_paths):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        try:
            # add .copy() to fix read-only bug
            pts_semantic_mask = [np.frombuffer(get(pts_semantic_mask_path, backend_args=self.backend_args),
                 dtype=np.int64).copy() for pts_semantic_mask_path in pts_semantic_mask_paths]
        except ConnectionError:
            pts_semantic_mask = [np.fromfile(pts_semantic_mask_path, dtype=np.int64)
                 for pts_semantic_mask_path in pts_semantic_mask_paths]
        pts_semantic_mask = np.concatenate(pts_semantic_mask, axis=0)

        results['pts_semantic_mask'] = pts_semantic_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['pts_semantic_mask'] = pts_semantic_mask
        return results
    
    def _load_sp_pts_3d(self, results, sp_pts_mask_paths):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        try:
            # add .copy() to fix read-only bug
            sp_pts_mask = [np.frombuffer(get(sp_pts_mask_path, backend_args=self.backend_args),
                 dtype=np.int64).copy() for sp_pts_mask_path in sp_pts_mask_paths]
        except ConnectionError:
            sp_pts_mask = [np.fromfile(sp_pts_mask_path, dtype=np.int64)
                 for sp_pts_mask_path in sp_pts_mask_paths]
        sp_pts_mask = np.array(sp_pts_mask)
        sp_pts_mask = np.concatenate(sp_pts_mask, axis=0)

        results['sp_pts_mask'] = sp_pts_mask
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['sp_pts_mask'] = sp_pts_mask
            results['eval_ann_info']['lidar_idx'] = \
                sp_pts_mask_paths[0].split("/")[-2]
        results['lidar_idx'] = sp_pts_mask_paths[0].split("/")[-2]
        return results
    
    def _load_rec_3d(self, results, pts_filenames):
        """Private function to load 3D superpoints mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        scene_name = os.path.basename(os.path.dirname(pts_filenames[0]))
        rec_pts_filename = os.path.join('data', self.dataset_type, 'points',
                                        f'{scene_name}.bin')
        rec_ins_path = os.path.join('data', self.dataset_type, 'instance_mask',
                                    f'{scene_name}.bin')
        rec_sem_path = os.path.join('data', self.dataset_type, 'semantic_mask',
                                    f'{scene_name}.bin')

        def _load_arr(filename: str, dtype):
            try:
                return np.frombuffer(
                    get(filename, backend_args=self.backend_args), dtype=dtype)
            except ConnectionError:
                return np.fromfile(filename, dtype=dtype)

        try:
            rec_pts = _load_arr(rec_pts_filename, np.float32)
            rec_ins = _load_arr(rec_ins_path, np.int64)
            rec_sem = _load_arr(rec_sem_path, np.int64).copy()
        except (FileNotFoundError, OSError):
            # MV 数据集通常只有逐帧文件：points/<scene>/<frame>.bin。
            # 当未准备 scene-level 的 rec 文件时，回退到使用第一帧作为 rec。
            frame_pts_filename = pts_filenames[0]
            if '/points/' in frame_pts_filename:
                frame_ins_path = frame_pts_filename.replace(
                    '/points/', '/instance_mask/', 1)
                frame_sem_path = frame_pts_filename.replace(
                    '/points/', '/semantic_mask/', 1)
            else:
                # 兜底：基于路径段替换 points 目录
                parts = frame_pts_filename.split(os.sep)
                if 'points' in parts:
                    pidx = parts.index('points')
                    frame_ins_path = os.sep.join(
                        parts[:pidx] + ['instance_mask'] + parts[pidx + 1:])
                    frame_sem_path = os.sep.join(
                        parts[:pidx] + ['semantic_mask'] + parts[pidx + 1:])
                else:
                    raise

            rec_pts = _load_arr(frame_pts_filename, np.float32)
            rec_ins = _load_arr(frame_ins_path, np.int64)
            rec_sem = _load_arr(frame_sem_path, np.int64).copy()
        
        # 初始化segment_ids，确保在所有情况下都有定义
        segment_ids = np.array([])
        
        if self.dataset_type == 'scannet' or self.dataset_type == 'scannet200':
            segment_path = os.path.join('data', self.dataset_type, 'scans',
                                        scene_name,
                                        f'{scene_name}_vh_clean_2.0.010000.segs.json')
            try:
                segment_ids = np.array(
                    json.load(open(segment_path))['segIndices'])
            except FileNotFoundError:
                segment_ids = np.array([])
        elif self.dataset_type == '3RScan':
            segment_path = os.path.join('data', self.dataset_type, '3RScan',
                                        scene_name,
                                        'mesh.refined.0.010000.segs.v2.json')
            try:
                segment_ids = np.array(
                    json.load(open(segment_path))['segIndices'])
            except FileNotFoundError:
                segment_ids = np.array([])
        elif self.dataset_type == 'scenenn':
            segment_path = os.path.join('data', self.dataset_type, 'mesh_segs',
                                        f'{scene_name}.segs.json')
            try:
                segment_ids = np.array(
                    json.load(open(segment_path))['segIndices'])
            except FileNotFoundError:
                segment_ids = np.array([])

        rec_pts = rec_pts.reshape(-1, self.load_dim).copy()
        if self.dataset_type == 'scenenn':
            rec_pts.flags.writeable = True
            rec_pts[:,:3] = np.dot(self.rotation_matrix, rec_pts[:,:3].T).T
        results['rec_xyz'] = rec_pts
        results['rec_instance_mask'] = rec_ins
        results['rec_semantic_mask'] = rec_sem
        # 'eval_ann_info' will be passed to evaluator
        if 'eval_ann_info' in results:
            results['eval_ann_info']['rec_xyz'] = rec_pts
            results['eval_ann_info']['segment_ids'] = segment_ids
        return results

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        assert len(results['pts_paths']) == len(results['super_pts_paths']) \
            == len(results['pts_semantic_mask_paths']) == len(results['pts_instance_mask_paths'])
        pts_file_paths = results['pts_paths']
        pts_instance_mask_paths = results['pts_instance_mask_paths']
        pts_semantic_mask_paths = results['pts_semantic_mask_paths']
        sp_pts_mask_paths = results['super_pts_paths']
        
        # 初始化img_file_paths和poses，确保在所有情况下都有定义
        img_file_paths = []
        poses = []
        
        if self.use_FF:
            img_file_paths = results['img_paths']
            poses = results['poses']
        else:
            if 'img_paths' in results:
                del results['img_paths']
            if 'poses' in results:
                del results['poses']

        if self.num_frames > 0:
            begin_idx = np.random.randint(0, len(pts_file_paths))
            keep_view_idx = np.arange(begin_idx, begin_idx + self.num_frames)
            keep_view_idx %= len(pts_file_paths)
            pts_file_paths = [pts_file_paths[idx] for idx in keep_view_idx]
            pts_instance_mask_paths = [pts_instance_mask_paths[idx] for idx in keep_view_idx]
            pts_semantic_mask_paths = [pts_semantic_mask_paths[idx] for idx in keep_view_idx]
            sp_pts_mask_paths = [sp_pts_mask_paths[idx] for idx in keep_view_idx]
            if self.use_FF:
                img_file_paths = [img_file_paths[idx] for idx in keep_view_idx]
                poses = [poses[idx] for idx in keep_view_idx]

        if self.max_frames > 0 and len(pts_file_paths) > self.max_frames:
            choose_seq = np.floor(np.linspace(0, len(pts_file_paths) - 1, num=self.max_frames)).astype(np.int_)
            pts_file_paths = [pts_file_paths[idx] for idx in choose_seq]
            pts_instance_mask_paths = [pts_instance_mask_paths[idx] for idx in choose_seq]
            pts_semantic_mask_paths = [pts_semantic_mask_paths[idx] for idx in choose_seq]
            sp_pts_mask_paths = [sp_pts_mask_paths[idx] for idx in choose_seq]
            results['pts_paths'] = pts_file_paths
            results['pts_instance_mask_paths'] = pts_instance_mask_paths
            results['pts_semantic_mask_paths'] = pts_semantic_mask_paths
            results['super_pts_paths'] = sp_pts_mask_paths
            if self.use_FF:
                img_file_paths = [img_file_paths[idx] for idx in choose_seq]
                results['img_paths'] = img_file_paths
                poses = [poses[idx] for idx in choose_seq]
                results['poses'] = poses

        points = self._load_points(pts_file_paths)
        points = points.reshape(-1, self.load_dim)
        if self.dataset_type == 'scenenn':
            points[:,:3] = np.dot(self.rotation_matrix, points[:,:3].T).T
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = {}
            attribute_dims['color'] = [  # type: ignore[assignment]
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
            ]

        if self.with_bbox_3d:
            raise NotImplementedError("bbox_3d is not needed for Online seg")
        
        if self.with_label_3d:
            raise NotImplementedError("label_3d is not needed for Online seg")

        if self.with_mask_3d:
            results = self._load_masks_3d(results, pts_instance_mask_paths)

        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results, pts_semantic_mask_paths)

        if self.with_sp_mask_3d:
            results = self._load_sp_pts_3d(results, sp_pts_mask_paths)
        
        if self.with_rec:
            results = self._load_rec_3d(results, pts_file_paths)
            if self.cat_rec:
                points = np.concatenate([points, results['rec_xyz']], axis=0)
        
        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        
        if self.use_FF:
            imgs = []
            for img_path in img_file_paths:
                load_dict = dict(img_path=img_path, img_prefix=None)
                _img_result = self.loader(load_dict)  # type: ignore
                if isinstance(_img_result, dict):
                    imgs.append(_img_result['img'])  # type: ignore[index]
                    for k, v in _img_result.items():
                        if k not in ['img', 'img_prefix', 'img_path']:
                            results[k] = v  # 其余 meta 信息写回
                else:
                    # fallback: 仅将返回值视为图像 Tensor
                    imgs.append(_img_result)  # type: ignore[arg-type]
            results['img'] = imgs
            results['img_paths'] = img_file_paths
            results['poses'] = poses
            if self.dataset_type == 'scenenn':  
                results['poses'] = [(self.transform_matrix @ pose) for pose in poses]
        results['num_frames'] = len(pts_file_paths) if self.num_frames == -1 else self.num_frames
        results['num_sample'] = self.num_sample
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str

@TRANSFORMS.register_module()
class LoadPointsFromFile_(BaseTransform):
    """Load Points From File.

    Required Keys:

    - lidar_points (dict)

        - lidar_path (str)

    Added Keys:

    - points (np.float32)

    Args:
        coord_type (str): The type of coordinates of points cloud.
            Available options includes:

            - 'LIDAR': Points in LiDAR coordinates.
            - 'DEPTH': Points in depth coordinates, usually for indoor dataset.
            - 'CAMERA': Points in camera coordinates.
        load_dim (int): The dimension of the loaded points. Defaults to 6.
        use_dim (list[int] | int): Which dimensions of the points to use.
            Defaults to [0, 1, 2]. For KITTI dataset, set use_dim=4
            or use_dim=[0, 1, 2, 3] to use the intensity dimension.
        shift_height (bool): Whether to use shifted height. Defaults to False.
        use_color (bool): Whether to use color features. Defaults to False.
        norm_intensity (bool): Whether to normlize the intensity. Defaults to
            False.
        norm_elongation (bool): Whether to normlize the elongation. This is
            usually used in Waymo dataset.Defaults to False.
        backend_args (dict, optional): Arguments to instantiate the
            corresponding backend. Defaults to None.
    """

    def __init__(self,
                 coord_type: str,
                 load_dim: int = 6,
                 use_dim: Union[int, List[int]] = [0, 1, 2],
                 shift_height: bool = False,
                 use_color: bool = False,
                 norm_intensity: bool = False,
                 norm_elongation: bool = False,
                 backend_args: Optional[dict] = None,
                 dataset_type='scannet') -> None:
        self.shift_height = shift_height
        self.use_color = use_color
        if isinstance(use_dim, int):
            use_dim = list(range(use_dim))
        assert max(use_dim) < load_dim, \
            f'Expect all used dimensions < {load_dim}, got {use_dim}'
        assert coord_type in ['CAMERA', 'LIDAR', 'DEPTH']

        self.coord_type = coord_type
        self.load_dim = load_dim
        self.use_dim = use_dim
        self.norm_intensity = norm_intensity
        self.norm_elongation = norm_elongation
        self.backend_args = backend_args
        self.dataset_type = dataset_type
        
        self.rotation_matrix = np.array([
                [1, 0, 0],
                [0, 0, -1],
                [0, 1, 0],
            ])
        self.transform_matrix = np.eye(4)
        self.transform_matrix[:3,:3] = self.rotation_matrix

    def _load_points(self, pts_filename: str) -> np.ndarray:
        """Private function to load point clouds data.

        Args:
            pts_filename (str): Filename of point clouds data.

        Returns:
            np.ndarray: An array containing point clouds data.
        """
        try:
            pts_bytes = get(pts_filename, backend_args=self.backend_args)
            points = np.frombuffer(pts_bytes, dtype=np.float32)
        except ConnectionError:
            mmengine.check_file_exist(pts_filename)
            if pts_filename.endswith('.npy'):
                points = np.load(pts_filename)
            else:
                points = np.fromfile(pts_filename, dtype=np.float32)

        return points.copy()

    def transform(self, results: dict) -> dict:
        """Method to load points data from file.

        Args:
            results (dict): Result dict containing point clouds data.

        Returns:
            dict: The result dict containing the point clouds data.
            Added key and value are described below.

                - points (:obj:`BasePoints`): Point clouds data.
        """
        pts_file_path = results['lidar_points']['lidar_path']
        points = self._load_points(pts_file_path)
        points = points.reshape(-1, self.load_dim)
        points.flags.writeable = True
        if self.dataset_type == 'scenenn':
            points[:,:3] = np.dot(self.rotation_matrix, points[:,:3].T).T
        points = points[:, self.use_dim]
        if self.norm_intensity:
            assert len(self.use_dim) >= 4, \
                f'When using intensity norm, expect used dimensions >= 4, got {len(self.use_dim)}'  # noqa: E501
            points[:, 3] = np.tanh(points[:, 3])
        if self.norm_elongation:
            assert len(self.use_dim) >= 5, \
                f'When using elongation norm, expect used dimensions >= 5, got {len(self.use_dim)}'  # noqa: E501
            points[:, 4] = np.tanh(points[:, 4])
        attribute_dims = None

        if self.shift_height:
            floor_height = np.percentile(points[:, 2], 0.99)
            height = points[:, 2] - floor_height
            points = np.concatenate(
                [points[:, :3],
                 np.expand_dims(height, 1), points[:, 3:]], 1)
            attribute_dims = dict(height=3)

        if self.use_color:
            assert len(self.use_dim) >= 6
            if attribute_dims is None:
                attribute_dims = {}
            attribute_dims['color'] = [  # type: ignore[assignment]
                    points.shape[1] - 3,
                    points.shape[1] - 2,
                    points.shape[1] - 1,
            ]

        points_class = get_points_type(self.coord_type)
        points = points_class(
            points, points_dim=points.shape[-1], attribute_dims=attribute_dims)
        results['points'] = points
        return results

    def __repr__(self) -> str:
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__ + '('
        repr_str += f'shift_height={self.shift_height}, '
        repr_str += f'use_color={self.use_color}, '
        repr_str += f'backend_args={self.backend_args}, '
        repr_str += f'load_dim={self.load_dim}, '
        repr_str += f'use_dim={self.use_dim})'
        repr_str += f'norm_intensity={self.norm_intensity})'
        repr_str += f'norm_elongation={self.norm_elongation})'
        return repr_str

@TRANSFORMS.register_module()
class LoadClipFeature(BaseTransform):
    """Load offline CLIP conv1 features (.pt) saved by ``precompute_clip_feats.py``.

    1. 对 *SV*（单帧）样本：
       - 优先读取 ``clip_feat_path``，若不存在则由 ``img_path`` 推断。
    2. 对 *MV*（多帧）样本：
       - 优先读取 ``clip_feat_paths``，否则由 ``img_paths`` 推断。

    添加字段：
        - ``clip_pix``   : Tensor 或 List[Tensor]，C=192, H/8, W/8
        - ``clip_global``: Tensor 或 List[Tensor]，dim=768

    若指定路径的 .pt 不存在，则对应位置写入 ``None``；模型内部可据此回退到在线计算。
    """
    def __init__(self, data_root: str, dtype: str = 'fp16') -> None:
        super().__init__()
        self.data_root = data_root
        self.dtype = dtype

    def _load_single(self, rel_path: Optional[str]):
        import torch, os
        if rel_path is None:
            return None, None
        full_path = os.path.join(self.data_root, rel_path)
        if not os.path.exists(full_path):
            return None, None
        data = torch.load(full_path)
        
        # 处理不同的数据格式
        if isinstance(data, dict):
            # 原始格式：字典包含 'pix' 和 'global'
            return data.get('pix', None), data.get('global', None)
        elif isinstance(data, torch.Tensor):
            # 新格式：直接是60×80的特征张量 (512, 60, 80)
            # 这就是我们需要的 'pix' 特征，没有 'global' 特征
            return data, None
        else:
            # 未知格式
            print(f"警告: 未知的CLIP特征格式: {type(data)} 在文件 {full_path}")
            return None, None

    def transform(self, results: dict) -> dict:
        # Determine SV or MV
        if 'img_paths' in results:  # MV
            clip_paths = results.get('clip_feat_paths', None)
            if clip_paths is None:
                clip_paths = [p.replace('2D', 'clip_feat').replace('.jpg', '.pt')
                              for p in results['img_paths']]
            clip_pix_list, clip_global_list = [], []
            for rel in clip_paths:
                pix, glob = self._load_single(rel)
                clip_pix_list.append(pix)
                clip_global_list.append(glob)
            results['clip_pix'] = clip_pix_list
            results['clip_global'] = clip_global_list
        else:  # SV
            clip_path = results.get('clip_feat_path', None)
            if clip_path is None and 'img_path' in results:
                clip_path = results['img_path'].replace('2D', 'clip_feat').replace('.jpg', '.pt')
            pix, glob = self._load_single(clip_path)
            results['clip_pix'] = pix
            results['clip_global'] = glob
        return results
  
@TRANSFORMS.register_module()
class LoadSingleImageFromFile(BaseTransform):
    """Load single image for BiFusion encoder.
    
    Adds:
        - 'imgs': List[Tensor] (C,H,W) 
        - 'cam_info': dict with intrinsics/extrinsics (single camera)
    """
    def __init__(self, backend_args: Optional[dict] = None, dataset_type: str = 'scannet200', keep_imgs: bool = True):
        self.backend_args = backend_args
        self.dataset_type = dataset_type
        self.keep_imgs = keep_imgs
        # 使用 mmcv 的标准图像加载器
        from mmcv.transforms import LoadImageFromFile
        self.loader = LoadImageFromFile(backend_args=backend_args)
    
    def transform(self, results: dict) -> dict:
        if 'img_path' in results:
            # Load image using mmcv's LoadImageFromFile
            temp_results = {'img_path': results['img_path']}
            temp_results = self.loader(temp_results)
            
            # 检查加载结果是否有效
            if temp_results is None or 'img' not in temp_results:
                # 图像路径无效 / 读取失败：默认直接报错（避免静默用“假图”污染训练）。
                # 如需容错继续训练，可设置环境变量 ALLOW_MISSING_IMG=1，将使用零图并标记 img_valid=False。
                import os
                if os.environ.get('ALLOW_MISSING_IMG', '') != '1':
                    raise FileNotFoundError(f"[LoadSingleImageFromFile] failed to load img: {results.get('img_path')}")
                import torch
                import numpy as np
                results['img'] = torch.zeros((3, 224, 224), dtype=torch.float32)
                # 仍然补齐 cam_info，便于上游定位（但下游 DINO strict 可选择直接中止）
                cam_info = {}
                if self.dataset_type in ['scannet', 'scannet200']:
                    intrinsics = [577.870605, 577.870605, 319.5, 239.5]
                elif self.dataset_type == 'scenenn':
                    intrinsics = [544.47329, 544.47329, 320.0, 240.0]
                else:
                    intrinsics = [577.870605, 577.870605, 319.5, 239.5]
                cam_info['intrinsics'] = torch.tensor(intrinsics, dtype=torch.float32)
                cam_info['img_size_dino'] = torch.tensor([224, 224], dtype=torch.int64)
                if 'pose' in results:
                    pose_t = torch.as_tensor(results['pose'], dtype=torch.float32)
                    cam_info['pose'] = pose_t
                    cam_info['extrinsics'] = pose_t
                cam_info['img_valid'] = False
                results['cam_info'] = cam_info
                return results

            import torch
            import numpy as np
            # 使用异常处理来避免类型检查错误
            try:
                img = temp_results['img']  # type: ignore
            except (KeyError, TypeError):
                import os
                if os.environ.get('ALLOW_MISSING_IMG', '') != '1':
                    raise FileNotFoundError(f"[LoadSingleImageFromFile] invalid img in loader output: {results.get('img_path')}")
                results['img'] = torch.zeros((3, 224, 224), dtype=torch.float32)
                cam_info = {}
                if self.dataset_type in ['scannet', 'scannet200']:
                    intrinsics = [577.870605, 577.870605, 319.5, 239.5]
                elif self.dataset_type == 'scenenn':
                    intrinsics = [544.47329, 544.47329, 320.0, 240.0]
                else:
                    intrinsics = [577.870605, 577.870605, 319.5, 239.5]
                cam_info['intrinsics'] = torch.tensor(intrinsics, dtype=torch.float32)
                cam_info['img_size_dino'] = torch.tensor([224, 224], dtype=torch.int64)
                if 'pose' in results:
                    pose_t = torch.as_tensor(results['pose'], dtype=torch.float32)
                    cam_info['pose'] = pose_t
                    cam_info['extrinsics'] = pose_t
                cam_info['img_valid'] = False
                results['cam_info'] = cam_info
                return results

            # 保留原始 numpy 图像用于 2D pipeline（H,W,C）
            img_np = img

            # 转换为 (C,H,W) tensor 供旧的 BiFusion / 多帧路径使用
            if isinstance(img, np.ndarray):
                if img.ndim == 3 and img.shape[-1] == 3:  # (H,W,C)
                    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
                else:
                    img_tensor = torch.from_numpy(img).float()
            elif isinstance(img, torch.Tensor):
                img_tensor = img.float()
                if img_tensor.dim() == 3 and img_tensor.shape[0] not in (1, 3):
                    img_tensor = img_tensor.permute(2, 0, 1).contiguous()
            else:
                img_tensor = torch.tensor(img).float()

            # 新的在线 DINO pipeline 使用 `results['img']`，
            # 后续 ColorJitterImg / ResizeForDINO 等会接管并转换为 tensor。
            results['img'] = img_np

            # 旧的 BiFusion 路径继续通过 `results['imgs']` 读入 C,H,W tensor
            if self.keep_imgs:
                results['imgs'] = [img_tensor]  # List format for batch compatibility
            
            # Prepare cam_info with intrinsics from PKL or defaults
            cam_info = {}
            
            # 🔧 核心修复：完全忽略PKL中的内参，统一使用ScanNet标准内参
            # 参考官方数据处理策略: load_scannet_sv_data_v2_fast.py 第152行
            # 官方使用: unify_intrinsic = adjust_intrinsic(make_intrinsic(577.870605,577.870605,319.5,239.5), [640,480], unify_dim)
            # 这解释了为什么不同场景有不同内参值 - 官方根本不使用场景特定内参！
            
            # ScanNet标准内参 (fx, fy, cx, cy) - 统一使用
            if self.dataset_type in ['scannet', 'scannet200']:
                intrinsics = [577.870605, 577.870605, 319.5, 239.5]
            elif self.dataset_type == 'scenenn':
                intrinsics = [544.47329, 544.47329, 320.0, 240.0]
            else:
                intrinsics = [577.870605, 577.870605, 319.5, 239.5]  # fallback
            
            # 🔧 只在初始化时打印一次，避免重复日志
            if not hasattr(self, '_intrinsics_logged'):
                # 多 worker 下每个 worker 都会各自初始化一个 transform 实例；
                # 这里仅在 worker0（或 num_workers=0）打印一次，减少训练日志噪声。
                try:
                    from torch.utils.data import get_worker_info
                    wi = get_worker_info()
                    should_print = (wi is None) or (getattr(wi, 'id', 0) == 0)
                except Exception:
                    should_print = True
                if should_print:
                    print(f"[LoadCamInfo] 使用固定标准内参: {intrinsics} (ScanNet官方策略)")
                self._intrinsics_logged = True
            
            # 统一成 tensor，避免 default_collate 产生“list(4) of Tensor(B)”这种易混淆结构
            cam_info['intrinsics'] = torch.tensor(intrinsics, dtype=torch.float32)
            # 记录当前图像尺寸（后续 ResizeForDINO 会覆盖为 DINO 输入尺寸）
            if isinstance(img_np, np.ndarray) and img_np.ndim >= 2:
                cam_info['img_size_dino'] = torch.tensor([int(img_np.shape[0]), int(img_np.shape[1])], dtype=torch.int64)
            cam_info['img_valid'] = True

            # Use pose as both extrinsics and pose (ScanNet format: pose = cam2world)
            if 'pose' in results:
                pose_data = results['pose']
                # 🔧 关键修复：处理多相机pose数据，只使用第一个相机
                if isinstance(pose_data, (list, tuple)) and len(pose_data) > 0:
                    # 多相机情况：使用第一个相机的pose
                    first_pose = pose_data[0]
                    pose_t = torch.as_tensor(first_pose, dtype=torch.float32)
                    cam_info['extrinsics'] = pose_t  # cam2world matrix
                    cam_info['pose'] = pose_t        # 单相机pose信息
                else:
                    # 单相机情况：直接使用
                    pose_t = torch.as_tensor(pose_data, dtype=torch.float32)
                    cam_info['extrinsics'] = pose_t
                    cam_info['pose'] = pose_t
            
            # 单相机：用 dict 形式即可；后续 `default_collate` 会自然 batch 成 dict(tensor[B,...])
            results['cam_info'] = cam_info
        
        return results


@TRANSFORMS.register_module()
class BGR2RGBImg(BaseTransform):
    """Convert loaded image from BGR to RGB.

    mmcv's `LoadImageFromFile` commonly returns BGR images (OpenCV convention),
    while DINOv2 and most foundation models expect RGB.

    This transform updates:
    - results['img']: numpy(H,W,3) or torch(3,H,W)/(H,W,3)
    - results['imgs']: optional list of torch tensors (C,H,W)
    """

    def __init__(self, *, apply_to_imgs: bool = True) -> None:
        super().__init__()
        self.apply_to_imgs = apply_to_imgs

    @staticmethod
    def _swap3(x):
        import numpy as np

        if x is None:
            return x
        if isinstance(x, np.ndarray):
            if x.ndim == 3 and x.shape[-1] == 3:
                # HWC
                return np.ascontiguousarray(x[..., [2, 1, 0]])
            if x.ndim == 3 and x.shape[0] == 3:
                # CHW
                return np.ascontiguousarray(x[[2, 1, 0], ...])
            return x
        if torch.is_tensor(x):
            if x.dim() == 3 and x.shape[0] == 3:
                return x[[2, 1, 0], ...].contiguous()
            if x.dim() == 3 and x.shape[-1] == 3:
                return x[..., [2, 1, 0]].contiguous()
            if x.dim() == 4 and x.shape[1] == 3:
                return x[:, [2, 1, 0], ...].contiguous()
            return x
        return x

    def transform(self, results: dict) -> dict:
        if 'img' in results:
            results['img'] = self._swap3(results['img'])
        if self.apply_to_imgs and 'imgs' in results and isinstance(results['imgs'], list):
            swapped = []
            for t in results['imgs']:
                swapped.append(self._swap3(t))
            results['imgs'] = swapped
        return results


@TRANSFORMS.register_module()
class ColorJitterImg(BaseTransform):
    """Apply color jitter to a single image.

    This transform only perturbs RGB values and will not touch any
    geometry-related fields (cam_info, pose, intrinsics, etc.).
    """

    def __init__(self,
                 brightness: float = 0.4,
                 contrast: float = 0.4,
                 saturation: float = 0.4,
                 hue: float = 0.1) -> None:
        super().__init__()
        self.jitter = T.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation,
            hue=hue)

    def transform(self, results: dict) -> dict:
        if 'img' not in results:
            return results

        img = results['img']

        # Convert to torch tensor (C,H,W), float32.
        if isinstance(img, torch.Tensor):
            tensor = img
            if tensor.dim() == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                tensor = tensor.permute(2, 0, 1).contiguous()
            elif tensor.dim() != 3:
                tensor = tensor.view(3, *tensor.shape[-2:])
            tensor = tensor.float()
        else:
            import numpy as np
            if isinstance(img, np.ndarray):
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                if img.ndim == 3 and img.shape[-1] in (1, 3):
                    tensor = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
                elif img.ndim == 3 and img.shape[0] in (1, 3):
                    tensor = torch.from_numpy(img).contiguous().float()
                else:
                    tensor = torch.from_numpy(np.asarray(img)).float()
                    if tensor.dim() == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                        tensor = tensor.permute(2, 0, 1).contiguous()
            else:
                tensor = torch.tensor(img).float()
                if tensor.dim() == 3 and tensor.shape[0] not in (1, 3) and tensor.shape[-1] in (1, 3):
                    tensor = tensor.permute(2, 0, 1).contiguous()

        # IMPORTANT: Use fixed 0..1 normalization, NOT per-image min/max rescaling.
        # DINOv2 expects standard RGB statistics; per-image stretching changes appearance semantics too much.
        # We keep the tensor in 0..255 (float) after jitter to match downstream expectations.
        assume_255 = False
        if tensor.dtype.is_floating_point:
            maxv = float(tensor.max().item()) if tensor.numel() > 0 else 0.0
            assume_255 = maxv > 1.5
        else:
            assume_255 = True

        x01 = (tensor / 255.0) if assume_255 else tensor
        x01 = x01.clamp(0.0, 1.0)
        x01 = self.jitter(x01)
        x01 = x01.clamp(0.0, 1.0)
        out = x01 * 255.0 if assume_255 else x01

        results['img'] = out
        return results


@TRANSFORMS.register_module()
class ResizeForDINO(BaseTransform):
    """Resize image to a fixed size for DINO and update intrinsics.

    - Resize img to (target_h, target_w) with aspect ratio preserved.
    - Scale intrinsics in `cam_info` accordingly and record `img_size_dino`.
    - This transform does not perform any flip.
    """

    def __init__(self,
                 target_size: Tuple[int, int] = (420, 560)) -> None:
        super().__init__()
        self.target_h, self.target_w = target_size

    def transform(self, results: dict) -> dict:
        if 'img' not in results:
            return results

        img = results['img']

        # Expect img in (C,H,W) tensor; also support numpy (H,W,C) from mmcv.
        if isinstance(img, torch.Tensor):
            if img.dim() == 3:
                # Prefer CHW; if HWC-like, convert.
                if img.shape[0] in (1, 3, 4):
                    _, H0, W0 = img.shape
                elif img.shape[-1] in (1, 3, 4):
                    # HWC -> CHW
                    H0, W0 = int(img.shape[0]), int(img.shape[1])
                    img = img.permute(2, 0, 1).contiguous()
                else:
                    # Fallback: treat as CHW
                    _, H0, W0 = img.shape
            else:
                H0, W0 = int(img.shape[-2]), int(img.shape[-1])
        else:
            # numpy or other array-like
            import numpy as np
            if isinstance(img, np.ndarray):
                # 处理由于翻转等操作导致的负 stride，先转为连续内存
                if not img.flags['C_CONTIGUOUS']:
                    img = np.ascontiguousarray(img)
                # Robustly infer original H,W from numpy layout.
                if img.ndim == 2:
                    H0, W0 = int(img.shape[0]), int(img.shape[1])
                    img = torch.from_numpy(img).unsqueeze(0).float()  # (1,H,W)
                elif img.ndim == 3:
                    # Most common: HWC (H,W,C)
                    if img.shape[-1] in (1, 3, 4):
                        H0, W0 = int(img.shape[0]), int(img.shape[1])
                        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
                    # Possible: CHW (C,H,W)
                    elif img.shape[0] in (1, 3, 4):
                        H0, W0 = int(img.shape[1]), int(img.shape[2])
                        img = torch.from_numpy(img).contiguous().float()
                    else:
                        # Unknown layout; fall back to (H,W,?) assumption for safety.
                        H0, W0 = int(img.shape[0]), int(img.shape[1])
                        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
                        if not hasattr(self, '_resize_layout_warned'):
                            print(f"[ResizeForDINO][warn] unknown numpy img layout {img.shape}; assuming HWC.")
                            self._resize_layout_warned = True
                else:
                    # Fallback for unusual dims: use last two dims as H,W.
                    H0, W0 = int(img.shape[-2]), int(img.shape[-1])
                    img = torch.from_numpy(img).float()
            else:
                tensor = torch.tensor(img).float()
                if tensor.dim() == 3 and tensor.shape[0] not in (1, 3):
                    tensor = tensor.permute(2, 0, 1).contiguous()
                img = tensor
                H0, W0 = img.shape[-2], img.shape[-1]

        H1, W1 = self.target_h, self.target_w
        scale_h = H1 / float(H0)
        scale_w = W1 / float(W0)

        # 保持近似等比例缩放，避免改变长宽比。
        if abs(scale_h - scale_w) > 1e-3:
            # 记录一个警告，但仍然继续使用各自的尺度，防止训练中断。
            print(f"[ResizeForDINO] non-uniform scale detected: "
                  f"H0={H0},W0={W0},H1={H1},W1={W1}, "
                  f"scale_h={scale_h:.4f}, scale_w={scale_w:.4f}")

        # 使用双线性插值缩放到目标大小。
        img_resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(H1, W1), mode='bilinear', align_corners=False
        ).squeeze(0)
        results['img'] = img_resized

        # 同步更新 cam_info 中的 intrinsics / img_size_dino（若存在）。
        cam_info = results.get('cam_info', None)
        if cam_info is not None:
            # 支持 cam_info 是 list[dict] 或 dict
            metas = cam_info if isinstance(cam_info, list) else [cam_info]
            for meta in metas:
                intr = meta.get('intrinsics', None)
                if intr is None:
                    continue
                # intrinsics 允许 list[float] 或 tensor(4,)
                if torch.is_tensor(intr):
                    if intr.numel() != 4:
                        continue
                    fx, fy, cx, cy = [float(x) for x in intr.reshape(-1)]
                    fx_new = fx * scale_w
                    fy_new = fy * scale_h
                    cx_new = cx * scale_w
                    cy_new = cy * scale_h
                    meta['intrinsics'] = intr.new_tensor([fx_new, fy_new, cx_new, cy_new]).to(torch.float32)
                    meta['img_size_dino'] = torch.tensor([int(H1), int(W1)], dtype=torch.int64)
                else:
                    if not (isinstance(intr, (list, tuple)) and len(intr) == 4):
                        continue
                    fx, fy, cx, cy = intr
                    fx_new = fx * scale_w
                    fy_new = fy * scale_h
                    cx_new = cx * scale_w
                    cy_new = cy * scale_h
                    # IMPORTANT: always keep cam_info as tensor to avoid ambiguous default_collate outputs
                    meta['intrinsics'] = torch.tensor(
                        [float(fx_new), float(fy_new), float(cx_new), float(cy_new)],
                        dtype=torch.float32)
                    meta['img_size_dino'] = torch.tensor([int(H1), int(W1)], dtype=torch.int64)
                # Guardrail: warn once if intrinsics look suspicious (often caused by wrong H/W inference).
                if (not hasattr(self, '_intrinsics_scale_warned') and
                        (float(fx_new) > 5000 or float(fy_new) > 5000 or
                         float(cx_new) > 2 * float(W1) or float(cy_new) > 2 * float(H1))):
                    print(
                        "[ResizeForDINO][warn] suspicious intrinsics after resize: "
                        f"(H0,W0)=({H0},{W0})->(H1,W1)=({H1},{W1}), "
                        f"scale_h={scale_h:.4f}, scale_w={scale_w:.4f}, "
                        f"intr_old={[float(fx), float(fy), float(cx), float(cy)]}, "
                        f"intr_new={[float(fx_new), float(fy_new), float(cx_new), float(cy_new)]}"
                    )
                    self._intrinsics_scale_warned = True

        return results
