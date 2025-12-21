# Adapted from mmdet3d/datasets/transforms/formating.py
import numpy as np
from mmengine.structures import InstanceData

from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.datasets.transforms.formating import to_tensor
from mmdet3d.registry import TRANSFORMS
from mmdet3d.structures import BaseInstance3DBoxes, Det3DDataSample, PointData
from mmdet3d.structures.points import BasePoints
import math
import PIL.Image as Image
import torch

@TRANSFORMS.register_module()
class Pack3DDetInputs_(Pack3DDetInputs):
    """Just add elastic_coords, sp_pts_mask, and gt_sp_masks.
    """
    INPUTS_KEYS = ['points', 'img', 'imgs', 'cam_info', 'elastic_coords', 'img_path', 'clip_pix', 'clip_global']
    SEG_KEYS = [
        'gt_seg_map',
        'pts_instance_mask',
        'pts_semantic_mask',
        'gt_semantic_seg',
        'sp_pts_mask',
    ]
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d',
        'gt_sp_masks'
    ]
    def __init__(self, keys, dataset_type='scannet'):
        super().__init__(keys)
        self.dataset_type = dataset_type

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor

        if 'pose' in results:
            depth2img = []
            unify_dim = (640, 480)
            if self.dataset_type == 'scannet' or self.dataset_type == 'scannet200':
                fx = 577.870605
                fy = 577.870605
                cx = 319.5
                cy = 239.5
            elif self.dataset_type == 'scenenn':
                fx = 544.47329
                fy = 544.47329
                cx = 320
                cy = 240
            intrinsic = adjust_intrinsic(make_intrinsic(fx, fy, cx, cy), [640,480], unify_dim)
            results['depth2img'] = intrinsic @ np.linalg.inv(results['pose'])
            del results['pose']
    
        if 'img' in results:
            img = results['img']
            # 统一将 img 转成 (C,H,W) 的 torch.Tensor，兼容 numpy / tensor / list 等多种格式
            if isinstance(img, list):
                # 多张图的情况（当前单帧流程一般不会走到这里）
                tensor_list = []
                for im in img:
                    if isinstance(im, torch.Tensor):
                        t = im.float()
                        if t.dim() == 2:
                            t = t.unsqueeze(0)
                        elif t.dim() == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
                            # (H,W,C) -> (C,H,W)
                            t = t.permute(2, 0, 1).contiguous()
                        tensor_list.append(t)
                    else:
                        arr = np.asarray(im)
                        if arr.ndim < 3:
                            arr = np.expand_dims(arr, -1)
                        if arr.flags.c_contiguous:
                            t = to_tensor(arr).permute(2, 0, 1).contiguous()
                        else:
                            t = to_tensor(
                                np.ascontiguousarray(arr.transpose(2, 0, 1)))
                        tensor_list.append(t)
                results['img'] = tensor_list
            else:
                if isinstance(img, torch.Tensor):
                    t = img.float()
                    if t.dim() == 2:
                        t = t.unsqueeze(0)
                    elif t.dim() == 3 and t.shape[0] not in (1, 3) and t.shape[-1] in (1, 3):
                        # (H,W,C) -> (C,H,W)
                        t = t.permute(2, 0, 1).contiguous()
                    results['img'] = t
                else:
                    arr = np.asarray(img)
                    if arr.ndim < 3:
                        arr = np.expand_dims(arr, -1)
                    # To improve the computational speed by by 3-5 times, apply:
                    # `torch.permute()` rather than `np.transpose()`.
                    # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
                    # for more details
                    if arr.flags.c_contiguous:
                        t = to_tensor(arr).permute(2, 0, 1).contiguous()
                    else:
                        t = to_tensor(
                            np.ascontiguousarray(arr.transpose(2, 0, 1)))
                    results['img'] = t

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'sp_pts_mask', 'gt_sp_masks',
                'elastic_coords', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        if 'img_path' in results:
            img_shape = Image.open(results['img_path']).size
            results['img_shape'] = (img_shape[1], img_shape[0])
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data_sample.img_metas = img_metas
        # data_sample.set_metainfo(img_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')
        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs

        # 轻量调试：仅前几次打印 inputs 键，确认 cam_info 是否被打包
        if not hasattr(self, '_debug_count'):
            self._debug_count = 0
        if self._debug_count < 3:
            # 多 worker 下每个 worker 都会各自触发 debug print；
            # 仅在 worker0（或 num_workers=0）输出，减少日志噪声。
            try:
                from torch.utils.data import get_worker_info
                wi = get_worker_info()
                should_print = (wi is None) or (getattr(wi, 'id', 0) == 0)
            except Exception:
                should_print = True
            if should_print:
                print(f"[Pack3DDetInputs_] keys={self.keys}, inputs_keys={list(inputs.keys())}")
            self._debug_count += 1

        return packed_results


@TRANSFORMS.register_module()
class Pack3DDetInputs_Online(Pack3DDetInputs):
    """Just add elastic_coords, sp_pts_mask, and gt_sp_masks.
    """
    INPUTS_KEYS = ['points', 'img', 'imgs', 'cam_info', 'elastic_coords', 'img_paths', 'clip_pix', 'clip_global']
    SEG_KEYS = [
        'gt_seg_map',
        'pts_instance_mask',
        'pts_semantic_mask',
        'gt_semantic_seg',
        'sp_pts_mask',
    ]
    INSTANCEDATA_3D_KEYS = [
        'gt_bboxes_3d', 'gt_labels_3d', 'attr_labels', 'depths', 'centers_2d',
        'gt_sp_masks'
    ]
    
    def __init__(self, keys, dataset_type='scannet'):
        super().__init__(keys)
        self.dataset_type = dataset_type

    def pack_single_results(self, results: dict) -> dict:
        """Method to pack the single input data. when the value in this dict is
        a list, it usually is in Augmentations Testing.

        Args:
            results (dict): Result dict from the data pipeline.

        Returns:
            dict: A dict contains

            - 'inputs' (dict): The forward data of models. It usually contains
              following keys:

                - points
                - img

            - 'data_samples' (:obj:`Det3DDataSample`): The annotation info
              of the sample.
        """
        # TODO: need to reshape: 'points', 'pts_semantic_mask', 'pts_instance_mask', 'sp_pts_mask', 'elastic_coords'
        # TODO: already temporal list: 'gt_labels_3d', 'gt_sp_masks'
        # Format 3D data
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = results['points'].tensor
            results['points'] = results['points'].reshape(results['num_frames'], results['num_sample'], -1)

        if 'poses' in results:
            depth2img = []
            unify_dim = (640, 480)
            if self.dataset_type == 'scannet' or self.dataset_type == 'scannet200':
                fx = 577.870605
                fy = 577.870605
                cx = 319.5
                cy = 239.5
            elif self.dataset_type == 'scenenn':
                fx = 544.47329
                fy = 544.47329
                cx = 320
                cy = 240
            intrinsic = adjust_intrinsic(make_intrinsic(fx, fy, cx, cy), [640,480], unify_dim)
            for pose in results['poses']:
                depth2img.append(
                    intrinsic @ np.linalg.inv(pose))
            results['depth2img'] = depth2img
        
        # if 'img' in results:
        #     if isinstance(results['img'], list):
        #         # process multiple imgs in single frame
        #         imgs = np.stack(results['img'], axis=0)
        #         if imgs.flags.c_contiguous:
        #             imgs = to_tensor(imgs).permute(0, 3, 1, 2).contiguous()
        #         else:
        #             imgs = to_tensor(
        #                 np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
        #         results['img'] = imgs
        #     elif len(results['img'].shape) == 4:
        #         imgs = results['img']
        #         if imgs.flags.c_contiguous:
        #             imgs = to_tensor(results['img']).permute(0, 3, 1, 2).contiguous()
        #         else:
        #             imgs = to_tensor(
        #                 np.ascontiguousarray(imgs.transpose(0, 3, 1, 2)))
        #         results['img'] = imgs
        #     else:
        #         img = results['img']
        #         if len(img.shape) < 3:
        #             img = np.expand_dims(img, -1)
        #         # To improve the computational speed by by 3-5 times, apply:
        #         # `torch.permute()` rather than `np.transpose()`.
        #         # Refer to https://github.com/open-mmlab/mmdetection/pull/9533
        #         # for more details
        #         if img.flags.c_contiguous:
        #             img = to_tensor(img).permute(2, 0, 1).contiguous()
        #         else:
        #             img = to_tensor(
        #                 np.ascontiguousarray(img.transpose(2, 0, 1)))
        #         results['img'] = img

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_bboxes_labels', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'sp_pts_mask', 'gt_sp_masks',
                'elastic_coords', 'centers_2d', 'depths', 'gt_labels_3d'
        ]:
            if key not in results:
                continue
            if isinstance(results[key], list):
                results[key] = [to_tensor(res) for res in results[key]]
            else:
                results[key] = to_tensor(results[key])
                if key in ['pts_semantic_mask', 'pts_instance_mask', 'sp_pts_mask', 'elastic_coords']:
                    new_shape = [results['num_frames'], results['num_sample']] + list(results[key].shape)[1:]
                    results[key] = results[key].reshape(new_shape)

        if 'gt_bboxes_3d' in results:
            if not isinstance(results['gt_bboxes_3d'], BaseInstance3DBoxes):
                results['gt_bboxes_3d'] = to_tensor(results['gt_bboxes_3d'])
                results['gt_bboxes_3d'] = [results['gt_bboxes_3d'] for i in range(results['num_frames'])]

        if 'gt_semantic_seg' in results:
            results['gt_semantic_seg'] = to_tensor(
                results['gt_semantic_seg'][None])
        if 'gt_seg_map' in results:
            results['gt_seg_map'] = results['gt_seg_map'][None, ...]

        data_sample = Det3DDataSample()
        gt_instances_3d = InstanceData()
        gt_instances = InstanceData()
        gt_pts_seg = PointData()

        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        img_metas['lidar_idx'] = results['lidar_idx']
        data_sample.img_metas = img_metas
        # data_sample.set_metainfo(img_metas)

        inputs = {}
        for key in self.keys:
            if key in results:
                if key in self.INPUTS_KEYS:
                    inputs[key] = results[key]
                elif key in self.INSTANCEDATA_3D_KEYS:
                    gt_instances_3d[self._remove_prefix(key)] = results[key]
                elif key in self.INSTANCEDATA_2D_KEYS:
                    if key == 'gt_bboxes_labels':
                        gt_instances['labels'] = results[key]
                    else:
                        gt_instances[self._remove_prefix(key)] = results[key]
                elif key in self.SEG_KEYS:
                    gt_pts_seg[self._remove_prefix(key)] = results[key]
                else:
                    raise NotImplementedError(f'Please modified '
                                              f'`Pack3DDetInputs` '
                                              f'to put {key} to '
                                              f'corresponding field')
        data_sample.gt_instances_3d = gt_instances_3d
        data_sample.gt_instances = gt_instances
        data_sample.gt_pts_seg = gt_pts_seg
        if 'eval_ann_info' in results:
            data_sample.eval_ann_info = results['eval_ann_info']
        else:
            data_sample.eval_ann_info = None

        packed_results = dict()
        packed_results['data_samples'] = data_sample
        packed_results['inputs'] = inputs
        return packed_results
    

def make_intrinsic(fx, fy, mx, my):
    intrinsic = np.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic
