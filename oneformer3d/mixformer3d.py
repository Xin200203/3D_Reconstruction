import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import MinkowskiEngine as ME
import pointops
import pdb, time
from functools import partial
from mmdet3d.registry import MODELS
from mmdet3d.structures import PointData
from mmdet3d.models import Base3DDetector
from mmdet3d.structures.bbox_3d import get_proj_mat_by_coord_type
from mmengine.structures import InstanceData
from .mask_matrix_nms import mask_matrix_nms
from .oneformer3d import ScanNetOneFormer3DMixin
from .instance_merge import ins_merge_mat, ins_cat, ins_merge, OnlineMerge
import numpy as np
from .img_backbone import point_sample
from .projection_utils import project_points_to_uv, SCANET_INTRINSICS, sample_img_feat
from .dino_sparse_fpn import build_sparse_fpn
from typing import Any, Dict, List, Tuple, Union, Optional, cast
from mmengine import ConfigDict

@MODELS.register_module()
class ScanNet200MixFormer3D(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(_cfg(backbone, 'backbone'))
        self.neck = MODELS.build(_cfg(neck, 'neck')) if neck is not None else None
        self.pool = MODELS.build(_cfg(pool, 'pool'))
        decoder_cfg = _cfg(decoder, 'decoder')
        self.decoder = MODELS.build(decoder_cfg)
        self.criterion = MODELS.build(_cfg(criterion, 'criterion'))
        self.test_cfg: Any = ConfigDict(test_cfg or {})
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        # DINO 调试计数（限制打印次数）
        self._dino_debug_count = 0

    def initialize_training_optimization(self):
        """Placeholder for compatibility; no-op in lightweight build."""
        return None

    def get_param_groups(self, base_lr: float = 1e-4) -> List[Dict]:
        """Return a single parameter group for default optimizer setup."""
        return [{'params': self.parameters(), 'lr': base_lr}]

    def update_training_progress(self, step: int):
        """Update training progress and apply scheduled modifications."""
        return None

    def _get_freeze_config_for_progress(self, progress: float) -> Dict[str, bool]:
        """Get freeze configuration based on training progress."""
        config = {}
        
        # Example progressive freeze schedule
        if progress < 0.1:
            # Early stage: freeze CLIP backbone completely
            config['clip_vit_all'] = True
        elif progress < 0.3:
            # Mid-early: unfreeze late layers, keep early layers frozen
            config['clip_vit_early'] = True
            config['clip_vit_late'] = False
        elif progress < 0.7:
            # Mid-late: unfreeze all ViT layers
            config['clip_vit_all'] = False
        # Late stage: everything unfrozen
        
        return config

    def extract_feat(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any]) -> Tuple[List[Any], List[torch.Tensor], Any]:  # type: ignore[override]
        """Extract features from sparse tensor."""
        # === DINO FPN 构建（单帧）===
        dino_feats: Optional[List[ME.SparseTensor]] = None
        # 轻量调试：查看关键键以及 cam_info 类型/长度
        if self._dino_debug_count < 5:
            cam_info_field = batch_inputs_dict.get('cam_info', None)
            if isinstance(cam_info_field, list):
                cam_len = len(cam_info_field)
            elif cam_info_field is None:
                cam_len = 0
            else:
                cam_len = 1
            has_keys = {
                'dino_fpn': 'dino_fpn' in batch_inputs_dict,
                'dino_feats': 'dino_feats' in batch_inputs_dict,
                'dino_point_feats': 'dino_point_feats' in batch_inputs_dict,
                'clip_pix': batch_inputs_dict.get('clip_pix', None) is not None,
                'cam_info': cam_len > 0,
            }
            clip_shape = None
            clip_val = batch_inputs_dict.get('clip_pix', None)
            if isinstance(clip_val, list):
                clip_val = clip_val[0] if len(clip_val) > 0 else None
            if torch.is_tensor(clip_val):
                clip_shape = clip_val.shape
            print(f"[DINO][debug] keys={has_keys}, clip_shape={clip_shape}, cam_info_type={type(cam_info_field)}, cam_len={cam_len}")
            self._dino_debug_count += 1
        if 'dino_fpn' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_fpn']
                if self._dino_debug_count < 3:
                    print(f"[DINO] use provided dino_fpn (single-frame)")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        elif 'dino_feats' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_feats']
                if self._dino_debug_count < 3:
                    print(f"[DINO] use provided dino_feats (single-frame)")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None

        if dino_feats is None and 'dino_point_feats' in batch_inputs_dict:
            try:
                coords_list, feats_list = [], []
                for b_idx, pts in enumerate(batch_inputs_dict['points']):
                    xyz = pts[:, :3]
                    feats_2d = batch_inputs_dict['dino_point_feats'][b_idx]
                    # 注意：sparse_collate 会自动添加 batch 维，这里只需要 (N, 3) 体素坐标
                    coords = (xyz / self.voxel_size).floor().to(torch.int32)
                    coords_list.append(coords.to(device=xyz.device))
                    feats_list.append(feats_2d.to(device=xyz.device))
                coords_batch, feats_batch = ME.utils.sparse_collate(coords_list, feats_list, device=feats_list[0].device)
                dino_feats = build_sparse_fpn(coords_batch, feats_batch)
                if self._dino_debug_count < 3:
                    shapes = [x.shape for x in dino_feats]
                    strides = [x.tensor_stride for x in dino_feats]
                    print(f"[DINO] build from dino_point_feats, shapes={shapes}, strides={strides}")
                    self._dino_debug_count += 1
            except Exception as e:
                if self._dino_debug_count < 3:
                    print(f"[DINO][error] build from dino_point_feats failed: {e}")
                    self._dino_debug_count += 1
                dino_feats = None

        if dino_feats is None and 'clip_pix' in batch_inputs_dict and 'cam_info' in batch_inputs_dict:
            try:
                dino_feats = self._build_dino_fpn_from_clip(batch_inputs_dict)
                if dino_feats is not None and self._dino_debug_count < 3:
                    shapes = [x.shape for x in dino_feats]
                    strides = [x.tensor_stride for x in dino_feats]
                    print(f"[DINO] build from clip_pix, shapes={shapes}, strides={strides}")
                    self._dino_debug_count += 1
            except Exception as e:
                if self._dino_debug_count < 3:
                    print(f"[DINO][error] build from clip_pix failed: {e}")
                    self._dino_debug_count += 1
                dino_feats = None

        if dino_feats is None and self._dino_debug_count < 3:
            print("[DINO] no dino_feats used (single-frame)")
            self._dino_debug_count += 1

        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates
        
        coordinates, features, *_ = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(), dino_feats=dino_feats)
        if self.with_neck:
            assert self.neck is not None
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c, f], dim=-1) for c, f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w, *_ = self.pool(x, sp_idx, all_xyz, with_xyz=False)

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features, point_features, all_xyz_w

    def _build_dino_fpn_from_clip(self, batch_inputs_dict: Dict[str, Any]):
        """从 clip_pix + cam_info 在线投影获取点级 DINO，并构建稀疏 FPN（单帧版）。"""
        points_list = batch_inputs_dict.get('points', None)
        clip = batch_inputs_dict.get('clip_pix', None)
        cam_info_list = batch_inputs_dict.get('cam_info', None)
        if points_list is None or clip is None or cam_info_list is None:
            if self._dino_debug_count < 3:
                print(f"[DINO][warn] skip build_from_clip: points={points_list is not None}, clip={clip is not None}, cam_info={cam_info_list is not None}")
                self._dino_debug_count += 1
            return None

        # 统一采样对齐策略：align_corners=False 与 grid_sample 配套，避免半像素偏移
        align_corners = False

        if isinstance(cam_info_list, list):
            cam_metas = cam_info_list
        else:
            cam_metas = [cam_info_list]
        # clip_pix 单帧：Tensor(C,H,W) 或 list 单元素
        if isinstance(clip, list):
            clip_tensor = clip[0] if len(clip) > 0 else None
        else:
            clip_tensor = clip
        if clip_tensor is None:
            if self._dino_debug_count < 3:
                print("[DINO][warn] clip_tensor is None")
                self._dino_debug_count += 1
            return None
        if not torch.is_tensor(clip_tensor):
            clip_tensor = torch.as_tensor(clip_tensor)
        clip_tensor = clip_tensor.to(device=points_list[0].device)
        C, Hf, Wf = clip_tensor.shape

        coords_list, feats_list = [], []
        for b_idx, pts in enumerate(points_list):
            if b_idx >= len(cam_metas):
                continue
            cam_meta = cam_metas[b_idx] if isinstance(cam_metas, list) else cam_metas
            xyz = pts[:, :3]

            # 统一使用标准 ScanNet 内参，避免 cam_info 中不同格式的 intrinsics 带来不确定性
            intr = SCANET_INTRINSICS
            max_depth = cam_meta.get('max_depth', 20.0) if isinstance(cam_meta, dict) else 20.0
            pose = None
            if isinstance(cam_meta, dict):
                pose = cam_meta.get('pose') or cam_meta.get('extrinsics')

            # 世界 -> 相机
            if pose is not None:
                pose_t = torch.as_tensor(pose, device=xyz.device, dtype=xyz.dtype)
                if pose_t.shape == (4, 4):
                    try:
                        w2c = torch.linalg.inv(pose_t)
                    except Exception:
                        w2c = torch.inverse(pose_t)
                    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
                    xyz_cam = (w2c @ xyz_h.t()).t()[:, :3]
                else:
                    xyz_cam = xyz
            else:
                xyz_cam = xyz

            try:
                uv_feat, valid = project_points_to_uv(
                    xyz_cam,
                    (Hf, Wf),
                    max_depth=max_depth,
                    standard_intrinsics=intr
                )
                sampled = sample_img_feat(clip_tensor.unsqueeze(0), uv_feat, valid, align_corners=align_corners)
                sampled = sampled.to(pts.device)
            except Exception as e:
                if self._dino_debug_count < 3:
                    print(f"[DINO][error] projection/sample failed: {e}")
                    self._dino_debug_count += 1
                continue

            # 注意：sparse_collate 会自动添加 batch 维，这里只使用 (N, 3) 体素坐标
            coords = (xyz / self.voxel_size).floor().to(torch.int32)
            coords_list.append(coords)
            feats_list.append(sampled)

        if not coords_list:
            return None
        # 调试：观察 collate 前各块形状
        if self._dino_debug_count < 5:
            try:
                print(f"[DINO][debug] build_from_clip before collate: "
                      f"n_chunks={len(coords_list)}, "
                      f"coords0={coords_list[0].shape}, feats0={feats_list[0].shape}")
            except Exception:
                pass
        try:
            coords_batch, feats_batch = ME.utils.sparse_collate(
                coords_list, feats_list, device=feats_list[0].device)
            if self._dino_debug_count < 5:
                print(f"[DINO][debug] after sparse_collate: "
                      f"coords_batch={coords_batch.shape}, feats_batch={feats_batch.shape}")
            fpn = build_sparse_fpn(coords_batch, feats_batch)
            if self._dino_debug_count < 5:
                shapes = [x.shape for x in fpn]
                strides = [x.tensor_stride for x in fpn]
                print(f"[DINO][debug] FPN built: shapes={shapes}, strides={strides}")
            return fpn
        except Exception as e:
            if self._dino_debug_count < 5:
                print(f"[DINO][error] sparse_collate/build_sparse_fpn failed: {repr(e)}")
            return None

    def _forward(self, *args, **kwargs) -> Any:  # type: ignore[override]
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any], **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Calculate losses from a batch of inputs dict and data samples."""
        # Backbone
        x, point_features, all_xyz_w = self.extract_feat(batch_inputs_dict, batch_data_samples)
        # GT preparation
        gt_instances = [s.gt_instances_3d for s in batch_data_samples]
        gt_point_instances = []
        for i in range(len(gt_instances)):
            ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask
            if torch.sum(ins == -1) != 0:
                ins[ins == -1] = torch.max(ins) + 1
                ins = F.one_hot(ins)[:, :-1]
            else:
                ins = F.one_hot(ins)
            ins = ins.bool().T
            gt_point = InstanceData()
            gt_point.p_masks = ins
            gt_point_instances.append(gt_point)
        queries, gt_instances, *_ = self._select_queries(x, gt_instances)
        # Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w)
        x = self.decoder(x=x, queries=queries, sp_feats=x, p_feats=point_features, super_points=super_points)
        # Loss
        return self.criterion(x, gt_instances, gt_point_instances, None, self.decoder.mask_pred_mode)

    def predict(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any], **kwargs) -> List[Any]:  # type: ignore[override]
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.
        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input samples. Each Det3DDataSample contains 'pred_pts_seg'.
            And the `pred_pts_seg` contains following keys.
                - instance_scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - instance_labels (Tensor): Labels of instances, has a shape
                    (num_instances, )
                - pts_instance_mask (Tensor): Instance mask, has a shape
                    (num_points, num_instances) of type bool.
        """
        assert len(batch_data_samples) == 1
        ## Backbone
        x, point_features, all_xyz_w = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w)
        x = self.decoder(x=x, queries=x, sp_feats=x, p_feats=point_features, super_points=super_points)
        ## Post-processing
        pred_pts_seg = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        batch_data_samples[0].pred_pts_seg = pred_pts_seg[0]
        return batch_data_samples
    
    def predict_by_feat_instance(self, out, superpoints, score_threshold):
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(int(self.test_cfg.topk_insts), scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()

        # 兼容SP/P两种掩码域：若当前掩码不是点域，则将SP域掩码映射到点域
        try:
            if isinstance(superpoints, torch.Tensor):
                n_raw = int(superpoints.numel())
                n_mask_dim = int(mask_pred_sigmoid.shape[1])
                # 如果掩码列数与原始点数不一致，且与超点数量一致，则按超点索引映射到点域
                n_sp = int(superpoints.max().item()) + 1 if superpoints.numel() > 0 else 0
                if n_mask_dim != n_raw and n_sp > 0 and n_mask_dim == n_sp:
                    mask_pred_sigmoid = mask_pred_sigmoid[:, superpoints]
        except Exception:
            # 映射失败则保持原样，不中断推理
            pass

        if self.test_cfg.get('obj_normalization', None):
            # 使用概率阈值而非logits符号，提升稳定性
            pos = (mask_pred_sigmoid >= 0.5).float()
            denom = pos.sum(1).clamp_min(1e-6)
            mask_scores = (mask_pred_sigmoid * pos).sum(1) / denom
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = str(self.test_cfg.matrix_nms_kernel)
            scores, labels, mask_pred_sigmoid, _ = mask_matrix_nms(  # type: ignore[arg-type]
                mask_pred_sigmoid, labels, scores, kernel=kernel)

        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_thr = int(self.test_cfg.npoint_thr)
        npoint_mask = mask_pointnum > npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        return mask_pred, labels, scores

@MODELS.register_module()
class ScanNet200MixFormer3D_FF(ScanNet200MixFormer3D):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if img_backbone is not None:
            self.img_backbone = MODELS.build(_cfg(img_backbone, 'img_backbone'))
        else:
            self.img_backbone = None
        self.backbone = MODELS.build(_cfg(backbone, 'backbone'))
        self.neck = MODELS.build(_cfg(neck, 'neck')) if neck is not None else None
        self.pool = MODELS.build(_cfg(pool, 'pool'))
        self.decoder = MODELS.build(_cfg(decoder, 'decoder'))
        self.criterion = MODELS.build(_cfg(criterion, 'criterion'))
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg: Any = ConfigDict(test_cfg or {})
        self.init_weights()

        # 初始化为None，将在运行时动态创建
        self.conv = None
    
    def init_weights(self):
        if hasattr(self, 'memory') and self.memory is not None:
            self.memory.init_weights()
        if hasattr(self, 'img_backbone') and self.img_backbone is not None:
            self.img_backbone.init_weights()
    
    def extract_feat(self, batch_inputs_dict, batch_data_samples):
        """Extract features from sparse tensor.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_pts_seg.sp_pts_mask`.

        Returns:
            Tuple:
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_channels).
                List[Tensor]: of len batch_size,
                    each of shape (n_points_i, n_classes + 1).
        """
        # extract image features
        with torch.no_grad():
            if getattr(self, 'img_backbone', None) is not None and 'img_path' in batch_inputs_dict:
                _ = self.img_backbone(batch_inputs_dict['img_path'])  # type: ignore[operator]
        img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
        
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates
        
        coordinates, features, *_ = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_metas, img_shape=img_metas[0]['img_shape']))
        if self.with_neck:
            assert self.neck is not None
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w, *_ = self.pool(x, sp_idx, all_xyz, with_xyz=False)

        # apply cls_layer
        features = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
        return features, point_features, all_xyz_w

    def _f(self, x, img_features, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):  # type: ignore[name-defined]
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        # 动态初始化卷积层
        if self.conv is None:
            self.conv = nn.Sequential(
                ME.MinkowskiConvolution(projected_features.shape[1], 32, kernel_size=1, dimension=3),
                ME.MinkowskiBatchNorm(32),
                ME.MinkowskiReLU(inplace=True)
            ).to(projected_features.device)
        
        projected_features = self.conv(projected_features)
        return projected_features + x

@MODELS.register_module()
class ScanNet200MixFormer3D_Online(ScanNetOneFormer3DMixin, Base3DDetector):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 map_to_rec_pcd=True,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(_cfg(backbone, 'backbone'))
        if memory is not None:
            self.memory = MODELS.build(_cfg(memory, 'memory'))
        self.neck = MODELS.build(_cfg(neck, 'neck')) if neck is not None else None
        self.pool = MODELS.build(_cfg(pool, 'pool'))
        dec_cfg = _cfg(decoder, 'decoder')
        self.decoder = MODELS.build(dec_cfg)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(_cfg(criterion, 'criterion'))
        self.decoder_online = dec_cfg['temporal_attn']
        self.use_bbox = dec_cfg['bbox_flag']
        self.sem_len = dec_cfg['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg: Any = ConfigDict(test_cfg or {})
        self.map_to_rec_pcd = map_to_rec_pcd
        self._dino_debug_count = 0
        self.init_weights()
    
    def init_weights(self):
        if hasattr(self, 'memory') and self.memory is not None:
            self.memory.init_weights()
        if hasattr(self, 'img_backbone') and self.img_backbone is not None:
            self.img_backbone.init_weights()

    def extract_feat(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any], frame_i: int) -> Tuple[List[Any], List[torch.Tensor], Any, List[Any]]:  # type: ignore[override]
        """Extract features from sparse tensor.
        DINO 三条链路（有且仅在可用时触发，缺一则退化为纯3D）：
          1) 外部直接传入 dino_fpn / dino_feats：[s1..s16]，无需再构建。
          2) 传入点级 dino_point_feats（与 points 对齐），在此处 voxel 化并 build_sparse_fpn。
          3) 如果有 clip_pix + cam_info，则在线投影 clip_pix 到点级 → voxel → build_sparse_fpn。
        未命中上述任一分支时，dino_feats=None，U-Net 注入自动跳过。
        """
        # 调试：查看 DINO 相关键是否存在（仅前几次打印）
        if self._dino_debug_count < 3:
            has_keys = {
                'dino_fpn': 'dino_fpn' in batch_inputs_dict,
                'dino_feats': 'dino_feats' in batch_inputs_dict,
                'dino_point_feats': 'dino_point_feats' in batch_inputs_dict,
                'clip_pix': 'clip_pix' in batch_inputs_dict,
                'cam_info': 'cam_info' in batch_inputs_dict,
            }
            clip_shapes = None
            if 'clip_pix' in batch_inputs_dict:
                cp = batch_inputs_dict['clip_pix']
                if isinstance(cp, list) and len(cp) > frame_i and torch.is_tensor(cp[frame_i]):
                    clip_shapes = cp[frame_i].shape
                elif torch.is_tensor(cp):
                    clip_shapes = cp.shape
            cam_info_len = len(batch_inputs_dict.get('cam_info', [])) if isinstance(batch_inputs_dict.get('cam_info', None), list) else int('cam_info' in batch_inputs_dict)
            print(f"[DINO][debug] keys={has_keys}, clip_shape={clip_shapes}, cam_info_len={cam_info_len}, frame={frame_i}")
        # 可选：外部传入的 DINO 稀疏金字塔或点级 DINO 特征，按帧索引取用
        dino_feats = None
        # 1) 直接传入的稀疏 FPN 列表 [s1..s16]
        if 'dino_fpn' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_fpn'][frame_i]
                if self._dino_debug_count < 5:
                    print(f"[DINO] use provided dino_fpn frame={frame_i}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        elif 'dino_feats' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_feats'][frame_i]
                if self._dino_debug_count < 5:
                    print(f"[DINO] use provided dino_feats frame={frame_i}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        # 2) 如果传入了点级 DINO 特征（与 points 同顺序），自动构建 FPN
        if dino_feats is None and 'dino_point_feats' in batch_inputs_dict:
            try:
                pt_feats_list = []
                coords_list = []
                for b_idx, pts in enumerate(batch_inputs_dict['points']):
                    xyz = pts[frame_i, :, :3]  # (N,3) 已包含 elastic 等增强
                    feats_2d = batch_inputs_dict['dino_point_feats'][b_idx][frame_i]  # (N,C_dino)
                    # voxel 坐标（向下取整），添加 batch 维
                    coords = torch.cat([
                        torch.full((xyz.shape[0], 1), b_idx, dtype=torch.int32, device=xyz.device),
                        (xyz / self.voxel_size).floor().to(torch.int32)
                    ], dim=1)
                    coords_list.append(coords.cpu())  # Minkowski 要求 int32，后续 sparse_collate 处理
                    pt_feats_list.append(feats_2d.cpu())
                # 批量稀疏拼接
                coords_batch, feats_batch = ME.utils.sparse_collate(
                    coords_list, pt_feats_list, device=pt_feats_list[0].device)
                fpn = build_sparse_fpn(coords_batch, feats_batch)
                dino_feats = fpn
                if self._dino_debug_count < 3:
                    print(f"[DINO] build from dino_point_feats frame={frame_i}, coords={coords_batch.shape}, feats={feats_batch.shape}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        # 3) 若仍为空且存在 clip_pix + cam_info，尝试在线投影得到点级 DINO
        if dino_feats is None and 'clip_pix' in batch_inputs_dict and 'cam_info' in batch_inputs_dict:
            try:
                dino_feats = self._build_dino_fpn_from_clip(batch_inputs_dict, frame_i)
                if dino_feats is not None and self._dino_debug_count < 3:
                    shapes = [x.shape for x in dino_feats]
                    strides = [x.tensor_stride for x in dino_feats]
                    print(f"[DINO] build from clip_pix frame={frame_i}, shapes={shapes}, strides={strides}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        # 4) 若最终仍未构建，打印一次提示
        if dino_feats is None and self._dino_debug_count < 5:
            print(f"[DINO] no dino_feats used at frame={frame_i}")
            self._dino_debug_count += 1

        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates

        coordinates, features, *_ = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(
            field.sparse(),
            dino_feats=dino_feats,
            memory=self.memory if hasattr(self, 'memory') else None)
        if self.with_neck:
            assert self.neck is not None
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c,f], dim=-1) for c,f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w, *_ = self.pool(x, sp_idx, all_xyz, with_xyz=False)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list

    def _build_dino_fpn_from_clip(self, batch_inputs_dict: Dict[str, Any], frame_i: int):
        """从 clip_pix + cam_info 在线投影获取点级 DINO，并构建稀疏 FPN。
        假设 clip_pix 为 List[Tensor]，cam_info 为 List[dict]，points 为 List[Tensor]。
        若缺失必要信息则返回 None。
        """
        if 'points' not in batch_inputs_dict:
            return None
        points_list = batch_inputs_dict['points']
        clip_list = batch_inputs_dict.get('clip_pix', None)
        cam_info_list = batch_inputs_dict.get('cam_info', None)
        if clip_list is None or cam_info_list is None:
            return None

        coords_list, feats_list = [], []
        for b_idx, pts in enumerate(points_list):
            if b_idx >= len(cam_info_list):
                continue
            cam_meta = cam_info_list[b_idx] if isinstance(cam_info_list, list) else cam_info_list
            clip = clip_list[b_idx] if isinstance(clip_list, list) else clip_list
            if clip is None:
                continue
            # clip_pix 期望 shape (C, H, W)
            if not torch.is_tensor(clip):
                clip = torch.as_tensor(clip)
            clip = clip.to(pts.device)
            C, Hf, Wf = clip.shape

            # 取当前帧点
            xyz = pts[frame_i, :, :3]  # (N,3)

            # 准备相机参数
            intr = cam_meta[0].get('intrinsics', SCANET_INTRINSICS) if isinstance(cam_meta, list) else cam_meta.get('intrinsics', SCANET_INTRINSICS)
            max_depth = cam_meta[0].get('max_depth', 20.0) if isinstance(cam_meta, list) else cam_meta.get('max_depth', 20.0)
            pose = None
            if isinstance(cam_meta, list) and len(cam_meta) > 0:
                pose = cam_meta[0].get('pose') or cam_meta[0].get('extrinsics')
            elif isinstance(cam_meta, dict):
                pose = cam_meta.get('pose') or cam_meta.get('extrinsics')

            # 世界坐标 -> 相机坐标（若提供 cam2world 则取逆）
            if pose is not None:
                pose_t = torch.as_tensor(pose, device=xyz.device, dtype=xyz.dtype)
                if pose_t.shape == (4, 4):
                    try:
                        w2c = torch.linalg.inv(pose_t)
                    except Exception:
                        w2c = torch.inverse(pose_t)
                    xyz_h = torch.cat([xyz, torch.ones_like(xyz[:, :1])], dim=1)
                    xyz_cam = (w2c @ xyz_h.t()).t()[:, :3]
                else:
                    xyz_cam = xyz
            else:
                xyz_cam = xyz

            # 投影到特征图坐标
            uv_feat, valid = project_points_to_uv(
                xyz_cam,
                (Hf, Wf),
                max_depth=max_depth,
                standard_intrinsics=tuple(intr) if not torch.is_tensor(intr) else tuple(intr.cpu().numpy().tolist())
            )
            # 采样特征 (N, C)
            sampled = sample_img_feat(clip.unsqueeze(0), uv_feat, valid, align_corners=True)
            sampled = sampled.to(pts.device)

            # 构造稀疏坐标（同当前点的 voxel 量化方式）
            coords = torch.cat([
                torch.full((xyz.shape[0], 1), b_idx, dtype=torch.int32, device=xyz.device),
                (xyz / self.voxel_size).floor().to(torch.int32)
            ], dim=1).cpu()
            coords_list.append(coords)
            feats_list.append(sampled.cpu())

        if not coords_list:
            return None
        coords_batch, feats_batch = ME.utils.sparse_collate(coords_list, feats_list, device=feats_list[0].device)
        fpn = build_sparse_fpn(coords_batch, feats_batch)
        if self._dino_debug_count < 3:
            shapes = [x.shape for x in fpn]
            strides = [x.tensor_stride for x in fpn]
            print(f"[DINO] proj+fpn frame={frame_i}, shapes={shapes}, strides={strides}")
            self._dino_debug_count += 1
        return fpn
    
    def _select_queries(self, x: List[Any], gt_instances: List[Any], sp_xyz: Optional[List[Any]] = None, frame_i: int = 0) -> Tuple[List[Any], List[Any], List[Any]]:  # type: ignore[override]
        """Select queries for train pass.
        """
        gt_instances_ = []
        for i in range(len(x)):
            temp = InstanceData()
            temp.labels_3d = gt_instances[i].labels_3d[frame_i].to(x[i].device)
            temp.sp_masks = gt_instances[i].sp_masks[frame_i].to(x[i].device)
            bboxes_3d = gt_instances[i].bboxes_3d[frame_i].to(x[i].device)
            temp.bboxes_3d = torch.cat([bboxes_3d, torch.zeros(self.sem_len, 7).to(x[i].device)])
            gt_instances_.append(temp)

        queries = []
        for i in range(len(x)):
            if self.query_thr < 1:
                n = (1 - self.query_thr) * torch.rand(1) + self.query_thr
                n = (n * len(x[i])).ceil().int()
                ids = torch.randperm(len(x[i]))[:n].to(x[i].device)
                queries.append(x[i][ids])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks[:, ids]
                if sp_xyz is not None:
                    sp_xyz[i] = sp_xyz[i][ids]
            else:
                queries.append(x[i])
                gt_instances_[i].query_masks = gt_instances_[i].sp_masks
        
        return queries, gt_instances_, sp_xyz if sp_xyz is not None else []

    def _forward(self, *args, **kwargs) -> Any:  # type: ignore[override]
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any], **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Calculate losses from a batch of inputs dict and data samples.
        """
        losses, merge_feat_n_frames, ins_masks_query_n_frames = {}, [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        if hasattr(self, 'memory'):
            self.memory.reset()
        for frame_i in range(num_frames):
            ## Backbone
            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## GT-prepare
            gt_instances = [s.gt_instances_3d for s in batch_data_samples]
            gt_point_instances, ins_masks_query_batch = [], []
            for i in range(len(gt_instances)):
                ins = batch_data_samples[i].gt_pts_seg.pts_instance_mask[frame_i]
                if torch.sum(ins == -1) != 0:
                    # Use global instance number for each frame
                    ins[ins == -1] = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    ins = F.one_hot(ins)[:, :-1]
                else:
                    ins = F.one_hot(ins)
                    max_ids = gt_instances[i].sp_masks[frame_i].shape[0] - self.sem_len
                    if ins.shape[1] < max_ids:
                        zero_pad = torch.zeros(ins.shape[0], max_ids - ins.shape[1]).to(ins.device)
                        ins = torch.cat([ins, zero_pad], dim=-1)
                ins = ins.bool().T
                gt_point = InstanceData()
                gt_point.p_masks = ins
                gt_point_instances.append(gt_point)

            queries, gt_instances, sp_xyz = self._select_queries(x, gt_instances, sp_xyz, frame_i)
            ## Decoder
            super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w)
            x = self.decoder(x=x, queries=queries, sp_feats=x, p_feats=point_features, super_points=super_points)
            ## Query projector
            for i in range(len(gt_instances)):
                ins_masks_query = gt_instances[i].query_masks[:-self.sem_len, :]
                ins_masks_query = [ins_masks_query[i].nonzero().flatten()
                        for i in range(ins_masks_query.shape[0])]
                ins_masks_query_batch.append(ins_masks_query)
            if hasattr(self, 'merge_head'):
                merge_feat = self.merge_head(x['queries'])
                merge_feat_n_frames.append(merge_feat)
                ins_masks_query_n_frames.append(ins_masks_query_batch)
            ## Loss
            loss = self.criterion(x, gt_instances, gt_point_instances, sp_xyz, self.decoder.mask_pred_mode)
            for key, value in loss.items():
                if key in losses:
                    losses[key] += value
                else:
                    losses[key] = value
        
        ## Query contrast
        if hasattr(self, 'merge_criterion'):
            merge_feat_n_frames = [[frame[i] for frame in merge_feat_n_frames]
                 for i in range(len(merge_feat_n_frames[0]))]
            ins_masks_query_n_frames = [[frame[i] for frame in ins_masks_query_n_frames]
                 for i in range(len(ins_masks_query_n_frames[0]))]
            loss = self.merge_criterion(merge_feat_n_frames, ins_masks_query_n_frames)
            losses.update(loss)
        return losses

    def predict(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any], **kwargs) -> List[Any]:  # type: ignore[override]
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        assert len(batch_data_samples) == 1
        results, query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list, cls_preds_list = [], [], [], [], [], []
        num_frames = batch_inputs_dict['points'][0].shape[0]
        
        # Initialize variables that may be used later
        mv_mask, mv_labels, mv_scores, mv_bboxes = None, None, None, None
        mv_mask2, mv_labels2, mv_scores2 = None, None, None
        mv_queries = None
        online_merger = None
        
        if hasattr(self, 'memory'):
            self.memory.reset()
        for frame_i in range(num_frames):
            ## Backbone
            x, point_features, all_xyz_w, sp_xyz = self.extract_feat(batch_inputs_dict, batch_data_samples, frame_i)
            ## Decoder
            super_points = ([bds.gt_pts_seg.sp_pts_mask[frame_i] for bds in batch_data_samples], all_xyz_w)
            x = self.decoder(x=x, queries=x, sp_feats=x, p_feats=point_features, super_points=super_points)
            ## Post-processing
            pred_pts_seg, mapping = self.predict_by_feat(
                x, batch_data_samples[0].gt_pts_seg.sp_pts_mask[frame_i])
            results.append(pred_pts_seg[0])
            ## Query projector, semantic and geometric information
            if hasattr(self, 'merge_head'):
                query_feats = self.merge_head(x['queries'][0])
                query_feats_list.append([query_feats[mapping[0]], query_feats[mapping[1]]])
                sem_preds = x['cls_preds'][0]
                sem_preds_list.append([sem_preds[mapping[0]], sem_preds[mapping[1]]])
                sp_xyz_list.append([sp_xyz[0][mapping[0]], sp_xyz[0][mapping[1]]])
                if self.use_bbox:
                    bbox_preds = x['bboxes'][0] # [N, 6]
                    bboxes_list.append([bbox_preds[mapping[0]], bbox_preds[mapping[1]]])
            ## Online merging
            if self.test_cfg.merge_type == 'learnable_online':
                if frame_i == 0:
                    online_merger = OnlineMerge(self.test_cfg.inscat_topk_insts, self.use_bbox)
                if online_merger is not None:
                    mv_mask, mv_labels, mv_scores, mv_queries, mv_bboxes = online_merger.merge(
                        results[-1].pop('pts_instance_mask')[0],
                        results[-1].pop('instance_labels')[0],
                        results[-1].pop('instance_scores')[0],
                        results[-1].pop('instance_queries')[0],
                        query_feats_list.pop(-1)[0],
                        sem_preds_list.pop(-1)[0],
                        sp_xyz_list.pop(-1)[0],
                        bboxes_list.pop(-1)[0] if self.use_bbox else None)
                    # Empty cache. Only offline merging requires the whole list.
                    torch.cuda.empty_cache()
                    if frame_i == num_frames - 1:
                        online_merger.clean() # Ignore panoptic segmentation
        
        ## Offline merging
        if self.test_cfg.merge_type == 'learnable':
            mv_mask, mv_labels, mv_scores = ins_merge_mat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                [res[0] for res in query_feats_list],
                [res[0] for res in sem_preds_list],
                [res[0] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge_mat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                [res[1] for res in query_feats_list],
                [res[1] for res in sem_preds_list],
                [res[1] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'concat':
            mv_mask, mv_labels, mv_scores = ins_cat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_cat(
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'geometric':
            mv_mask, mv_labels, mv_scores = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
            mv_mask2, mv_labels2, mv_scores2 = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][1] for res in results],
                [res['instance_labels'][1] for res in results],
                [res['instance_scores'][1] for res in results],
                [res['instance_queries'][1] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'learnable_online':
            pass
        else:
            raise NotImplementedError("Unknown merge_type.")

        ## Offline panoptic segmentation
        mv_sem = torch.cat([res['pts_semantic_mask'][0] for res in results])
        
        # Ensure variables are not None before using them
        if mv_mask is None or mv_labels is None or mv_scores is None:
            # If no merging was performed, use the first result
            mv_mask = results[0]['pts_instance_mask'][0]
            mv_labels = results[0]['instance_labels'][0]
            mv_scores = results[0]['instance_scores'][0]
        
        if self.use_bbox and mv_bboxes is not None:
            batch_data_samples[0].pred_bbox = mv_bboxes.cpu().numpy()
        
        # Not mapping to reconstructed point clouds, return directly for visualization
        if not self.map_to_rec_pcd:
            merged_result = PointData(
                pts_semantic_mask=[mv_sem.cpu().numpy()],
                pts_instance_mask=[mv_mask.cpu().numpy()],
                instance_labels=mv_labels.cpu().numpy(),
                instance_scores=mv_scores.cpu().numpy())
            batch_data_samples[0].pred_pts_seg = merged_result
            return batch_data_samples
        
        ## Mapping to reconstructed point clouds for evaluation
        mv_xyz = batch_inputs_dict['points'][0][:, :, :3].reshape(-1, 3)
        rec_xyz = torch.tensor(batch_data_samples[0].eval_ann_info['rec_xyz'])[:, :3]
        target_coord = rec_xyz.to(mv_xyz.device).contiguous().float()
        target_offset = torch.tensor(target_coord.shape[0]).to(mv_xyz.device).float()
        source_coord = mv_xyz.contiguous().float()
        source_offset = torch.tensor(source_coord.shape[0]).to(mv_xyz.device).float()
        indices, _ = pointops.knn_query(  # type: ignore[misc]
            1, source_coord, source_offset, target_coord, target_offset)
        indices = indices.reshape(-1).long()

        merged_result = PointData(
            pts_semantic_mask=[mv_sem[indices].cpu().numpy()],
            pts_instance_mask=[mv_mask[:, indices].cpu().numpy()],
            instance_labels=mv_labels.cpu().numpy(),
            instance_scores=mv_scores.cpu().numpy())

        # Ensemble the predictions with mesh segments (eval_ann_info['segment_ids'])
        if 'segment_ids' in batch_data_samples[0].eval_ann_info:
            merged_result = self.segment_smooth(merged_result, mv_xyz.device,
                batch_data_samples[0].eval_ann_info['segment_ids'])
        batch_data_samples[0].pred_pts_seg = merged_result
        
        return batch_data_samples
    
    def segment_smooth(self, results, device, segment_ids):
        unique_ids = np.unique(segment_ids)
        new_segment_ids = np.zeros_like(segment_ids)
        for i, ids in enumerate(unique_ids):
            new_segment_ids[segment_ids == ids] = i
        segment_ids = new_segment_ids
        segment_ids = torch.from_numpy(segment_ids).to(device)
        sem_mask = torch.from_numpy(results.pts_semantic_mask[0]).to(device)
        ins_mask = torch.from_numpy(results.pts_instance_mask[0]).to(device)
        sem_mask = scatter_mean(F.one_hot(sem_mask).float(), segment_ids, dim=0)
        sem_mask = sem_mask.argmax(dim=1)[segment_ids]
        ins_mask = scatter_mean(ins_mask.float(), segment_ids, dim=1)
        ins_mask = (ins_mask > 0.5)[:, segment_ids]
        results.pts_semantic_mask[0] = sem_mask.cpu().numpy()
        results.pts_instance_mask[0] = ins_mask.cpu().numpy()
        return results
    
    def predict_by_feat(self, out: Dict[str, Any], superpoints: Any) -> Tuple[List[PointData], List[torch.Tensor]]:  # type: ignore[override]
        """Predict instance, semantic, and panoptic masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `sem_preds` of shape (n_queries, n_semantic_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            List[PointData]: of len 1 with `pts_semantic_mask`,
                `pts_instance_mask`, `instance_labels`, `instance_scores`.
        """
        inst_res = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.inst_score_thr)
        sem_res = self.predict_by_feat_semantic(out, superpoints)

        sem_map2 = self.predict_by_feat_semantic(
            out, superpoints, self.test_cfg.stuff_classes)
        inst_res2 = self.predict_by_feat_instance(
            out, superpoints, self.test_cfg.pan_score_thr)

        pts_semantic_mask = [sem_res, sem_map2]
        pts_instance_mask = [inst_res[0].bool(), inst_res2[0].bool()]
        instance_labels = [inst_res[1], inst_res2[1]]
        instance_scores = [inst_res[2], inst_res2[2]]
        instance_queries = [inst_res[3], inst_res2[3]]
        mapping = [inst_res[4], inst_res2[4]]
      
        return [
            PointData(
                pts_semantic_mask=pts_semantic_mask,
                pts_instance_mask=pts_instance_mask,
                instance_labels=instance_labels,
                instance_scores=instance_scores,
                instance_queries=instance_queries)], mapping
    
    def predict_by_feat_instance(self, out: Dict[str, Any], superpoints: Any, score_threshold: float) -> Tuple[Any, torch.Tensor, torch.Tensor, Any, torch.Tensor]:  # type: ignore[override]
        """Predict instance masks for a single scene.

        Args:
            out (Dict): Decoder output, each value is List of len 1. Keys:
                `cls_preds` of shape (n_queries, n_instance_classes + 1),
                `masks` of shape (n_queries, n_points),
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
            score_threshold (float): minimal score for predicted object.
        
        Returns:
            Tuple:
                Tensor: mask_preds of shape (n_preds, n_raw_points),
                Tensor: labels of shape (n_preds,),
                Tensor: scors of shape (n_preds,).
        """
        mapping = torch.arange(len(out['cls_preds'][0])).to(superpoints.device)
        cls_preds = out['cls_preds'][0]
        pred_masks = out['masks'][0]
        queries = out['queries'][0]
        assert self.num_classes == 1 or self.num_classes == cls_preds.shape[1] - 1

        scores = F.softmax(cls_preds, dim=-1)[:, :-1]
        if out['scores'][0] is not None:
            scores *= out['scores'][0]
        if self.num_classes == 1:
            scores = scores.sum(-1, keepdim=True)
        labels = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)
        topk_num = min(int(self.test_cfg.topk_insts), scores.shape[0] * scores.shape[1])
        scores, topk_idx = scores.flatten(0, 1).topk(topk_num, sorted=False)
        labels = labels[topk_idx]

        topk_idx = torch.div(topk_idx, self.num_classes, rounding_mode='floor')
        mask_pred = pred_masks
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        queries = queries[topk_idx]
        mapping = mapping[topk_idx]

        if self.test_cfg.get('obj_normalization', None):
            mask_scores = (mask_pred_sigmoid * (mask_pred > 0)).sum(1) / \
                ((mask_pred > 0).sum(1) + 1e-6)
            scores = scores * mask_scores

        if self.test_cfg.get('nms', None):
            kernel = str(self.test_cfg.matrix_nms_kernel)
            scores, labels, mask_pred_sigmoid, keep_inds = mask_matrix_nms(  # type: ignore[arg-type]
                mask_pred_sigmoid, labels, scores, kernel=kernel)
            queries = queries[keep_inds]
            mapping = mapping[keep_inds]

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries = queries[score_mask]
        mapping = mapping[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_thr = int(self.test_cfg.npoint_thr)
        npoint_mask = mask_pointnum > npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries = queries[npoint_mask]
        mapping = mapping[npoint_mask]

        return mask_pred, labels, scores, queries, mapping
    
    def predict_by_feat_panoptic(self, sem_map: torch.Tensor, mask_pred: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Predict panoptic masks for a single scene.

        Args:
            sem_map: Semantic map tensor
            mask_pred: Predicted instance masks tensor
            labels: Predicted class labels tensor
            scores: Predicted confidence scores tensor
                `scores` of shape (n_queris, 1) or None.
            superpoints (Tensor): of shape (n_raw_points,).
        
        Returns:
            Tuple:
                Tensor: semantic mask of shape (n_raw_points,),
                Tensor: instance mask of shape (n_raw_points,).
        """
        if mask_pred.shape[0] == 0:
            return sem_map, sem_map

        scores, idxs = scores.sort()
        labels = labels[idxs]
        mask_pred = mask_pred[idxs]

        n_stuff_classes = len(self.test_cfg.stuff_classes)
        inst_idxs = torch.arange(
            n_stuff_classes, 
            mask_pred.shape[0] + n_stuff_classes, 
            device=mask_pred.device).view(-1, 1)
        insts = inst_idxs * mask_pred
        things_inst_mask, idxs = insts.max(dim=0)
        things_sem_mask = labels[idxs] + n_stuff_classes

        inst_idxs, num_pts = things_inst_mask.unique(return_counts=True)
        for inst, pts in zip(inst_idxs, num_pts):
            if pts <= self.test_cfg.npoint_thr and inst != 0:
                things_inst_mask[things_inst_mask == inst] = 0

        things_sem_mask[things_inst_mask == 0] = 0
      
        sem_map[things_inst_mask != 0] = 0
        inst_map = sem_map.clone()
        inst_map += things_inst_mask
        sem_map += things_sem_mask

        return sem_map, inst_map

@MODELS.register_module()
class ScanNet200MixFormer3D_FF_Online(ScanNet200MixFormer3D_Online):
    """OneFormer3D for ScanNet200 dataset.
    
    Args:
        voxel_size (float): Voxel size.
        num_classes (int): Number of classes.
        query_thr (float): Min percent of queries.
        backbone (ConfigDict): Config dict of the backbone.
        neck (ConfigDict, optional): Config dict of the neck.
        decoder (ConfigDict): Config dict of the decoder.
        criterion (ConfigDict): Config dict of the criterion.
        matcher (ConfigDict): To match superpoints to objects.
        train_cfg (dict, optional): Config dict of training hyper-parameters.
            Defaults to None.
        test_cfg (dict, optional): Config dict of test hyper-parameters.
            Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (dict or ConfigDict, optional): the config to control the
            initialization. Defaults to None.
    """

    def __init__(self,
                 voxel_size,
                 num_classes,
                 query_thr,
                 img_backbone=None,
                 backbone=None,
                 memory=None,
                 neck=None,
                 pool=None,
                 decoder=None,
                 merge_head=None,
                 merge_criterion=None,
                 criterion=None,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        if img_backbone is not None:
            self.img_backbone = MODELS.build(_cfg(img_backbone, 'img_backbone'))
        else:
            self.img_backbone = None
        self.backbone = MODELS.build(_cfg(backbone, 'backbone'))
        if memory is not None:
            self.memory = MODELS.build(_cfg(memory, 'memory'))
        self.neck = MODELS.build(_cfg(neck, 'neck')) if neck is not None else None
        self.pool = MODELS.build(_cfg(pool, 'pool'))
        dec_cfg = _cfg(decoder, 'decoder')
        self.decoder = MODELS.build(dec_cfg)
        if merge_head is not None:
            self.merge_head = MODELS.build(merge_head)
        if merge_criterion is not None:
            self.merge_criterion = MODELS.build(merge_criterion)
        self.criterion = MODELS.build(_cfg(criterion, 'criterion'))
        self.decoder_online = dec_cfg['temporal_attn']
        self.use_bbox = dec_cfg['bbox_flag']
        self.sem_len = dec_cfg['num_semantic_classes'] + 1 # 201
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        self.test_cfg: Any = ConfigDict(test_cfg or {})
        self.init_weights()
        
        # 初始化为None，将在运行时动态创建
        self.conv = None
    
    def init_weights(self):
        if hasattr(self, 'memory') and self.memory is not None:
            self.memory.init_weights()
        if hasattr(self, 'img_backbone') and self.img_backbone is not None:
            self.img_backbone.init_weights()

    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i):
        """Extract features from sparse tensor.
        """
        # extract image features
        with torch.no_grad():
            if getattr(self, 'img_backbone', None) is not None and 'img_path' in batch_inputs_dict:
                _ = self.img_backbone(batch_inputs_dict['img_path'])  # type: ignore[operator]
        img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]
    
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            if 'elastic_coords' in batch_inputs_dict:
                coordinates.append(
                    batch_inputs_dict['elastic_coords'][i][frame_i] * self.voxel_size)
            else:
                coordinates.append(batch_inputs_dict['points'][i][frame_i, :, :3])
            features.append(batch_inputs_dict['points'][i][frame_i, :, 3:])
        all_xyz = coordinates
        
        coordinates, features, *_ = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(),
                          partial(self._f, img_features=img_metas, img_shape=img_metas[0]['img_shape']),
                          memory=self.memory if hasattr(self,'memory') else None)
        if self.with_neck:
            assert self.neck is not None
            x = self.neck(x)
        x = x.slice(field)
        point_features = [torch.cat([c, f], dim=-1) for c, f in zip(all_xyz, x.decomposed_features)]  # [B, N, 3+D]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask[frame_i]
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w, *_ = self.pool(x, sp_idx, all_xyz, with_xyz=False)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list

    def _f(self, x, img_features, img_shape):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):  # type: ignore[name-defined]
            coord_type = 'DEPTH'
            img_scale_factor = (
                point.new_tensor(img_meta['scale_factor'][:2])
                if 'scale_factor' in img_meta.keys() else 1)
            #img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_flip = False
            img_crop_offset = (
                point.new_tensor(img_meta['img_crop_offset'])
                if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, coord_type)
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_shape[-2:],
                img_shape=img_shape[-2:],
                aligned=True,
                padding_mode='zeros',
                align_corners=True))
 
        projected_features = torch.cat(projected_features, dim=0)
        projected_features = ME.SparseTensor(
            projected_features,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager)
        
        # 动态初始化卷积层
        if self.conv is None:
            self.conv = nn.Sequential(
                ME.MinkowskiConvolution(projected_features.shape[1], 32, kernel_size=1, dimension=3),
                ME.MinkowskiBatchNorm(32),
                ME.MinkowskiReLU(inplace=True)
            ).to(projected_features.device)
        
        projected_features = self.conv(projected_features)
        return projected_features + x

@MODELS.register_module()
class ScanNet200MixFormer3D_Stream(ScanNet200MixFormer3D_Online):
    def extract_feat(self, batch_inputs_dict, batch_data_samples, frame_i=None):
        """Extract features from sparse tensor.
        """
        # construct tensor field
        coordinates, features = [], []
        for i in range(len(batch_inputs_dict['points'])):
            coordinates.append(batch_inputs_dict['points'][i][:, :3])
            features.append(batch_inputs_dict['points'][i][:, 3:])
        all_xyz = coordinates

        coordinates, features, *_ = ME.utils.batch_sparse_collate(
            [(c / self.voxel_size, f) for c, f in zip(coordinates, features)],
            device=coordinates[0].device)
        field = ME.TensorField(coordinates=coordinates, features=features)

        # forward of backbone and neck
        x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None)
        map_index = None; x_voxel = None
        x = x.slice(field)
        point_features = [torch.cat([c, f], dim=-1) for c, f in zip(all_xyz, x.decomposed_features)]
        x = x.features

        # apply scatter_mean
        sp_pts_masks, n_super_points = [], []
        for data_sample in batch_data_samples:
            sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
            sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
            n_super_points.append(sp_pts_mask.max() + 1)
        sp_idx = torch.cat(sp_pts_masks)
        x, all_xyz_w, *_ = self.pool(x, sp_idx, all_xyz, with_xyz=False)

        # apply cls_layer
        features = []
        sp_xyz_list = []
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end, :-3])
            sp_xyz_list.append(x[begin: end, -3:])
        return features, point_features, all_xyz_w, sp_xyz_list
    
    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        """Predict results from a batch of inputs and data samples with post-
        processing.
        """
        assert len(batch_data_samples) == 1
        results, query_feats_list, sem_preds_list, sp_xyz_list, bboxes_list, cls_preds_list = [], [], [], [], [], []
        
        # Initialize variables that may be used later
        mv_mask, mv_labels, mv_scores, mv_bboxes = None, None, None, None
        
        if hasattr(self, 'memory'):
            self.memory.reset()

        ## Backbone
        x, point_features, all_xyz_w, sp_xyz = self.extract_feat(
            batch_inputs_dict, batch_data_samples)
        ## Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w)
        x = self.decoder(x=x, queries=x, sp_feats=x, p_feats=point_features, super_points=super_points)
        ## Post-processing
        pred_pts_seg, mapping = self.predict_by_feat(
            x, batch_data_samples[0].gt_pts_seg.sp_pts_mask)
        results.append(pred_pts_seg[0])
        ## Query projector, semantic and geometric information
        if hasattr(self, 'merge_head'):
            query_feats = self.merge_head(x['queries'][0])
            query_feats_list.append([query_feats[mapping[0]], query_feats[mapping[1]]])
            sem_preds = x['cls_preds'][0]
            sem_preds_list.append([sem_preds[mapping[0]], sem_preds[mapping[1]]])
            sp_xyz_list.append([sp_xyz[0][mapping[0]], sp_xyz[0][mapping[1]]])
            if self.use_bbox:
                bbox_preds = x['bboxes'][0] # [N, 6]
                bboxes_list.append([bbox_preds[mapping[0]], bbox_preds[mapping[1]]])
        ## Online merging
        if self.test_cfg.merge_type == 'learnable_online':
            if not hasattr(self, 'online_merger'):
                self.online_merger = OnlineMerge(self.test_cfg.inscat_topk_insts, self.use_bbox)
            mv_mask, mv_labels, mv_scores, _, mv_bboxes = self.online_merger.merge(
                results[-1].pop('pts_instance_mask')[0],
                results[-1].pop('instance_labels')[0],
                results[-1].pop('instance_scores')[0],
                results[-1].pop('instance_queries')[0],
                query_feats_list.pop(-1)[0],
                sem_preds_list.pop(-1)[0],
                sp_xyz_list.pop(-1)[0],
                bboxes_list.pop(-1)[0] if self.use_bbox else None)
        ## Clean. Empty cache. Only offline merging requires the whole list.
        if self.test_cfg.merge_type == 'learnable_online':
            torch.cuda.empty_cache()
        #     t1 = time.time()
        #     self.time_list.append(t1-t0)
        # print(sum(self.time_list) / len(self.time_list))
        ## Offline merging
        if self.test_cfg.merge_type == 'learnable':
            mv_mask, mv_labels, mv_scores = ins_merge_mat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                [res[0] for res in query_feats_list],
                [res[0] for res in sem_preds_list],
                [res[0] for res in sp_xyz_list],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'concat':
            mv_mask, mv_labels, mv_scores = ins_cat(
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'geometric':
            mv_mask, mv_labels, mv_scores = ins_merge(
                [points for points in batch_inputs_dict['points'][0]],
                [res['pts_instance_mask'][0] for res in results],
                [res['instance_labels'][0] for res in results],
                [res['instance_scores'][0] for res in results],
                [res['instance_queries'][0] for res in results],
                self.test_cfg.inscat_topk_insts)
        elif self.test_cfg.merge_type == 'learnable_online':
            pass
        else:
            raise NotImplementedError("Unknown merge_type.")

        ## Offline semantic segmentation
        mv_sem = torch.cat([res['pts_semantic_mask'][0] for res in results])
        
        if self.use_bbox and mv_bboxes is not None:
            batch_data_samples[0].pred_bbox = mv_bboxes.cpu().numpy()
        
        # Ensure variables are not None before using them
        if mv_mask is None or mv_labels is None or mv_scores is None:
            # If no merging was performed, use the first result
            mv_mask = results[0]['pts_instance_mask'][0]
            mv_labels = results[0]['instance_labels'][0]
            mv_scores = results[0]['instance_scores'][0]
        
        # Not mapping to reconstructed point clouds, return directly for visualization
        merged_result = PointData(
            pts_semantic_mask=[mv_sem.cpu().numpy()],
            pts_instance_mask=[mv_mask.cpu().numpy()],
            instance_labels=mv_labels.cpu().numpy(),
            instance_scores=mv_scores.cpu().numpy())
        batch_data_samples[0].pred_pts_seg = merged_result
        return batch_data_samples

# -----------------------------------------------------------------------------
# Utility helpers for static-type friendly module construction
# -----------------------------------------------------------------------------

def _cfg(cfg: Any, name: str) -> Dict[str, Any]:
    """Ensure *cfg* is a dict so that static analyzers don't complain.

    This wrapper raises a clear error at runtime if the user forgets to
    provide a config section, while also narrowing the type for Pyright so
    that calls like ``MODELS.build`` won't emit *Unknown | None* diagnostics.
    """
    if cfg is None:
        raise ValueError(f'Config for "{name}" must be provided, but got None.')
    if not isinstance(cfg, dict):
        raise TypeError(f'Config for "{name}" must be a dict, got {type(cfg)}.')
    return cast(Dict[str, Any], cfg)
