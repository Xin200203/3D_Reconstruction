import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
import MinkowskiEngine as ME
import pointops
import pdb, time
import json
import os
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
from .img_backbone import point_sample, apply_3d_transformation
from .projection_utils import project_points_to_uv, SCANET_INTRINSICS, sample_img_feat, MIN_DEPTH
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
                 dino_cfg=None,
                 dino_require: bool = False,
                 dino_online_only: bool = False,
                 data_preprocessor=None,
                 init_cfg=None):
        super(Base3DDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)

        self.backbone = MODELS.build(_cfg(backbone, 'backbone'))
        self.neck = MODELS.build(_cfg(neck, 'neck')) if neck is not None else None
        self.pool = MODELS.build(_cfg(pool, 'pool'))
        decoder_cfg = _cfg(decoder, 'decoder')
        self.decoder = MODELS.build(decoder_cfg)
        # 可选 DINOv2 Backbone（在线 2D 特征），保持完全冻结
        self.dino = MODELS.build(_cfg(dino_cfg, 'dino')) if dino_cfg is not None else None
        # 训练期严格约束：用于强制 DINO 在线注入链路必须可用，否则直接报错终止训练。
        # - dino_require=True: 必须使用 DINO（不能退化为纯3D）
        # - dino_online_only=True: 禁止使用外部离线特征键（dino_fpn/dino_feats/dino_point_feats/clip_pix）
        self.dino_require = bool(dino_require)
        self.dino_online_only = bool(dino_online_only)
        self.criterion = MODELS.build(_cfg(criterion, 'criterion'))
        self.test_cfg: Any = ConfigDict(test_cfg or {})
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg
        # DINO 调试计数（限制打印次数）
        self._dino_debug_count = 0
        # 仅在需要时开启调试日志，避免污染纯 3D baseline 的训练/测试输出
        self._dino_debug = os.environ.get('DINO_DEBUG', '') == '1'

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
        points_list: List[torch.Tensor] = batch_inputs_dict['points']

        # === DINO FPN 构建（单帧）===
        dino_feats: Optional[List[ME.SparseTensor]] = None
        backbone_use_dino = bool(getattr(self.backbone, 'use_dino', False))
        if (self.dino_require or self.dino_online_only or getattr(self, 'dino', None) is not None) and (not backbone_use_dino):
            raise RuntimeError(
                "DINO is enabled in model config, but the 3D backbone is not configured for DINO injection. "
                "Set `model.backbone.dino_dim` (e.g. 1024) to enable `backbone.use_dino=True`."
            )

        if getattr(self, 'dino_online_only', False):
            unexpected = []
            for k in ('dino_fpn', 'dino_feats', 'dino_point_feats'):
                if k in batch_inputs_dict:
                    unexpected.append(k)
            # clip_pix 可能存在但为空，这里只在非 None 时认为“使用了离线”
            if batch_inputs_dict.get('clip_pix', None) is not None:
                unexpected.append('clip_pix')
            if unexpected:
                raise RuntimeError(
                    f"DINO online-only: unexpected offline inputs present: {unexpected}. "
                    "Remove related pipeline steps / keys to ensure only online DINO is used."
                )
        # 轻量调试：明确区分“离线/外部特征是否提供” vs “是否将在线构建”
        if self._dino_debug and self._dino_debug_count < 5:
            cam_info_field = batch_inputs_dict.get('cam_info', None)
            if isinstance(cam_info_field, list):
                cam_len = len(cam_info_field)
            elif cam_info_field is None:
                cam_len = 0
            else:
                cam_len = 1

            imgs_field = batch_inputs_dict.get('img', None)
            img_present = imgs_field is not None
            img_type = type(imgs_field).__name__ if img_present else 'None'
            img_shape = getattr(imgs_field, 'shape', None)
            if isinstance(img_shape, torch.Size):
                img_shape = tuple(img_shape)
            if isinstance(imgs_field, list):
                img_list_len = len(imgs_field)
                img0_shape = getattr(imgs_field[0], 'shape', None) if img_list_len > 0 else None
                if isinstance(img0_shape, torch.Size):
                    img0_shape = tuple(img0_shape)
            else:
                img_list_len = None
                img0_shape = None

            provided = {
                # 这些字段是“离线/外部直接提供的 DINO 特征”，在线链路下通常应该为 False。
                'dino_fpn': 'dino_fpn' in batch_inputs_dict,
                'dino_feats': 'dino_feats' in batch_inputs_dict,
                'dino_point_feats': 'dino_point_feats' in batch_inputs_dict,
                'clip_pix': batch_inputs_dict.get('clip_pix', None) is not None,
            }
            online_ready = (getattr(self, 'dino', None) is not None) and img_present and (cam_len > 0)
            print(
                "[DINO][debug] "
                f"provided_offline={provided}, "
                f"online_ready={online_ready}, "
                f"img_type={img_type}, img_shape={img_shape}, img_list_len={img_list_len}, img0_shape={img0_shape}, "
                f"cam_info_type={type(cam_info_field)}, cam_len={cam_len}"
            )
            self._dino_debug_count += 1

        # 1) 优先使用外部提供的 dino_fpn / dino_feats（完全跳过内部构建）
        if backbone_use_dino and 'dino_fpn' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_fpn']
                if self._dino_debug and self._dino_debug_count < 3:
                    print("[DINO] use provided dino_fpn (single-frame)")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        elif backbone_use_dino and 'dino_feats' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_feats']
                if self._dino_debug and self._dino_debug_count < 3:
                    print("[DINO] use provided dino_feats (single-frame)")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None

        # 2) 若未提供，且模型中挂载了 DINOv2Backbone，则在线从 img + cam_info 构建 FPN
        if backbone_use_dino and dino_feats is None and getattr(self, 'dino', None) is not None:
            imgs = batch_inputs_dict.get('img', None)
            cam_raw = batch_inputs_dict.get('cam_info', None)
            if getattr(self, 'dino_require', False) and (imgs is None or cam_raw is None):
                # 给出可定位的信息：batch_inputs_dict keys + 前几条样本路径/增强信息
                sample_infos = []
                for i, s in enumerate(batch_data_samples[:min(3, len(batch_data_samples))]):
                    m = getattr(s, 'img_metas', None)
                    if not isinstance(m, dict):
                        m = {}
                    sample_infos.append({
                        'i': i,
                        'img_path': m.get('img_path', None),
                        'lidar_path': m.get('lidar_path', None),
                        'flow': m.get('transformation_3d_flow', None),
                        'flip': m.get('flip', None),
                        'pcd_hflip': m.get('pcd_horizontal_flip', None),
                        'pcd_vflip': m.get('pcd_vertical_flip', None),
                    })
                raise RuntimeError(
                    "DINO required but missing inputs: "
                    f"img_present={imgs is not None}, cam_info_present={cam_raw is not None}. "
                    f"batch_inputs_keys={list(batch_inputs_dict.keys())}. "
                    f"samples_head={sample_infos}"
                )
            if imgs is not None and cam_raw is not None:
                try:
                    # Det3DDataPreprocessor_ 通常会把单帧 img collate 成 list[Tensor(C,H,W)]，
                    # 而 DINOv2Backbone 期望 (B,3,H,W) Tensor。这里做一次鲁棒规范化。
                    if isinstance(imgs, list):
                        norm_imgs: List[torch.Tensor] = []
                        for it in imgs:
                            x = it
                            if isinstance(x, tuple) and len(x) > 0:
                                x = x[0]
                            if not torch.is_tensor(x):
                                x = torch.as_tensor(x)
                            if x.dim() == 4 and x.shape[0] == 1:
                                x = x[0]
                            # 若为 HWC，则转 CHW
                            if x.dim() == 3 and x.shape[0] not in (1, 3) and x.shape[-1] in (1, 3):
                                x = x.permute(2, 0, 1).contiguous()
                            norm_imgs.append(x)
                        imgs = torch.stack(norm_imgs, dim=0)
                    elif torch.is_tensor(imgs):
                        if imgs.dim() == 3:
                            imgs = imgs.unsqueeze(0)
                        elif imgs.dim() == 4:
                            pass
                        else:
                            imgs = imgs.view(-1, 3, *imgs.shape[-2:])
                    else:
                        imgs = torch.as_tensor(imgs)
                        if imgs.dim() == 3:
                            imgs = imgs.unsqueeze(0)

                    cam_metas = self._normalize_cam_info(cam_raw, len(points_list))
                    # 若某些样本图像无效（例如缺失文件但 ALLOW_MISSING_IMG=1），在 strict 模式下直接中止
                    if getattr(self, 'dino_require', False):
                        for i, cm in enumerate(cam_metas):
                            if cm.get('img_valid', True) is False:
                                s = batch_data_samples[i]
                                m = getattr(s, 'img_metas', None)
                                if not isinstance(m, dict):
                                    m = {}
                                raise RuntimeError(
                                    f"DINO required but img_valid=False at sample {i}: img_path={m.get('img_path')} lidar_path={m.get('lidar_path')}"
                                )
                    feat_maps = self.dino(imgs)  # type: ignore[operator]
                    if getattr(self, 'dino_require', False):
                        # 期望 420×560 with patch=14 -> 30×40（不允许在模型内再 resize）
                        if (not torch.is_tensor(feat_maps) or feat_maps.dim() != 4 or
                                tuple(feat_maps.shape[-2:]) != (30, 40)):
                            raise RuntimeError(
                                f"online DINO returned unexpected feat_maps shape={getattr(feat_maps, 'shape', None)}; "
                                "check ResizeForDINO(target_size=(420,560)) is applied in dataset pipeline."
                            )
                    elastic_coords = batch_inputs_dict.get('elastic_coords', None)
                    dino_feats = self._build_dino_fpn_online(
                        points_list, feat_maps, cam_metas, batch_data_samples,
                        elastic_coords=elastic_coords)
                    if self._dino_debug and dino_feats is not None and self._dino_debug_count < 3:
                        shapes = [x.shape for x in dino_feats]
                        strides = [x.tensor_stride for x in dino_feats]
                        print(f"[DINO] build from online DINO backbone, shapes={shapes}, strides={strides}")
                        self._dino_debug_count += 1
                except Exception as e:
                    if getattr(self, 'dino_require', False):
                        raise RuntimeError(f"DINO required but online build failed: {repr(e)}")
                    if self._dino_debug and self._dino_debug_count < 3:
                        print(f"[DINO][error] build from online DINO failed: {e}")
                        self._dino_debug_count += 1
                    dino_feats = None

        # 若要求必须使用 DINO，则不允许退化为纯3D
        if getattr(self, 'dino_require', False):
            if getattr(self, 'dino', None) is None:
                raise RuntimeError("DINO required but model.dino is None (missing dino_cfg).")
            if dino_feats is None:
                raise RuntimeError(
                    "DINO required but dino_feats is None. "
                    "Check dataset pipeline provides img+cam_info and that online DINO build succeeds."
                )
            if not isinstance(dino_feats, (list, tuple)) or len(dino_feats) < 4:
                raise RuntimeError(
                    f"DINO required but got invalid dino_feats type/len: {type(dino_feats)} / {getattr(dino_feats, '__len__', lambda: 'NA')()}"
                )

        # 3) 兼容旧逻辑：直接给出点级 DINO 特征或 clip_pix
        if backbone_use_dino and dino_feats is None and 'dino_point_feats' in batch_inputs_dict:
            try:
                coords_list, feats_list = [], []
                for b_idx, pts in enumerate(points_list):
                    xyz = pts[:, :3]
                    feats_2d = batch_inputs_dict['dino_point_feats'][b_idx]
                    coords = (xyz / self.voxel_size).floor().to(torch.int32)
                    coords_list.append(coords.to(device=xyz.device))
                    feats_list.append(feats_2d.to(device=xyz.device))
                coords_batch, feats_batch = ME.utils.sparse_collate(
                    coords_list, feats_list, device=feats_list[0].device)
                dino_feats = build_sparse_fpn(coords_batch, feats_batch)
                if self._dino_debug and self._dino_debug_count < 3:
                    shapes = [x.shape for x in dino_feats]
                    strides = [x.tensor_stride for x in dino_feats]
                    print(f"[DINO] build from dino_point_feats, shapes={shapes}, strides={strides}")
                    self._dino_debug_count += 1
            except Exception as e:
                if self._dino_debug and self._dino_debug_count < 3:
                    print(f"[DINO][error] build from dino_point_feats failed: {e}")
                    self._dino_debug_count += 1
                dino_feats = None

        if backbone_use_dino and dino_feats is None and 'clip_pix' in batch_inputs_dict and 'cam_info' in batch_inputs_dict:
            try:
                dino_feats = self._build_dino_fpn_from_clip(batch_inputs_dict)
                if self._dino_debug and dino_feats is not None and self._dino_debug_count < 3:
                    shapes = [x.shape for x in dino_feats]
                    strides = [x.tensor_stride for x in dino_feats]
                    print(f"[DINO] build from clip_pix, shapes={shapes}, strides={strides}")
                    self._dino_debug_count += 1
            except Exception as e:
                if self._dino_debug and self._dino_debug_count < 3:
                    print(f"[DINO][error] build from clip_pix failed: {e}")
                    self._dino_debug_count += 1
                dino_feats = None

        if self._dino_debug and dino_feats is None and self._dino_debug_count < 3:
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

    def _normalize_cam_info(self, cam_raw: Any, num_samples: int) -> List[Dict]:
        """Normalize various cam_info formats to a per-sample list[dict].

        需要同时处理：
        - DataLoader collate 后出现的嵌套 list 结构；
        - 当 cam_info 被 batch 到单个 dict（各字段第一维为 B）时，按样本拆分；
        - intrinsics / img_size_dino / pose 等字段被包装成 0/1 维 torch.Tensor。
        """
        import os
        from typing import Iterable
        import numpy as np

        def _summarize_value(v: Any) -> str:
            try:
                if torch.is_tensor(v):
                    return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, device={v.device})"
                if isinstance(v, np.ndarray):
                    return f"ndarray(shape={v.shape}, dtype={v.dtype})"
                if isinstance(v, (list, tuple)):
                    head = v[0] if len(v) > 0 else None
                    return f"{type(v).__name__}(len={len(v)}, head={_summarize_value(head) if head is not None else 'None'})"
                return f"{type(v).__name__}({str(v)[:80]})"
            except Exception:
                return f"{type(v).__name__}"

        debug = os.environ.get('DEBUG_CAMINFO_NORM', '') == '1'
        strict = os.environ.get('STRICT_CAMINFO_NORM', '') == '1'
        if debug and not hasattr(self, '_caminfo_debug_count'):
            self._caminfo_debug_count = 0  # type: ignore[attr-defined]
        if debug and self._caminfo_debug_count < 3:  # type: ignore[attr-defined]
            print(f"[CamInfo][raw] num_samples={num_samples}, type={type(cam_raw)}")
            if isinstance(cam_raw, dict):
                for k, v in cam_raw.items():
                    print(f"  - {k}: {_summarize_value(v)}")
            elif isinstance(cam_raw, list):
                print(f"  - list_len={len(cam_raw)} head={_summarize_value(cam_raw[0]) if len(cam_raw)>0 else 'None'}")
            self._caminfo_debug_count += 1  # type: ignore[attr-defined]

        cam_metas: List[Dict] = []
        if isinstance(cam_raw, list):
            for item in cam_raw:
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], dict):
                    cam_metas.append(item[0])
                elif isinstance(item, dict):
                    cam_metas.append(item)
        elif isinstance(cam_raw, dict):
            # IMPORTANT:
            # - `cam_raw` 可能是 “collate 后的 batched dict”（各字段第0维为 B）；
            # - 也可能是单样本/单相机 dict（B=1）。
            # 这里不要直接复制 num_samples 份（会导致每个 sample 仍持有 batched pose/intrinsics，
            # 后续若再用 `.reshape(-1,4,4)[0]` 会错误地对所有样本使用第0个 pose）。
            # 统一先当作 1 个 dict，交由下面的 split 逻辑按 batch 维拆分。
            cam_metas = [cam_raw]

        # 特殊情况：collate 后 cam_info 变成长度 1 的 dict，
        # 其中每个字段的第 0 维为 batch 维（B）。
        if len(cam_metas) == 1 and num_samples > 1 and isinstance(cam_metas[0], dict):
            batched_meta = cam_metas[0]
            split_metas: List[Dict] = []
            for b in range(num_samples):
                m: Dict[str, Any] = {}
                for k, v in batched_meta.items():
                    if torch.is_tensor(v):
                        if v.ndim >= 1 and v.shape[0] == num_samples:
                            m[k] = v[b]
                        else:
                            m[k] = v
                    elif isinstance(v, np.ndarray):
                        if v.ndim >= 1 and v.shape[0] == num_samples:
                            m[k] = v[b]
                        else:
                            m[k] = v
                    elif isinstance(v, (list, tuple)):
                        # 常见情况：某些字段在 collate 后会变成 “长度为 B 的 python list”，
                        # 每个元素是该样本的值（例如 img_size_dino / intrinsics 的嵌套 list）。
                        if len(v) == num_samples and k in {'img_size_dino', 'pose', 'extrinsics'}:
                            m[k] = v[b]
                            continue
                        if k == 'intrinsics' and len(v) == num_samples and len(v) > 0:
                            # intrinsics 有两种常见 collate 形态：
                            # A) per-sample: list[B]，每个元素是 (4,)（list/tuple/ndarray/tensor）
                            # B) component-wise: list[4]，每个元素是 (B,)（tensor/ndarray）
                            # 当 batch_size==4 时，A/B 在 len(v) 上会“撞车”，必须用元素形态区分。
                            is_component_wise = (
                                len(v) == 4 and all(
                                    (torch.is_tensor(elem) and elem.ndim >= 1 and elem.shape[0] == num_samples) or
                                    (isinstance(elem, np.ndarray) and elem.ndim >= 1 and elem.shape[0] == num_samples)
                                    for elem in v
                                )
                            )
                            if not is_component_wise:
                                sample_v = v[b]
                                if isinstance(sample_v, (list, tuple)) and len(sample_v) == 4:
                                    m[k] = sample_v
                                    continue
                                if torch.is_tensor(sample_v) and sample_v.numel() == 4:
                                    m[k] = sample_v
                                    continue
                                if isinstance(sample_v, np.ndarray) and sample_v.size == 4:
                                    m[k] = sample_v
                                    continue
                        if len(v) > 0 and (torch.is_tensor(v[0]) or isinstance(v[0], np.ndarray)):
                            per_list = []
                            for elem in v:
                                if torch.is_tensor(elem) and elem.ndim >= 1 and elem.shape[0] == num_samples:
                                    per_list.append(elem[b])
                                elif isinstance(elem, np.ndarray) and elem.ndim >= 1 and elem.shape[0] == num_samples:
                                    per_list.append(elem[b])
                                else:
                                    per_list.append(elem)
                            m[k] = per_list
                        else:
                            m[k] = list(v)
                    else:
                        m[k] = v
                split_metas.append(m)
            cam_metas = split_metas

        if len(cam_metas) != num_samples:
            raise RuntimeError(
                f"_normalize_cam_info: got {len(cam_metas)} metas for {num_samples} samples")

        # 逐样本清理 tensor 包装，统一为 Python 标量 / 2D 矩阵
        def _to_scalar(x):
            if isinstance(x, (list, tuple)):
                if len(x) == 0:
                    return float('nan')
                return _to_scalar(x[0])
            if isinstance(x, torch.Tensor):
                return float(x.detach().cpu().reshape(-1)[0].item())
            if isinstance(x, np.ndarray):
                return float(x.reshape(-1)[0])
            return float(x)

        def _to_matrix4x4(x):
            if isinstance(x, torch.Tensor):
                t = x.detach()
                if t.ndim == 3 and t.shape[0] == 1:
                    t = t[0]
                return t
            if isinstance(x, np.ndarray):
                a = x
                if a.ndim == 3 and a.shape[0] == 1:
                    a = a[0]
                return a
            return x

        for meta in cam_metas:
            intr = meta.get('intrinsics', None)
            # intrinsics 可能以 list/tuple 或 (4,) tensor/ndarray 形式存在
            if isinstance(intr, (list, tuple)) and len(intr) == 4:
                meta['intrinsics'] = [_to_scalar(v) for v in intr]
            elif isinstance(intr, (list, tuple)) and len(intr) == 1:
                # 兼容误包装：intr=[tensor(4,)] 或 intr=[list(4,)]
                x = intr[0]
                if torch.is_tensor(x) and x.numel() == 4:
                    meta['intrinsics'] = [_to_scalar(v) for v in x.reshape(-1)]
                elif isinstance(x, np.ndarray) and x.size == 4:
                    meta['intrinsics'] = [_to_scalar(v) for v in x.reshape(-1)]
                elif isinstance(x, (list, tuple)) and len(x) == 4:
                    meta['intrinsics'] = [_to_scalar(v) for v in x]
            elif torch.is_tensor(intr) and intr.numel() == 4:
                meta['intrinsics'] = [_to_scalar(v) for v in intr.reshape(-1)]
            elif isinstance(intr, np.ndarray) and intr.size == 4:
                meta['intrinsics'] = [_to_scalar(v) for v in intr.reshape(-1)]
            # img_size_dino 可能是 [tensor([H]), tensor([W])] 或 (H,W)
            img_size = meta.get('img_size_dino', None)
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                H, W = img_size
                try:
                    meta['img_size_dino'] = (int(_to_scalar(H)), int(_to_scalar(W)))
                except Exception:
                    pass
            elif isinstance(img_size, (list, tuple)) and len(img_size) == 1:
                # 兼容误包装：img_size_dino=[tensor(2,)] / [ndarray(2,)] / [(H,W)]
                x = img_size[0]
                if torch.is_tensor(x) and x.numel() == 2:
                    try:
                        meta['img_size_dino'] = (int(_to_scalar(x.reshape(-1)[0])),
                                                int(_to_scalar(x.reshape(-1)[1])))
                    except Exception:
                        pass
                elif isinstance(x, np.ndarray) and x.size == 2:
                    try:
                        flat = x.reshape(-1)
                        meta['img_size_dino'] = (int(_to_scalar(flat[0])), int(_to_scalar(flat[1])))
                    except Exception:
                        pass
                elif isinstance(x, (list, tuple)) and len(x) == 2:
                    try:
                        meta['img_size_dino'] = (int(_to_scalar(x[0])), int(_to_scalar(x[1])))
                    except Exception:
                        pass
            elif torch.is_tensor(img_size) and img_size.numel() == 2:
                try:
                    meta['img_size_dino'] = (int(_to_scalar(img_size.reshape(-1)[0])),
                                            int(_to_scalar(img_size.reshape(-1)[1])))
                except Exception:
                    pass
            elif isinstance(img_size, np.ndarray) and img_size.size == 2:
                try:
                    flat = img_size.reshape(-1)
                    meta['img_size_dino'] = (int(_to_scalar(flat[0])), int(_to_scalar(flat[1])))
                except Exception:
                    pass

            if 'pose' in meta:
                meta['pose'] = _to_matrix4x4(meta['pose'])
            if 'extrinsics' in meta:
                meta['extrinsics'] = _to_matrix4x4(meta['extrinsics'])

            if strict:
                intr = meta.get('intrinsics', None)
                if not (isinstance(intr, (list, tuple)) and len(intr) == 4):
                    raise RuntimeError(f"[CamInfo][strict] invalid intrinsics after normalize: {intr}")
                pose = meta.get('pose', meta.get('extrinsics', None))
                if pose is None:
                    raise RuntimeError("[CamInfo][strict] missing pose/extrinsics after normalize")
                img_size = meta.get('img_size_dino', None)
                if img_size is not None and not (isinstance(img_size, (list, tuple)) and len(img_size) == 2):
                    raise RuntimeError(f"[CamInfo][strict] invalid img_size_dino after normalize: {img_size}")
                if torch.is_tensor(pose):
                    if pose.ndim != 2 or pose.shape != (4, 4):
                        raise RuntimeError(f"[CamInfo][strict] invalid pose tensor shape: {tuple(pose.shape)}")
                elif isinstance(pose, np.ndarray):
                    if pose.ndim != 2 or pose.shape != (4, 4):
                        raise RuntimeError(f"[CamInfo][strict] invalid pose ndarray shape: {pose.shape}")

        if debug and hasattr(self, '_caminfo_debug_count') and self._caminfo_debug_count < 6:  # type: ignore[attr-defined]
            print(f"[CamInfo][norm] got {len(cam_metas)} metas")
            for i, m in enumerate(cam_metas[:min(2, len(cam_metas))]):
                print(f"  - meta[{i}] intrinsics={m.get('intrinsics')} img_size_dino={m.get('img_size_dino')} "
                      f"pose={_summarize_value(m.get('pose', m.get('extrinsics', None)))}")
            self._caminfo_debug_count += 1  # type: ignore[attr-defined]

        return cam_metas

    def _build_dino_fpn_online(
        self,
        points_list: List[torch.Tensor],
        feat_maps: torch.Tensor,
        cam_metas: List[Dict],
        batch_data_samples: List[Any],
        elastic_coords: Optional[List[Any]] = None,
    ) -> Optional[List[ME.SparseTensor]]:
        """从在线 DINOv2 特征 (feat_maps) 构建稀疏 FPN。

        Args:
            points_list: list[Tensor(N_i, C)]，每个样本的点云（增强坐标系）。
            feat_maps: (B, C, H_p, W_p) 的 DINO patch 特征。
            cam_metas: list[dict]，每个样本的 cam_info，包含 intrinsics / img_size_dino 等。
            batch_data_samples: list[Det3DDataSample]，用于读取 img_meta（含 3D 增强 flow）。
            elastic_coords: 可选，list[Tensor/ndarray]，每个样本的 elastic voxel coords（单位：voxel）。
                当启用 elastic 时，backbone 使用 elastic_coords 构建 Minkowski 坐标；
                这里若仍用 xyz_train/voxel_size，会导致 DINO 注入坐标系错配，features_at_coordinates 大量 miss。
        """
        if feat_maps is None or len(points_list) == 0:
            return None

        coords_list: List[torch.Tensor] = []
        feats_list: List[torch.Tensor] = []
        valid_counts: List[int] = []
        total_counts: List[int] = []
        coords_sources: List[str] = []
        skip_reasons: List[str] = []
        B = len(points_list)
        assert feat_maps.size(0) == B, \
            f"feat_maps batch size {feat_maps.size(0)} != points_list length {B}"

        for b_idx in range(B):
            skip_reason = "ok"
            pts = points_list[b_idx]
            if pts.numel() == 0:
                skip_reason = "empty_points"
                skip_reasons.append(skip_reason)
                continue
            # 为避免 CPU/GPU 混用导致的错误，统一将几何运算放到与 DINO 特征相同的 device 上
            feat_map = feat_maps[b_idx:b_idx + 1]  # 1×C×H_p×W_p
            device = feat_map.device
            pts = pts.to(device=device)
            xyz_train = pts[:, :3]
            # Pack3DDetInputs_ 将原始 meta 写在 data_sample.img_metas 中，
            # 而 data_preprocessor 额外写的 pad/batch 信息在 metainfo。
            # DINO 投影与 3D 反解必须使用包含 transformation_3d_flow/pcd_* 的那份 meta。
            sample = batch_data_samples[b_idx]
            img_meta = getattr(sample, 'img_metas', None)
            if not isinstance(img_meta, dict) or 'transformation_3d_flow' not in img_meta:
                img_meta = sample.metainfo if hasattr(sample, 'metainfo') else {}
            cam_meta = cam_metas[b_idx]

            # 1) 反解 3D 增强：训练坐标 -> world 坐标
            #    这里使用与 3DMV / BiFusion 一致的 apply_3d_transformation，只撤销
            #    RandomFlip3D / GlobalRotScaleTrans 等刚性增强，不改变“场景坐标系”本身。
            try:
                xyz_world_np = apply_3d_transformation(
                    xyz_train.clone(), coord_type='DEPTH', img_meta=img_meta, reverse=True)
                xyz_world = torch.as_tensor(xyz_world_np, device=xyz_train.device, dtype=xyz_train.dtype)
            except Exception:
                # 若反解失败，则退回使用训练坐标系（至少保证不中断训练）
                xyz_world = xyz_train

            # 1.5) world -> camera：使用 cam_info 中的 cam2world 外参（或 pose），
            #      与 vis_demo/dino_rgb_vis_3d.ipynb 中的可视化逻辑保持一致。
            pose = cam_meta.get('pose', None)
            if pose is None:
                pose = cam_meta.get('extrinsics', None)

            if pose is not None:
                try:
                    T_c2w = torch.as_tensor(pose, device=xyz_world.device, dtype=xyz_world.dtype)
                    if T_c2w.shape == (4, 4):
                        # world -> cam: w2c = (cam2world)^-1
                        try:
                            T_w2c = torch.linalg.inv(T_c2w)
                        except Exception:
                            T_w2c = T_c2w.inverse()
                        ones = torch.ones((xyz_world.shape[0], 1),
                                          device=xyz_world.device,
                                          dtype=xyz_world.dtype)
                        xyz_h = torch.cat([xyz_world, ones], dim=1)  # (N,4)
                        xyz_cam = (xyz_h @ T_w2c.t())[:, :3]
                    else:
                        # 形状异常时退回 world 坐标（至少不中断）
                        xyz_cam = xyz_world
                except Exception:
                    xyz_cam = xyz_world
            else:
                # 若未提供外参，则近似认为 world≈camera 坐标
                xyz_cam = xyz_world

            # 2) 使用更新后的 intrinsics + DINO 特征图尺寸投影到 patch 网格
            intr = cam_meta.get('intrinsics', None)
            if torch.is_tensor(intr) and intr.numel() == 4:
                intr = [float(x) for x in intr.reshape(-1)]
            elif isinstance(intr, (list, tuple)) and len(intr) == 1 and torch.is_tensor(intr[0]) and intr[0].numel() == 4:
                intr = [float(x) for x in intr[0].reshape(-1)]
            if intr is None or not (isinstance(intr, (list, tuple)) and len(intr) == 4):
                if getattr(self, 'dino_require', False):
                    raise RuntimeError(
                        f"[DINO][strict] missing/invalid intrinsics at sample={b_idx}: intr={intr} "
                        f"cam_keys={list(cam_meta.keys())}"
                    )
                skip_reason = "bad_intrinsics"
                skip_reasons.append(skip_reason)
                continue
            fx_img, fy_img, cx_img, cy_img = intr
            # img_size_dino 为 DINO 输入图像大小（例如 420×560）
            img_size = cam_meta.get('img_size_dino', None)
            def _parse_hw(x):
                if x is None:
                    return None
                if torch.is_tensor(x):
                    if x.numel() == 2:
                        flat = x.detach().reshape(-1)
                        return int(flat[0].item()), int(flat[1].item())
                    return None
                if isinstance(x, (list, tuple)):
                    if len(x) == 2:
                        try:
                            return int(x[0]), int(x[1])
                        except Exception:
                            return int(float(x[0])), int(float(x[1]))
                    if len(x) == 1:
                        return _parse_hw(x[0])
                    return None
                return None

            hw = _parse_hw(img_size)
            if hw is None:
                if getattr(self, 'dino_require', False) and img_size is not None:
                    s = batch_data_samples[b_idx]
                    m = getattr(s, 'img_metas', None)
                    if not isinstance(m, dict):
                        m = {}
                    raise RuntimeError(
                        "[DINO][strict] invalid img_size_dino format "
                        f"sample={b_idx}, img_size_dino={img_size}, "
                        f"type={type(img_size)}, cam_keys={list(cam_meta.keys())}, "
                        f"lidar_path={m.get('lidar_path')}, img_path={m.get('img_path')}"
                    )
                # 若未显式记录，默认认为与 BASE_IMAGE_SIZE 一致
                H_img, W_img = BASE_IMAGE_SIZE  # type: ignore[name-defined]
            else:
                H_img, W_img = hw

            # DINO 特征图大小（patch 网格），通常为 30×40
            H_feat, W_feat = feat_map.shape[-2], feat_map.shape[-1]
            scale_w = float(W_feat) / float(W_img)
            scale_h = float(H_feat) / float(H_img)
            fx_feat = fx_img * scale_w
            fy_feat = fy_img * scale_h
            # 为了精确对齐中心，将主点坐标也映射到 patch 网格坐标系下
            cx_feat = (cx_img + 0.5) * scale_w - 0.5
            cy_feat = (cy_img + 0.5) * scale_h - 0.5

            try:
                uv, valid = project_points_to_uv(
                    xyz_cam,
                    feat_hw=(H_feat, W_feat),
                    max_depth=cam_meta.get('max_depth', 20.0),
                    standard_intrinsics=(fx_feat, fy_feat, cx_feat, cy_feat),
                    already_scaled=True)
            except Exception:
                if getattr(self, 'dino_require', False):
                    s = batch_data_samples[b_idx]
                    m = getattr(s, 'img_metas', None)
                    if not isinstance(m, dict):
                        m = {}
                    raise RuntimeError(
                        "[DINO][strict] project_points_to_uv failed "
                        f"sample={b_idx}, lidar_path={m.get('lidar_path')}, img_path={m.get('img_path')}, "
                        f"img_size_dino={(H_img, W_img)}, feat_hw={(H_feat, W_feat)}, intr={intr}"
                    )
                skip_reason = "proj_exception"
                skip_reasons.append(skip_reason)
                continue

            # 记录当前样本的 2D 可见性统计，用于 valid rate 评估
            if isinstance(valid, torch.Tensor):
                v_cnt = int(valid.sum().item())
                t_cnt = int(valid.numel())
                valid_counts.append(v_cnt)
                total_counts.append(t_cnt)

            # 若 2D 图像在 pipeline 中做过水平翻转，则在像素坐标上做镜像
            if img_meta.get('img_flip', False) or img_meta.get('flip', False):
                u = uv[:, 0]
                v = uv[:, 1]
                u = float(W_feat - 1) - u
                uv = torch.stack([u, v], dim=-1)

            # 3) 在 DINO 特征图上采样点级特征
            feats_2d = sample_img_feat(feat_map, uv, valid, align_corners=False)

            # 4) 构造 Minkowski 稀疏坐标
            # 关键：当启用 elastic 时，backbone 的 sparse coords 来自 elastic_coords。
            # 为了让 DINO 注入在训练中稳定命中，DINO sparse coords 必须与 backbone 一致。
            coords: torch.Tensor
            used_src = 'xyz_train'
            if elastic_coords is not None and b_idx < len(elastic_coords) and elastic_coords[b_idx] is not None:
                try:
                    e = elastic_coords[b_idx]
                    if torch.is_tensor(e):
                        e_t = e.to(device=xyz_train.device, dtype=xyz_train.dtype)
                    else:
                        e_t = torch.as_tensor(e, device=xyz_train.device, dtype=xyz_train.dtype)
                    # elastic_coords 本身是 voxel units 的连续坐标；ME 的 quantize 逻辑等价于 floor
                    coords = torch.floor(e_t).to(torch.int32)
                    used_src = 'elastic'
                except Exception:
                    coords = (xyz_train / self.voxel_size).floor().to(torch.int32)
            else:
                coords = (xyz_train / self.voxel_size).floor().to(torch.int32)
            batch_col = torch.full(
                (coords.shape[0], 1), b_idx, dtype=torch.int32, device=coords.device)
            coords_batched = torch.cat([batch_col, coords], dim=1)

            coords_list.append(coords_batched)
            feats_list.append(feats_2d.to(device=coords.device))
            coords_sources.append(used_src)
            skip_reasons.append(skip_reason)

        if not coords_list:
            # 若一个 batch 内所有样本都无法构建投影，则清空统计信息
            self._last_dino_valid_rate = None  # type: ignore[attr-defined]
            self._last_dino_valid_rate_per_sample = []  # type: ignore[attr-defined]
            self._last_dino_coords_source = []  # type: ignore[attr-defined]
            if getattr(self, 'dino_require', False):
                # 严格模式：禁止静默退化为 None，必须给出可定位的原因
                raise RuntimeError(f"[DINO][strict] build_dino_fpn_online produced empty coords_list; reasons={skip_reasons}")
            return None

        # 聚合 valid rate 统计，保存到成员变量，便于在 notebook 中查看
        if total_counts and sum(total_counts) > 0:
            total_points = float(sum(total_counts))
            total_valid = float(sum(valid_counts))
            valid_rate = total_valid / total_points
            per_sample_rates = [
                (float(v) / float(t)) if t > 0 else 0.0
                for v, t in zip(valid_counts, total_counts)
            ]
            self._last_dino_valid_rate = valid_rate  # type: ignore[attr-defined]
            self._last_dino_valid_rate_per_sample = per_sample_rates  # type: ignore[attr-defined]
        else:
            self._last_dino_valid_rate = None  # type: ignore[attr-defined]
            self._last_dino_valid_rate_per_sample = []  # type: ignore[attr-defined]

        # 记录本次 dino_fpn 构建使用的坐标来源（elastic / xyz_train），便于定位错配问题
        self._last_dino_coords_source = coords_sources  # type: ignore[attr-defined]

        coords_batch = torch.cat(coords_list, dim=0)
        feats_batch = torch.cat(feats_list, dim=0)
        return build_sparse_fpn(coords_batch, feats_batch)

    def _build_dino_fpn_from_clip(self, batch_inputs_dict: Dict[str, Any]):
        """从 clip_pix + cam_info 在线投影获取点级 DINO，并构建稀疏 FPN（单帧版）。"""
        points_list = batch_inputs_dict.get('points', None)
        clip_raw = batch_inputs_dict.get('clip_pix', None)
        cam_raw = batch_inputs_dict.get('cam_info', None)
        if points_list is None or clip_raw is None or cam_raw is None:
            if self._dino_debug and self._dino_debug_count < 3:
                print(f"[DINO][warn] skip build_from_clip: points={points_list is not None}, clip={clip_raw is not None}, cam_info={cam_raw is not None}")
                self._dino_debug_count += 1
            return None

        # 统一采样对齐策略：align_corners=False 与 grid_sample 配套，避免半像素偏移
        align_corners = False

        # 规范 cam_info 为 per-sample 列表
        if isinstance(cam_raw, list):
            cam_metas = cam_raw
        elif isinstance(cam_raw, dict):
            cam_metas = [cam_raw for _ in range(len(points_list))]
        else:
            return None

        # 规范 clip_pix 为 per-sample 列表
        clip_list: List[torch.Tensor] = []
        if isinstance(clip_raw, list):
            clip_list = clip_raw
        elif torch.is_tensor(clip_raw):
            if clip_raw.dim() == 4 and clip_raw.shape[0] == len(points_list):
                clip_list = [clip_raw[b] for b in range(clip_raw.shape[0])]
            elif clip_raw.dim() == 3:
                clip_list = [clip_raw for _ in range(len(points_list))]
        else:
            clip_tensor = torch.as_tensor(clip_raw)
            if clip_tensor.dim() == 4 and clip_tensor.shape[0] == len(points_list):
                clip_list = [clip_tensor[b] for b in range(clip_tensor.shape[0])]
            elif clip_tensor.dim() == 3:
                clip_list = [clip_tensor for _ in range(len(points_list))]

        if not clip_list:
            if self._dino_debug and self._dino_debug_count < 3:
                print("[DINO][warn] clip_list empty after normalization")
                self._dino_debug_count += 1
            return None

        coords_list, feats_list = [], []
        for b_idx, pts in enumerate(points_list):
            if b_idx >= len(cam_metas):
                continue
            cam_meta = cam_metas[b_idx] if isinstance(cam_metas, list) else cam_metas
            clip_tensor = clip_list[b_idx] if b_idx < len(clip_list) else clip_list[0]
            if clip_tensor is None:
                continue
            if clip_tensor.dim() == 4:
                clip_tensor = clip_tensor[b_idx] if clip_tensor.shape[0] > b_idx else clip_tensor[0]
            if not torch.is_tensor(clip_tensor):
                clip_tensor = torch.as_tensor(clip_tensor)
            clip_tensor = clip_tensor.to(device=pts.device)
            if clip_tensor.dim() != 3:
                continue
            C, Hf, Wf = clip_tensor.shape

            xyz = pts[:, :3]

            # 固定使用标准 ScanNet 内参（不依赖 cam_info 内参）
            intr = SCANET_INTRINSICS
            raw_intr = intr
            max_depth = cam_meta.get('max_depth', 20.0) if isinstance(cam_meta, dict) else 20.0
            pose = None
            if isinstance(cam_meta, dict):
                pose = cam_meta.get('pose') or cam_meta.get('extrinsics')
                frame_idx = cam_meta.get('frame_idx', cam_meta.get('frame_id', b_idx))
            else:
                frame_idx = b_idx

            # 世界 -> 相机
            if pose is not None:
                try:
                    pose_arr = np.asarray(pose)
                except Exception:
                    pose_arr = pose
                # 若 pose 是多帧序列，按 frame_idx 取对应帧，不足时报错
                if isinstance(pose_arr, np.ndarray) and pose_arr.ndim == 3 and pose_arr.shape[-2:] == (4, 4):
                    if frame_idx >= pose_arr.shape[0]:
                        raise RuntimeError(f"pose sequence len={pose_arr.shape[0]} but frame_idx={frame_idx}")
                    pose_arr = pose_arr[frame_idx]
                elif isinstance(pose_arr, (list, tuple)):
                    pose_arr_np = np.asarray(pose_arr)
                    if pose_arr_np.ndim == 3 and pose_arr_np.shape[-2:] == (4, 4):
                        if frame_idx >= pose_arr_np.shape[0]:
                            raise RuntimeError(f"pose sequence len={pose_arr_np.shape[0]} but frame_idx={frame_idx}")
                        pose_arr = pose_arr_np[frame_idx]
                pose_t = torch.as_tensor(pose_arr, device=xyz.device, dtype=xyz.dtype)
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
                if not hasattr(self, '_dino_caminfo_log'):
                    self._dino_caminfo_log = 0
                if self._dino_debug and self._dino_caminfo_log < 3:
                    intr_arr = np.asarray(raw_intr)
                    pose_shape = None if pose is None else np.asarray(pose).shape
                    print(f"[DINO][cam_debug] b={b_idx} intr_raw_shape={intr_arr.shape} intr_raw={intr_arr} pose_shape={pose_shape} max_depth={max_depth} feat_hw=({Hf},{Wf})")
                    self._dino_caminfo_log += 1
                if not hasattr(self, '_dino_proj_log_count'):
                    self._dino_proj_log_count = 0
                if self._dino_debug and self._dino_proj_log_count < 5:
                    depth = xyz_cam[:, 2]
                    depth_valid = (depth > MIN_DEPTH) & (depth < max_depth)
                    depth_ratio = float(depth_valid.float().mean().item()) if depth.numel() > 0 else 0.0
                    valid_ratio = float(valid.float().mean().item()) if valid.numel() > 0 else 0.0
                    u_valid = uv_feat[:, 0][valid]
                    v_valid = uv_feat[:, 1][valid]
                    try:
                        u_min, u_max = float(u_valid.min().item()), float(u_valid.max().item()) if u_valid.numel() > 0 else (0.0, 0.0)
                        v_min, v_max = float(v_valid.min().item()), float(v_valid.max().item()) if v_valid.numel() > 0 else (0.0, 0.0)
                    except Exception:
                        u_min = u_max = v_min = v_max = 0.0
                    print(f"[DINO][proj_stats] b={b_idx} depth_ok={depth_ratio:.3f} valid_ratio={valid_ratio:.3f} "
                          f"u=[{u_min:.1f},{u_max:.1f}] v=[{v_min:.1f},{v_max:.1f}] feat_hw=({Hf},{Wf})")
                    self._dino_proj_log_count += 1
                sampled = sample_img_feat(clip_tensor.unsqueeze(0), uv_feat, valid, align_corners=align_corners)
                sampled = sampled.to(pts.device)
            except Exception as e:
                if self._dino_debug and self._dino_debug_count < 3:
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
        if self._dino_debug and self._dino_debug_count < 5:
            try:
                print(f"[DINO][debug] build_from_clip before collate: "
                      f"n_chunks={len(coords_list)}, "
                      f"coords0={coords_list[0].shape}, feats0={feats_list[0].shape}")
            except Exception:
                pass
        try:
            coords_batch, feats_batch = ME.utils.sparse_collate(
                coords_list, feats_list, device=feats_list[0].device)
            if self._dino_debug and self._dino_debug_count < 5:
                print(f"[DINO][debug] after sparse_collate: "
                      f"coords_batch={coords_batch.shape}, feats_batch={feats_batch.shape}")
            fpn = build_sparse_fpn(coords_batch, feats_batch)
            if self._dino_debug and self._dino_debug_count < 5:
                shapes = [x.shape for x in fpn]
                strides = [x.tensor_stride for x in fpn]
                print(f"[DINO][debug] FPN built: shapes={shapes}, strides={strides}")
            return fpn
        except Exception as e:
            if self._dino_debug and self._dino_debug_count < 5:
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

        labels_all = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)

        # Selection / rerank before top-K (quality-aware selection).
        # - `scores` is kept as the *base score* for downstream scoring/AP.
        # - `select_scores` is used for top-K selection and diagnostics (oracle/rank).
        sel_cfg = self.test_cfg.get('selection', None) or {}
        base_scores = scores  # (Q, C)
        select_scores = base_scores
        if bool(sel_cfg.get('enable', False)):
            mode = str(sel_cfg.get('mode', 'cls'))
            if mode not in ('cls', 'cls_stab_size'):
                mode = 'cls'
            if mode == 'cls_stab_size':
                # Compute quality proxies on point-domain masks (robust to SP/P domains).
                sp_thr = float(self.test_cfg.sp_score_thr)
                stability_offset = float(sel_cfg.get('stability_offset', 0.1))
                gamma = float(sel_cfg.get('gamma', 1.0))
                size_ref = float(sel_cfg.get('size_ref', 300.0))
                delta = float(sel_cfg.get('delta', 0.3))
                size_clip_min = float(sel_cfg.get('size_clip_min', 0.5))
                size_clip_max = float(sel_cfg.get('size_clip_max', 2.0))

                mask_sigmoid_all = pred_masks.sigmoid()
                try:
                    if isinstance(superpoints, torch.Tensor):
                        n_raw = int(superpoints.numel())
                        n_mask_dim = int(mask_sigmoid_all.shape[1])
                        n_sp = int(superpoints.max().item()) + 1 if superpoints.numel() > 0 else 0
                        if n_mask_dim != n_raw and n_sp > 0 and n_mask_dim == n_sp:
                            mask_sigmoid_all = mask_sigmoid_all[:, superpoints]
                except Exception:
                    pass

                # Stability proxy: IoU(thr-Δ, thr+Δ). Since (thr+Δ) ⊆ (thr-Δ), IoU == |hi|/|lo|.
                thr_lo = max(0.0, sp_thr - stability_offset)
                thr_hi = min(1.0, sp_thr + stability_offset)
                lo = mask_sigmoid_all > thr_lo
                hi = mask_sigmoid_all > thr_hi
                lo_sum = lo.sum(1).float().clamp_min(1.0)
                hi_sum = hi.sum(1).float()
                q_stab = (hi_sum / lo_sum).clamp(0.0, 1.0)

                # Size/coverage guardrail (avoid "too small & pure" dominating the ranking).
                n_fg = (mask_sigmoid_all > sp_thr).sum(1).float()
                size_ref = max(size_ref, 1.0)
                q_size = (n_fg / size_ref).clamp(min=size_clip_min, max=size_clip_max)

                q = (q_stab.pow(gamma) * q_size.pow(delta)).to(base_scores.dtype)
                select_scores = base_scores * q.unsqueeze(1)

            keep_topk = int(sel_cfg.get('keep_topk', int(self.test_cfg.topk_insts)))
            keep_topk = max(0, keep_topk)
            fallback_topk = int(sel_cfg.get('fallback_topk', 0))
            fallback_topk = max(0, fallback_topk)
            max_candidates = int(sel_cfg.get('max_candidates', keep_topk + fallback_topk))
            max_candidates = max(keep_topk, max_candidates)

            base_flat = base_scores.flatten(0, 1)
            select_flat = select_scores.flatten(0, 1)
            k_main = min(int(keep_topk), int(select_flat.numel()))
            if k_main > 0:
                _, idx_main = select_flat.topk(k_main, sorted=False)
            else:
                idx_main = select_flat.new_zeros((0,), dtype=torch.long)
            idx = idx_main
            if fallback_topk > 0:
                k_fb = min(int(fallback_topk), int(base_flat.numel()))
                if k_fb > 0:
                    _, idx_fb = base_flat.topk(k_fb, sorted=False)
                    idx = torch.unique(torch.cat([idx_main, idx_fb], dim=0))
                else:
                    idx = idx_main
                if idx.numel() > max_candidates:
                    sub = select_flat[idx]
                    _, ord2 = sub.topk(max_candidates, sorted=False)
                    idx = idx[ord2]

            select_scores = select_flat[idx]
            scores = base_flat[idx]
            labels = labels_all[idx]
            topk_idx = torch.div(idx, self.num_classes, rounding_mode='floor')
        else:
            topk_num = min(int(self.test_cfg.topk_insts), base_scores.shape[0] * base_scores.shape[1])
            base_flat = base_scores.flatten(0, 1)
            if topk_num > 0:
                scores, idx = base_flat.topk(topk_num, sorted=False)
                labels = labels_all[idx]
                topk_idx = torch.div(idx, self.num_classes, rounding_mode='floor')
                select_scores = scores.clone()
            else:
                empty = base_flat.new_zeros((0,))
                scores, idx = empty, base_flat.new_zeros((0,), dtype=torch.long)
                labels = labels_all.new_zeros((0,), dtype=torch.long)
                topk_idx = idx
                select_scores = empty

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
        select_scores = select_scores[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_thr = int(self.test_cfg.npoint_thr)
        npoint_mask = mask_pointnum > npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        select_scores = select_scores[npoint_mask]

        # Optional: high-IoU copy suppression (explicit de-dup; actually drops masks).
        cs_cfg = self.test_cfg.get('copy_suppress', None) or {}
        if bool(cs_cfg.get('enable', False)) and mask_pred.shape[0] > 1:
            iou_thr = float(cs_cfg.get('iou_thr', 0.9))
            max_num = cs_cfg.get('max_num', None)
            pre_max_num = cs_cfg.get('pre_max_num', None)
            allow_replace = bool(cs_cfg.get('allow_replace', True))
            refill = bool(cs_cfg.get('refill', True))
            if pre_max_num is not None:
                pre_max_num = int(pre_max_num)
            if max_num is not None:
                max_num = int(max_num)

            sort_by = str(cs_cfg.get('sort_by', 'scores'))
            if sort_by not in ('scores', 'select_scores'):
                sort_by = 'scores'
            prefer_by = str(cs_cfg.get('prefer_by', 'scores'))
            if prefer_by not in ('scores', 'select_scores'):
                prefer_by = 'scores'

            rank_scores = scores if sort_by == 'scores' else select_scores
            prefer_scores = scores if prefer_by == 'scores' else select_scores

            order = torch.argsort(rank_scores, descending=True)
            if pre_max_num is not None and pre_max_num > 0 and order.numel() > pre_max_num:
                order = order[:pre_max_num]

            m = mask_pred[order].float()
            areas = m.sum(1).clamp_min(1.0)
            inter = m @ m.t()
            union = areas[:, None] + areas[None, :] - inter
            iou = inter / union.clamp_min(1.0)

            prefer_vals = prefer_scores[order]
            keep: list[int] = []

            for i in range(int(order.numel())):
                if not keep:
                    keep.append(i)
                    continue

                prev = torch.tensor(keep, device=iou.device, dtype=torch.long)
                ov = iou[i, prev] > iou_thr
                if torch.any(ov):
                    if allow_replace:
                        prev_ov = prev[ov]
                        ov_iou = iou[i, prev_ov]
                        j = int(prev_ov[int(torch.argmax(ov_iou).item())].item())
                        if float(prefer_vals[i].item()) > float(prefer_vals[j].item()):
                            keep[keep.index(j)] = i
                    continue

                if max_num is None or len(keep) < max_num:
                    keep.append(i)
                    continue

                if not refill:
                    break

            keep_t = torch.tensor(keep, device=order.device, dtype=torch.long)
            keep_inds = order[keep_t]
            scores = scores[keep_inds]
            labels = labels[keep_inds]
            mask_pred = mask_pred[keep_inds]
            select_scores = select_scores[keep_inds]

        return mask_pred, labels, scores, select_scores

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
        batch_size = len(batch_data_samples)
        img_metas = [batch_data_sample.img_metas.copy() for batch_data_sample in batch_data_samples]

        # 提取图像特征并规范为 per-sample list（若不可用则回退为 None，并打印提示）
        img_feats: Optional[List[torch.Tensor]] = None
        if getattr(self, 'img_backbone', None) is not None:
            with torch.no_grad():
                raw_feats = None
                if 'img' in batch_inputs_dict:
                    raw_feats = self.img_backbone(batch_inputs_dict['img'])  # type: ignore[operator]
                elif 'imgs' in batch_inputs_dict:
                    raw_feats = self.img_backbone(batch_inputs_dict['imgs'])  # type: ignore[operator]
                elif 'img_path' in batch_inputs_dict:
                    raw_feats = self.img_backbone(batch_inputs_dict['img_path'])  # type: ignore[operator]

            # 规范 raw_feats -> List[Tensor(C,H,W)] 与 batch 对齐
            if raw_feats is not None:
                if isinstance(raw_feats, list):
                    img_feats = [f for f in raw_feats]
                elif torch.is_tensor(raw_feats):
                    if raw_feats.dim() == 4 and raw_feats.shape[0] == batch_size:
                        img_feats = [raw_feats[i] for i in range(raw_feats.shape[0])]
                    elif raw_feats.dim() == 3:
                        img_feats = [raw_feats for _ in range(batch_size)]
                # 若长度不匹配则放弃使用 2D 特征
                if img_feats is not None and len(img_feats) != batch_size:
                    img_feats = None
        if getattr(self, 'img_backbone', None) is not None and img_feats is None:
            print("[MixFormer3D_FF] img_backbone enabled but no valid 2D features; fallback to pure 3D.")
        
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
        if img_feats is not None:
            x = self.backbone(field.sparse(),
                              partial(self._f, img_features=img_feats, img_metas=img_metas))
        else:
            x = self.backbone(field.sparse())
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

    def _f(self, x, img_features, img_metas):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
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
            # 规范 img_feature -> (C,H,W)
            if img_feature.dim() == 4:
                img_feature = img_feature[0]
            if not torch.is_tensor(img_feature):
                img_feature = torch.as_tensor(img_feature, device=point.device)
            img_pad_shape = img_feature.shape[-2:]
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_pad_shape,
                img_shape=img_pad_shape,
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
        self._dino_debug = os.environ.get('DINO_DEBUG', '') == '1'
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
        batch_size = len(batch_data_samples)
        backbone_use_dino = bool(getattr(self.backbone, 'use_dino', False))
        # 调试：明确区分“外部是否提供离线 DINO” vs “是否具备在线/投影构建条件”（仅前几次打印）
        if self._dino_debug and self._dino_debug_count < 3:
            provided_offline = {
                'dino_fpn': 'dino_fpn' in batch_inputs_dict,
                'dino_feats': 'dino_feats' in batch_inputs_dict,
                'dino_point_feats': 'dino_point_feats' in batch_inputs_dict,
                'clip_pix': 'clip_pix' in batch_inputs_dict,
            }
            clip_shapes = None
            if 'clip_pix' in batch_inputs_dict:
                cp = batch_inputs_dict['clip_pix']
                if isinstance(cp, list) and len(cp) > frame_i and torch.is_tensor(cp[frame_i]):
                    clip_shapes = cp[frame_i].shape
                elif torch.is_tensor(cp):
                    clip_shapes = cp.shape
            cam_raw = batch_inputs_dict.get('cam_info', None)
            cam_info_len = len(cam_raw) if isinstance(cam_raw, list) else (1 if cam_raw is not None else 0)
            proj_ready = ('clip_pix' in batch_inputs_dict) and (cam_raw is not None)
            print(
                "[DINO][debug] "
                f"provided_offline={provided_offline}, "
                f"backbone_use_dino={backbone_use_dino}, proj_ready={proj_ready}, "
                f"clip_shape={clip_shapes}, cam_info_len={cam_info_len}, frame={frame_i}"
            )
            self._dino_debug_count += 1
        # 可选：外部传入的 DINO 稀疏金字塔或点级 DINO 特征，按帧索引取用
        dino_feats = None
        # 1) 直接传入的稀疏 FPN 列表 [s1..s16]
        if backbone_use_dino and 'dino_fpn' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_fpn'][frame_i]
                if self._dino_debug and self._dino_debug_count < 5:
                    print(f"[DINO] use provided dino_fpn frame={frame_i}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        elif backbone_use_dino and 'dino_feats' in batch_inputs_dict:
            try:
                dino_feats = batch_inputs_dict['dino_feats'][frame_i]
                if self._dino_debug and self._dino_debug_count < 5:
                    print(f"[DINO] use provided dino_feats frame={frame_i}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        # 2) 如果传入了点级 DINO 特征（与 points 同顺序），自动构建 FPN
        if backbone_use_dino and dino_feats is None and 'dino_point_feats' in batch_inputs_dict:
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
                if self._dino_debug and self._dino_debug_count < 3:
                    print(f"[DINO] build from dino_point_feats frame={frame_i}, coords={coords_batch.shape}, feats={feats_batch.shape}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        # 3) 若仍为空且存在 clip_pix + cam_info，尝试在线投影得到点级 DINO
        if backbone_use_dino and dino_feats is None and 'clip_pix' in batch_inputs_dict and 'cam_info' in batch_inputs_dict:
            try:
                dino_feats = self._build_dino_fpn_from_clip(batch_inputs_dict, frame_i)
                if self._dino_debug and dino_feats is not None and self._dino_debug_count < 3:
                    shapes = [x.shape for x in dino_feats]
                    strides = [x.tensor_stride for x in dino_feats]
                    print(f"[DINO] build from clip_pix frame={frame_i}, shapes={shapes}, strides={strides}")
                    self._dino_debug_count += 1
            except Exception:
                dino_feats = None
        # 4) 若最终仍未构建，打印一次提示
        if self._dino_debug and dino_feats is None and self._dino_debug_count < 5:
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
        sp_xyz = scatter_mean(torch.cat(all_xyz, dim=0), sp_idx, dim=0)
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
            sp_xyz_list.append(sp_xyz[begin: end])
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
            # cam_meta 可能是按帧列表或单个 dict，这里按 frame_i 取对应帧
            if isinstance(cam_meta, list) and len(cam_meta) > 0:
                meta_t = cam_meta[frame_i] if frame_i < len(cam_meta) else cam_meta[0]
            elif isinstance(cam_meta, dict):
                meta_t = cam_meta
            else:
                meta_t = {}

            # Online 也统一固定内参，避免 cam_info 中被缩放过
            intr = SCANET_INTRINSICS
            max_depth = meta_t.get('max_depth', 20.0) if isinstance(meta_t, dict) else 20.0
            pose = None
            if isinstance(meta_t, dict):
                pose = meta_t.get('pose') or meta_t.get('extrinsics')

            # 世界坐标 -> 相机坐标（若提供 cam2world 则取逆）
            if pose is not None:
                try:
                    pose_arr = np.asarray(pose)
                except Exception:
                    pose_arr = pose
                # 若 pose 为多帧序列，按 frame_i 取对应帧，否则取首帧
                if isinstance(pose_arr, np.ndarray) and pose_arr.ndim == 3 and pose_arr.shape[-2:] == (4, 4):
                    if frame_i >= pose_arr.shape[0]:
                        raise RuntimeError(f"pose sequence len={pose_arr.shape[0]} but frame_i={frame_i}")
                    pose_arr = pose_arr[frame_i]
                elif isinstance(pose_arr, (list, tuple)):
                    pose_arr_np = np.asarray(pose_arr)
                    if pose_arr_np.ndim == 3 and pose_arr_np.shape[-2:] == (4, 4):
                        if frame_i >= pose_arr_np.shape[0]:
                            raise RuntimeError(f"pose sequence len={pose_arr_np.shape[0]} but frame_i={frame_i}")
                        pose_arr = pose_arr_np[frame_i]
                pose_t = torch.as_tensor(pose_arr, device=xyz.device, dtype=xyz.dtype)
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
            if not hasattr(self, '_dino_caminfo_log'):
                self._dino_caminfo_log = 0
            if self._dino_debug and self._dino_caminfo_log < 3:
                intr_arr = np.asarray(meta_t.get('intrinsics', intr) if isinstance(meta_t, dict) else intr)
                pose_shape = None if pose is None else np.asarray(pose).shape
                print(f"[DINO][cam_debug] frame={frame_i} b={b_idx} intr_raw_shape={intr_arr.shape} intr_raw={intr_arr} pose_shape={pose_shape} max_depth={max_depth} feat_hw=({Hf},{Wf})")
                self._dino_caminfo_log += 1
            if not hasattr(self, '_dino_proj_log_count'):
                self._dino_proj_log_count = 0
            if self._dino_debug and self._dino_proj_log_count < 5:
                depth = xyz_cam[:, 2]
                depth_valid = (depth > MIN_DEPTH) & (depth < max_depth)
                depth_ratio = float(depth_valid.float().mean().item()) if depth.numel() > 0 else 0.0
                valid_ratio = float(valid.float().mean().item()) if valid.numel() > 0 else 0.0
                u_valid = uv_feat[:, 0][valid]
                v_valid = uv_feat[:, 1][valid]
                try:
                    u_min, u_max = float(u_valid.min().item()), float(u_valid.max().item()) if u_valid.numel() > 0 else (0.0, 0.0)
                    v_min, v_max = float(v_valid.min().item()), float(v_valid.max().item()) if v_valid.numel() > 0 else (0.0, 0.0)
                except Exception:
                    u_min = u_max = v_min = v_max = 0.0
                print(f"[DINO][proj_stats] frame={frame_i} b={b_idx} depth_ok={depth_ratio:.3f} valid_ratio={valid_ratio:.3f} "
                      f"u=[{u_min:.1f},{u_max:.1f}] v=[{v_min:.1f},{v_max:.1f}] feat_hw=({Hf},{Wf})")
                self._dino_proj_log_count += 1
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
        if self._dino_debug and self._dino_debug_count < 3:
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
        scene_metrics = []
        
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
                    mv_mask, mv_labels, mv_scores, mv_queries, mv_bboxes, stats = online_merger.merge(
                        results[-1].pop('pts_instance_mask')[0],
                        results[-1].pop('instance_labels')[0],
                        results[-1].pop('instance_scores')[0],
                        results[-1].pop('instance_queries')[0],
                        query_feats_list.pop(-1)[0],
                        sem_preds_list.pop(-1)[0],
                        sp_xyz_list.pop(-1)[0],
                        bboxes_list.pop(-1)[0] if self.use_bbox else None)
                    
                    # 🆕 Collect Online Metrics
                    if self.test_cfg.get('collect_online_metrics', False):
                        try:
                            # 1. Get GT masks for current frame
                            gt_ins = batch_data_samples[0].gt_pts_seg.pts_instance_mask[frame_i]
                            # Convert to binary masks (N_gt, P)
                            gt_ids = torch.unique(gt_ins)
                            gt_ids = gt_ids[gt_ids != -1] # Remove ignore
                            
                            matched_gt_count = 0
                            if len(gt_ids) > 0:
                                # Create binary masks for each GT instance (N_gt, P)
                                gt_masks = (gt_ins.unsqueeze(0) == gt_ids.unsqueeze(1))
                                
                                # 2. Get Memory masks for current frame
                                # online_merger.cur_masks is (N_mem, Total_points)
                                # points_per_frame can vary if points are not fixed
                                points_per_frame = batch_inputs_dict['points'][0][frame_i].shape[0]
                                mem_masks = online_merger.cur_masks[:, -points_per_frame:]
                                
                                # 3. Calculate IoU
                                # Intersection: (N_gt, N_mem)
                                intersection = (gt_masks.unsqueeze(1) & mem_masks.unsqueeze(0)).sum(-1).float()
                                # Union: (N_gt, N_mem)
                                union = (gt_masks.unsqueeze(1) | mem_masks.unsqueeze(0)).sum(-1).float()
                                ious = intersection / (union + 1e-6)
                                
                                # 4. Check matches > 0.1
                                max_ious, _ = ious.max(dim=1) # Max IoU for each GT
                                matched_gt_count = (max_ious > 0.1).sum().item()
                            
                            stats['GT_matched_0.1'] = matched_gt_count
                            stats['GT_total'] = len(gt_ids)
                            stats['frame_idx'] = frame_i
                            scene_metrics.append(stats)
                        except Exception as e:
                            print(f"Error collecting online metrics: {e}")

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

        # 🆕 Save Online Metrics
        if self.test_cfg.get('collect_online_metrics', False) and len(scene_metrics) > 0:
            try:
                save_dir = os.path.join(self.test_cfg.get('work_dir', 'work_dirs'), 'online_metrics')
                os.makedirs(save_dir, exist_ok=True)
                # Try to get scene ID from metainfo
                if 'sample_idx' in batch_data_samples[0].metainfo:
                    scene_id = str(batch_data_samples[0].metainfo['sample_idx'])
                elif 'file_name' in batch_data_samples[0].metainfo:
                    scene_id = os.path.splitext(os.path.basename(batch_data_samples[0].metainfo['file_name']))[0]
                else:
                    scene_id = f"scene_{int(time.time())}"
                
                with open(os.path.join(save_dir, f"{scene_id}.json"), 'w') as f:
                    json.dump(scene_metrics, f, indent=2)
            except Exception as e:
                print(f"Error saving online metrics: {e}")

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
        instance_select_scores = []
        if isinstance(inst_res, (tuple, list)) and len(inst_res) > 5:
            instance_select_scores.append(inst_res[5])
        if isinstance(inst_res2, (tuple, list)) and len(inst_res2) > 5:
            instance_select_scores.append(inst_res2[5])
        instance_queries = [inst_res[3], inst_res2[3]]
        mapping = [inst_res[4], inst_res2[4]]
      
        pd_kwargs = dict(
            pts_semantic_mask=pts_semantic_mask,
            pts_instance_mask=pts_instance_mask,
            instance_labels=instance_labels,
            instance_scores=instance_scores,
            instance_queries=instance_queries,
        )
        if len(instance_select_scores) == 2:
            pd_kwargs["instance_select_scores"] = instance_select_scores

        return [PointData(**pd_kwargs)], mapping
    
    def predict_by_feat_instance(self, out: Dict[str, Any], superpoints: Any, score_threshold: float) -> Tuple[Any, torch.Tensor, torch.Tensor, Any, torch.Tensor, torch.Tensor]:  # type: ignore[override]
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
        labels_all = torch.arange(
            self.num_classes,
            device=scores.device).unsqueeze(0).repeat(
                len(cls_preds), 1).flatten(0, 1)

        sel_cfg = self.test_cfg.get('selection', None) or {}
        base_scores = scores
        select_scores_mat = base_scores
        if bool(sel_cfg.get('enable', False)):
            mode = str(sel_cfg.get('mode', 'cls'))
            if mode not in ('cls', 'cls_stab_size'):
                mode = 'cls'
            if mode == 'cls_stab_size':
                sp_thr = float(self.test_cfg.sp_score_thr)
                stability_offset = float(sel_cfg.get('stability_offset', 0.1))
                gamma = float(sel_cfg.get('gamma', 1.0))
                size_ref = float(sel_cfg.get('size_ref', 300.0))
                delta = float(sel_cfg.get('delta', 0.3))
                size_clip_min = float(sel_cfg.get('size_clip_min', 0.5))
                size_clip_max = float(sel_cfg.get('size_clip_max', 2.0))

                mask_sigmoid_all = pred_masks.sigmoid()
                try:
                    if isinstance(superpoints, torch.Tensor):
                        n_raw = int(superpoints.numel())
                        n_mask_dim = int(mask_sigmoid_all.shape[1])
                        n_sp = int(superpoints.max().item()) + 1 if superpoints.numel() > 0 else 0
                        if n_mask_dim != n_raw and n_sp > 0 and n_mask_dim == n_sp:
                            mask_sigmoid_all = mask_sigmoid_all[:, superpoints]
                except Exception:
                    pass

                thr_lo = max(0.0, sp_thr - stability_offset)
                thr_hi = min(1.0, sp_thr + stability_offset)
                lo = mask_sigmoid_all > thr_lo
                hi = mask_sigmoid_all > thr_hi
                lo_sum = lo.sum(1).float().clamp_min(1.0)
                hi_sum = hi.sum(1).float()
                q_stab = (hi_sum / lo_sum).clamp(0.0, 1.0)

                n_fg = (mask_sigmoid_all > sp_thr).sum(1).float()
                size_ref = max(size_ref, 1.0)
                q_size = (n_fg / size_ref).clamp(min=size_clip_min, max=size_clip_max)

                q = (q_stab.pow(gamma) * q_size.pow(delta)).to(base_scores.dtype)
                select_scores_mat = base_scores * q.unsqueeze(1)

            keep_topk = int(sel_cfg.get('keep_topk', int(self.test_cfg.topk_insts)))
            keep_topk = max(0, keep_topk)
            fallback_topk = int(sel_cfg.get('fallback_topk', 0))
            fallback_topk = max(0, fallback_topk)
            max_candidates = int(sel_cfg.get('max_candidates', keep_topk + fallback_topk))
            max_candidates = max(keep_topk, max_candidates)

            base_flat = base_scores.flatten(0, 1)
            select_flat = select_scores_mat.flatten(0, 1)
            k_main = min(int(keep_topk), int(select_flat.numel()))
            if k_main > 0:
                _, idx_main = select_flat.topk(k_main, sorted=False)
            else:
                idx_main = select_flat.new_zeros((0,), dtype=torch.long)
            idx = idx_main
            if fallback_topk > 0:
                k_fb = min(int(fallback_topk), int(base_flat.numel()))
                if k_fb > 0:
                    _, idx_fb = base_flat.topk(k_fb, sorted=False)
                    idx = torch.unique(torch.cat([idx_main, idx_fb], dim=0))
                else:
                    idx = idx_main
                if idx.numel() > max_candidates:
                    sub = select_flat[idx]
                    _, ord2 = sub.topk(max_candidates, sorted=False)
                    idx = idx[ord2]

            select_scores = select_flat[idx]
            scores, topk_idx_flat = base_flat[idx], idx
            labels = labels_all[topk_idx_flat]
        else:
            topk_num = min(int(self.test_cfg.topk_insts), base_scores.shape[0] * base_scores.shape[1])
            base_flat = base_scores.flatten(0, 1)
            if topk_num > 0:
                scores, topk_idx_flat = base_flat.topk(topk_num, sorted=False)
                labels = labels_all[topk_idx_flat]
                select_scores = scores.clone()
            else:
                empty = base_flat.new_zeros((0,))
                scores, topk_idx_flat = empty, base_flat.new_zeros((0,), dtype=torch.long)
                labels = labels_all.new_zeros((0,), dtype=torch.long)
                select_scores = empty

        topk_idx = torch.div(topk_idx_flat, self.num_classes, rounding_mode='floor')
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
            select_scores = select_scores[keep_inds]

        mask_pred_sigmoid = mask_pred_sigmoid[:, ...]
        mask_pred = mask_pred_sigmoid > self.test_cfg.sp_score_thr

        # score_thr
        score_mask = scores > score_threshold
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]
        queries = queries[score_mask]
        mapping = mapping[score_mask]
        select_scores = select_scores[score_mask]

        # npoint_thr
        mask_pointnum = mask_pred.sum(1)
        npoint_thr = int(self.test_cfg.npoint_thr)
        npoint_mask = mask_pointnum > npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]
        queries = queries[npoint_mask]
        mapping = mapping[npoint_mask]
        select_scores = select_scores[npoint_mask]

        cs_cfg = self.test_cfg.get('copy_suppress', None) or {}
        if bool(cs_cfg.get('enable', False)) and mask_pred.shape[0] > 1:
            iou_thr = float(cs_cfg.get('iou_thr', 0.9))
            max_num = cs_cfg.get('max_num', None)
            pre_max_num = cs_cfg.get('pre_max_num', None)
            allow_replace = bool(cs_cfg.get('allow_replace', True))
            refill = bool(cs_cfg.get('refill', True))
            if pre_max_num is not None:
                pre_max_num = int(pre_max_num)
            if max_num is not None:
                max_num = int(max_num)

            sort_by = str(cs_cfg.get('sort_by', 'scores'))
            if sort_by not in ('scores', 'select_scores'):
                sort_by = 'scores'
            prefer_by = str(cs_cfg.get('prefer_by', 'scores'))
            if prefer_by not in ('scores', 'select_scores'):
                prefer_by = 'scores'

            rank_scores = scores if sort_by == 'scores' else select_scores
            prefer_scores = scores if prefer_by == 'scores' else select_scores

            order = torch.argsort(rank_scores, descending=True)
            if pre_max_num is not None and pre_max_num > 0 and order.numel() > pre_max_num:
                order = order[:pre_max_num]
            m = mask_pred[order].float()
            areas = m.sum(1).clamp_min(1.0)
            inter = m @ m.t()
            union = areas[:, None] + areas[None, :] - inter
            iou = inter / union.clamp_min(1.0)

            prefer_vals = prefer_scores[order]
            keep: list[int] = []
            for i in range(int(order.numel())):
                if not keep:
                    keep.append(i)
                    continue

                prev = torch.tensor(keep, device=iou.device, dtype=torch.long)
                ov = iou[i, prev] > iou_thr
                if torch.any(ov):
                    if allow_replace:
                        prev_ov = prev[ov]
                        ov_iou = iou[i, prev_ov]
                        j = int(prev_ov[int(torch.argmax(ov_iou).item())].item())
                        if float(prefer_vals[i].item()) > float(prefer_vals[j].item()):
                            keep[keep.index(j)] = i
                    continue

                if max_num is None or len(keep) < max_num:
                    keep.append(i)
                    continue

                if not refill:
                    break

            keep_t = torch.tensor(keep, device=order.device, dtype=torch.long)
            keep_inds = order[keep_t]
            scores = scores[keep_inds]
            labels = labels[keep_inds]
            mask_pred = mask_pred[keep_inds]
            queries = queries[keep_inds]
            mapping = mapping[keep_inds]
            select_scores = select_scores[keep_inds]

        return mask_pred, labels, scores, queries, mapping, select_scores
    
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

        # 提取并规范图像特征（若可用），否则记录回退
        img_feats: Optional[List[torch.Tensor]] = None
        if getattr(self, 'img_backbone', None) is not None:
            with torch.no_grad():
                raw_feats = None
                if 'img' in batch_inputs_dict:
                    raw_feats = self.img_backbone(batch_inputs_dict['img'])  # type: ignore[operator]
                elif 'imgs' in batch_inputs_dict:
                    raw_feats = self.img_backbone(batch_inputs_dict['imgs'])  # type: ignore[operator]
                elif 'img_paths' in batch_inputs_dict:
                    raw_feats = self.img_backbone(batch_inputs_dict['img_paths'])  # type: ignore[operator]

            if raw_feats is not None:
                if isinstance(raw_feats, list):
                    img_feats = [f for f in raw_feats]
                elif torch.is_tensor(raw_feats):
                    if raw_feats.dim() == 4 and raw_feats.shape[0] == batch_size:
                        img_feats = [raw_feats[i] for i in range(raw_feats.shape[0])]
                    elif raw_feats.dim() == 3:
                        img_feats = [raw_feats for _ in range(batch_size)]
                if img_feats is not None and len(img_feats) != batch_size:
                    img_feats = None
        if getattr(self, 'img_backbone', None) is not None and img_feats is None:
            print("[MixFormer3D_FF_Online] img_backbone enabled but no valid 2D features; fallback to pure 3D.")

        # forward of backbone and neck
        if img_feats is not None:
            x = self.backbone(field.sparse(),
                              partial(self._f, img_features=img_feats, img_metas=img_metas),
                              memory=self.memory if hasattr(self,'memory') else None)
        else:
            x = self.backbone(field.sparse(),
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
        sp_xyz = scatter_mean(torch.cat(all_xyz, dim=0), sp_idx, dim=0)
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
            sp_xyz_list.append(sp_xyz[begin: end])
        return features, point_features, all_xyz_w, sp_xyz_list

    def _f(self, x, img_features, img_metas):
        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size
        projected_features = []
        for point, img_feature, img_meta in zip(points, img_features, img_metas):
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
            if img_feature.dim() == 4:
                img_feature = img_feature[0]
            if not torch.is_tensor(img_feature):
                img_feature = torch.as_tensor(img_feature, device=point.device)
            img_pad_shape = img_feature.shape[-2:]
            projected_features.append(point_sample(
                img_meta=img_meta,
                img_features=img_feature.unsqueeze(0),
                points=point,
                proj_mat=point.new_tensor(proj_mat),
                coord_type=coord_type,
                img_scale_factor=img_scale_factor,
                img_crop_offset=img_crop_offset,
                img_flip=img_flip,
                img_pad_shape=img_pad_shape,
                img_shape=img_pad_shape,
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
        sp_xyz = scatter_mean(torch.cat(all_xyz, dim=0), sp_idx, dim=0)
        for i in range(len(n_super_points)):
            begin = sum(n_super_points[:i])
            end = sum(n_super_points[:i + 1])
            features.append(x[begin: end])
            sp_xyz_list.append(sp_xyz[begin: end])
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
