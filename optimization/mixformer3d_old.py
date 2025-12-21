import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean, scatter
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
from .instance_merge import ins_merge_mat, ins_cat, ins_merge, OnlineMerge, GTMerge
import numpy as np
from .img_backbone import point_sample
from .projection_utils import project_points_to_uv, splat_to_grid, SCANET_INTRINSICS
import os
from typing import Any, Dict, List, Tuple, Union, Optional, cast
from mmengine import ConfigDict


@MODELS.register_module(name='ScanNet200MixFormer3D_Old')
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
                 img_backbone=None,
                 backbone=None,
                 bi_encoder=None,
                 clip_criterion=None,
                 alpha_regularizers=None,  # NEW: Alpha regularization
                 training_optimization=None,  # NEW: Training optimization config
                 neck=None,
                 pool=None,
                 decoder=None,
                 criterion=None,
                 two_d_losses=None,
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
        self.bi_encoder = MODELS.build(_cfg(bi_encoder, 'bi_encoder')) if bi_encoder is not None else None
        self.neck = MODELS.build(_cfg(neck, 'neck')) if neck is not None else None
        self.pool = MODELS.build(_cfg(pool, 'pool'))
        decoder_cfg = _cfg(decoder, 'decoder')
        self.decoder = MODELS.build(decoder_cfg)
        self.criterion = MODELS.build(_cfg(criterion, 'criterion'))
        self.clip_criterion = MODELS.build(_cfg(clip_criterion, 'clip_criterion')) if clip_criterion is not None else None
        
        # Alpha regularizers (optional)
        self.alpha_regularizers = {}
        if alpha_regularizers is not None:
            for reg_name, reg_config in alpha_regularizers.items():
                if reg_config is not None:
                    self.alpha_regularizers[reg_name] = MODELS.build(_cfg(reg_config, f'alpha_{reg_name}'))
                else:
                    self.alpha_regularizers[reg_name] = None
        
        # Training optimization components (disabled in lightweight build)
        self.training_optimization = training_optimization
        self.progress_scheduler = None
        self.progressive_freeze_manager = None
        self.enhanced_loss_recorder = None
        
        self.test_cfg: Any = ConfigDict(test_cfg or {})
        self.voxel_size = voxel_size
        self.num_classes = num_classes
        self.query_thr = query_thr
        self.train_cfg = train_cfg

        # 2D supervision placeholders
        self._two_d_hidden_dim = decoder_cfg.get('in_channels', 128)
        self._num_semantic_classes_2d = decoder_cfg.get('num_semantic_classes', 0)
        self.seg2d_head: Optional[nn.Module] = None
        self.recon2d_head: Optional[nn.Module] = None
        self._clip_channels: Optional[int] = None
        default_two_d_cfg = {
            'enable_recon': False,
            'enable_seg': False,
            'w_recon': 0.0,
            'w_seg': 0.0,
            'w_align': 0.0,
            'recon_tau': 1.0,
            'seg_conf': 1.0,
            'depth_tol': 0.05,
            'grid_hw': (60, 80),
            'alpha_max': 1.0,
            'alpha_warmup': 0,
            'recon_warmup': 0,
            'seg_warmup': 0,
            'align_warmup': 0,
        }
        if two_d_losses is not None:
            default_two_d_cfg.update(two_d_losses)
        self.two_d_loss_cfg = default_two_d_cfg
        self._loss_call_counter = 0
        self._loss_call_counter = 0

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

    def _ensure_2d_heads(self, clip_channels: Optional[int], device: torch.device) -> None:
        hidden = self._two_d_hidden_dim

        if self.two_d_loss_cfg.get('enable_seg', False) and self._num_semantic_classes_2d > 0:
            if self.seg2d_head is None:
                self.seg2d_head = nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=1),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, self._num_semantic_classes_2d + 1, kernel_size=1)
                ).to(device)

        if self.two_d_loss_cfg.get('enable_recon', False):
            if clip_channels is None:
                return
            if self.recon2d_head is None or self._clip_channels != clip_channels:
                self._clip_channels = clip_channels
                self.recon2d_head = nn.Sequential(
                    nn.Conv2d(hidden, hidden, kernel_size=1),
                    nn.GroupNorm(8, hidden),
                    nn.GELU(),
                    nn.Conv2d(hidden, clip_channels, kernel_size=1)
                ).to(device)

    def extract_feat(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any]) -> Tuple[List[Any], List[torch.Tensor], Any]:  # type: ignore[override]
        """Extract features from sparse tensor."""
        if self.bi_encoder is not None and 'imgs' in batch_inputs_dict:
            # === BiFusion path ===
            # 使 ElasticTransfrom 在 BiFusion 路径下也生效：
            # 若存在 'elastic_coords'，将其替换到 points 的 xyz（世界坐标系）后再送入 bi_encoder。
            points_list = batch_inputs_dict['points']
            if 'elastic_coords' in batch_inputs_dict and isinstance(points_list, list):
                replaced_points = []
                elastic_list = batch_inputs_dict['elastic_coords']
                for i in range(len(points_list)):
                    pts = points_list[i]
                    ec = elastic_list[i]
                    if not torch.is_tensor(ec):
                        ec = torch.as_tensor(ec, dtype=pts.dtype, device=pts.device)
                    else:
                        ec = ec.to(device=pts.device, dtype=pts.dtype)
                    # ElasticTransfrom 产生的是体素坐标，需乘以 voxel_size 还原到世界坐标
                    xyz_world = ec * self.voxel_size
                    # 保留原有颜色等特征通道
                    feats_rest = pts[:, 3:]
                    pts_new = torch.cat([xyz_world, feats_rest], dim=1)
                    replaced_points.append(pts_new)
                points_for_encoder = replaced_points
            else:
                points_for_encoder = points_list

            encoder_out = self.bi_encoder(
                points_for_encoder,
                batch_inputs_dict['imgs'],
                batch_inputs_dict['cam_info']
            )
            
            # Add clip_global from batch_inputs to encoder_out for loss computation
            if 'clip_global' in batch_inputs_dict:
                encoder_out['clip_global'] = batch_inputs_dict['clip_global']
            
            self._encoder_out = encoder_out  # cache for loss
            fused_list = encoder_out['feat_fusion']
            # 与送入 bi_encoder 的坐标保持一致（若替换为 elastic 后，这里也同步使用）
            all_xyz = [pts[:, :3] for pts in points_for_encoder]

            # 基本长度检查（bi_encoder已经有详细验证）
            if len(fused_list) != len(batch_data_samples):
                raise RuntimeError(f"Fused features length {len(fused_list)} != batch samples {len(batch_data_samples)}")

            # concatenate all fused features
            x = torch.cat(fused_list, dim=0)

            # superpoint pooling
            sp_pts_masks, n_super_points = [], []
            for i, data_sample in enumerate(batch_data_samples):
                sp_pts_mask = data_sample.gt_pts_seg.sp_pts_mask
                sp_pts_masks.append(sp_pts_mask + sum(n_super_points))
                n_super_points.append(sp_pts_mask.max() + 1)
            sp_idx = torch.cat(sp_pts_masks)

            x, all_xyz_w = self.pool(x, sp_idx, all_xyz, with_xyz=False)

            # split per sample features
            features = []
            for i in range(len(n_super_points)):
                begin = sum(n_super_points[:i])
                end = sum(n_super_points[:i + 1])
                features.append(x[begin: end])

            point_features = [torch.cat([c, f], dim=-1) for c, f in zip(all_xyz, fused_list)]

            return features, point_features, all_xyz_w

        # === Original path ===
        # 如果提供了图像骨干网络，则先提取图像特征（仅占位，避免 OptionalCall 报错）
        with torch.no_grad():
            if self.img_backbone is not None and 'img_path' in batch_inputs_dict:
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
        x = self.backbone(field.sparse())
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

    def _forward(self, *args, **kwargs) -> Any:  # type: ignore[override]
        """Implement abstract method of Base3DDetector."""
        pass

    def loss(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any], **kwargs) -> Dict[str, torch.Tensor]:  # type: ignore[override]
        """Calculate losses from a batch of inputs dict and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                `points` key.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It includes information such as
                `gt_instances_3d` and `gt_sem_seg_3d`.
        Returns:
            dict: A dictionary of loss components.
        """
        ## Backbone
        x, point_features, all_xyz_w = self.extract_feat(batch_inputs_dict, batch_data_samples)
        ## GT-prepare
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
        ## Decoder
        super_points = ([bds.gt_pts_seg.sp_pts_mask for bds in batch_data_samples], all_xyz_w)
        x = self.decoder(x=x, queries=queries, sp_feats=x, p_feats=point_features, super_points=super_points)
        ## Loss computation with v1 optimizations
        losses = self.criterion(x, gt_instances, gt_point_instances, None, self.decoder.mask_pred_mode)
        
        self._loss_call_counter = getattr(self, '_loss_call_counter', 0) + 1
        current_step = self._loss_call_counter

        # Enhanced CLIP consistency loss with progress-aware scheduling
        clip_loss_value = 0.0
        if self.clip_criterion is not None and hasattr(self, '_encoder_out'):
            # Update step counter for CLIP criterion
            self.clip_criterion.step()
            
            # Update training progress if scheduler is available
            if self.progress_scheduler is not None:
                self.progress_scheduler.step()
            
            # Use valid projection mask for point-level consistency
            valid_mask = None
            if (self.bi_encoder is not None and 
                hasattr(self.bi_encoder, '_debug_stats') and 
                self.bi_encoder._debug_stats):
                # Extract valid projection masks from debug stats if available
                valid_mask = [torch.tensor(stats.get('valid_points', []), dtype=torch.bool, device=self._encoder_out['feat_fusion'][0].device) 
                             for stats in self.bi_encoder._debug_stats]
            
            loss_clip = self.clip_criterion(
                self._encoder_out['feat_fusion'], 
                self._encoder_out['clip_global'],
                valid_projection_mask=valid_mask
            )
            losses.update(dict(loss_clip=loss_clip))
            clip_loss_value = float(loss_clip) if isinstance(loss_clip, torch.Tensor) else loss_clip

        two_d_cfg = getattr(self, 'two_d_loss_cfg', None)
        if two_d_cfg is not None and hasattr(self, 'bi_encoder') and self.bi_encoder is not None:
            base_device = next(self.parameters()).device

            def _effective_weight(name: str, warm_name: str) -> float:
                base = float(two_d_cfg.get(name, 0.0))
                warm = max(int(two_d_cfg.get(warm_name, 0)), 0)
                if warm > 0:
                    return base * min(1.0, current_step / warm)
                return base

            w_recon_eff = _effective_weight('w_recon', 'recon_warmup')
            w_seg_eff = _effective_weight('w_seg', 'seg_warmup')
            w_align_eff = _effective_weight('w_align', 'align_warmup')
            recon_enabled_cfg = bool(two_d_cfg.get('enable_recon', False)) and w_recon_eff > 0
            seg_enabled_cfg = bool(two_d_cfg.get('enable_seg', False)) and w_seg_eff > 0 and self._num_semantic_classes_2d > 0
            align_enabled_cfg = w_align_eff > 0

            alpha_max = float(two_d_cfg.get('alpha_max', 1.0))
            alpha_warmup = max(int(two_d_cfg.get('alpha_warmup', 0)), 0)
            if alpha_warmup > 0:
                alpha_factor = min(1.0, current_step / alpha_warmup)
            else:
                alpha_factor = 1.0
            alpha_value = alpha_max * alpha_factor if (recon_enabled_cfg or seg_enabled_cfg or align_enabled_cfg) else 0.0
            if hasattr(self.bi_encoder, 'set_alpha_2d'):
                self.bi_encoder.set_alpha_2d(alpha_value)
            losses['alpha_2d'] = torch.tensor(alpha_value, dtype=torch.float32, device=base_device)

            if hasattr(self, '_encoder_out') and (recon_enabled_cfg or seg_enabled_cfg or align_enabled_cfg):
                encoder_out = getattr(self, '_encoder_out', {})
                feat_list = encoder_out.get('feat_fusion', [])
                proj3d_list = encoder_out.get('proj_3d_points', [])
                proj2d_list = encoder_out.get('proj_2d_points', [])
                valid_masks = encoder_out.get('valid_projection_mask', [])
                points_list = batch_inputs_dict.get('points', [])
                cam_infos = batch_inputs_dict.get('cam_info', [])
                num_samples = min(
                    len(feat_list),
                    len(proj3d_list),
                    len(proj2d_list),
                    len(valid_masks),
                    len(points_list),
                    len(batch_data_samples)
                )
                if num_samples > 0:
                    if not cam_infos:
                        cam_infos = [{} for _ in range(num_samples)]
                    elif len(cam_infos) == 1 and num_samples > 1:
                        cam_infos = [cam_infos[0] for _ in range(num_samples)]
                    elif len(cam_infos) > num_samples:
                        cam_infos = cam_infos[:num_samples]

                    device = feat_list[0].device
                    H, W = two_d_cfg.get('grid_hw', (60, 80))
                    recon_tau = float(two_d_cfg.get('recon_tau', 1.0))
                    seg_conf = float(two_d_cfg.get('seg_conf', 1.0))

                    recon_cos_total = torch.zeros((), device=device)
                    recon_mse_total = torch.zeros((), device=device)
                    recon_total = torch.zeros((), device=device)
                    recon_count = 0
                    cover_mean_total = torch.zeros((), device=device)
                    cover_count = 0

                    seg_total = torch.zeros((), device=device)
                    seg_count = 0
                    supervised_pixel_ratio = torch.zeros((), device=device)

                    align_total = torch.zeros((), device=device)
                    align_count = 0

                    for idx in range(num_samples):
                        feat_pts = feat_list[idx]
                        proj3d_pts = proj3d_list[idx]
                        proj2d_pts = proj2d_list[idx]
                        valid_mask = valid_masks[idx].to(feat_pts.device)
                        pts = points_list[idx].to(feat_pts.device)
                        xyz = pts[:, :3]
                        cam_meta = cam_infos[idx] if idx < len(cam_infos) else {}

                        pose = None
                        if hasattr(self.bi_encoder, '_extract_pose_matrix'):
                            pose = self.bi_encoder._extract_pose_matrix(cam_meta, sample_idx=0)
                        if pose is None:
                            continue

                        xyz_cam = self.bi_encoder._transform_coordinates(xyz, pose)
                        if xyz_cam is None:
                            continue

                        uv, proj_valid = project_points_to_uv(
                            xyz_cam,
                            (H, W),
                            max_depth=getattr(self.bi_encoder, 'max_depth', 20.0),
                            standard_intrinsics=getattr(self.bi_encoder, 'standard_scannet_intrinsics', SCANET_INTRINSICS)
                        )
                        proj_valid = proj_valid.to(feat_pts.device)
                        combined_valid = proj_valid & valid_mask
                        if not combined_valid.any():
                            continue

                        F2D, cover = splat_to_grid(
                            uv=uv,
                            z=xyz_cam[:, 2],
                            feats=feat_pts,
                            valid=combined_valid,
                            H=H,
                            W=W,
                            mode='bilinear'
                        )
                        cover_mask = cover.squeeze(0) >= recon_tau

                        clip_channels = None
                        clip_pix = cam_meta.get('clip_pix') if isinstance(cam_meta, dict) else None
                        if isinstance(clip_pix, (list, tuple)):
                            clip_candidates = [c for c in clip_pix if c is not None]
                            clip_pix = clip_candidates[0] if clip_candidates else None
                        clip_tensor = None
                        if isinstance(clip_pix, torch.Tensor):
                            clip_tensor = clip_pix.to(device=feat_pts.device, dtype=F2D.dtype)
                        elif isinstance(clip_pix, np.ndarray):
                            clip_tensor = torch.from_numpy(clip_pix).to(device=feat_pts.device, dtype=F2D.dtype)
                        if clip_tensor is not None:
                            clip_channels = clip_tensor.shape[0]

                        if recon_enabled_cfg and clip_tensor is not None and clip_tensor.shape[-2:] == (H, W):
                            self._ensure_2d_heads(clip_channels, feat_pts.device)
                            if self.recon2d_head is not None and cover_mask.any():
                                pred = self.recon2d_head(F2D.unsqueeze(0))[0]
                                mask_flat = cover_mask.view(-1)
                                pred_flat = pred.view(pred.shape[0], -1)[:, mask_flat]
                                target_flat = clip_tensor.view(clip_tensor.shape[0], -1)[:, mask_flat]
                                if pred_flat.shape[1] > 0:
                                    pred_norm = F.normalize(pred_flat.transpose(0, 1), dim=1)
                                    target_norm = F.normalize(target_flat.transpose(0, 1), dim=1)
                                    cos_loss = (1.0 - (pred_norm * target_norm).sum(dim=1).clamp(-1.0, 1.0)).mean()
                                    mse_loss = F.mse_loss(pred_flat, target_flat)
                                    combined_loss = 0.7 * cos_loss + 0.3 * mse_loss
                                    recon_cos_total = recon_cos_total + cos_loss
                                    recon_mse_total = recon_mse_total + mse_loss
                                    recon_total = recon_total + combined_loss
                                    recon_count += 1
                                    cover_mean_total = cover_mean_total + cover_mask.float().mean()
                                    cover_count += 1

                        if seg_enabled_cfg:
                            self._ensure_2d_heads(None, feat_pts.device)
                            if self.seg2d_head is not None:
                                labels = batch_data_samples[idx].gt_pts_seg.pts_semantic_mask.to(feat_pts.device)
                                ignore_index = self._num_semantic_classes_2d
                                label_valid = combined_valid & (labels != ignore_index)
                                if label_valid.any():
                                    rows = torch.round(uv[:, 1]).clamp(0, H - 1).long()
                                    cols = torch.round(uv[:, 0]).clamp(0, W - 1).long()
                                    flat = rows * W + cols
                                    flat_valid = flat[label_valid]
                                    labels_valid = labels[label_valid]
                                    depths_valid = xyz_cam[:, 2][label_valid]
                                    order = torch.argsort(depths_valid)
                                    flat_sorted = flat_valid[order]
                                    label_sorted = labels_valid[order]
                                    keep_mask = torch.ones_like(flat_sorted, dtype=torch.bool)
                                    keep_mask[1:] = flat_sorted[1:] != flat_sorted[:-1]
                                    chosen_flat = flat_sorted[keep_mask]
                                    chosen_label = label_sorted[keep_mask]
                                    pseudo_label = torch.full((H * W,), ignore_index, device=feat_pts.device, dtype=torch.long)
                                    pseudo_label[chosen_flat] = chosen_label
                                    counts = torch.zeros(H * W, device=feat_pts.device)
                                    counts.index_add_(0, flat_valid, torch.ones_like(flat_valid, dtype=torch.float32))
                                seg_mask = counts >= seg_conf
                                seg_mask = seg_mask.view(H, W)
                                seg_mask = seg_mask & cover_mask
                                if seg_mask.any():
                                    logits_2d = self.seg2d_head(F2D.unsqueeze(0))[0]
                                    ce_map = F.cross_entropy(
                                        logits_2d.unsqueeze(0),
                                        pseudo_label.view(1, H, W),
                                        ignore_index=ignore_index,
                                        reduction='none'
                                    )[0]
                                    seg_loss = ce_map[seg_mask].mean()
                                    seg_total = seg_total + seg_loss
                                    seg_count += 1
                                    supervised_pixel_ratio = supervised_pixel_ratio + seg_mask.float().mean()

                    if align_enabled_cfg:
                        mask = valid_mask & torch.isfinite(proj3d_pts.sum(dim=1)) & torch.isfinite(proj2d_pts.sum(dim=1))
                        if mask.any():
                            proj3d_sel = F.normalize(proj3d_pts[mask], dim=1)
                            proj2d_sel = F.normalize(proj2d_pts[mask], dim=1)
                            cos_sim = (proj3d_sel * proj2d_sel).sum(dim=1).clamp(-1.0, 1.0)
                            align_loss = (1.0 - cos_sim).mean()
                            align_total = align_total + align_loss
                            align_count += 1

                if recon_enabled_cfg:
                    if recon_count > 0:
                        avg_cos = recon_cos_total / recon_count
                        avg_mse = recon_mse_total / recon_count
                        avg_recon = recon_total / recon_count
                        losses['loss_2d_recon_cos'] = avg_cos
                        losses['loss_2d_recon_mse'] = avg_mse
                        losses['loss_2d_recon'] = avg_recon * w_recon_eff
                        if cover_count > 0:
                            losses['cover_mean'] = cover_mean_total / cover_count
                    else:
                        zero = torch.zeros((), device=device)
                        losses.setdefault('loss_2d_recon_cos', zero)
                        losses.setdefault('loss_2d_recon_mse', zero)
                        losses.setdefault('loss_2d_recon', zero)
                        losses.setdefault('cover_mean', zero)

                if seg_enabled_cfg:
                    if seg_count > 0:
                        avg_seg = seg_total / seg_count
                        losses['loss_2d_seg_ce'] = avg_seg * w_seg_eff
                        losses['loss_2d_seg_ce_raw'] = avg_seg
                        losses['supervised_pixel_ratio'] = supervised_pixel_ratio / seg_count
                    else:
                        zero = torch.zeros((), device=device)
                        losses.setdefault('loss_2d_seg_ce', zero)
                        losses.setdefault('loss_2d_seg_ce_raw', zero)
                        losses.setdefault('supervised_pixel_ratio', zero)

                if align_enabled_cfg:
                    if align_count > 0:
                        avg_align = align_total / align_count
                        losses['loss_align_raw'] = avg_align
                        losses['loss_align'] = avg_align * w_align_eff
                    else:
                        zero = torch.zeros((), device=device)
                        losses.setdefault('loss_align', zero)
                        losses.setdefault('loss_align_raw', zero)
            else:
                base_device = next(self.parameters()).device
                zero = torch.zeros((), device=base_device)
                if two_d_cfg.get('enable_recon', False):
                    losses.setdefault('loss_2d_recon', zero)
                    losses.setdefault('loss_2d_recon_cos', zero)
                    losses.setdefault('loss_2d_recon_mse', zero)
                    losses.setdefault('cover_mean', zero)
                if two_d_cfg.get('enable_seg', False):
                    losses.setdefault('loss_2d_seg_ce', zero)
                    losses.setdefault('loss_2d_seg_ce_raw', zero)
                    losses.setdefault('supervised_pixel_ratio', zero)
                if two_d_cfg.get('w_align', 0.0) > 0:
                    losses.setdefault('loss_align', zero)
                    losses.setdefault('loss_align_raw', zero)
        else:
            if hasattr(self.bi_encoder, 'set_alpha_2d'):
                self.bi_encoder.set_alpha_2d(0.0)
        
        # Alpha regularization losses (if enabled)
        if hasattr(self, 'alpha_regularizers') and self.alpha_regularizers:
            if hasattr(self, '_encoder_out') and 'alpha_values' in self._encoder_out:
                alpha_values = self._encoder_out['alpha_values']
                
                for reg_name, regularizer in self.alpha_regularizers.items():
                    if regularizer is not None:
                        regularizer.step()  # Update step counter
                        reg_loss = regularizer(alpha_values)
                        losses.update({f'alpha_{reg_name}': reg_loss})
        
        # Ensure all losses are in FP32 for numerical stability
        for key, value in losses.items():
            if isinstance(value, torch.Tensor) and value.dtype != torch.float32:
                losses[key] = value.float()
        
        # 收集融合统计信息用于日志输出（enhanced with more metrics）
        if self.bi_encoder is not None and hasattr(self.bi_encoder, '_fusion_stats') and self.bi_encoder._fusion_stats:
            fusion_stats = self.bi_encoder._fusion_stats

            # Core fusion statistics
            stat_keys = [
                'avg_confidence',
                'valid_ratio',
                'norm_ratio_2d_over_3d',
                'cos_2d3d_mean',
                'cos_2d3d_mean_ln',
                'norm_2d_mean',
                'norm_3d_mean',
                'feat3d_mean_abs',
                'feat3d_std',
                'feat3d_nonzero_ratio',
                'feat2d_mean_abs',
                'feat2d_std',
                'feat2d_nonzero_ratio',
                'fused_mean_abs',
                'fused_std',
                'grad_norm_feat3d',
                'grad_norm_feat2d',
                'grad_norm_feat3d_raw',
                'grad_norm_feat2d_raw',
                'grad_norm_fusion_raw',
                'grad_params_feat2d',
                'grad_params_feat3d',
                'grad_params_fusion',
                'grad_params_decoder',
                'grad_ratio_2d_over_3d'
            ]
            tensor_stats = {
                key: torch.tensor(fusion_stats[key], dtype=torch.float32)
                for key in stat_keys if key in fusion_stats
            }
            losses.update(tensor_stats)
        
        # Enhanced loss recording and anomaly detection
        return losses

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
        self.init_weights()
    
    def init_weights(self):
        if hasattr(self, 'memory') and self.memory is not None:
            self.memory.init_weights()
        if hasattr(self, 'img_backbone') and self.img_backbone is not None:
            self.img_backbone.init_weights()

    def extract_feat(self, batch_inputs_dict: Dict[str, Any], batch_data_samples: List[Any], frame_i: int) -> Tuple[List[Any], List[torch.Tensor], Any, List[Any]]:  # type: ignore[override]
        """Extract features from sparse tensor.
        """
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
        x = self.backbone(field.sparse(), memory=self.memory if hasattr(self,'memory') else None)
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
        # 收集融合统计信息用于日志输出
        if self.bi_encoder is not None and hasattr(self.bi_encoder, '_fusion_stats'):
            fusion_stats = self.bi_encoder._fusion_stats
            monitor_keys = [
                'avg_confidence',
                'valid_ratio',
                'norm_ratio_2d_over_3d',
                'cos_2d3d_mean',
                'cos_2d3d_mean_ln',
                'feat3d_mean_abs',
                'feat3d_std',
                'feat3d_nonzero_ratio',
                'feat2d_mean_abs',
                'feat2d_std',
                'feat2d_nonzero_ratio',
                'fused_mean_abs',
                'fused_std',
                'grad_norm_feat3d',
                'grad_norm_feat2d'
            ]
            monitor_tensors = {
                key: torch.tensor(fusion_stats[key])
                for key in monitor_keys if key in fusion_stats
            }
            losses.update(monitor_tensors)

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
