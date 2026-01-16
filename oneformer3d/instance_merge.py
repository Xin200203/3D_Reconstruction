import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
from mmdet3d.structures import AxisAlignedBboxOverlaps3D
import pdb
from sklearn.cluster import AgglomerativeClustering
import networkx as nx
from mmdet3d.registry import MODELS
from .time_divided_transformer import TimeDividedTransformer
from .track_manager import TrackManager

# This function is deprecated by OnlineMerge. No update anymore.
def ins_merge_mat(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, inscat_topk_insts):
    """Merge multiview instances according to geometry and query feature
    """
    weights = [0.4,0.4,0.2]
    threshold = 0.75
    frame_num = len(masks)
    points_per_mask = masks[0].shape[1]
    cur_masks, cur_labels, cur_scores, cur_queries, cur_query_feats, cur_sem_preds, cur_xyz = \
        masks[0], labels[0], scores[0], queries[0], query_feats[0], sem_preds[0], xyz_list[0]
    for i in range(1, frame_num):
        next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
            masks[i], labels[i], scores[i], queries[i], query_feats[i], sem_preds[i], xyz_list[i]
        query_feat_scores = (cur_query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
        sem_pred_scores = F.cosine_similarity(cur_sem_preds.unsqueeze(1), next_sem_preds.unsqueeze(0), dim=2)
        xyz_dists = torch.cdist(cur_xyz, next_xyz, p=2)
        xyz_scores = 1 / (xyz_dists + 1e-6)
        
        mix_scores = weights[0] * query_feat_scores + weights[1] * sem_pred_scores + weights[2] * xyz_scores
        mix_scores = torch.where(mix_scores > threshold, mix_scores, torch.zeros_like(mix_scores))
        if mix_scores.shape[0] < mix_scores.shape[1]:
            mix_scores = torch.cat((mix_scores, torch.zeros((mix_scores.shape[1]
                    - mix_scores.shape[0], mix_scores.shape[1])).to(mix_scores.device)), dim=0)
        # Hungarian assign
        row_ind, col_ind = linear_sum_assignment(-mix_scores.cpu())
        row_ind = torch.tensor(row_ind).to(mix_scores.device)
        col_ind = torch.tensor(col_ind).to(mix_scores.device)
        mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
        row_ind = row_ind[mix_scores_mask]
        col_ind = col_ind[mix_scores_mask]

        temp = torch.zeros(cur_masks.shape[0]).bool().to(cur_masks.device)
        temp[row_ind] = True
        temp = temp.unsqueeze(1)
        temp_masks = torch.zeros((cur_masks.shape[0], points_per_mask)).bool().to(cur_masks.device)
        temp_masks[row_ind] = next_masks[col_ind]
        next_masks_ = torch.where(temp, temp_masks,
                                    torch.zeros((cur_masks.shape[0],points_per_mask)).bool().to(next_masks.device))
        cur_masks = torch.cat((cur_masks, next_masks_), dim=1)
        no_merge_masks = torch.tensor(np.setdiff1d(np.arange(next_masks.shape[0]),
                col_ind.cpu())).to(next_masks.device)
        former_padding = torch.zeros((no_merge_masks.shape[0], points_per_mask * i)).bool().to(next_masks.device)
        new_masks = torch.cat((former_padding, next_masks[no_merge_masks]), dim=1)
        cur_masks = torch.cat((cur_masks, new_masks), dim=0)
        
        cur_scores[row_ind] = (cur_scores[row_ind] * i + next_scores[col_ind]) / (i + 1)
        cur_scores = torch.cat((cur_scores, next_scores[no_merge_masks]), dim=0)
        cur_queries[row_ind] = (cur_queries[row_ind] * i + next_queries[col_ind]) / (i + 1)
        cur_queries = torch.cat((cur_queries, next_queries[no_merge_masks]), dim=0)
        cur_query_feats[row_ind] = (cur_query_feats[row_ind] * i + next_query_feats[col_ind]) / (i + 1)
        cur_query_feats = torch.cat((cur_query_feats, next_query_feats[no_merge_masks]), dim=0)
        cur_sem_preds[row_ind] = (cur_sem_preds[row_ind] * i + next_sem_preds[col_ind]) / (i + 1)
        cur_sem_preds = torch.cat((cur_sem_preds, next_sem_preds[no_merge_masks]), dim=0)
        cur_xyz[row_ind] = (cur_xyz[row_ind] * i + next_xyz[col_ind]) / (i + 1)
        cur_xyz = torch.cat((cur_xyz, next_xyz[no_merge_masks]), dim=0)
    
    if len(cur_scores) > inscat_topk_insts:
        _, kept_ins = cur_scores.topk(inscat_topk_insts)
    else:
        kept_ins = torch.arange(cur_scores.shape[0], device=cur_scores.device)
    cur_masks, cur_scores = cur_masks[kept_ins], cur_scores[kept_ins]
    cur_labels = torch.zeros_like(cur_scores).long()
    return cur_masks, cur_labels, cur_scores
       
def ins_cat(masks, labels, scores, inscat_topk_insts):
    """Directly stack multiview instances without mask merging"""
    frame_num = len(masks)
    labels = torch.cat(labels)
    scores = torch.cat(scores)
    if len(scores) > inscat_topk_insts:
        _, kept_ins = scores.topk(inscat_topk_insts)
    else:
        kept_ins = torch.arange(scores.shape[0], device=scores.device)
    labels, scores = labels[kept_ins], scores[kept_ins]
    ins_num = [mask.shape[0] for mask in masks]
    frame_indicator = torch.cat([torch.ones(num)*i for i, num in enumerate(ins_num)])
    frame_indicator = frame_indicator.to(scores.device)[kept_ins]
    masks = torch.cat(masks, dim=0)[kept_ins]
    new_mask = masks.new_zeros(size=(masks.shape[0], frame_num*masks.shape[1]))
    for ids in range(len(ins_num)):
        this_frame = (frame_indicator == ids)
        new_mask[this_frame, ids*masks.shape[1]:(ids+1)*masks.shape[1]] = masks[this_frame]
    return new_mask, labels, scores

def ins_merge(points, masks, labels, scores, queries, inscat_topk_insts):
    """Merge multiview instances according to geometry and query feature"""
    frame_num = len(points)
    pts_per_frame = points[0].shape[0]
    cur_instances = [InstanceQuery(mask, label, score, query) for mask, label, score, query \
            in zip(masks[0], labels[0], scores[0], queries[0])]
    cur_points = points[0]
    for i in range(1, frame_num):
        for mask, label, score, query in zip(masks[i], labels[i], scores[i], queries[i]):
            is_merge = False
            for InsQ in cur_instances:
                # merged ins
                if InsQ.compare(cur_points, points[i], mask, label, score, query):
                    InsQ.merge(mask, label, score, query, i)
                    is_merge = True
                    break
            # new ins
            if not is_merge:
                mask = torch.cat([mask.new_zeros(pts_per_frame*i).bool(), mask])
                cur_instances.append(InstanceQuery(mask, label, score, query))
        cur_points = torch.cat([cur_points, points[i]])
        # not merged ins
        for InsQ in cur_instances:
            if len(InsQ.mask) < cur_points.shape[0]:
                InsQ.pad(pts_per_frame)
    merged_mask = torch.stack([InsQ.mask for InsQ in cur_instances], dim=0)
    merged_labels = torch.tensor([InsQ.label for InsQ in cur_instances]).to(merged_mask.device)
    merged_scores = torch.tensor([InsQ.score for InsQ in cur_instances]).to(merged_mask.device)
    if len(merged_scores) > inscat_topk_insts:
        _, kept_ins = merged_scores.topk(inscat_topk_insts)
    else:
        kept_ins = torch.arange(merged_scores.shape[0], device=merged_scores.device)
    merged_mask, merged_labels, merged_scores = \
        merged_mask[kept_ins], merged_labels[kept_ins], merged_scores[kept_ins]
    return merged_mask, merged_labels, merged_scores

class GTMerge():
    def __init__(self):
        self.cur_queries = None
        self.fi = 0
        self.merge_counts = None
    
    def clean(self):
        self.cur_queries = None
        self.merge_counts = None
    
    # weighted sum according to count of merge, rather than frame
    def merge(self, queries, cls_preds, query_ins_masks):
        batch_size = len(queries)
        ins_query_list = []
        merge_count_list = []
        # Intra-frame merge: choose one with max score
        for i in range(batch_size):
            n_instances = len(query_ins_masks[i])
            if n_instances == 0:
                return None
            ins_query = []
            merge_count = []
            for j in range(n_instances):
                temp_idx = query_ins_masks[i][j]
                # ins_query.append(queries[i][temp_idx].mean(0) if
                #      len(temp_idx) != 0 else torch.zeros_like(queries[i][0]))
                # merge_count.append(len(temp_idx))
                fg_scores = cls_preds[i][temp_idx].softmax(-1)[:,:-1].sum(-1)
                ins_query.append(queries[i][temp_idx][fg_scores.argmax()] if
                     len(temp_idx) != 0 else torch.zeros_like(queries[i][0]))
                merge_count.append(1 if len(temp_idx) != 0 else 0)
            ins_query_list.append(torch.stack(ins_query, dim=0))
            merge_count_list.append(torch.tensor(merge_count, device=temp_idx.device).unsqueeze(-1))
        if self.cur_queries is None:
            self.cur_queries = ins_query_list
            self.merge_counts = merge_count_list
        else:
            # Static typing：确保非 None
            assert self.cur_queries is not None and self.merge_counts is not None
            # Inter-frame merge: mean across frame
            for i in range(batch_size):
                self.cur_queries[i] = (self.cur_queries[i] * self.merge_counts[i] + ins_query_list[i]
                     * merge_count_list[i]) / (self.merge_counts[i] + merge_count_list[i] + 1e-6)
                self.merge_counts[i] = self.merge_counts[i] + merge_count_list[i]
        output_queries = []
        for i in range(batch_size):
            output_queries.append(self.cur_queries[i][self.cur_queries[i].sum(-1) != 0])
        self.fi += 1
        return output_queries


class OnlineMerge():
    def __init__(
        self,
        inscat_topk_insts,
        use_bbox=False,
        merge_type="count",
        tformer_cfg=None,
        iou_thr=0.1,
        absorb_cfg=None,
        support_cfg=None,
        rescue_cfg=None,
        birth_cfg=None,
        dedup_cfg=None,
        track_feat_cfg=None,
        geom_diag_cfg=None,
        monitor: bool = False,
    ):
        assert merge_type in ['count', 'frame']
        self.merge_type = merge_type
        self.inscat_topk_insts = inscat_topk_insts
        self.use_bbox = use_bbox
        self.iou_thr = iou_thr  # IoU预剪枝阈值
        self.monitor = bool(monitor)
        self.last_stats = None
        self.absorb_cfg = absorb_cfg if isinstance(absorb_cfg, dict) else {}
        self.support_cfg = support_cfg if isinstance(support_cfg, dict) else {}
        self.birth_cfg = birth_cfg if isinstance(birth_cfg, dict) else {}
        self.track_feat_cfg = track_feat_cfg if isinstance(track_feat_cfg, dict) else {}
        self.geom_diag_cfg = geom_diag_cfg if isinstance(geom_diag_cfg, dict) else {}
        # Avoid bloating per-frame JSON by default.
        self.record_kept_track_ids = bool(self.support_cfg.get("record_kept_track_ids", False)) or bool(
            self.absorb_cfg.get("record_kept_track_ids", False)
        )
        if self.use_bbox:
            self.iou_calculator = AxisAlignedBboxOverlaps3D()
        # 初始化跨帧 Transformer
        if tformer_cfg is not None:
            self.tformer = MODELS.build(tformer_cfg)
        else:
            self.tformer = None
        self.tracks = TrackManager(merge_type=merge_type)
        # Kept (top-K) track ids corresponding to the last returned outputs.
        self.output_track_ids = None
        # Semantic head top1 outputs for kept tracks (decoder semantic head).
        self.output_sem_labels = None
        self.output_sem_confs = None
        self.output_bg_confs = None
        # Optional: geometry diagnostics (det↔track/track↔track matrices).
        self.geom_diag_records = []
        self.rescue_cfg = rescue_cfg if isinstance(rescue_cfg, dict) else {}
        self.dedup_cfg = dedup_cfg if isinstance(dedup_cfg, dict) else {}
    
    def clean(self):
        self.tracks.reset()
        self.last_stats = None
        self.output_track_ids = None
        self.output_sem_labels = None
        self.output_sem_confs = None
        self.output_bg_confs = None
        self.geom_diag_records = []
    
    def merge(
        self,
        masks,
        labels,
        scores,
        queries,
        query_feats,
        sem_preds,
        xyz_list,
        bboxes,
        det_dino_feats=None,
        det_feat3d=None,
        det_gt=None,
        det_gt_conf=None,
        det_gt_inter=None,
        det_gt_size=None,
        det_gt_nvalid=None,
        det_voxels=None,
        rescue_masks=None,
        rescue_scores=None,
        rescue_dino_feats=None,
        rescue_feat3d=None,
        rescue_gt=None,
        rescue_gt_conf=None,
        rescue_gt_inter=None,
        rescue_gt_size=None,
        rescue_gt_nvalid=None,
        rescue_voxels=None,
        frame_i: int = -1,
    ):
        # Online behavior stats (optional; written to self.last_stats).
        det_to_merge = int(masks.shape[0])
        prev_mem_size = int(self.tracks.num_tracks)
        matched_cnt = 0
        birth_cnt = det_to_merge
        absorbed_cnt = 0
        supporters_cnt = 0
        absorbed_det_idx = []
        absorbed_mem_idx = []
        supporters_det_idx = []
        supporters_mem_idx = []
        matched_mem_idx = []
        matched_det_idx = []
        matched_track_ids = []
        birth_det_idx = []
        birth_track_ids = []
        drop_cnt = 0
        drop_det_idx = []
        rescued_cnt = 0
        rescued_from_l = 0
        rescued_from_uh = 0
        rescue_ambiguous = 0
        rescue_reject = {}
        rescue_delta_mask_pts = []
        rescue_new_vox_ratio = []
        rescue_new_vox_cnt = []
        rescue_det_vox_cnt = []
        rescue_trk_vox_cnt = []
        rescue_det_contain = []
        rescue_det_dino_cos = []
        rescue_det_feat3d_cos = []
        rescue_det_purity = []
        rescue_det_cov = []
        rescue_det_iou = []
        rescue_mem_idx = []
        rescue_det_idx = []
        rescue_is_l = []
        # L-pool portrait (pre-gate) and drop attribution (L-only).
        l_pool_n = 0
        l_pool_npoints = []
        l_pool_det_purity = []
        l_pool_det_cov = []
        l_pool_det_iou = []
        l_pool_useful_any05_rate = 0.0
        l_pool_reject = {"total": 0, "size": 0, "score": 0, "no_vox": 0, "geom": 0, "dino": 0, "feat3d": 0, "ambiguous": 0, "pass": 0}
        tentative_created = 0
        tentative_confirmed = 0
        tentative_deleted = 0
        dedup_stats = None

        points_per_mask = masks.shape[1]
        # masks, labels, scores, queries, query_feats, sem_preds, xyz_list = \
        #     self.intra_frame_merge(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes, q)
        if not self.tracks.initialized:
            cur_xyz = self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            # Optional: normalize det_dino_feats before initializing tracks (keeps behavior stable when disabled).
            if det_dino_feats is not None:
                try:
                    dino_cfg = self.track_feat_cfg.get("dino", {}) if isinstance(self.track_feat_cfg, dict) else {}
                    if isinstance(dino_cfg, dict) and bool(dino_cfg.get("normalize", True)):
                        det_dino_feats = torch.nn.functional.normalize(det_dino_feats, dim=1, eps=1e-6)
                except Exception:
                    pass
            # Optional: normalize det_feat3d before initializing tracks.
            if det_feat3d is not None:
                try:
                    feat3d_cfg = self.track_feat_cfg.get("feat3d", {}) if isinstance(self.track_feat_cfg, dict) else {}
                    if isinstance(feat3d_cfg, dict) and bool(feat3d_cfg.get("normalize", True)):
                        det_feat3d = torch.nn.functional.normalize(det_feat3d, dim=1, eps=1e-6)
                except Exception:
                    pass
            self.tracks.init_first_frame(
                masks=masks,
                labels=labels,
                scores=scores,
                queries=queries,
                query_feats=query_feats,
                sem_preds=sem_preds,
                xyz=cur_xyz,
                bboxes=bboxes if self.use_bbox else None,
                dino_feats=det_dino_feats,
                feat3d=det_feat3d,
                det_voxels=det_voxels,
            )
        else:
            # Static typing：确保前一帧已经初始化完毕
            assert self.tracks.labels is not None and self.tracks.query_feats is not None and \
                   self.tracks.sem_preds is not None and self.tracks.xyz is not None and \
                   self.tracks.merge_counts is not None and self.tracks.queries is not None and \
                   self.tracks.masks is not None and self.tracks.scores is not None and \
                   self.tracks.track_ids is not None
            self.tracks.begin_next_frame()
            next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
                masks, labels, scores, queries, query_feats, sem_preds, \
                self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            # 步骤1: IoU预剪枝 (统一计算，TDT和传统方法共用)
            if self.use_bbox:
                iou_matrix = self.iou_calculator(self.tracks.xyz, next_xyz, is_aligned=False)
                # IoU计算器返回的是(Memory, Current) = (Nm, Nc)
                xyz_scores = iou_matrix
            else:
                xyz_dists = torch.cdist(self.tracks.xyz, next_xyz, p=2)
                # cdist返回的是(Memory, Current) = (Nm, Nc)
                xyz_scores = 1 / (xyz_dists + 1e-6)
            
            # 预剪枝掩码（Memory x Current）= (Nm, Nc)
            attention_mask_mem_cur = xyz_scores > self.iou_thr  # True=允许匹配，False=禁止
            # Transformer attention mask 常见约定为 (B, tgt, src) = (1, Nc, Nm)
            attention_mask_tgt_src = attention_mask_mem_cur.T.unsqueeze(0)
            
            if self.tformer is not None and self.tracks.queries is not None:
                # 步骤2a: 使用Time Divided Transformer with attention mask
                # 构造几何向量 p_c / p_m
                def build_geom(xyz_or_bbox, bbox=None):
                    """构造9维几何特征向量
                    Args:
                        xyz_or_bbox: (N, 3) 位置坐标 或 (N, 6) bbox坐标
                        bbox: (N, 6) 或 (N, 7) bbox，或者None
                    Returns:
                        geom: (N, 9) 几何特征 [xyz, sin(xyz), size_xyz]
                    """
                    # 首先提取3维xyz坐标
                    if xyz_or_bbox.shape[-1] == 3:
                        # 输入是3维坐标
                        xyz = xyz_or_bbox
                    elif xyz_or_bbox.shape[-1] == 6:
                        # 输入是6维bbox [x1, y1, z1, x2, y2, z2]，提取中心点
                        xyz = (xyz_or_bbox[:, :3] + xyz_or_bbox[:, 3:6]) / 2.0
                    else:
                        # 其他格式，尝试取前3维作为坐标
                        xyz = xyz_or_bbox[:, :3]
                    
                    # 然后处理size信息
                    if bbox is None:
                        if xyz_or_bbox.shape[-1] == 6:
                            # 从bbox中提取size
                            size = xyz_or_bbox[:, 3:6] - xyz_or_bbox[:, 0:3]  # [w, h, l]
                        else:
                            # 使用默认尺寸
                            size = torch.ones_like(xyz) * 0.5  # 默认0.5m尺寸
                    else:
                        # 统一处理bbox维度，确保size总是3维
                        if bbox.shape[-1] == 6:
                            # 格式：[x1, y1, z1, x2, y2, z2]
                            size = bbox[:, 3:6] - bbox[:, 0:3]  # [w, h, l]
                        elif bbox.shape[-1] == 7:
                            # 格式：[center_x, center_y, center_z, w, h, l, angle]
                            size = bbox[:, 3:6]  # [w, h, l]
                        else:
                            # 其他格式，使用默认尺寸
                            size = torch.ones_like(xyz) * 0.5
                    
                    # 确保size是3维
                    if size.shape[-1] != 3:
                        size = size[:, :3] if size.shape[-1] > 3 else torch.ones_like(xyz) * 0.5
                    
                    # 构造9维几何特征：[xyz(3), sin(xyz)(3), size(3)]
                    geom = torch.cat([xyz, torch.sin(xyz), size], dim=-1)
                    
                    # 最终验证
                    assert geom.shape[-1] == 9, f"几何特征维度错误: {geom.shape[-1]} != 9, xyz: {xyz.shape}, size: {size.shape}"
                    return geom

                p_m = build_geom(self.tracks.xyz) if self.tracks.xyz is not None else None
                p_c = build_geom(next_xyz) if next_xyz is not None else None
                if p_m is None or p_c is None:
                    # fallback to zeros
                    p_m = torch.zeros(self.tracks.queries.shape[0], 9, device=self.tracks.queries.device)
                    p_c = torch.zeros(next_queries.shape[0], 9, device=next_queries.device)

                attn_mat, updated_queries = self.tformer(
                    next_queries.unsqueeze(0), 
                    self.tracks.queries.unsqueeze(0),
                    p_c.unsqueeze(0), 
                    p_m.unsqueeze(0),
                    mask_mem=torch.ones(1, self.tracks.queries.shape[0], dtype=torch.bool, device=next_queries.device),
                    attention_mask=attention_mask_tgt_src  # (1, Nc, Nm)
                )
                # attn_mat: (1, Nc, Nm) -> Hungarian expects (Nm, Nc) for memory-row assignment
                mix_scores = attn_mat.squeeze(0).T  # Nm x Nc
                
                # 🆕 使用TDT更新的特征来更新Memory (EMA更新)
                if hasattr(self, 'ema_alpha'):
                    alpha = self.ema_alpha
                else:
                    alpha = 0.9  # 默认EMA系数
                
                # 注意：这里updated_queries是当前帧的更新特征，应该用于后续的Memory更新
                self._updated_next_queries = updated_queries.squeeze(0)
            else:
                # 步骤2b: 使用传统特征匹配方法
                # 确保特征均已初始化，静态检查不再报 None
                if self.tracks.query_feats is None or next_query_feats is None:
                    raise RuntimeError('query_feats is None when merging instances')
                if self.tracks.sem_preds is None or next_sem_preds is None:
                    raise RuntimeError('sem_preds is None when merging instances')
                if self.tracks.xyz is None or next_xyz is None:
                    raise RuntimeError('xyz is None when merging instances')

                # Keep matrix layout consistent with xyz_scores: (Nm, Nc)
                query_feat_scores = (self.tracks.query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
                sem_pred_scores = F.cosine_similarity(
                    next_sem_preds.unsqueeze(1), self.tracks.sem_preds.unsqueeze(0), dim=2)

                mix_scores = query_feat_scores * xyz_scores
                # 应用IoU预剪枝mask
                mix_scores = torch.where(attention_mask_mem_cur, mix_scores, torch.zeros_like(mix_scores))
            
            # 确保标签匹配矩阵的维度与mix_scores一致 (Nm, Nc)
            inst_label_scores = torch.where(
                self.tracks.labels.unsqueeze(1) == next_labels.unsqueeze(0),
                torch.ones((self.tracks.labels.shape[0], next_labels.shape[0])).to(self.tracks.labels.device),
                torch.zeros((self.tracks.labels.shape[0], next_labels.shape[0])).to(self.tracks.labels.device)
            )
            
            mix_scores = torch.where(mix_scores > 0, mix_scores, torch.zeros_like(mix_scores))
            mix_scores = mix_scores * inst_label_scores

            # ------------------------------------------------------------
            # H/L pool split (online tracking):
            # - H pool: high-confidence detections used for 1-to-1 Hungarian association.
            # - L pool: low-confidence detections reserved for stage-2 rescue; never used in Hungarian.
            #
            # We reuse `birth_cfg` thresholds as the definition of H pool (strict and stable).
            # This keeps the identity backbone robust while still allowing a larger association
            # candidate pool via `assoc_filter` (handled upstream).
            # ------------------------------------------------------------
            det_points = None
            try:
                det_points = next_masks.sum(dim=1)
            except Exception:
                det_points = None

            h_mask = None
            try:
                if isinstance(self.birth_cfg, dict):
                    score_thr_birth = float(self.birth_cfg.get('score_thr', -1.0))
                    npoint_thr_birth = int(self.birth_cfg.get('npoint_thr', -1))
                    if score_thr_birth > 0 or npoint_thr_birth > 0:
                        h_mask = torch.ones((int(next_masks.shape[0]),), device=next_masks.device, dtype=torch.bool)
                        if score_thr_birth > 0:
                            h_mask = h_mask & (next_scores >= score_thr_birth)
                        if npoint_thr_birth > 0 and det_points is not None:
                            h_mask = h_mask & (det_points > int(npoint_thr_birth))
            except Exception:
                h_mask = None

            if h_mask is not None and int(h_mask.numel()) == int(mix_scores.shape[1]):
                # Hard-disable non-H columns for Hungarian by zeroing their affinity.
                mix_scores = mix_scores * h_mask.unsqueeze(0)

            # Optional: tentative-track penalty in Hungarian (delayed birth).
            # This reduces the chance that tentative tracks steal associations from confirmed tracks.
            try:
                tent_cfg = self.birth_cfg.get("tentative", {}) if isinstance(self.birth_cfg, dict) else {}
                tent_enable = bool(tent_cfg.get("enable", False))
                tent_weight = float(tent_cfg.get("match_weight", 1.0))
                if tent_enable and tent_weight < 1.0 and getattr(self.tracks, "track_states", None) is not None:
                    ts = getattr(self.tracks, "track_states")
                    if torch.is_tensor(ts) and int(ts.numel()) == int(mix_scores.shape[0]):
                        tent_rows = (ts == 1)
                        if bool(tent_rows.any().item()):
                            mix_scores[tent_rows] = mix_scores[tent_rows] * float(tent_weight)
            except Exception:
                pass

            # Hungarian assign (supports rectangular matrices)
            row_ind, col_ind = linear_sum_assignment(-mix_scores.detach().cpu())
            row_ind = torch.tensor(row_ind).to(mix_scores.device)
            col_ind = torch.tensor(col_ind).to(mix_scores.device)
            
            # 只保留有效的匹配分数
            if len(row_ind) > 0:
                mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
                row_ind = row_ind[mix_scores_mask]
                col_ind = col_ind[mix_scores_mask]

            # Optional: update track GT votes for diagnostics (matched pairs only).
            try:
                if det_gt is not None:
                    gt_conf_thr = float(self.geom_diag_cfg.get("det_gt_conf_thr", 0.5))
                    weight_by_conf = bool(self.geom_diag_cfg.get("det_gt_weight_by_conf", True))
                    self.tracks.update_track_gt_votes(
                        row_ind=row_ind,
                        col_ind=col_ind,
                        det_gt=det_gt,
                        det_gt_conf=det_gt_conf,
                        conf_thr=gt_conf_thr,
                        weight_by_conf=weight_by_conf,
                    )
            except Exception:
                pass

            # Online matching statistics (post-filter).
            matched_cnt = int(row_ind.numel())
            matched_mem_idx = row_ind.detach().cpu().tolist()
            matched_det_idx = col_ind.detach().cpu().tolist()
            try:
                matched_track_ids = self.tracks.track_ids[row_ind].detach().cpu().tolist()
            except Exception:
                matched_track_ids = []

            temp_masks = torch.zeros((self.tracks.masks.shape[0], points_per_mask)).bool().to(self.tracks.masks.device)
            temp_masks[row_ind] = next_masks[col_ind]
            no_merge_masks = torch.ones(next_masks.shape[0]).bool().to(next_masks.device)
            no_merge_masks[col_ind] = False

            # Update tentative-hit counts based on this frame's 1-to-1 matches.
            try:
                self.tracks.update_tentative_hits(matched_mem_idx=row_ind)
            except Exception:
                pass

            # ------------------------------------------------------------
            # Optional: many-to-one "rescue write" (stage-2).
            # Goal: allow extra dets (from a broader association pool) to be written into
            # *eligible* tracks, without creating new births, using hard multi-modal gating.
            #
            # Design (per user plan):
            # - Identity backbone remains Hungarian 1-to-1 on H pool only.
            # - Rescue candidates come from (L pool + unmatched H), but never birth from L.
            # - Eligible tracks default to matched tracks of this frame.
            # - Hard gating AND: contain_det(voxel) + DINO cos + 3D pooled cos
            # - Require uniqueness: |S(det)| == 1
            # - Write behavior: OR into current frame mask (temp_masks) + suppress birth (no_merge_masks=False)
            # ------------------------------------------------------------
            rescue_enable = bool(self.rescue_cfg.get("enable", False))
            rescued_cnt = 0
            rescued_from_l = 0
            rescued_from_uh = 0
            rescue_ambiguous = 0
            rescue_reject = {
                "no_feat": 0,
                "no_vox": 0,
                "size": 0,
                "score": 0,
                "geom": 0,
                "dino": 0,
                "feat3d": 0,
                "no_pass": 0,
                "ambiguous": 0,
            }
            rescue_mem_idx = []
            rescue_det_idx = []
            rescue_is_l = []
            if rescue_enable and no_merge_masks.any():
                try:
                    # Eligibility: by default, only tracks that are already matched by Hungarian this frame.
                    eligible_mode = str(self.rescue_cfg.get("eligible_tracks", "matched")).lower()
                    if eligible_mode not in ("matched", "all"):
                        eligible_mode = "matched"

                    if eligible_mode == "matched":
                        eligible_mem = row_ind.detach().cpu().tolist()
                    else:
                        eligible_mem = list(range(int(self.tracks.num_tracks)))

                    if not eligible_mem:
                        rescue_enable = False
                except Exception:
                    rescue_enable = False

            if rescue_enable and no_merge_masks.any():
                # Require per-det voxel ids and per-track voxel cache for geometry gating.
                track_voxels = getattr(self.tracks, "track_voxels", None)
                det_vox_ok = det_voxels is not None
                rescue_vox_ok = rescue_voxels is not None and isinstance(rescue_voxels, list)
                if track_voxels is None or (not det_vox_ok and not rescue_vox_ok):
                    rescue_reject["no_vox"] += int(torch.nonzero(no_merge_masks, as_tuple=False).numel())
                    rescue_enable = False
                # Require multi-modal features if configured.
                det_dino_ok = det_dino_feats is not None and torch.is_tensor(det_dino_feats)
                det_f3d_ok = det_feat3d is not None and torch.is_tensor(det_feat3d)
                rescue_det_dino_ok = rescue_dino_feats is not None and torch.is_tensor(rescue_dino_feats)
                rescue_det_f3d_ok = rescue_feat3d is not None and torch.is_tensor(rescue_feat3d)
                trk_dino_ok = getattr(self.tracks, "dino_feats", None) is not None and torch.is_tensor(getattr(self.tracks, "dino_feats", None))
                trk_f3d_ok = getattr(self.tracks, "feat3d", None) is not None and torch.is_tensor(getattr(self.tracks, "feat3d", None))
                # H candidates require det feats; pre-pool candidates require rescue feats.
                if (not trk_dino_ok) or (not trk_f3d_ok) or (
                    (not det_dino_ok or not det_f3d_ok) and (not rescue_det_dino_ok or not rescue_det_f3d_ok)
                ):
                    rescue_reject["no_feat"] += int(torch.nonzero(no_merge_masks, as_tuple=False).numel())
                    rescue_enable = False

            # L-pool portrait (pre-gate): summarize npoints and det->GT purity/cov/IoU.
            portrait_enable = False
            try:
                portrait_cfg = self.rescue_cfg.get("l_pool_portrait", {}) if isinstance(self.rescue_cfg, dict) else {}
                portrait_enable = bool(portrait_cfg.get("enable", False))
            except Exception:
                portrait_enable = False
            if portrait_enable:
                try:
                    if rescue_masks is not None and torch.is_tensor(rescue_masks) and rescue_masks.numel() > 0:
                        l_pool_n = int(rescue_masks.shape[0])
                        pts = rescue_masks.sum(dim=1).to(torch.float32)
                        l_pool_npoints = pts.detach().cpu().tolist()
                        if rescue_gt is not None and rescue_gt_conf is not None and rescue_gt_inter is not None and rescue_gt_size is not None and rescue_gt_nvalid is not None:
                            inter = rescue_gt_inter.to(torch.float32).detach()
                            gsz = rescue_gt_size.to(torch.float32).clamp_min(0.0).detach()
                            nvalid = rescue_gt_nvalid.to(torch.float32).clamp_min(0.0).detach()
                            purity = rescue_gt_conf.to(torch.float32).detach()
                            cov = torch.where(gsz > 0, inter / (gsz + 1e-6), torch.zeros_like(inter))
                            iou = torch.where((nvalid + gsz - inter) > 0, inter / (nvalid + gsz - inter + 1e-6), torch.zeros_like(inter))
                            l_pool_det_purity = purity.detach().cpu().tolist()
                            l_pool_det_cov = cov.detach().cpu().tolist()
                            l_pool_det_iou = iou.detach().cpu().tolist()
                            if iou.numel():
                                useful = ((cov >= 0.5) | (iou >= 0.5)).to(torch.float32)
                                l_pool_useful_any05_rate = float(useful.mean().item())
                except Exception:
                    l_pool_n = 0
                    l_pool_npoints = []
                    l_pool_det_purity = []
                    l_pool_det_cov = []
                    l_pool_det_iou = []
                    l_pool_useful_any05_rate = 0.0

            if rescue_enable and no_merge_masks.any():
                try:
                    # Candidate dets: L pool + unmatched H dets.
                    # H definition follows birth_cfg thresholds (same as Hungarian H pool).
                    score_thr_birth = float(self.birth_cfg.get('score_thr', -1.0)) if isinstance(self.birth_cfg, dict) else -1.0
                    npoint_thr_birth = int(self.birth_cfg.get('npoint_thr', -1)) if isinstance(self.birth_cfg, dict) else -1
                    h_mask2 = torch.ones((int(next_masks.shape[0]),), device=next_masks.device, dtype=torch.bool)
                    if score_thr_birth > 0:
                        h_mask2 = h_mask2 & (next_scores >= score_thr_birth)
                    if npoint_thr_birth > 0:
                        pts = next_masks.sum(dim=1)
                        h_mask2 = h_mask2 & (pts > int(npoint_thr_birth))

                    use_l = bool(self.rescue_cfg.get("use_l", True))
                    use_unmatched_h = bool(self.rescue_cfg.get("use_unmatched_h", True))
                    cand = []
                    cand_is_l = []
                    has_rescue_pool = bool(
                        use_l
                        and rescue_masks is not None
                        and torch.is_tensor(rescue_masks)
                        and rescue_scores is not None
                        and torch.is_tensor(rescue_scores)
                        and rescue_voxels is not None
                        and isinstance(rescue_voxels, list)
                        and rescue_dino_feats is not None
                        and torch.is_tensor(rescue_dino_feats)
                        and rescue_feat3d is not None
                        and torch.is_tensor(rescue_feat3d)
                    )
                    if use_l:
                        if has_rescue_pool:
                            # L pool from pre-inst_thr pre_pool (never birth).
                            cand_l = list(range(int(rescue_masks.shape[0])))
                            cand.extend(cand_l)
                            cand_is_l.extend([True] * len(cand_l))
                        else:
                            # Backward compatible fallback: L pool inside current det set (below birth thresholds).
                            cand_l = torch.nonzero(no_merge_masks & (~h_mask2), as_tuple=False).reshape(-1).detach().cpu().tolist()
                            cand.extend(cand_l)
                            cand_is_l.extend([True] * len(cand_l))
                    if use_unmatched_h:
                        cand_h = torch.nonzero(no_merge_masks & h_mask2, as_tuple=False).reshape(-1).detach().cpu().tolist()
                        cand.extend(cand_h)
                        cand_is_l.extend([False] * len(cand_h))

                    if not cand:
                        rescue_enable = False
                except Exception:
                    rescue_enable = False

            if rescue_enable and no_merge_masks.any():
                try:
                    # Thresholds / constraints.
                    contain_thr = float(self.rescue_cfg.get("contain_det_thr", 0.30))
                    dino_thr = float(self.rescue_cfg.get("dino_cos_thr", 0.75))
                    f3d_thr = float(self.rescue_cfg.get("feat3d_cos_thr", 0.80))
                    min_pts = int(self.rescue_cfg.get("min_mask_points", 20))
                    max_pts = int(self.rescue_cfg.get("max_mask_points", 500))
                    min_score = float(self.rescue_cfg.get("min_score", -1.0))
                    require_unique = bool(self.rescue_cfg.get("require_unique", True))
                    update_mask = bool(self.rescue_cfg.get("update_mask", True))

                    # Track voxel cap for updates.
                    cap_trk = int(self.rescue_cfg.get("V_trk_cap", self.geom_diag_cfg.get("V_trk_cap", 1024)))

                    trk_dino = getattr(self.tracks, "dino_feats", None)
                    trk_f3d = getattr(self.tracks, "feat3d", None)
                    assert trk_dino is not None and trk_f3d is not None
                    trk_dino = trk_dino.to(device=next_masks.device)
                    trk_f3d = trk_f3d.to(device=next_masks.device)
                    det_dino = det_dino_feats.to(device=next_masks.device)
                    det_f3d = det_feat3d.to(device=next_masks.device)
                    det_dino_l = rescue_dino_feats.to(device=next_masks.device) if (rescue_dino_feats is not None and torch.is_tensor(rescue_dino_feats)) else None
                    det_f3d_l = rescue_feat3d.to(device=next_masks.device) if (rescue_feat3d is not None and torch.is_tensor(rescue_feat3d)) else None
                    use_rescue_pool = bool(
                        rescue_masks is not None
                        and torch.is_tensor(rescue_masks)
                        and rescue_scores is not None
                        and torch.is_tensor(rescue_scores)
                        and rescue_voxels is not None
                        and isinstance(rescue_voxels, list)
                        and det_dino_l is not None
                        and det_f3d_l is not None
                    )

                    # Helper: containment(det -> track) on hashed voxel ids.
                    def _contain_det_track(det_vox: torch.Tensor, trk_vox: torch.Tensor) -> float:
                        if det_vox.numel() == 0 or trk_vox.numel() == 0:
                            return 0.0
                        return float(torch.isin(det_vox, trk_vox).float().mean().item())

                    for dj, is_l in zip(cand, cand_is_l):
                        if bool(is_l):
                            l_pool_reject["total"] = int(l_pool_reject.get("total", 0)) + 1
                        if bool(is_l) and use_rescue_pool:
                            pm = rescue_masks[int(dj)]
                            pts = int(pm.sum().item())
                            sc = float(rescue_scores[int(dj)].item())
                            dv = rescue_voxels[int(dj)]
                            dg = rescue_gt
                            dg_conf = rescue_gt_conf
                            dg_inter = rescue_gt_inter
                            dg_size = rescue_gt_size
                            dg_nvalid = rescue_gt_nvalid
                        else:
                            pm = next_masks[int(dj)]
                            pts = int(pm.sum().item())
                            sc = float(next_scores[int(dj)].item())
                            dv = det_voxels[int(dj)]
                            dg = det_gt
                            dg_conf = det_gt_conf
                            dg_inter = det_gt_inter
                            dg_size = det_gt_size
                            dg_nvalid = det_gt_nvalid
                        if pts < min_pts or (max_pts > 0 and pts > max_pts):
                            rescue_reject["size"] += 1
                            if bool(is_l):
                                l_pool_reject["size"] = int(l_pool_reject.get("size", 0)) + 1
                            continue
                        if min_score >= 0 and float(sc) < min_score:
                            rescue_reject["score"] += 1
                            if bool(is_l):
                                l_pool_reject["score"] = int(l_pool_reject.get("score", 0)) + 1
                            continue
                        if dv is None or (torch.is_tensor(dv) and dv.numel() == 0):
                            rescue_reject["no_vox"] += 1
                            if bool(is_l):
                                l_pool_reject["no_vox"] = int(l_pool_reject.get("no_vox", 0)) + 1
                            continue
                        # Evaluate gates over eligible tracks with structured reject reasons.
                        geom_pass = []
                        contain_cache = {}
                        for mi in eligible_mem:
                            if mi < 0 or mi >= len(track_voxels):
                                continue
                            tv = track_voxels[int(mi)]
                            c = _contain_det_track(dv, tv)
                            contain_cache[int(mi)] = float(c)
                            if c >= contain_thr:
                                geom_pass.append(int(mi))
                        if not geom_pass:
                            rescue_reject["geom"] += 1
                            rescue_reject["no_pass"] += 1
                            if bool(is_l):
                                l_pool_reject["geom"] = int(l_pool_reject.get("geom", 0)) + 1
                            continue

                        dino_pass = []
                        dino_cache = {}
                        for mi in geom_pass:
                            if bool(is_l) and use_rescue_pool and det_dino_l is not None:
                                cd = float((det_dino_l[int(dj)] * trk_dino[int(mi)]).sum().item())
                            else:
                                cd = float((det_dino[int(dj)] * trk_dino[int(mi)]).sum().item())
                            dino_cache[int(mi)] = float(cd)
                            if cd >= dino_thr:
                                dino_pass.append(int(mi))
                        if not dino_pass:
                            rescue_reject["dino"] += 1
                            rescue_reject["no_pass"] += 1
                            if bool(is_l):
                                l_pool_reject["dino"] = int(l_pool_reject.get("dino", 0)) + 1
                            continue

                        f3d_pass = []
                        f3d_cache = {}
                        for mi in dino_pass:
                            if bool(is_l) and use_rescue_pool and det_f3d_l is not None:
                                c3 = float((det_f3d_l[int(dj)] * trk_f3d[int(mi)]).sum().item())
                            else:
                                c3 = float((det_f3d[int(dj)] * trk_f3d[int(mi)]).sum().item())
                            f3d_cache[int(mi)] = float(c3)
                            if c3 >= f3d_thr:
                                f3d_pass.append(int(mi))
                                if not require_unique:
                                    break
                        if not f3d_pass:
                            rescue_reject["feat3d"] += 1
                            rescue_reject["no_pass"] += 1
                            if bool(is_l):
                                l_pool_reject["feat3d"] = int(l_pool_reject.get("feat3d", 0)) + 1
                            continue
                        if require_unique and len(f3d_pass) != 1:
                            rescue_reject["ambiguous"] += 1
                            rescue_ambiguous += 1
                            if bool(is_l):
                                l_pool_reject["ambiguous"] = int(l_pool_reject.get("ambiguous", 0)) + 1
                            continue
                        best_i = int(f3d_pass[0])
                        # Record gating scores (chosen track).
                        try:
                            rescue_det_contain.append(float(contain_cache.get(best_i, 0.0)))
                            rescue_det_dino_cos.append(float(dino_cache.get(best_i, 0.0)))
                            rescue_det_feat3d_cos.append(float(f3d_cache.get(best_i, 0.0)))
                        except Exception:
                            pass
                        # Write: suppress birth + (optional) union current-frame mask into temp_masks.
                        if not bool(is_l):
                            no_merge_masks[int(dj)] = False
                        if update_mask:
                            try:
                                pre_pts = int(temp_masks[best_i].sum().item())
                                merged = temp_masks[best_i] | pm
                                post_pts = int(merged.sum().item())
                                temp_masks[best_i] = merged
                                rescue_delta_mask_pts.append(float(max(post_pts - pre_pts, 0)))
                            except Exception:
                                pass
                        # Record voxel novelty (det voxels not present in track before update).
                        try:
                            tv = track_voxels[int(best_i)]
                            if torch.is_tensor(tv) and torch.is_tensor(dv) and dv.numel() > 0:
                                in_mask = torch.isin(dv, tv)
                                new_cnt = int((~in_mask).sum().item())
                                det_cnt = int(dv.numel())
                                trk_cnt = int(tv.numel())
                                rescue_new_vox_cnt.append(float(new_cnt))
                                rescue_det_vox_cnt.append(float(det_cnt))
                                rescue_trk_vox_cnt.append(float(trk_cnt))
                                rescue_new_vox_ratio.append(float(new_cnt) / float(det_cnt + 1e-6))
                        except Exception:
                            pass
                        # GT quality portrait for rescued dets (optional; only when provided).
                        try:
                            if dg is not None and dg_conf is not None and dg_inter is not None and dg_size is not None and dg_nvalid is not None:
                                gid = int(dg[int(dj)].item())
                                if gid >= 0:
                                    gsz = int(dg_size[int(dj)].item())
                                    inter = int(dg_inter[int(dj)].item())
                                    nvalid = int(dg_nvalid[int(dj)].item())
                                    if gsz > 0 and nvalid > 0 and inter > 0:
                                        purity = float(dg_conf[int(dj)].item())
                                        cov = float(inter) / float(gsz + 1e-6)
                                        iou = float(inter) / float(nvalid + gsz - inter + 1e-6)
                                        rescue_det_purity.append(purity)
                                        rescue_det_cov.append(cov)
                                        rescue_det_iou.append(iou)
                        except Exception:
                            pass
                        rescued_cnt += 1
                        if is_l:
                            rescued_from_l += 1
                        else:
                            rescued_from_uh += 1
                        rescue_mem_idx.append(int(best_i))
                        rescue_det_idx.append(int(dj))
                        rescue_is_l.append(bool(is_l))
                        if bool(is_l):
                            l_pool_reject["pass"] = int(l_pool_reject.get("pass", 0)) + 1
                except Exception:
                    # Safety: never crash the main online pipeline due to rescue logic.
                    rescued_cnt = rescued_from_l = rescued_from_uh = rescue_ambiguous = 0
                    rescue_mem_idx = []
                    rescue_det_idx = []
                    rescue_is_l = []
                    rescue_enable = False
                    rescue_delta_mask_pts = []
                    rescue_new_vox_ratio = []
                    rescue_new_vox_cnt = []
                    rescue_det_vox_cnt = []
                    rescue_trk_vox_cnt = []
                    rescue_det_contain = []
                    rescue_det_dino_cos = []
                    rescue_det_feat3d_cos = []
                    rescue_det_purity = []
                    rescue_det_cov = []
                    rescue_det_iou = []

            # Optional: many-to-one absorb (stage-2) to reduce fragmentation without birthing new tracks.
            absorb_enable = bool(self.absorb_cfg.get('enable', False))
            if absorb_enable and no_merge_masks.any():
                try:
                    # thresholds
                    only_to_matched = bool(self.absorb_cfg.get('only_to_matched', True))
                    max_absorb = int(self.absorb_cfg.get('max_absorb_per_frame', 0))
                    iou_thr_absorb = float(self.absorb_cfg.get('iou_thr', 0.5))
                    score_thr_absorb = float(self.absorb_cfg.get('score_thr', 0.0))
                    min_pts = int(self.absorb_cfg.get('min_mask_points', 20))
                    max_pts = int(self.absorb_cfg.get('max_mask_points', 500))
                    update_score = str(self.absorb_cfg.get('update_score', 'max'))
                    if update_score not in ('max', 'none'):
                        update_score = 'max'

                    Nm, Nc = int(mix_scores.shape[0]), int(mix_scores.shape[1])
                    if Nm > 0 and Nc > 0:
                        cand_det = torch.nonzero(no_merge_masks, as_tuple=False).reshape(-1)
                        if cand_det.numel() > 0:
                            if only_to_matched and row_ind.numel() > 0:
                                allow_mem = torch.zeros((Nm,), device=mix_scores.device, dtype=torch.bool)
                                allow_mem[row_ind] = True
                            else:
                                allow_mem = torch.ones((Nm,), device=mix_scores.device, dtype=torch.bool)
                            # absorb at most max_absorb (0 means no limit)
                            absorbed_this = 0
                            for dj in cand_det.tolist():
                                if max_absorb > 0 and absorbed_this >= max_absorb:
                                    break
                                pm = next_masks[int(dj)]
                                pts = int(pm.sum().item())
                                if pts < min_pts or (max_pts > 0 and pts > max_pts):
                                    continue
                                col = mix_scores[:, int(dj)]
                                col = torch.where(allow_mem, col, torch.zeros_like(col))
                                best_i = int(torch.argmax(col).item())
                                best_s = float(col[best_i].item())
                                if best_s <= score_thr_absorb:
                                    continue
                                geo = float(xyz_scores[best_i, int(dj)].item())
                                if geo < iou_thr_absorb:
                                    continue
                                # absorb into temp_masks (current frame segment)
                                temp_masks[best_i] = temp_masks[best_i] | pm
                                no_merge_masks[int(dj)] = False
                                absorbed_cnt += 1
                                absorbed_this += 1
                                absorbed_det_idx.append(int(dj))
                                absorbed_mem_idx.append(int(best_i))
                                if update_score == 'max':
                                    try:
                                        assert self.tracks.scores is not None
                                        self.tracks.scores[best_i] = torch.maximum(self.tracks.scores[best_i], next_scores[int(dj)])
                                    except Exception:
                                        pass
                except Exception:
                    pass

            # Optional: one-to-many supporters association (stage-2, suppress-birth only).
            # This is a *principle-check* mechanism: a track may accept multiple supporters
            # within the same frame, while each det can be assigned to at most one track.
            # Unlike `absorb_cfg`, it does NOT union masks or update track features/scores.
            support_enable = bool(self.support_cfg.get('enable', False))
            if support_enable and no_merge_masks.any():
                try:
                    mode = str(self.support_cfg.get('mode', 'mask_overlap')).lower()
                    only_to_matched = bool(self.support_cfg.get('only_to_matched', True))
                    max_support = int(self.support_cfg.get('max_support_per_frame', 0))
                    # NOTE: in bbox-containment mode, `cov_thr` is interpreted as det-bbox containment ratio.
                    cov_thr = float(self.support_cfg.get('cov_thr', 0.8))
                    small_ratio_max = float(self.support_cfg.get('small_ratio_max', 0.3))
                    score_thr = float(self.support_cfg.get('score_thr', 0.0))
                    min_pts = int(self.support_cfg.get('min_mask_points', 20))
                    max_pts = int(self.support_cfg.get('max_mask_points', 500))

                    # Optional geometry safety valves.
                    center_norm_thr = float(self.support_cfg.get('center_norm_thr', -1.0))
                    require_positive_overlap = bool(self.support_cfg.get('require_positive_overlap', True))

                    Nm = int(mix_scores.shape[0])
                    if Nm > 0 and next_masks.numel() > 0:
                        cand_det = torch.nonzero(no_merge_masks, as_tuple=False).reshape(-1)
                        if cand_det.numel() > 0:
                            if only_to_matched and row_ind.numel() > 0:
                                allow_mem = torch.zeros((Nm,), device=mix_scores.device, dtype=torch.bool)
                                allow_mem[row_ind] = True
                            else:
                                allow_mem = torch.ones((Nm,), device=mix_scores.device, dtype=torch.bool)

                    def _center_and_diag(bb6: torch.Tensor):
                        c = (bb6[:, 0:3] + bb6[:, 3:6]) * 0.5
                        d = torch.linalg.norm((bb6[:, 3:6] - bb6[:, 0:3]).clamp_min(0), dim=1)
                        return c, d

                    # Precompute bbox centers/diags if available (used as safety valve).
                    if center_norm_thr > 0 and self.use_bbox and self.tracks.xyz is not None and next_xyz is not None \
                            and self.tracks.xyz.numel() > 0 and next_xyz.numel() > 0 \
                            and int(self.tracks.xyz.shape[-1]) == 6 and int(next_xyz.shape[-1]) == 6:
                        mem_center, mem_diag = _center_and_diag(self.tracks.xyz)  # (Nm,3),(Nm,)
                        det_center, _ = _center_and_diag(next_xyz)  # (Nc,3)
                    else:
                        mem_center = mem_diag = det_center = None

                    supported_this = 0
                    for dj in cand_det.tolist():
                        if max_support > 0 and supported_this >= max_support:
                            break
                        pm = next_masks[int(dj)]
                        pts_det = int(pm.sum().item())
                        if pts_det < min_pts or (max_pts > 0 and pts_det > max_pts):
                            continue
                        if float(next_scores[int(dj)].item()) < score_thr:
                            continue

                        if mode == 'bbox_contain' and self.use_bbox and self.tracks.xyz is not None and next_xyz is not None \
                                and int(self.tracks.xyz.shape[-1]) == 6 and int(next_xyz.shape[-1]) == 6:
                            # BBox containment-based supporters (calibration-friendly).
                            det_bb = next_xyz[int(dj)]  # (6,)
                            mem_bb = self.tracks.xyz  # (Nm,6)

                            # Intersection bbox (Nm,6): [max(mins), min(maxs)]
                            inter_min = torch.maximum(mem_bb[:, 0:3], det_bb[0:3].unsqueeze(0))
                            inter_max = torch.minimum(mem_bb[:, 3:6], det_bb[3:6].unsqueeze(0))
                            inter_sz = (inter_max - inter_min).clamp_min(0)
                            inter_vol = inter_sz[:, 0] * inter_sz[:, 1] * inter_sz[:, 2]  # (Nm,)

                            det_sz = (det_bb[3:6] - det_bb[0:3]).clamp_min(0)
                            det_vol = float((det_sz[0] * det_sz[1] * det_sz[2]).item())
                            if det_vol <= 0:
                                continue

                            mem_sz = (mem_bb[:, 3:6] - mem_bb[:, 0:3]).clamp_min(0)
                            mem_vol = mem_sz[:, 0] * mem_sz[:, 1] * mem_sz[:, 2] + 1e-6  # (Nm,)

                            contain_det = inter_vol / (det_vol + 1e-6)  # (Nm,)
                            vol_ratio = float(det_vol) / mem_vol  # (Nm,)

                            valid = allow_mem & (contain_det >= cov_thr) & (vol_ratio <= small_ratio_max)
                            if require_positive_overlap:
                                valid = valid & (inter_vol > 0)

                            # Optional center-distance normalized by track bbox diag (safety valve).
                            if center_norm_thr > 0 and mem_center is not None and mem_diag is not None and det_center is not None:
                                dc = torch.linalg.norm(mem_center - det_center[int(dj)].unsqueeze(0), dim=1)  # (Nm,)
                                dn = dc / (mem_diag + 1e-6)
                                valid = valid & (dn <= center_norm_thr)

                            if not bool(valid.any().item()):
                                continue
                            # Choose best by containment (tie-break by smaller vol_ratio).
                            score_sel = torch.where(valid, contain_det, torch.zeros_like(contain_det))
                            best_i = int(torch.argmax(score_sel).item())
                        else:
                            # Fallback: mask-overlap coverage against current-frame track segment (may miss complementary fragments).
                            track_masks = temp_masks  # (Nm, P) bool
                            track_pts = track_masks.sum(dim=1).to(torch.float32)  # (Nm,)
                            ov = (track_masks & pm.unsqueeze(0)).sum(dim=1).to(torch.float32)  # (Nm,)
                            if require_positive_overlap:
                                ov = torch.where(ov > 0, ov, torch.zeros_like(ov))
                            cov = ov / max(float(pts_det), 1.0)  # (Nm,)
                            ratio = float(pts_det) / (track_pts + 1e-6)  # (Nm,)
                            valid = allow_mem & (track_pts > 0) & (ratio <= small_ratio_max) & (cov >= cov_thr)
                            if center_norm_thr > 0 and mem_center is not None and mem_diag is not None and det_center is not None:
                                dc = torch.linalg.norm(mem_center - det_center[int(dj)].unsqueeze(0), dim=1)  # (Nm,)
                                dn = dc / (mem_diag + 1e-6)
                                valid = valid & (dn <= center_norm_thr)
                            if not bool(valid.any().item()):
                                continue
                            score_cov = torch.where(valid, cov, torch.zeros_like(cov))
                            best_i = int(torch.argmax(score_cov).item())

                        # Suppress birth: do not create a new track for this det.
                        no_merge_masks[int(dj)] = False
                        supporters_cnt += 1
                        supported_this += 1
                        supporters_det_idx.append(int(dj))
                        supporters_mem_idx.append(int(best_i))
                except Exception:
                    pass

            # Optional: geometry diagnostics (det↔track / track↔track matrices).
            try:
                geom_enable = bool(self.geom_diag_cfg.get("enable", False))
            except Exception:
                geom_enable = False
            if geom_enable and det_gt is not None and det_voxels is not None:
                try:
                    frame_stride = int(self.geom_diag_cfg.get("frame_stride", 1))
                except Exception:
                    frame_stride = 1
                try:
                    frame_allowlist = self.geom_diag_cfg.get("frame_allowlist", None)
                except Exception:
                    frame_allowlist = None
                if frame_i >= 0 and frame_stride > 1 and (frame_i % frame_stride) != 0:
                    geom_enable = False
                if isinstance(frame_allowlist, (list, tuple)) and frame_i >= 0 and frame_i not in frame_allowlist:
                    geom_enable = False
            if geom_enable and det_gt is not None and det_voxels is not None:
                try:
                    det_gt_conf_thr = float(self.geom_diag_cfg.get("det_gt_conf_thr", 0.5))
                    track_gt_min_strength = float(self.geom_diag_cfg.get("track_gt_min_strength", 1.0))
                    max_gt_per_frame = int(self.geom_diag_cfg.get("max_gt_per_frame", 3))
                    neg_tracks_k = int(self.geom_diag_cfg.get("neg_tracks_k", 32))
                    if not torch.is_tensor(det_gt):
                        det_gt_t = torch.as_tensor(det_gt)
                    else:
                        det_gt_t = det_gt
                    if det_gt_conf is None:
                        det_conf_t = torch.ones_like(det_gt_t, dtype=torch.float32)
                    elif torch.is_tensor(det_gt_conf):
                        det_conf_t = det_gt_conf
                    else:
                        det_conf_t = torch.as_tensor(det_gt_conf, dtype=torch.float32)

                    # Prepare track GT ids/strength and voxel cache (pre-update).
                    track_gt_ids = getattr(self.tracks, "track_gt_ids", None)
                    track_gt_strength = getattr(self.tracks, "track_gt_strength", None)
                    track_voxels = getattr(self.tracks, "track_voxels", None)
                    track_ids = getattr(self.tracks, "track_ids", None)
                    if track_gt_ids is not None and track_voxels is not None and track_ids is not None:
                        # Build per-GT det index list (only confident GT assignments).
                        valid_det = (det_gt_t >= 0) & (det_conf_t >= det_gt_conf_thr)
                        det_ids_all = torch.nonzero(valid_det, as_tuple=False).reshape(-1).detach().cpu().tolist()
                        if det_ids_all:
                            det_gt_cpu = det_gt_t.detach().cpu()
                            # Count dets per GT and select top-k.
                            gt_counts = {}
                            for dj in det_ids_all:
                                gid = int(det_gt_cpu[int(dj)].item())
                                gt_counts[gid] = gt_counts.get(gid, 0) + 1
                            gt_sorted = sorted(gt_counts.items(), key=lambda x: (-x[1], x[0]))
                            if max_gt_per_frame > 0:
                                gt_sorted = gt_sorted[:max_gt_per_frame]
                            gt_targets = [gid for gid, _ in gt_sorted]

                            # Helper: containment(det -> track)
                            def _contain_det_track(det_vox: torch.Tensor, trk_vox: torch.Tensor) -> float:
                                if det_vox.numel() == 0 or trk_vox.numel() == 0:
                                    return 0.0
                                dv = det_vox.detach().cpu()
                                tv = trk_vox.detach().cpu()
                                return float(torch.isin(dv, tv).float().mean().item())

                            # Build matrices for each GT.
                            for gid in gt_targets:
                                det_idx = [dj for dj in det_ids_all if int(det_gt_cpu[int(dj)].item()) == int(gid)]
                                if not det_idx:
                                    continue
                                # Positive tracks: track_gt_ids == gid and strength >= threshold.
                                pos_tracks = []
                                for ti, tg in enumerate(track_gt_ids):
                                    if int(tg) != int(gid):
                                        continue
                                    if track_gt_strength is not None and float(track_gt_strength[ti]) < track_gt_min_strength:
                                        continue
                                    pos_tracks.append(ti)
                                if not pos_tracks:
                                    continue
                                # Negative tracks: not this GT.
                                neg_tracks = [ti for ti, tg in enumerate(track_gt_ids) if int(tg) != int(gid)]
                                if neg_tracks_k > 0:
                                    neg_tracks = neg_tracks[:neg_tracks_k]

                                # Matrices.
                                M_pos = np.zeros((len(det_idx), len(pos_tracks)), dtype=np.float32)
                                for ri, dj in enumerate(det_idx):
                                    dv = det_voxels[int(dj)]
                                    for ci, ti in enumerate(pos_tracks):
                                        M_pos[ri, ci] = _contain_det_track(dv, track_voxels[ti])
                                M_neg = None
                                if neg_tracks:
                                    M_neg = np.zeros((len(det_idx), len(neg_tracks)), dtype=np.float32)
                                    for ri, dj in enumerate(det_idx):
                                        dv = det_voxels[int(dj)]
                                        for ci, ti in enumerate(neg_tracks):
                                            M_neg[ri, ci] = _contain_det_track(dv, track_voxels[ti])
                                # Compute full asymmetric overlap matrix (i in j)
                                # M_tt[i, j] = |T_i \cap T_j| / |T_i|
                                M_tt = None
                                if len(pos_tracks) >= 2:
                                    M_tt = np.zeros((len(pos_tracks), len(pos_tracks)), dtype=np.float32)
                                    for i, ti in enumerate(pos_tracks):
                                        for j, tj in enumerate(pos_tracks):
                                            if i == j:
                                                M_tt[i, j] = 1.0
                                            else:
                                                # Use asymmetric containment directly
                                                M_tt[i, j] = _contain_det_track(track_voxels[ti], track_voxels[tj])

                                # Track-track negatives: pos_tracks vs sampled neg_tracks.
                                # Split into two matrices: M_tt_neg_ij (Pos in Neg) and M_tt_neg_ji (Neg in Pos)
                                M_tt_neg_ij = None
                                M_tt_neg_ji = None
                                pos_tracks_for_neg = pos_tracks
                                try:
                                    max_pos_tracks_for_neg = int(self.geom_diag_cfg.get("neg_pos_tracks_k", 16))
                                except Exception:
                                    max_pos_tracks_for_neg = 16
                                if max_pos_tracks_for_neg > 0 and len(pos_tracks_for_neg) > max_pos_tracks_for_neg:
                                    pos_tracks_for_neg = pos_tracks_for_neg[:max_pos_tracks_for_neg]
                                
                                if neg_tracks and pos_tracks_for_neg:
                                    M_tt_neg_ij = np.zeros((len(pos_tracks_for_neg), len(neg_tracks)), dtype=np.float32)
                                    M_tt_neg_ji = np.zeros((len(pos_tracks_for_neg), len(neg_tracks)), dtype=np.float32)
                                    for i, ti in enumerate(pos_tracks_for_neg):
                                        for j, tj in enumerate(neg_tracks):
                                            ci = _contain_det_track(track_voxels[ti], track_voxels[tj])
                                            cj = _contain_det_track(track_voxels[tj], track_voxels[ti])
                                            M_tt_neg_ij[i, j] = ci
                                            M_tt_neg_ji[i, j] = cj

                                # Optional: semantic similarity diagnostics (DINO / 3D pooled feats).
                                use_dino_diag = bool(self.geom_diag_cfg.get("diag_dino", False))
                                use_feat3d_diag = bool(self.geom_diag_cfg.get("diag_feat3d", False))
                                M_dino_pos = M_dino_neg = M_dino_tt = M_dino_tt_neg = None
                                M_feat3d_pos = M_feat3d_neg = M_feat3d_tt = M_feat3d_tt_neg = None

                                def _cos_mat(a, b):
                                    if a is None or b is None:
                                        return None
                                    if a.numel() == 0 or b.numel() == 0:
                                        return None
                                    an = a / (a.norm(dim=1, keepdim=True) + 1e-6)
                                    bn = b / (b.norm(dim=1, keepdim=True) + 1e-6)
                                    return (an @ bn.t()).detach().cpu().numpy()

                                # DINO cosine matrices.
                                if use_dino_diag:
                                    try:
                                        det_dino = det_dino_feats
                                        trk_dino = getattr(self.tracks, "dino_feats", None)
                                        if det_dino is not None and trk_dino is not None:
                                            det_sel = det_dino[det_idx] if det_idx else None
                                            trk_pos = trk_dino[pos_tracks] if pos_tracks else None
                                            trk_neg = trk_dino[neg_tracks] if neg_tracks else None
                                            trk_pos_neg = trk_dino[pos_tracks_for_neg] if pos_tracks_for_neg else None
                                            M_dino_pos = _cos_mat(det_sel, trk_pos)
                                            M_dino_neg = _cos_mat(det_sel, trk_neg)
                                            if trk_pos is not None and trk_pos.shape[0] >= 2:
                                                M_dino_tt = _cos_mat(trk_pos, trk_pos)
                                            M_dino_tt_neg = _cos_mat(trk_pos_neg, trk_neg)
                                    except Exception:
                                        pass

                                # 3D pooled feature cosine matrices.
                                if use_feat3d_diag:
                                    try:
                                        det_f3d = det_feat3d
                                        trk_f3d = getattr(self.tracks, "feat3d", None)
                                        if det_f3d is not None and trk_f3d is not None:
                                            det_sel = det_f3d[det_idx] if det_idx else None
                                            trk_pos = trk_f3d[pos_tracks] if pos_tracks else None
                                            trk_neg = trk_f3d[neg_tracks] if neg_tracks else None
                                            trk_pos_neg = trk_f3d[pos_tracks_for_neg] if pos_tracks_for_neg else None
                                            M_feat3d_pos = _cos_mat(det_sel, trk_pos)
                                            M_feat3d_neg = _cos_mat(det_sel, trk_neg)
                                            if trk_pos is not None and trk_pos.shape[0] >= 2:
                                                M_feat3d_tt = _cos_mat(trk_pos, trk_pos)
                                            M_feat3d_tt_neg = _cos_mat(trk_pos_neg, trk_neg)
                                    except Exception:
                                        pass

                                rec = {
                                    "frame_i": int(frame_i),
                                    "gt_id": int(gid),
                                    "det_idx": det_idx,
                                    "track_pos_idx": pos_tracks,
                                    "track_pos_id": track_ids[pos_tracks].detach().cpu().tolist() if torch.is_tensor(track_ids) else [],
                                    "track_neg_idx": neg_tracks,
                                    "track_neg_id": track_ids[neg_tracks].detach().cpu().tolist() if torch.is_tensor(track_ids) else [],
                                    "det_gt_conf": det_conf_t[det_idx].detach().cpu().tolist(),
                                    "track_gt_strength": [float(track_gt_strength[ti]) for ti in pos_tracks]
                                    if track_gt_strength is not None else [],
                                    "M_pos": M_pos,
                                    "M_neg": M_neg,
                                    "M_tt_ij": M_tt, # Renamed to indicate asymmetric (Row in Col)
                                    "track_pos_idx_neg": pos_tracks_for_neg,
                                    "M_tt_neg_ij": M_tt_neg_ij,
                                    "M_tt_neg_ji": M_tt_neg_ji,
                                    "M_dino_pos": M_dino_pos,
                                    "M_dino_neg": M_dino_neg,
                                    "M_dino_tt": M_dino_tt,
                                    "M_dino_tt_neg": M_dino_tt_neg,
                                    "M_feat3d_pos": M_feat3d_pos,
                                    "M_feat3d_neg": M_feat3d_neg,
                                    "M_feat3d_tt": M_feat3d_tt,
                                    "M_feat3d_tt_neg": M_feat3d_tt_neg,
                                }
                                self.geom_diag_records.append(rec)
                except Exception:
                    pass

            dino_cfg = {}
            try:
                tf = self.track_feat_cfg.get("dino", {}) if isinstance(self.track_feat_cfg, dict) else {}
                if isinstance(tf, dict):
                    dino_cfg = tf
            except Exception:
                dino_cfg = {}
            feat3d_cfg = {}
            try:
                tf3d = self.track_feat_cfg.get("feat3d", {}) if isinstance(self.track_feat_cfg, dict) else {}
                if isinstance(tf3d, dict):
                    feat3d_cfg = tf3d
            except Exception:
                feat3d_cfg = {}

            # Birth filtering: allow more detections for association, but be stricter when
            # spawning new tracks. This reduces pseudo-birth under one-to-one constraints.
            if isinstance(self.birth_cfg, dict) and no_merge_masks.any():
                try:
                    score_thr_birth = float(self.birth_cfg.get('score_thr', -1.0))
                except Exception:
                    score_thr_birth = -1.0
                try:
                    npoint_thr_birth = int(self.birth_cfg.get('npoint_thr', -1))
                except Exception:
                    npoint_thr_birth = -1
                if score_thr_birth > 0 or npoint_thr_birth > 0:
                    allow = torch.ones((int(next_masks.shape[0]),), device=next_masks.device, dtype=torch.bool)
                    if score_thr_birth > 0:
                        allow = allow & (next_scores >= score_thr_birth)
                    if npoint_thr_birth > 0:
                        pts = next_masks.sum(dim=1)
                        allow = allow & (pts > int(npoint_thr_birth))
                    suppress = no_merge_masks & (~allow)
                    if bool(suppress.any().item()):
                        drop_det_idx = torch.nonzero(suppress, as_tuple=False).reshape(-1).detach().cpu().tolist()
                        drop_cnt = int(len(drop_det_idx))
                        no_merge_masks[suppress] = False
            birth_cnt, birth_det_idx, birth_track_ids = self.tracks.apply_assignment_and_update(
                points_per_mask=points_per_mask,
                row_ind=row_ind,
                col_ind=col_ind,
                temp_masks=temp_masks,
                next_masks=next_masks,
                no_merge_masks=no_merge_masks,
                next_labels=next_labels,
                next_scores=next_scores,
                next_queries=next_queries,
                next_query_feats=next_query_feats,
                next_sem_preds=next_sem_preds,
                next_xyz=next_xyz,
                det_dino_feats=det_dino_feats,
                dino_cfg=dino_cfg,
                det_feat3d=det_feat3d,
                feat3d_cfg=feat3d_cfg,
            )
            # Extend GT votes for newly born tracks and update voxel cache.
            try:
                self.tracks._append_track_gt_votes(int(birth_cnt))
            except Exception:
                pass
            # Tentative birth (delayed confirmation): mark new births as tentative here,
            # but defer prune/confirm until after all index-based updates (e.g. voxel cache).
            tent_enable = False
            tent_window = 0
            tent_min_hits = 0
            try:
                tent_cfg = self.birth_cfg.get("tentative", {}) if isinstance(self.birth_cfg, dict) else {}
                tent_enable = bool(tent_cfg.get("enable", False))
                tent_window = int(tent_cfg.get("window", 5))
                tent_min_hits = int(tent_cfg.get("min_hits", 4))
                if tent_enable and int(birth_cnt) > 0:
                    self.tracks.mark_last_births_tentative(birth_cnt=int(birth_cnt))
                    tentative_created = int(birth_cnt)
            except Exception:
                tent_enable = False
            try:
                if det_voxels is not None:
                    cap = int(self.geom_diag_cfg.get("V_trk_cap", 1024))
                    self.tracks.update_track_voxels(
                        row_ind=row_ind,
                        col_ind=col_ind,
                        det_voxels=det_voxels,
                        birth_det_idx=birth_det_idx,
                        cap=cap,
                    )
            except Exception:
                pass
            # Update voxel cache for rescue writes (optional).
            try:
                if rescue_mem_idx and rescue_det_idx and rescue_is_l:
                    cap_r = int(self.rescue_cfg.get("V_trk_cap", self.geom_diag_cfg.get("V_trk_cap", 1024)))
                    # H rescues (existing det set): update from det_voxels.
                    if det_voxels is not None:
                        mem_h = [m for m, is_l in zip(rescue_mem_idx, rescue_is_l) if not is_l]
                        det_h = [d for d, is_l in zip(rescue_det_idx, rescue_is_l) if not is_l]
                        if mem_h and det_h:
                            self.tracks.update_track_voxels_pairs(
                                mem_idx=mem_h,
                                det_idx=det_h,
                                det_voxels=det_voxels,
                                cap=cap_r,
                            )
                    # L rescues (pre-pool): update from rescue_voxels if provided.
                    if rescue_voxels is not None and isinstance(rescue_voxels, list):
                        mem_l = [m for m, is_l in zip(rescue_mem_idx, rescue_is_l) if is_l]
                        det_l = [d for d, is_l in zip(rescue_det_idx, rescue_is_l) if is_l]
                        if mem_l and det_l:
                            self.tracks.update_track_voxels_pairs(
                                mem_idx=mem_l,
                                det_idx=det_l,
                                det_voxels=rescue_voxels,
                                cap=cap_r,
                            )
            except Exception:
                pass

            # Now it's safe to prune/confirm tentative tracks (may change indexing).
            try:
                if tent_enable:
                    tentative_confirmed, tentative_deleted = self.tracks.prune_tentatives(
                        window=int(tent_window), min_hits=int(tent_min_hits)
                    )
            except Exception:
                pass

            # Optional: track-to-track dedup (periodic, hard gating).
            dedup_enable = bool(self.dedup_cfg.get("enable", False))
            dedup_stats = {"pairs_checked": 0, "pairs_merged": 0, "reject_no_feat": 0, "reject_no_vox": 0, "reject_geom": 0}
            if dedup_enable:
                try:
                    interval = int(self.dedup_cfg.get("interval", 10))
                except Exception:
                    interval = 10
                # Skip fi=0 (first frame) by default.
                if interval <= 0:
                    dedup_enable = False
                if int(self.tracks.fi) <= 0:
                    dedup_enable = False
                if interval > 0 and (int(self.tracks.fi) % int(interval)) != 0:
                    dedup_enable = False

            if dedup_enable:
                try:
                    # Only dedup confirmed tracks by default.
                    confirmed = self.tracks.get_confirmed_mask()
                    if confirmed is None:
                        confirmed = torch.ones((int(self.tracks.num_tracks),), device=next_masks.device, dtype=torch.bool)
                    idx_all = torch.nonzero(confirmed, as_tuple=False).reshape(-1).detach().cpu().tolist()
                    if len(idx_all) < 2:
                        dedup_enable = False
                except Exception:
                    dedup_enable = False

            if dedup_enable:
                try:
                    tv = getattr(self.tracks, "track_voxels", None)
                    trk_dino = getattr(self.tracks, "dino_feats", None)
                    trk_f3d = getattr(self.tracks, "feat3d", None)
                    if tv is None or trk_dino is None or trk_f3d is None:
                        dedup_stats["reject_no_vox"] += 1
                        dedup_enable = False
                    else:
                        # thresholds
                        contain_thr = float(self.dedup_cfg.get("contain_thr", 0.20))
                        dino_thr = float(self.dedup_cfg.get("dino_cos_thr", 0.80))
                        f3d_thr = float(self.dedup_cfg.get("feat3d_cos_thr", 0.85))
                        pre_iou_thr = float(self.dedup_cfg.get("prefilter_iou_thr", 0.01))
                        pre_center_thr = float(self.dedup_cfg.get("prefilter_center_thr", 1.0))
                        voxel_cap = int(self.dedup_cfg.get("V_trk_cap", self.geom_diag_cfg.get("V_trk_cap", 1024)))

                        # Move feature tensors to local device.
                        trk_dino = trk_dino.to(device=next_masks.device)
                        trk_f3d = trk_f3d.to(device=next_masks.device)

                        # Candidate prefilter pairs.
                        cand_pairs = []
                        xyz = getattr(self.tracks, "xyz", None)
                        if torch.is_tensor(xyz) and int(xyz.shape[0]) >= 2:
                            if int(xyz.shape[-1]) == 6 and self.use_bbox:
                                try:
                                    # AABB IoU between confirmed tracks.
                                    b = xyz[torch.as_tensor(idx_all, device=xyz.device)]
                                    iou_tt = self.iou_calculator(b, b, is_aligned=False)
                                    for a_i, mi in enumerate(idx_all):
                                        for b_i, mj in enumerate(idx_all):
                                            if mj <= mi:
                                                continue
                                            if float(iou_tt[a_i, b_i].item()) > pre_iou_thr:
                                                cand_pairs.append((mi, mj))
                                except Exception:
                                    cand_pairs = []
                            else:
                                # Center distance prefilter.
                                try:
                                    c = xyz[torch.as_tensor(idx_all, device=xyz.device)][:, :3]
                                    d = torch.cdist(c, c, p=2)
                                    for a_i, mi in enumerate(idx_all):
                                        for b_i, mj in enumerate(idx_all):
                                            if mj <= mi:
                                                continue
                                            if float(d[a_i, b_i].item()) < pre_center_thr:
                                                cand_pairs.append((mi, mj))
                                except Exception:
                                    cand_pairs = []
                        else:
                            # Fallback: brute-force on confirmed indices.
                            for ii, mi in enumerate(idx_all):
                                for mj in idx_all[ii + 1 :]:
                                    cand_pairs.append((mi, mj))

                        # Helper: directional containment between tracks (hashed voxels).
                        def _contain(a: torch.Tensor, b: torch.Tensor) -> float:
                            if a.numel() == 0 or b.numel() == 0:
                                return 0.0
                            return float(torch.isin(a, b).float().mean().item())

                        removed = set()
                        for mi, mj in cand_pairs:
                            if mi in removed or mj in removed:
                                continue
                            if mi < 0 or mj < 0 or mi >= len(tv) or mj >= len(tv):
                                continue
                            dedup_stats["pairs_checked"] += 1
                            # Require same instance label (robust guard in non-CA mode).
                            try:
                                if self.tracks.labels is not None and int(self.tracks.labels[mi].item()) != int(self.tracks.labels[mj].item()):
                                    dedup_stats["reject_geom"] += 1
                                    continue
                            except Exception:
                                pass
                            # Geometry gate: either directional containment passes.
                            ci = _contain(tv[mi], tv[mj])
                            cj = _contain(tv[mj], tv[mi])
                            if max(ci, cj) < contain_thr:
                                dedup_stats["reject_geom"] += 1
                                continue
                            # Feature gates (AND).
                            cd = float((trk_dino[mi] * trk_dino[mj]).sum().item())
                            c3 = float((trk_f3d[mi] * trk_f3d[mj]).sum().item())
                            if cd < dino_thr or c3 < f3d_thr:
                                dedup_stats["reject_no_feat"] += 1
                                continue
                            # Merge: keep higher-score track for stability.
                            keep_idx, drop_idx = mi, mj
                            try:
                                if self.tracks.scores is not None and float(self.tracks.scores[mj].item()) > float(self.tracks.scores[mi].item()):
                                    keep_idx, drop_idx = mj, mi
                            except Exception:
                                pass
                            self.tracks.merge_tracks_inplace(keep_idx=int(keep_idx), drop_idx=int(drop_idx), voxel_cap=int(voxel_cap))
                            removed.add(int(drop_idx))
                            dedup_stats["pairs_merged"] += 1

                        if removed:
                            keep = torch.ones((int(self.tracks.num_tracks),), device=next_masks.device, dtype=torch.bool)
                            for di in removed:
                                if 0 <= di < int(keep.numel()):
                                    keep[di] = False
                            self.tracks._prune_by_mask(keep)
                except Exception:
                    pass
            
        include_tentative_in_output = True
        try:
            tent_cfg = self.birth_cfg.get("tentative", {}) if isinstance(self.birth_cfg, dict) else {}
            if bool(tent_cfg.get("enable", False)):
                include_tentative_in_output = bool(tent_cfg.get("include_in_output", False))
        except Exception:
            include_tentative_in_output = True
        kept_ins, cur_masks, cur_labels, cur_scores, cur_queries, cur_bboxes = self.tracks.select_topk_outputs(
            topk=self.inscat_topk_insts, use_bbox=self.use_bbox, include_tentative=include_tentative_in_output
        )
        self.output_track_ids = self.tracks.output_track_ids
        # Save semantic top1+confidence for downstream analysis/gating (independent of CA mode).
        try:
            sem = getattr(self.tracks, "sem_preds", None)
            if torch.is_tensor(sem):
                logits = sem[kept_ins]
                probs = torch.softmax(logits, dim=-1)
                bg = probs[:, -1]
                fg = probs[:, :-1]
                conf, label = torch.max(fg, dim=1)
                self.output_sem_labels = label.detach().cpu().numpy().astype(np.int64)
                self.output_sem_confs = conf.detach().cpu().numpy().astype(np.float32)
                self.output_bg_confs = bg.detach().cpu().numpy().astype(np.float32)
        except Exception:
            self.output_sem_labels = None
            self.output_sem_confs = None
            self.output_bg_confs = None

        # cur_labels = torch.zeros_like(self.cur_scores).long()

        if self.monitor:
            def _stat(vals):
                if not vals:
                    return {"mean": 0.0, "p50": 0.0, "p90": 0.0}
                try:
                    x = np.asarray(vals, dtype=np.float32)
                    return {
                        "mean": float(np.mean(x)),
                        "p50": float(np.percentile(x, 50)),
                        "p90": float(np.percentile(x, 90)),
                    }
                except Exception:
                    return {"mean": 0.0, "p50": 0.0, "p90": 0.0}

            mem_full = int(self.tracks.num_tracks)
            mem_kept = int(kept_ins.numel()) if torch.is_tensor(kept_ins) else int(len(kept_ins))
            self.last_stats = {
                "fi": int(self.tracks.fi),
                "det_to_merge": int(det_to_merge),
                "matched": int(matched_cnt),
                "birth": int(birth_cnt),
                "drop": int(drop_cnt),
                "absorbed": int(absorbed_cnt),
                "supporters": int(supporters_cnt),
                "rescued": int(rescued_cnt),
                "rescued_from_l": int(rescued_from_l),
                "rescued_from_uh": int(rescued_from_uh),
                "rescue_ambiguous": int(rescue_ambiguous),
                "rescue_reject": dict(rescue_reject),
                "rescue_delta_mask_pts": _stat(rescue_delta_mask_pts),
                "rescue_new_vox_ratio": _stat(rescue_new_vox_ratio),
                "rescue_new_vox_cnt": _stat(rescue_new_vox_cnt),
                "rescue_det_vox_cnt": _stat(rescue_det_vox_cnt),
                "rescue_trk_vox_cnt": _stat(rescue_trk_vox_cnt),
                "rescue_gate_contain": _stat(rescue_det_contain),
                "rescue_gate_dino": _stat(rescue_det_dino_cos),
                "rescue_gate_feat3d": _stat(rescue_det_feat3d_cos),
                "rescue_det_purity": _stat(rescue_det_purity),
                "rescue_det_cov": _stat(rescue_det_cov),
                "rescue_det_iou": _stat(rescue_det_iou),
                # L-pool portrait (pre-gate).
                "l_pool_n": int(l_pool_n),
                "l_pool_npoints": _stat(l_pool_npoints),
                "l_pool_det_purity": _stat(l_pool_det_purity),
                "l_pool_det_cov": _stat(l_pool_det_cov),
                "l_pool_det_iou": _stat(l_pool_det_iou),
                "l_pool_useful_any05_rate": float(l_pool_useful_any05_rate),
                "l_pool_reject": dict(l_pool_reject),
                "mem_size_prev": int(prev_mem_size),
                "mem_size_full": int(mem_full),
                "mem_size_kept": int(mem_kept),
                "topk_drop": int(max(mem_full - mem_kept, 0)),
                "matched_mem_idx": matched_mem_idx,
                "matched_det_idx": matched_det_idx,
                "matched_track_ids": matched_track_ids,
                "birth_det_idx": birth_det_idx,
                "birth_track_ids": birth_track_ids,
                "drop_det_idx": drop_det_idx,
                "absorbed_mem_idx": absorbed_mem_idx,
                "absorbed_det_idx": absorbed_det_idx,
                "supporters_mem_idx": supporters_mem_idx,
                "supporters_det_idx": supporters_det_idx,
                "rescue_mem_idx": rescue_mem_idx,
                "rescue_det_idx": rescue_det_idx,
                "tentative_created": int(tentative_created),
                "tentative_confirmed": int(tentative_confirmed),
                "tentative_deleted": int(tentative_deleted),
            }
            if isinstance(self.output_track_ids, list):
                if self.record_kept_track_ids:
                    self.last_stats["kept_track_ids"] = list(self.output_track_ids)
            if isinstance(dedup_stats, dict):
                self.last_stats["dedup"] = dict(dedup_stats)
        return cur_masks, cur_labels, cur_scores, cur_queries, cur_bboxes
    
    @staticmethod
    def _bbox_pred_to_bbox(points, bbox_pred):
        """Transform predicted bbox parameters to bbox.
        """
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + bbox_pred[:, 0]
        y_center = points[:, 1] + bbox_pred[:, 1]
        z_center = points[:, 2] + bbox_pred[:, 2]
        bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 3],
            bbox_pred[:, 4],
            bbox_pred[:, 5]], -1)

        # axis-aligned case: x, y, z, w, h, l -> x1, y1, z1, x2, y2, z2
        return torch.stack(
            (bbox[..., 0] - bbox[..., 3] / 2, bbox[..., 1] - bbox[..., 4] / 2,
             bbox[..., 2] - bbox[..., 5] / 2, bbox[..., 0] + bbox[..., 3] / 2,
             bbox[..., 1] + bbox[..., 4] / 2, bbox[..., 2] + bbox[..., 5] / 2),
            dim=-1)


class InstanceQuery():
    def __init__(self, mask, label, score, query):
        self.mask = mask
        self.label = label
        self.score = score
        self.query = query
        self.merge_count = 1
    
    def pad(self, pts_num):
        self.mask = torch.cat([self.mask, self.mask.new_zeros(pts_num).bool()])
    
    def compare(self, cur_points, points, mask, label, score, query, pts_thr=0.05, thr=0.1):
        if cur_points.shape[0] != len(self.mask):
            return False
        if self.label != label:
            return False
        cur_xyz = cur_points[self.mask, :3].unsqueeze(1) # Mx3
        if cur_xyz.shape[0] > 10000:
            sample_idx = torch.randperm(cur_xyz.shape[0])[:10000]
            cur_xyz = cur_xyz[sample_idx]
        xyz = points[mask, :3].unsqueeze(0) # Nx3
        if xyz.shape[0] > 10000:
            sample_idx = torch.randperm(xyz.shape[0])[:10000]
            xyz = xyz[sample_idx]
        dist_mat = cur_xyz - xyz # MxNx3
        dist_mat = (dist_mat ** 2).sum(-1).sqrt() # MxN
        min_dist1 = dist_mat.min(-1).values # M
        min_dist2 = dist_mat.min(0).values # N
        ratio1 = (min_dist1 < pts_thr).sum() / len(min_dist1)
        ratio2 = (min_dist2 < pts_thr).sum() / len(min_dist2)
        if max(ratio1, ratio2) > thr:
            return True
        else:
            return False
    
    def merge(self, mask, label, score, query, frame_i):
        self.mask = torch.cat([self.mask, mask])
        self.score = (self.score * frame_i + score) / (frame_i + 1)
        self.query = (self.query * frame_i + query) / (frame_i + 1)
        self.merge_count += 1
