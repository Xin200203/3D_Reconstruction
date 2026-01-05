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
        if self.use_bbox:
            self.iou_calculator = AxisAlignedBboxOverlaps3D()
        # 初始化跨帧 Transformer
        if tformer_cfg is not None:
            self.tformer = MODELS.build(tformer_cfg)
        else:
            self.tformer = None
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.cur_bboxes = None
        self.fi = 0
        self.merge_counts = None
    
    def clean(self):
        self.cur_masks = None
        self.cur_labels = None
        self.cur_scores = None
        self.cur_queries = None
        self.cur_query_feats = None
        self.cur_sem_preds = None
        self.cur_xyz = None
        self.cur_bboxes = None
        self.merge_counts = None
        self.last_stats = None
    
    def merge(self, masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes):
        # Online behavior stats (optional; written to self.last_stats).
        det_to_merge = int(masks.shape[0])
        prev_mem_size = int(self.cur_scores.shape[0]) if self.cur_scores is not None else 0
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

        points_per_mask = masks.shape[1]
        # masks, labels, scores, queries, query_feats, sem_preds, xyz_list = \
        #     self.intra_frame_merge(masks, labels, scores, queries, query_feats, sem_preds, xyz_list, bboxes, q)
        if self.cur_masks is None:
            self.cur_masks = masks
            self.cur_labels = labels
            self.cur_scores = scores
            self.cur_queries = queries
            self.cur_query_feats = query_feats
            self.cur_sem_preds = sem_preds
            self.cur_xyz = self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            if self.use_bbox:
                self.cur_bboxes = bboxes
            self.merge_counts = torch.zeros_like(scores).long()
        else:
            # Static typing：确保前一帧已经初始化完毕
            assert self.cur_labels is not None and self.cur_query_feats is not None and \
                   self.cur_sem_preds is not None and self.cur_xyz is not None and \
                   self.merge_counts is not None and self.cur_queries is not None and \
                   self.cur_masks is not None and self.cur_scores is not None
            self.fi += 1
            next_masks, next_labels, next_scores, next_queries, next_query_feats, next_sem_preds, next_xyz = \
                masks, labels, scores, queries, query_feats, sem_preds, \
                self._bbox_pred_to_bbox(xyz_list, bboxes) if self.use_bbox else xyz_list
            # 步骤1: IoU预剪枝 (统一计算，TDT和传统方法共用)
            if self.use_bbox:
                iou_matrix = self.iou_calculator(self.cur_xyz, next_xyz, is_aligned=False)
                # IoU计算器返回的是(Memory, Current) = (Nm, Nc)
                xyz_scores = iou_matrix
            else:
                xyz_dists = torch.cdist(self.cur_xyz, next_xyz, p=2)
                # cdist返回的是(Memory, Current) = (Nm, Nc)
                xyz_scores = 1 / (xyz_dists + 1e-6)
            
            # 预剪枝掩码（Memory x Current）= (Nm, Nc)
            attention_mask_mem_cur = xyz_scores > self.iou_thr  # True=允许匹配，False=禁止
            # Transformer attention mask 常见约定为 (B, tgt, src) = (1, Nc, Nm)
            attention_mask_tgt_src = attention_mask_mem_cur.T.unsqueeze(0)
            
            if self.tformer is not None and self.cur_queries is not None:
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

                p_m = build_geom(self.cur_xyz) if self.cur_xyz is not None else None
                p_c = build_geom(next_xyz) if next_xyz is not None else None
                if p_m is None or p_c is None:
                    # fallback to zeros
                    p_m = torch.zeros(self.cur_queries.shape[0], 9, device=self.cur_queries.device)
                    p_c = torch.zeros(next_queries.shape[0], 9, device=next_queries.device)

                attn_mat, updated_queries = self.tformer(
                    next_queries.unsqueeze(0), 
                    self.cur_queries.unsqueeze(0),
                    p_c.unsqueeze(0), 
                    p_m.unsqueeze(0),
                    mask_mem=torch.ones(1, self.cur_queries.shape[0], dtype=torch.bool, device=next_queries.device),
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
                if self.cur_query_feats is None or next_query_feats is None:
                    raise RuntimeError('query_feats is None when merging instances')
                if self.cur_sem_preds is None or next_sem_preds is None:
                    raise RuntimeError('sem_preds is None when merging instances')
                if self.cur_xyz is None or next_xyz is None:
                    raise RuntimeError('xyz is None when merging instances')

                # Keep matrix layout consistent with xyz_scores: (Nm, Nc)
                query_feat_scores = (self.cur_query_feats.unsqueeze(1) * next_query_feats.unsqueeze(0)).sum(2)
                sem_pred_scores = F.cosine_similarity(
                    next_sem_preds.unsqueeze(1), self.cur_sem_preds.unsqueeze(0), dim=2)

                mix_scores = query_feat_scores * xyz_scores
                # 应用IoU预剪枝mask
                mix_scores = torch.where(attention_mask_mem_cur, mix_scores, torch.zeros_like(mix_scores))
            
            # 确保标签匹配矩阵的维度与mix_scores一致 (Nm, Nc)
            inst_label_scores = torch.where(
                self.cur_labels.unsqueeze(1) == next_labels.unsqueeze(0),
                torch.ones((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device),
                torch.zeros((self.cur_labels.shape[0], next_labels.shape[0])).to(self.cur_labels.device)
            )
            
            mix_scores = torch.where(mix_scores > 0, mix_scores, torch.zeros_like(mix_scores))
            mix_scores = mix_scores * inst_label_scores

            # Hungarian assign (supports rectangular matrices)
            row_ind, col_ind = linear_sum_assignment(-mix_scores.detach().cpu())
            row_ind = torch.tensor(row_ind).to(mix_scores.device)
            col_ind = torch.tensor(col_ind).to(mix_scores.device)
            
            # 只保留有效的匹配分数
            if len(row_ind) > 0:
                mix_scores_mask = mix_scores[row_ind, col_ind].gt(0)
                row_ind = row_ind[mix_scores_mask]
                col_ind = col_ind[mix_scores_mask]

            # Online matching statistics (post-filter).
            matched_cnt = int(row_ind.numel())
            matched_mem_idx = row_ind.detach().cpu().tolist()
            matched_det_idx = col_ind.detach().cpu().tolist()

            temp = torch.zeros(self.cur_masks.shape[0]).bool().to(self.cur_masks.device)
            temp[row_ind] = True
            temp = temp.unsqueeze(1)
            temp_masks = torch.zeros((self.cur_masks.shape[0], points_per_mask)).bool().to(self.cur_masks.device)
            temp_masks[row_ind] = next_masks[col_ind]
            no_merge_masks = torch.ones(next_masks.shape[0]).bool().to(next_masks.device)
            no_merge_masks[col_ind] = False

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
                                if pts < min_pts or pts > max_pts:
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
                                        self.cur_scores[best_i] = torch.maximum(self.cur_scores[best_i], next_scores[int(dj)])
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
                    if center_norm_thr > 0 and self.use_bbox and self.cur_xyz is not None and next_xyz is not None \
                            and self.cur_xyz.numel() > 0 and next_xyz.numel() > 0 \
                            and int(self.cur_xyz.shape[-1]) == 6 and int(next_xyz.shape[-1]) == 6:
                        mem_center, mem_diag = _center_and_diag(self.cur_xyz)  # (Nm,3),(Nm,)
                        det_center, _ = _center_and_diag(next_xyz)  # (Nc,3)
                    else:
                        mem_center = mem_diag = det_center = None

                    supported_this = 0
                    for dj in cand_det.tolist():
                        if max_support > 0 and supported_this >= max_support:
                            break
                        pm = next_masks[int(dj)]
                        pts_det = int(pm.sum().item())
                        if pts_det < min_pts or pts_det > max_pts:
                            continue
                        if float(next_scores[int(dj)].item()) < score_thr:
                            continue

                        if mode == 'bbox_contain' and self.use_bbox and self.cur_xyz is not None and next_xyz is not None \
                                and int(self.cur_xyz.shape[-1]) == 6 and int(next_xyz.shape[-1]) == 6:
                            # BBox containment-based supporters (calibration-friendly).
                            det_bb = next_xyz[int(dj)]  # (6,)
                            mem_bb = self.cur_xyz  # (Nm,6)

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

            next_masks_ = torch.where(temp, temp_masks,
                                     torch.zeros((self.cur_masks.shape[0],points_per_mask)).bool().to(next_masks.device))
            self.cur_masks = torch.cat((self.cur_masks, next_masks_), dim=1)

            birth_cnt = int(no_merge_masks.sum().item()) if no_merge_masks.numel() else 0
            former_padding = torch.zeros((no_merge_masks.nonzero().shape[0], points_per_mask * self.fi)).bool().to(next_masks.device)
            new_masks = torch.cat((former_padding, next_masks[no_merge_masks]), dim=1)
            self.cur_masks = torch.cat((self.cur_masks, new_masks), dim=0)

            self.merge_counts[row_ind] += 1  # type: ignore[index]
            if len(no_merge_masks) > 0:
                self.merge_counts = torch.cat((self.merge_counts,
                     torch.zeros(no_merge_masks.shape[0], dtype=torch.long, device=self.merge_counts.device)), dim=0)
            
            if self.merge_type == 'count':
                count = self.merge_counts[row_ind]
            else: count = self.fi
            
            self.cur_scores[row_ind] = (self.cur_scores[row_ind] * count + next_scores[col_ind]) / (count + 1)
            self.cur_scores = torch.cat((self.cur_scores, next_scores[no_merge_masks]), dim=0)
            if self.merge_type == 'count':
                count = count.unsqueeze(-1)  # type: ignore[attr-defined]
            self.cur_labels = torch.cat((self.cur_labels, next_labels[no_merge_masks]), dim=0)
            self.cur_queries[row_ind] = (self.cur_queries[row_ind] * count + next_queries[col_ind]) / (count + 1)
            self.cur_queries = torch.cat((self.cur_queries, next_queries[no_merge_masks]), dim=0)
            self.cur_query_feats[row_ind] = (self.cur_query_feats[row_ind] * count + next_query_feats[col_ind]) / (count + 1)
            self.cur_query_feats = torch.cat((self.cur_query_feats, next_query_feats[no_merge_masks]), dim=0)
            self.cur_sem_preds[row_ind] = (self.cur_sem_preds[row_ind] * count + next_sem_preds[col_ind]) / (count + 1)
            self.cur_sem_preds = torch.cat((self.cur_sem_preds, next_sem_preds[no_merge_masks]), dim=0)
            self.cur_xyz[row_ind] = (self.cur_xyz[row_ind] * count + next_xyz[col_ind]) / (count + 1)
            self.cur_xyz = torch.cat((self.cur_xyz, next_xyz[no_merge_masks]), dim=0)
            
        if len(self.cur_scores) > self.inscat_topk_insts:
            _, kept_ins = self.cur_scores.topk(self.inscat_topk_insts)
        else:
            kept_ins = torch.arange(self.cur_scores.shape[0], device=self.cur_scores.device)
        cur_masks, cur_scores = self.cur_masks[kept_ins], self.cur_scores[kept_ins]
        cur_labels = self.cur_labels[kept_ins]
        cur_queries = self.cur_queries[kept_ins]
        cur_bboxes = self.cur_xyz[kept_ins] if self.use_bbox else None

        # cur_labels = torch.zeros_like(self.cur_scores).long()

        if self.monitor:
            mem_full = int(self.cur_scores.shape[0]) if self.cur_scores is not None else 0
            mem_kept = int(kept_ins.numel()) if torch.is_tensor(kept_ins) else int(len(kept_ins))
            self.last_stats = {
                "fi": int(self.fi),
                "det_to_merge": int(det_to_merge),
                "matched": int(matched_cnt),
                "birth": int(birth_cnt),
                "absorbed": int(absorbed_cnt),
                "supporters": int(supporters_cnt),
                "mem_size_prev": int(prev_mem_size),
                "mem_size_full": int(mem_full),
                "mem_size_kept": int(mem_kept),
                "topk_drop": int(max(mem_full - mem_kept, 0)),
                "matched_mem_idx": matched_mem_idx,
                "matched_det_idx": matched_det_idx,
                "absorbed_mem_idx": absorbed_mem_idx,
                "absorbed_det_idx": absorbed_det_idx,
                "supporters_mem_idx": supporters_mem_idx,
                "supporters_det_idx": supporters_det_idx,
            }
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
