import torch
import torch.nn as nn
import pdb, time
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from torch_scatter import scatter_mean, scatter_add


class CrossAttentionLayer(BaseModule):
    """Cross attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout, fix=False):
        super().__init__()
        self.fix = fix
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        """Init weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, sources, queries, attn_masks=None):
        """Forward pass.

        Args:
            sources (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).
            queries (List[Tensor]): of len batch_size,
                each of shape(n_queries_i, d_model).
            attn_masks (List[Tensor] or None): of len batch_size,
                each of shape (n_queries, n_points).
        
        Return:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        outputs = []
        for i in range(len(sources)):
            k = v = sources[i]
            attn_mask = attn_masks[i] if attn_masks is not None else None
            # MultiheadAttention: bool mask where True means "masked".
            # 若某一 query 行全为 True，会导致 softmax 全 -inf -> NaN。
            if attn_mask is not None:
                attn_mask = attn_mask.to(device=queries[i].device)
                if attn_mask.dtype is not torch.bool:
                    attn_mask = attn_mask.bool()
                if attn_mask.ndim == 2:
                    all_true = attn_mask.all(dim=1)
                    if all_true.any():
                        attn_mask = attn_mask.clone()
                        attn_mask[all_true] = False
            output, _ = self.attn(queries[i], k, v, attn_mask=attn_mask)
            if self.fix:
                output = self.dropout(output)
            output = output + queries[i]
            if self.fix:
                output = self.norm(output)
            outputs.append(output)
        return outputs


class SelfAttentionLayer(BaseModule):
    """Self attention layer.

    Args:
        d_model (int): Model dimension.
        num_heads (int): Number of heads.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model, num_heads, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z, _ = self.attn(y, y, y)
            z = self.dropout(z) + y
            z = self.norm(z)
            out.append(z)
        return out


class FFN(BaseModule):
    """Feed forward network.

    Args:
        d_model (int): Model dimension.
        hidden_dim (int): Hidden dimension.
        dropout (float): Dropout rate.
        activation_fn (str): 'relu' or 'gelu'.
    """

    def __init__(self, d_model, hidden_dim, dropout, activation_fn):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.ReLU() if activation_fn == 'relu' else nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """Forward pass.

        Args:
            x (List[Tensor]): Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        
        Returns:
            List[Tensor]: Queries of len batch_size,
                each of shape(n_queries_i, d_model).
        """
        out = []
        for y in x:
            z = self.net(y)
            z = z + y
            z = self.norm(z)
            out.append(z)
        return out


class QueryDecoder(BaseModule):
    """Query decoder for SPFormer.

    Args:
        num_layers (int): Number of transformer layers.
        num_instance_queries (int): Number of instance queries.
        num_semantic_queries (int): Number of semantic queries.
        num_classes (int): Number of classes.
        in_channels (int): Number of input channels.
        d_model (int): Number of channels for model layers.
        num_heads (int): Number of head in attention layer.
        hidden_dim (int): Dimension of attention layer.
        dropout (float): Dropout rate for transformer layer.
        activation_fn (str): 'relu' of 'gelu'.
        iter_pred (bool): Whether to predict iteratively.
        attn_mask (bool): Whether to use mask attention.
        pos_enc_flag (bool): Whether to use positional enconding.
    """

    def __init__(self, num_layers, num_instance_queries, num_semantic_queries,
                 num_classes, in_channels, d_model, num_heads, hidden_dim,
                 dropout, activation_fn, iter_pred, attn_mask, fix_attention,
                 objectness_flag, **kwargs):
        super().__init__()
        self.objectness_flag = objectness_flag
        self.expected_in_channels = in_channels  # 保存配置的预期维度
        self.d_model = d_model
        
        # 修复：在初始化时直接创建input_proj，避免延迟初始化导致的权重加载问题
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, d_model), 
            nn.LayerNorm(d_model), 
            nn.ReLU()
        )
        self.input_adapter = None  # 保留适配器接口，但默认不使用
        
        if num_instance_queries + num_semantic_queries > 0:
            self.query = nn.Embedding(num_instance_queries + num_semantic_queries, d_model)
        if num_instance_queries == 0:
            # query_proj也需要类似处理，但先用原始逻辑，出错时再修复
            self.query_proj = nn.Sequential(
                nn.Linear(in_channels, d_model), nn.ReLU(),
                nn.Linear(d_model, d_model))
        self.cross_attn_layers = nn.ModuleList([])
        self.self_attn_layers = nn.ModuleList([])
        self.ffn_layers = nn.ModuleList([])
        for i in range(num_layers):
            self.cross_attn_layers.append(
                CrossAttentionLayer(
                    d_model, num_heads, dropout, fix_attention))
            self.self_attn_layers.append(
                SelfAttentionLayer(d_model, num_heads, dropout))
            self.ffn_layers.append(
                FFN(d_model, hidden_dim, dropout, activation_fn))
        self.out_norm = nn.LayerNorm(d_model)
        self.out_cls = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, num_classes + 1))
        if objectness_flag:
            self.out_score = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(
            nn.Linear(in_channels, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model))
        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
    
    def _get_queries(self, queries=None, batch_size=None):
        """Get query tensor.

        Args:
            queries (List[Tensor], optional): of len batch_size,
                each of shape (n_queries_i, in_channels).
            batch_size (int, optional): batch size.
        
        Returns:
            List[Tensor]: of len batch_size, each of shape
                (n_queries_i, d_model).
        """
        if batch_size is None:
            if queries is not None:
                batch_size = len(queries)
            else:
                batch_size = 1  # 默认batch_size
        
        result_queries = []
        for i in range(batch_size):
            result_query = []
            if hasattr(self, 'query'):
                result_query.append(self.query.weight)
            if queries is not None:
                q_in = queries[i]
                if self.input_adapter is not None and q_in.shape[-1] != self.expected_in_channels:
                    q_in = self.input_adapter(q_in)
                result_query.append(self.query_proj(q_in))
            result_queries.append(torch.cat(result_query))
        return result_queries

    def _forward_head(self, queries, mask_feats, *args, **kwargs):
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, pred_scores, pred_masks, attn_masks = [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        return cls_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries=None, **kwargs):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats)
        return dict(
            cls_preds=cls_preds,
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries=None, **kwargs):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and aux_outputs.
        """
        cls_preds, pred_scores, pred_masks = [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats)
        cls_preds.append(cls_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            cls_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats)
            cls_preds.append(cls_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            {'cls_preds': cls_pred, 'masks': masks, 'scores': scores}
            for cls_pred, scores, masks in zip(
                cls_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)

    def forward(self, x, queries=None, **kwargs):
        """Forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self.forward_iter_pred(x, queries)
        else:
            return self.forward_simple(x, queries)


@MODELS.register_module()
class ScanNetQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, **kwargs)
        assert num_semantic_linears in [1, 2]
        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, last_flag=False, **kwargs):  # type: ignore[override]
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks = [], [], [], [], []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag \
                else None
            pred_scores.append(pred_score)
            pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks

    def forward_simple(self, x, queries=None, **kwargs):
        """Simple forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, and scores.
        """
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
        cls_preds, sem_preds, pred_scores, pred_masks, _ = self._forward_head(
            queries, mask_feats, last_flag=True)
        return dict(
            cls_preds=cls_preds,
            sem_preds=sem_preds if sem_preds is not None else [],
            masks=pred_masks,
            scores=pred_scores)

    def forward_iter_pred(self, x, queries=None, **kwargs):
        """Iterative forward pass.
        
        Args:
            x (List[Tensor]): of len batch_size, each of shape
                (n_points_i, in_channels).
            queries (List[Tensor], optional): of len batch_size, each of shape
                (n_points_i, in_channles).
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores,
                and aux_outputs.
        """
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        inst_feats = [self.input_proj(y) for y in x]
        mask_feats = [self.x_mask(y) for y in x]
        queries = self._get_queries(queries, len(x))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
            queries, mask_feats, last_flag=False)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred if sem_pred is not None else [])
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        for i in range(len(self.cross_attn_layers)):
            queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask = self._forward_head(
                queries, mask_feats, last_flag)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred if sem_pred is not None else [])
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)

        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores)
            for cls_pred, sem_pred, scores, masks in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            aux_outputs=aux_outputs)


@MODELS.register_module()
class ScanNetMixQueryDecoder(QueryDecoder):
    """We simply add semantic prediction for each instance query.
    """
    def __init__(self, num_instance_classes, num_semantic_classes,
                 d_model, num_semantic_linears, in_channels, share_attn_mlp, share_mask_mlp,
                 cross_attn_mode, mask_pred_mode, temporal_attn=False, bbox_flag=False, **kwargs):
        super().__init__(
            num_classes=num_instance_classes, d_model=d_model, in_channels=in_channels, **kwargs)
        assert num_semantic_linears in [1, 2]
        assert isinstance(cross_attn_mode, list)
        assert isinstance(mask_pred_mode, list)
        assert mask_pred_mode[-1] == "P"

        self.cross_attn_mode = cross_attn_mode
        self.mask_pred_mode = mask_pred_mode
        self.temporal_attn = temporal_attn

        self.share_attn_mlp = share_attn_mlp
        if not share_attn_mlp:
            if "P" in self.cross_attn_mode:
                self.input_pts_proj = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.LayerNorm(d_model), nn.ReLU())

        self.share_mask_mlp = share_mask_mlp
        if not share_mask_mlp:
            if "P" in self.mask_pred_mode:
                self.x_pts_mask = nn.Sequential(
                    nn.Linear(3 + in_channels, d_model), nn.ReLU(),
                    nn.Linear(d_model, d_model))

        self.bbox_flag = bbox_flag
        if self.bbox_flag:
            self.out_reg = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, 6))

        if num_semantic_linears == 2:
            self.out_sem = nn.Sequential(
                nn.Linear(d_model, d_model), nn.ReLU(),
                nn.Linear(d_model, num_semantic_classes + 1))
        else:
            self.out_sem = nn.Linear(d_model, num_semantic_classes + 1)

    def _forward_head(self, queries, mask_feats, mask_pts_feats, last_flag, layer):  # type: ignore[override]
        """Prediction head forward.

        Args:
            queries (List[Tensor] | Tensor): List of len batch_size,
                each of shape (n_queries_i, d_model). Or tensor of
                shape (batch_size, n_queries, d_model).
            mask_feats (List[Tensor]): of len batch_size,
                each of shape (n_points_i, d_model).

        Returns:
            Tuple:
                List[Tensor]: Classification predictions of len batch_size,
                    each of shape (n_queries_i, n_instance_classes + 1).
                List[Tensor] or None: Semantic predictions of len batch_size,
                    each of shape (n_queries_i, n_semantic_classes + 1).
                List[Tensor]: Confidence scores of len batch_size,
                    each of shape (n_queries_i, 1).
                List[Tensor]: Predicted masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
                List[Tensor] or None: Attention masks of len batch_size,
                    each of shape (n_queries_i, n_points_i).
        """
        cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, pred_bboxes = [], [], [], [], [], []
        object_queries = []
        for i in range(len(queries)):
            norm_query = self.out_norm(queries[i])
            object_queries.append(norm_query)
            cls_preds.append(self.out_cls(norm_query))
            if last_flag:
                sem_preds.append(self.out_sem(norm_query))
            pred_score = self.out_score(norm_query) if self.objectness_flag else None
            pred_scores.append(pred_score)
            if self.bbox_flag:
                reg_final = self.out_reg(norm_query)
                reg_distance = torch.exp(reg_final[:, 3:6])
                pred_bbox = torch.cat([reg_final[:, :3], reg_distance], dim=1)
            else: pred_bbox = None
            pred_bboxes.append(pred_bbox)
            if self.mask_pred_mode[layer] == "SP":
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_feats[i])
            elif self.mask_pred_mode[layer] == "P":
                pred_mask = torch.einsum('nd,md->nm', norm_query, mask_pts_feats[i])
            else:
                raise NotImplementedError("Query decoder not implemented!")
            if self.attn_mask:
                attn_mask = (pred_mask.sigmoid() < 0.5).bool()
                attn_mask[torch.where(
                    attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                attn_mask = attn_mask.detach()
                attn_masks.append(attn_mask)
            pred_masks.append(pred_mask)
        attn_masks = attn_masks if self.attn_mask else None
        sem_preds = sem_preds if last_flag else None
        return cls_preds, sem_preds, pred_scores, pred_masks, attn_masks, object_queries, pred_bboxes

    def forward_iter_pred(self, x=None, queries=None, sp_feats=None, p_feats=None, super_points=None, prev_queries=None, **kwargs):
        """Iterative forward pass - supports both interfaces.
        
        Args:
            x: Legacy compatibility (redirected to sp_feats if provided)
            queries: Query tensors
            sp_feats: Super-point features
            p_feats: Point features
            super_points: Super-point information
            prev_queries: Previous queries (optional)
        
        Returns:
            Dict: with instance scores, semantic scores, masks, scores, and aux_outputs.
        """
        # Handle legacy interface
        if x is not None and sp_feats is None:
            sp_feats = x
        if sp_feats is None or p_feats is None or super_points is None:
            raise ValueError("ScanNetMixQueryDecoder requires sp_feats, p_feats, and super_points")
            
        return self._forward_iter_pred_impl(sp_feats, p_feats, queries, super_points, prev_queries)
    
    def _forward_iter_pred_impl(self, sp_feats, p_feats, queries, super_points, prev_queries=None):
        """Implementation of iterative forward pass."""
        cls_preds, sem_preds, pred_scores, pred_masks = [], [], [], []
        object_queries, pred_bboxes = [], []
        
        # 修复：移除动态初始化逻辑，input_proj已在__init__中创建
        # 检查维度兼容性，如果需要可以添加适配器
        if "SP" in self.cross_attn_mode and sp_feats:
            actual_in_channels = sp_feats[0].shape[-1]
            
            # 如果维度不匹配且没有适配器，创建适配器
            if actual_in_channels != self.expected_in_channels and self.input_adapter is None:
                print(f"[QueryDecoder] 维度不匹配: 输入{actual_in_channels} vs 期望{self.expected_in_channels}，创建适配器")
                self.input_adapter = nn.Sequential(
                    nn.Linear(actual_in_channels, self.expected_in_channels),
                    nn.ReLU(),
                    nn.LayerNorm(self.expected_in_channels)
                ).to(sp_feats[0].device)
            # elif actual_in_channels == self.expected_in_channels:
            #     # 移除冗余日志输出：使用预初始化的input_proj
            #     pass
        
        # 应用input_proj，考虑适配器
        if "SP" in self.cross_attn_mode:
            if self.input_adapter is not None:
                # 需要先适配维度
                inst_feats = [self.input_proj(self.input_adapter(y)) for y in sp_feats]
            else:
                # 直接使用input_proj
                inst_feats = [self.input_proj(y) for y in sp_feats]
        else:
            inst_feats = None

        # 修复：处理点特征，input_proj已在__init__中创建，只需检查适配器
        if "P" in self.cross_attn_mode and p_feats:
            actual_in_channels = p_feats[0].shape[-1]
            
            # 如果维度不匹配且没有适配器，创建适配器
            if actual_in_channels != self.expected_in_channels and self.input_adapter is None:
                print(f"[QueryDecoder] 点特征维度不匹配: 输入{actual_in_channels} vs 期望{self.expected_in_channels}，创建适配器")
                self.input_adapter = nn.Sequential(
                    nn.Linear(actual_in_channels, self.expected_in_channels),
                    nn.ReLU(),
                    nn.LayerNorm(self.expected_in_channels)
                ).to(p_feats[0].device)
            
            # 应用点特征投影
            if self.share_attn_mlp:
                if self.input_adapter is not None:
                    inst_pts_feats = [self.input_proj(self.input_adapter(y)) for y in p_feats]
                else:
                    inst_pts_feats = [self.input_proj(y) for y in p_feats]
            else:
                inst_pts_feats = [self.input_pts_proj(y) for y in p_feats]
        else:
            inst_pts_feats = None

        if "SP" in self.mask_pred_mode:
            if self.input_adapter is not None and sp_feats and sp_feats[0].shape[-1] != self.expected_in_channels:
                mask_feats = [self.x_mask(self.input_adapter(y)) for y in sp_feats]
            else:
                mask_feats = [self.x_mask(y) for y in sp_feats]
        else:
            mask_feats = None
        mask_pts_feats = [self.x_mask(y) if self.share_mask_mlp else self.x_pts_mask(y)
             for y in p_feats] if "P" in self.mask_pred_mode else None
        queries = self._get_queries(queries, len(sp_feats))
        cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
             self._forward_head(queries, mask_feats, mask_pts_feats, last_flag=False, layer=0)
        cls_preds.append(cls_pred)
        sem_preds.append(sem_pred)
        pred_scores.append(pred_score)
        pred_masks.append(pred_mask)
        object_queries.append(object_query)
        pred_bboxes.append(pred_bbox)
        for i in range(len(self.cross_attn_layers)):
            if self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "SP":
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            elif self.cross_attn_mode[i+1] == "SP" and self.mask_pred_mode[i] == "P":   # current method, change P mask to SP
                if attn_mask is not None:
                    xyz_weights = torch.chunk(super_points[1], len(super_points[0]), dim=0)
                    attn_mask_score = [scatter_mean(att.float() * xyz_w.view(1, -1), sp, dim=1)
                         for att, sp, xyz_w in zip(attn_mask, super_points[0], xyz_weights)]
                    attn_mask_score = [torch.nan_to_num(att, nan=0.0, posinf=1.0, neginf=0.0)
                                       for att in attn_mask_score]
                    attn_mask = [(att > 0.5).bool() for att in attn_mask_score] # > 0.5, not <
                    # If attn_mask has all-True row, the result of CA will be nan
                    for j in range(len(attn_mask)):
                        mask = ~(attn_mask_score[j] == attn_mask_score[j].min(dim=1, keepdim=True)[0])
                        attn_mask[j] = attn_mask[j] & mask
                        all_true = attn_mask[j].all(dim=1)
                        if all_true.any():
                            attn_mask[j] = attn_mask[j].clone()
                            attn_mask[j][all_true] = False
                queries = self.cross_attn_layers[i](inst_feats, queries, attn_mask)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "SP":
                if attn_mask is not None:
                    attn_mask = [att[:, sp] for att, sp in zip(attn_mask, super_points[0])]
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            elif self.cross_attn_mode[i+1] == "P" and self.mask_pred_mode[i] == "P":
                queries = self.cross_attn_layers[i](inst_pts_feats, queries, attn_mask)
            else:
                raise NotImplementedError("Not support yet!")
            queries = self.self_attn_layers[i](queries)
            queries = self.ffn_layers[i](queries)
            last_flag = i == len(self.cross_attn_layers) - 1
            cls_pred, sem_pred, pred_score, pred_mask, attn_mask, object_query, pred_bbox = \
                 self._forward_head(queries, mask_feats, mask_pts_feats, last_flag, layer=i+1)
            cls_preds.append(cls_pred)
            sem_preds.append(sem_pred)
            pred_scores.append(pred_score)
            pred_masks.append(pred_mask)
            object_queries.append(object_query)
            pred_bboxes.append(pred_bbox)

        aux_outputs = [
            dict(
                cls_preds=cls_pred, sem_preds=sem_pred, masks=masks, scores=scores, bboxes=bboxes)
            for cls_pred, sem_pred, scores, masks, bboxes in zip(
                cls_preds[:-1], sem_preds[:-1], pred_scores[:-1], pred_masks[:-1], pred_bboxes[:-1])]
        return dict(
            cls_preds=cls_preds[-1],
            sem_preds=sem_preds[-1],
            masks=pred_masks[-1],
            scores=pred_scores[-1],
            queries=object_queries[-1],
            bboxes=pred_bboxes[-1],
            aux_outputs=aux_outputs)
    
    def forward(self, x=None, queries=None, sp_feats=None, p_feats=None, super_points=None, prev_queries=None, **kwargs):
        """Forward pass for ScanNetMixQueryDecoder - supports both interfaces.
        
        Args:
            x: Legacy interface compatibility (unused)
            queries: Query tensors
            sp_feats: Super-point features  
            p_feats: Point features
            super_points: Super-point information
            prev_queries: Previous queries (optional)
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        # Support legacy interface by redirecting to new interface
        if sp_feats is not None and p_feats is not None and super_points is not None:
            return self.forward_mix(sp_feats, p_feats, queries, super_points, prev_queries)
        else:
            raise NotImplementedError("ScanNetMixQueryDecoder requires sp_feats, p_feats, and super_points")
    
    def forward_mix(self, sp_feats, p_feats, queries, super_points, prev_queries=None):
        """Forward pass implementation for ScanNetMixQueryDecoder.
        
        Args:
            sp_feats: Super-point features
            p_feats: Point features  
            queries: Query tensors
            super_points: Super-point information
            prev_queries: Previous queries (optional)
        
        Returns:
            Dict: with labels, masks, scores, and possibly aux_outputs.
        """
        if self.iter_pred:
            return self._forward_iter_pred_impl(sp_feats, p_feats, queries, super_points, prev_queries)
        else:
            raise NotImplementedError("No simple forward!!!")


@MODELS.register_module()
class S3DISQueryDecoder(QueryDecoder):
    # Does it have any differences with QueryDecoder?
    pass
