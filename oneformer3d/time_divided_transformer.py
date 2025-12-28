import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from mmdet3d.registry import MODELS


class _GeomBiasAttnLayer(nn.Module):
    """单层 Geometry–biased Cross-Attention + Self-Attention + FFN.

    Args:
        d_model (int): hidden dim.
        nhead (int): number of heads.
        dropout (float): dropout prob.
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        # QKV projection
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        # geometry bias MLP: input 18 → 1
        self.geo_mlp = nn.Sequential(
            nn.Linear(18, d_model), nn.ReLU(), nn.Linear(d_model, 1)
        )
        self.beta = nn.Parameter(torch.zeros(1))
        self.dropout = nn.Dropout(dropout)
        # GRU for query update (updates per token)
        self.gru = nn.GRUCell(d_model, d_model)
        # point-wise self attn on queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4*d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(4*d_model, d_model)
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, q, k, p_c, p_m, mem_mask=None, attention_mask=None):
        """Args:
            q,k: (B, N, D), (B, M, D)
            p_c,p_m: (B, N, 9), (B, M, 9)
            mem_mask: (B, M)  bool 1=valid (传统Memory掩码)
            attention_mask: (B, N, M) bool True=允许注意力，False=禁止 (IoU预剪枝掩码)
        Returns:
            q_new: (B,N,D)
            attn: (B,N,M)  softmax along M
        """
        B, Nc, D = q.shape
        Nm = k.shape[1]
        q_proj = self.q_proj(q)  # (B,Nc,D)
        k_proj = self.k_proj(k)
        v_proj = self.v_proj(k)
        # reshape for heads
        def split_heads(x):
            return x.view(B, -1, self.nhead, D // self.nhead).transpose(1, 2)  # (B,h,N,dh)
        qh = split_heads(q_proj)
        kh = split_heads(k_proj)
        vh = split_heads(v_proj)
        # scaled dot
        attn_logits = torch.einsum('bhnd,bhmd->bhnm', qh, kh) / sqrt(D // self.nhead)
        # geometry bias
        pc = p_c.unsqueeze(2)  # B,Nc,1,9
        pc = pc.expand(-1, -1, Nm, -1)  # B,Nc,M,9
        pm = p_m.unsqueeze(1).expand(-1, Nc, -1, -1)      # B,Nc,M,9
        geo_in = torch.cat([pc, pm], dim=-1)               # B,Nc,M,18
        geo_bias = self.geo_mlp(geo_in).squeeze(-1)        # B,Nc,M
        attn_logits = attn_logits + self.beta * geo_bias.unsqueeze(1)  # broadcast to heads
        
        # 🆕 应用IoU预剪枝掩码
        if attention_mask is not None:
            # attention_mask: (B, Nc, Nm), True=允许注意力，False=禁止
            iou_mask = ~attention_mask.bool()  # 转换为禁止掩码
            attn_logits = attn_logits.masked_fill(iou_mask[:, None, :, :], -1e9)  # broadcast to heads
        
        # 应用传统Memory掩码
        if mem_mask is not None:
            mask = ~mem_mask.bool()  # True where invalid
            attn_logits = attn_logits.masked_fill(mask[:, None, None, :], -1e9)
            
        attn = F.softmax(attn_logits, dim=-1)              # B,h,Nc,Nm
        attn = self.dropout(attn)
        out = torch.einsum('bhnm,bhmd->bhnd', attn, vh)    # B,h,Nc,dh
        out = out.transpose(1, 2).contiguous().view(B, Nc, D)  # restore
        # GRU update
        q_reshaped = q.reshape(B*Nc, D)
        out_reshaped = out.reshape(B*Nc, D)
        q_updated = self.gru(out_reshaped, q_reshaped)
        q_updated = q_updated.view(B, Nc, D)
        # Self-Attention + FFN on updated queries
        sa_out, _ = self.self_attn(q_updated, q_updated, q_updated)
        q_sa = self.norm1(q_updated + self.dropout(sa_out))
        ffn_out = self.ffn(q_sa)
        q_final = self.norm3(q_sa + self.dropout(ffn_out))
        # average heads attn for output matrix
        attn_mean = attn.mean(1)  # B,Nc,Nm
        return q_final, attn_mean


@MODELS.register_module()
class TimeDividedTransformer(nn.Module):
    """Inter-frame Slot-Transformer (跨帧匹配模块).

    forward 返回 (attn_matrix, updated_query)
    """
    def __init__(self,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 3,
                 dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            _GeomBiasAttnLayer(d_model, nhead, dropout) for _ in range(num_layers)
        ])

    def forward(self, q, k, p_c, p_m, mask_mem=None, attention_mask=None):
        """Args:
            q, k: (B, N_c, D), (B, N_m, D)
            p_c, p_m: (B, N_c, 9), (B, N_m, 9)
            mask_mem: (B, N_m) 1=valid (传统Memory掩码)
            attention_mask: (B, N_c, N_m) IoU预剪枝掩码，True=允许注意力，False=禁止
        Returns:
            attn: (B, N_c, N_m)   softmax matrix of last layer
            q_new: (B, N_c, D)
        """
        for layer in self.layers:
            q, attn = layer(q, k, p_c, p_m, mask_mem, attention_mask)
        return attn, q 