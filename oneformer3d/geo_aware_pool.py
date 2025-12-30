from turtle import forward
import torch
import torch.nn as nn
import pdb, time
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from torch_scatter import scatter_mean, scatter


@MODELS.register_module()
class GeoAwarePooling(BaseModule):
    """Pool point features to super points.
    """
    def __init__(self, channel_proj):
        super().__init__()
        self.channel_proj = channel_proj
        self.pts_proj1 = nn.Sequential(
            nn.Linear(3, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, channel_proj),
            nn.LayerNorm(channel_proj)
        )
        self.pts_proj2 = nn.Sequential(
            nn.Linear(2 * channel_proj, channel_proj),
            nn.LayerNorm(channel_proj),
            nn.ReLU(),
            nn.Linear(channel_proj, 1, bias=False),
            nn.Sigmoid()
        )
        
        # 输入特征适配层 (延迟初始化)
        self.input_adapter = None
        self._adapter_logged = False
    
    def scatter_norm(self, points, idx):
        ''' Normalize positions of same-segment in a unit sphere of diameter 1
        Code is copied from SPT
        '''
        min_segment = scatter(points, idx, dim=0, reduce='min')
        max_segment = scatter(points, idx, dim=0, reduce='max')
        diameter_segment = (max_segment - min_segment).max(dim=1).values
        center_segment = scatter(points, idx, dim=0, reduce='mean')
        center = center_segment[idx]
        diameter = diameter_segment[idx]
        points = (points - center) / (diameter.view(-1, 1) + 1e-2)
        return points, diameter_segment.view(-1, 1)

    def forward(self, x, sp_idx, all_xyz, with_xyz=False):
        # 检查输入特征维度并适配
        input_dim = x.shape[1]
        if input_dim != self.channel_proj:
            # 需要适配输入特征维度
            if self.input_adapter is None:
                # 延迟初始化适配层
                self.input_adapter = nn.Sequential(
                    nn.Linear(input_dim, self.channel_proj),
                    nn.ReLU(),
                    nn.LayerNorm(self.channel_proj)
                ).to(x.device)
                if not self._adapter_logged:
                    print(f"[GeoAwarePooling] Created input adapter: {input_dim} -> {self.channel_proj}")
                    self._adapter_logged = True
            
            # 应用适配层
            x_adapted = self.input_adapter(x)
        else:
            x_adapted = x
            
        # 原有的坐标处理逻辑
        all_xyz_ = torch.cat(all_xyz)
        all_xyz, _ = self.scatter_norm(all_xyz_, sp_idx)
        all_xyz = self.pts_proj1(all_xyz)
        all_xyz_segment = scatter(all_xyz, sp_idx, dim=0, reduce='max')
        all_xyz = torch.cat([all_xyz, all_xyz_segment[sp_idx]], dim=-1)
        all_xyz_w = self.pts_proj2(all_xyz) * 2
        
        if with_xyz:
            x_final = torch.cat([x_adapted * all_xyz_w, all_xyz_], dim=-1)
            x_final = scatter_mean(x_final, sp_idx, dim=0)
            x_final[:, :-3] = x_final[:, :-3] + all_xyz_segment
        else:
            # 确保scatter_mean结果和all_xyz_segment维度匹配
            x_pooled = scatter_mean(x_adapted * all_xyz_w, sp_idx, dim=0)
            # 检查维度是否匹配，如果不匹配则调整all_xyz_segment
            if x_pooled.shape[0] != all_xyz_segment.shape[0]:
                # 重新计算all_xyz_segment以确保维度匹配
                num_segments = x_pooled.shape[0]
                if all_xyz_segment.shape[0] > num_segments:
                    all_xyz_segment = all_xyz_segment[:num_segments]
                else:
                    # 如果all_xyz_segment太小，则用零填充
                    pad_size = num_segments - all_xyz_segment.shape[0]
                    padding = torch.zeros(pad_size, all_xyz_segment.shape[1], 
                                        device=all_xyz_segment.device, dtype=all_xyz_segment.dtype)
                    all_xyz_segment = torch.cat([all_xyz_segment, padding], dim=0)
            x_final = x_pooled + all_xyz_segment

        # NOTE: pool 输出通道应与 `channel_proj`（或 with_xyz=True 时为 channel_proj+3）保持一致，
        # 下游 decoder/heads 依赖该维度；不要再映射回原始输入维度。
        return x_final, all_xyz_w
