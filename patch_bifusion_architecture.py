#!/usr/bin/env python3
"""
BiFusion架构级修复补丁
直接修复坐标变换和融合机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def patch_bifusion_encoder():
    """应用BiFusion的架构修复补丁"""
    
    from oneformer3d.bi_fusion_encoder import BiFusionEncoder
    
    # 保存原始方法
    original_process_single = BiFusionEncoder._process_single
    
    def fixed_process_single(self, points, img, cam_meta, feat2d_map=None, clip_global=None):
        """修复的_process_single方法"""
        
        # 1. 减少坐标变换：直接使用相机坐标
        xyz_cam = points[:, :3]
        
        # 2. 简化投影，减少误差累积
        if cam_meta.get('intrinsics', None) is not None:
            intr = cam_meta['intrinsics']
            if not torch.is_tensor(intr):
                intr = torch.as_tensor(intr, dtype=xyz_cam.dtype, device=xyz_cam.device)
            
            # 直接投影，避免双重变换
            valid, uv = self.build_uv_index(xyz_cam, intr, feat2d_map.shape[-2:])
            
            # 3. 改进特征采样：用最近邻填充
            sampled2d = xyz_cam.new_zeros((xyz_cam.shape[0], 256))
            if valid.any():
                f2d_vis = self.sample_img_feat(feat2d_map.unsqueeze(0), uv[valid])
                sampled2d[valid] = f2d_vis.to(sampled2d.dtype)
                
                # 新增：用最近邻填充无效点
                if (~valid).any():
                    # 找到每个无效点最近的有效点
                    invalid_idx = torch.where(~valid)[0]
                    valid_idx = torch.where(valid)[0]
                    
                    if len(valid_idx) > 0:
                        # 计算距离矩阵
                        dist = torch.cdist(xyz_cam[invalid_idx], xyz_cam[valid_idx])
                        nearest_idx = dist.argmin(dim=1)
                        sampled2d[invalid_idx] = sampled2d[valid_idx[nearest_idx]]
        else:
            # 无相机信息时，使用零向量
            sampled2d = xyz_cam.new_zeros((xyz_cam.shape[0], 256))
            valid = torch.zeros(xyz_cam.shape[0], dtype=torch.bool, device=xyz_cam.device)
        
        # 4. 3D分支处理（保持原逻辑但增加稳定性）
        xyz_world = xyz_cam  # 简化：直接使用相机坐标作为世界坐标
        coords_int = torch.round(xyz_world / self.voxel_size).to(torch.int32)
        coords = torch.cat([torch.zeros(coords_int.size(0), 1, dtype=torch.int32, device=coords_int.device),
                             coords_int], dim=1)
        feats = points[:, 3:6].contiguous()
        
        # 添加特征标准化
        feats = F.normalize(feats, dim=-1)
        
        field = ME.TensorField(coordinates=coords, features=feats)
        sparse_tensor = field.sparse()
        feat3d_sparse = self.backbone3d(sparse_tensor)
        feat3d = feat3d_sparse.slice(field).features
        feat3d = self.backbone_adapter(feat3d)
        
        # 5. 几何编码
        bbox_size = torch.zeros_like(xyz_world)
        pose_delta = torch.zeros(9, device=xyz_world.device, dtype=xyz_world.dtype)
        height = xyz_world[:, 2:3]
        pe = self.pe_mlp(build_geo_pe(xyz_world, bbox_size, pose_delta, height))
        
        # 6. 特征处理
        feat3d = self.simple_neck(feat3d)
        feat3d = self.lin3d(feat3d)
        feat3d = self.ln3d(feat3d)
        
        feat2d = self.lin2d(sampled2d)
        feat2d = self.ln2d(feat2d)
        
        # 7. 改进的特征融合
        f2d_final = self.lin2d_final(torch.cat([feat2d, pe], dim=-1))
        f3d_final = self.lin3d_final(torch.cat([feat3d, pe], dim=-1))
        
        # 新的自适应融合策略
        if self.use_enhanced_gate:
            # 计算特征质量得分
            f2d_quality = torch.sigmoid(self.quality_mlp_2d(f2d_final))  # 需要添加这个层
            f3d_quality = torch.sigmoid(self.quality_mlp_3d(f3d_final))  # 需要添加这个层
            
            # 结合有效性和质量
            valid_weight = valid.float().unsqueeze(-1)
            adaptive_weight = valid_weight * f2d_quality / (f2d_quality + f3d_quality + 1e-8)
            
            fused = adaptive_weight * f2d_final + (1 - adaptive_weight) * f3d_final
            conf = adaptive_weight
        else:
            # 回退到改进的简单融合
            gate_input = torch.cat([f2d_final, f3d_final], dim=-1)
            gate = torch.sigmoid(self.gate_mlp(gate_input))
            valid_weight = valid.float().unsqueeze(-1)
            # 更保守的无效权重：从0.2降到0.1
            gate = gate * valid_weight + 0.1 * (1 - valid_weight)
            fused = gate * f2d_final + (1 - gate) * f3d_final
            conf = gate
        
        return fused, conf, pe, clip_global
    
    # 应用补丁
    BiFusionEncoder._process_single = fixed_process_single
    print("🔧 BiFusion架构修复补丁已应用")

if __name__ == "__main__":
    patch_bifusion_encoder()
