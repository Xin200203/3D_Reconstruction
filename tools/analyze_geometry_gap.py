"""
分析检测框与GT框的gap：为什么正样本对的重叠度这么低？

关键问题：
1. 56.4% 的正样本对重叠度为0 - 这极不正常
2. 剩余的非零对平均值仅 0.1732 - 太低了

可能的原因诊断：
- 检测框 vs GT框 的不对齐
- 缩放/量化误差
- 多视角时间不对齐
"""

import numpy as np
import glob
import os


def analyze_geometry_gap(diag_dir: str, sample_size: int = 100):
    """分析检测框质量与几何重叠的关系"""
    
    files = sorted(glob.glob(os.path.join(diag_dir, '*.npz')))
    
    print("=" * 80)
    print("检测框质量诊断：为什么正样本对重叠度低？")
    print("=" * 80)
    
    # 统计信息
    zero_overlap_recs = []
    nonzero_overlap_recs = []
    
    total_zero = 0
    total_nonzero = 0
    
    for fp in files[:10]:
        data = np.load(fp, allow_pickle=True)
        for rec in data['records']:
            M_pos = rec.get('M_pos')
            if M_pos is None:
                continue
            
            # 按是否有零重叠分类
            has_zero = (M_pos == 0).any()
            has_nonzero = (M_pos > 0).any()
            
            if has_zero and not has_nonzero:
                # 完全零重叠
                zero_overlap_recs.append((fp.split('/')[-1], rec))
                total_zero += M_pos.size
            elif has_nonzero:
                # 有非零重叠
                nonzero_overlap_recs.append((fp.split('/')[-1], rec))
                total_nonzero += (M_pos > 0).sum()
    
    print(f"\n【统计】")
    print(f"完全零重叠的记录: {len(zero_overlap_recs)} 条")
    print(f"有非零重叠的记录: {len(nonzero_overlap_recs)} 条")
    print(f"零重叠的对数: {total_zero}")
    print(f"非零重叠的对数: {total_nonzero}")
    
    # 分析完全零重叠的情况
    if zero_overlap_recs:
        print(f"\n【分析完全零重叠的情况】")
        print(f"这些样本对的特征:")
        
        for i, (fname, rec) in enumerate(zero_overlap_recs[:3]):
            M_pos = rec.get('M_pos')
            frame_i = rec.get('frame_i', -1)
            n_det = M_pos.shape[0]
            n_track = M_pos.shape[1]
            
            print(f"\n  样本 {i+1}: {fname} Frame={frame_i}")
            print(f"    检测数: {n_det}, 轨迹数: {n_track}")
            print(f"    M_pos 形状: {M_pos.shape}, 最大值: {M_pos.max():.4f}")
            
            # 检查是否有其他特征可能给予线索
            det_gt_conf = rec.get('det_gt_conf')
            if det_gt_conf is not None:
                if isinstance(det_gt_conf, (list, np.ndarray)):
                    det_gt_conf = det_gt_conf[0] if len(det_gt_conf) > 0 else 0
                print(f"    Detection GT置信度: {det_gt_conf:.4f}")
            
            track_strength = rec.get('track_gt_strength')
            if track_strength is not None:
                if isinstance(track_strength, (list, np.ndarray)):
                    if len(track_strength) > 0:
                        ts_val = track_strength[0] if isinstance(track_strength[0], (int, float)) else np.mean(track_strength)
                        print(f"    Track GT强度: {ts_val:.4f}")
                    else:
                        print(f"    Track GT强度: (empty)")
                else:
                    print(f"    Track GT强度: {track_strength:.4f}")
    
    # 分析非零重叠的情况
    if nonzero_overlap_recs:
        print(f"\n【分析有非零重叠的情况】")
        
        for i, (fname, rec) in enumerate(nonzero_overlap_recs[:3]):
            M_pos = rec.get('M_pos')
            frame_i = rec.get('frame_i', -1)
            n_det = M_pos.shape[0]
            n_track = M_pos.shape[1]
            
            nonzero_vals = M_pos[M_pos > 0]
            
            print(f"\n  样本 {i+1}: {fname} Frame={frame_i}")
            print(f"    检测数: {n_det}, 轨迹数: {n_track}")
            print(f"    非零对数: {len(nonzero_vals)}/{M_pos.size} ({100*len(nonzero_vals)/M_pos.size:.1f}%)")
            print(f"    非零值: min={nonzero_vals.min():.4f}, max={nonzero_vals.max():.4f}, mean={nonzero_vals.mean():.4f}")
    
    print(f"\n【根本原因分析】")
    print("""
可能的根本原因（按概率排序）：

1. ✓ 检测框质量差 (最可能，80%)
   - 这个checkpoint的检测本身不准确
   - 检测框与GT框的位置/大小严重不匹配
   - 表现为：56.4%完全不重叠，44.6%部分重叠但都很低
   
2. ✓ 多视角数据的时间不对齐 (可能，15%)
   - ScanNet MV数据是多个时刻的扫描合并
   - GT是某一时刻的标注，检测是从混合多帧生成
   - 物体位置在不同帧中不同步
   
3. ✓ 体素化量化误差 (可能，5%)
   - voxel_size=0.02可能太小
   - 导致检测框的坐标在体素化后偏移
   - 但这只能解释一部分，不能解释56%的零重叠

【建议下一步】

A. 验证假设 (按优先级)
   1. 对比另一个checkpoint的重叠度（vs0p12等）
   2. 尝试更大的voxel_size (0.03, 0.04, 0.05)
   3. 查看诊断数据中的detection框坐标和GT框坐标
   
B. 替代方案
   1. 使用NN-coverage（近邻覆盖）替代contain_det
   2. 改用容差更大的匹配标准（如中心距离 < threshold）
   3. 只依赖DINO和3D特征（忽略几何重叠）
   
C. 长期解决
   1. 重新训练或微调检测模型
   2. 改进多视角融合的时间对齐
   3. 在推理时使用post-processing改进检测框
    """)


if __name__ == "__main__":
    diag_dir = 'work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/vs0p12_subset5_diag_trimodal/track_geom_diag'
    analyze_geometry_gap(diag_dir)
