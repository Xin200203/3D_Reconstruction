# V2: ScanNet数据处理脚本 - 通过2D点+3D实例+KNN生成2D实例和语义标签
# 数据流: 2d_point+3d_ins+knn-->2d_ins-->2d_sem

import enum
import cv2
import shutil
import numpy as np
import math
from scipy import stats
import os
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pdb
import torch
from tqdm import tqdm

# 导入依赖模块 - 可能导致错误的地方
try:
    from segment_anything import build_sam, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    print("警告: Segment Anything Model (SAM) 未安装，请运行: pip install segment-anything")
    SAM_AVAILABLE = False

try:
    import pointops
    POINTOPS_AVAILABLE = True
except ImportError:
    print("警告: pointops 模块未安装，请确保已正确编译安装")
    POINTOPS_AVAILABLE = False

try:
    from load_scannet_data import export
    SCANNET_EXPORT_AVAILABLE = True
except ImportError:
    print("警告: load_scannet_data 模块未找到，请确保该文件在当前目录")
    SCANNET_EXPORT_AVAILABLE = False

def make_intrinsic(fx, fy, mx, my): 
    """
    构建相机内参矩阵
    
    Args:
        fx, fy: 焦距参数
        mx, my: 主点坐标
    
    Returns:
        内参矩阵 [4x4]
    """
    intrinsic = np.eye(4)   
    intrinsic[0][0] = fx    # 水平方向焦距
    intrinsic[1][1] = fy    # 垂直方向焦距
    intrinsic[0][2] = mx    # 主点x坐标
    intrinsic[1][2] = my    # 主点y坐标
    return intrinsic

def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    """
    根据图像尺寸调整内参矩阵
    
    Args:
        intrinsic: 原始内参矩阵
        intrinsic_image_dim: 原始图像尺寸 [w, h]
        image_dim: 目标图像尺寸 [w, h]
    
    Returns:
        调整后的内参矩阵
    """
    if intrinsic_image_dim == image_dim:
        return intrinsic
    
    # 计算resize后的宽度，保持长宽比
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    
    # 调整焦距参数
    intrinsic[0, 0] *= float(resize_width) / float(intrinsic_image_dim[0])
    intrinsic[1, 1] *= float(image_dim[1]) / float(intrinsic_image_dim[1])
    
    # 调整主点坐标，考虑裁剪
    intrinsic[0, 2] *= float(image_dim[0] - 1) / float(intrinsic_image_dim[0] - 1)
    intrinsic[1, 2] *= float(image_dim[1] - 1) / float(intrinsic_image_dim[1] - 1)
    return intrinsic


def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """
    对点云进行随机采样
    
    Args:
        pc: 输入点云 [N, C]
        num_sample: 采样点数
        replace: 是否允许重复采样
        return_choices: 是否返回采样索引
    
    Returns:
        采样后的点云 [num_sample, C]
    """
    if replace is None: 
        replace = (pc.shape[0] < num_sample)  # 如果点数不足则允许重复采样
        
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]

def load_matrix_from_txt(path, shape=(4, 4)):
    """
    从文本文件加载矩阵
    
    Args:
        path: 文件路径
        shape: 矩阵形状
    
    Returns:
        矩阵 numpy array
    """
    try:
        with open(path) as f:
            txt = f.readlines()
        txt = ''.join(txt).replace('\n', ' ')
        matrix = [float(v) for v in txt.split()]
        return np.array(matrix).reshape(shape)
    except Exception as e:
        print(f"错误: 无法读取矩阵文件 {path}: {e}")
        return None


def convert_from_uvd(u, v, depth, intr, pose):
    """
    将UV深度图坐标转换为世界坐标系下的3D点
    
    Args:
        u: 像素u坐标 (宽度索引)
        v: 像素v坐标 (高度索引) 
        depth: 深度值
        intr: 相机内参矩阵 [4x4]
        pose: 相机外参矩阵 [4x4]
    
    Returns:
        世界坐标系下的3D点 [N, 3]
    """
    # ScanNet的深度缩放因子为1000 (mm -> m)
    depth_scale = 1000
    z = depth / depth_scale

    # 构建齐次坐标
    u = np.expand_dims(u, axis=0)
    v = np.expand_dims(v, axis=0)
    padding = np.ones_like(u)
    
    # 像素坐标转齐次坐标 [u, v, 1]
    uv = np.concatenate([u, v, padding], axis=0)
    
    # 反投影到相机坐标系: xyz_camera = K^(-1) * [u, v, 1] * z
    xyz = (np.linalg.inv(intr[:3, :3]) @ uv) * np.expand_dims(z, axis=0)
    
    # 转换为齐次坐标
    xyz = np.concatenate([xyz, padding], axis=0)
    
    # 变换到世界坐标系: xyz_world = pose * xyz_camera
    xyz = pose @ xyz
    
    # 齐次坐标归一化
    xyz[:3, :] /= xyz[3, :] 
    return xyz[:3, :].T

def export_one_scan(scan_name):    
    """
    导出单个场景的3D数据，包括网格、实例标签、边界框等
    
    Args:
        scan_name: 场景名称
    
    Returns:
        aligned_mesh_vertices: 对齐后的网格顶点
        instance_labels: 实例标签
        label_map: 标签映射
        object_id_to_label: 对象ID到标签的映射
        bboxes: 边界框
        bbox_instance_labels: 边界框实例标签
    """
    if not SCANNET_EXPORT_AVAILABLE:
        raise ImportError("load_scannet_data模块不可用，无法导出场景数据")
        
    # 构建各种文件路径
    mesh_file = os.path.join('3D', scan_name, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join('3D', scan_name, scan_name + '.aggregation.json')
    seg_file = os.path.join('3D', scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join('3D', scan_name, scan_name + '.txt')  # 包含轴对齐信息
    
    # 调用导出函数
    aligned_mesh_vertices, instance_labels, bboxes, label_map, object_id_to_label = \
        export(mesh_file, agg_file, seg_file, meta_file, 'meta_data/scannetv2-labels.combined.tsv', scannet200=True)
    
    # 生成边界框实例标签
    bbox_instance_labels = np.arange(1, bboxes.shape[0] + 1)

    return aligned_mesh_vertices, instance_labels, label_map, object_id_to_label, bboxes, bbox_instance_labels


def select_points_in_bbox(xyz, ins, bboxes, bbox_instance_labels):
    """
    在边界框内选择点，并清理边界框外的实例标签
    
    Args:
        xyz: 3D点坐标 [N, 3]
        ins: 实例标签 [N]
        bboxes: 边界框参数 [M, 6] (x, y, z, w, h, d)
        bbox_instance_labels: 边界框实例标签 [M]
    
    Returns:
        ins: 清理后的实例标签
        object_num: 对象数量
    """
    # 为边界框添加小边距
    delta = 0.05
    
    for i in range(bboxes.shape[0]):
        instance_target = bbox_instance_labels[i]
        
        # 计算边界框的最大最小范围 (中心+尺寸/2)
        x_max = bboxes[i, 0] + bboxes[i, 3]/2 + delta
        x_min = bboxes[i, 0] - bboxes[i, 3]/2 - delta
        y_max = bboxes[i, 1] + bboxes[i, 4]/2 + delta
        y_min = bboxes[i, 1] - bboxes[i, 4]/2 - delta
        z_max = bboxes[i, 2] + bboxes[i, 5]/2 + delta
        z_min = bboxes[i, 2] - bboxes[i, 5]/2 - delta
        
        max_range = np.array([x_max, y_max, z_max])
        min_range = np.array([x_min, y_min, z_min])
        
        # 计算点是否在边界框内
        margin_positive = xyz - min_range
        margin_negative = xyz - max_range
        in_criterion = margin_positive * margin_negative
        
        zero = np.zeros(in_criterion.shape)
        one = np.ones(in_criterion.shape)
        
        # 在边界框内的点: margin_positive > 0 且 margin_negative < 0
        in_criterion = np.where(in_criterion <= 0, one, zero)
        mask_inbox = in_criterion[:, 0] * in_criterion[:, 1] * in_criterion[:, 2]
        mask_inbox = mask_inbox.astype(bool)
        
        # 找到属于当前实例但在边界框外的点，将其标签设为0
        mask_ins = np.in1d(ins, instance_target)
        ins[mask_ins * (~mask_inbox)] = 0
        
    unique_ins = np.unique(ins)
    # 计算对象数量 (包括地板和墙壁，不区分前景和背景)
    object_num = len(unique_ins) - int(0 in unique_ins)
    return ins, object_num

def format_result(result):
    annotations = []
    n = len(result.masks.data)
    for i in range(n):
        annotation = {}
        mask = result.masks.data[i] == 1.0

        annotation['id'] = i
        annotation['segmentation'] = mask.cpu().numpy()
        annotation['bbox'] = result.boxes.data[i]
        annotation['score'] = result.boxes.conf[i]
        annotation['area'] = annotation['segmentation'].sum()
        annotations.append(annotation)
    return annotations

def process_cur_scan(cur_scan, mask_generator):
    """
    处理当前场景，生成2D-3D对应的实例和语义标签
    
    主要步骤:
    1. 读取场景的RGB-D图像序列和相机参数
    2. 使用SAM生成2D超点分割
    3. 将深度图转换为3D点云
    4. 通过KNN将3D实例标签传播到2D点
    5. 从实例标签生成语义标签
    6. 保存处理结果
    
    Args:
        cur_scan: 包含场景信息的字典
        mask_generator: SAM自动掩码生成器
    """
    scan_name_index = cur_scan["scan_name_index"]
    scan_name = cur_scan["scan_name"]
    path_dict = cur_scan["path_dict"]
    scan_num = cur_scan["scan_num"]
    print(f"正在处理场景: {scan_name} ({scan_name_index+1}/{scan_num})")

    # 解析路径配置
    DATA_PATH = path_dict["DATA_PATH"]
    INS_DATA_PATH = path_dict["INS_DATA_PATH"]
    TARGET_DIR = path_dict["TARGET_DIR"]
    AXIS_ALIGN_MATRIX_PATH = path_dict["AXIS_ALIGN_MATRIX_PATH"]

    scan_name = scan_name.strip("\n")
    scan_path = os.path.join(DATA_PATH, scan_name)
    ins_data_path = os.path.join(INS_DATA_PATH, scan_name)
    path_dict["scan_path"] = scan_path

    # 读取轴对齐矩阵
    axis_align_matrix_path = os.path.join(AXIS_ALIGN_MATRIX_PATH, "%s" % (scan_name), "%s.txt" % (scan_name))
    
    try:
        lines = open(axis_align_matrix_path).readlines()
        axis_align_matrix = None
        for line in lines:
            if 'axisAlignment' in line:
                axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
                break
        
        if axis_align_matrix is None:
            print(f"警告: 在 {axis_align_matrix_path} 中未找到axisAlignment信息")
            axis_align_matrix = np.eye(4).flatten().tolist()  # 使用单位矩阵作为默认值
            
        axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    except Exception as e:
        print(f"错误: 无法读取轴对齐矩阵 {axis_align_matrix_path}: {e}")
        axis_align_matrix = np.eye(4)  # 使用单位矩阵作为默认值

    # 统一的图像尺寸和内参
    unify_dim = (640, 480)
    unify_intrinsic = adjust_intrinsic(make_intrinsic(577.870605, 577.870605, 319.5, 239.5), [640, 480], unify_dim)
    
    # 按数字顺序排序文件列表 (0, 20, 40, 60, 80, 100, 120...)
    try:
        POSE_txt_list = sorted(os.listdir(os.path.join(scan_path, 'pose')), key=lambda s: int(s[:-4]))
        rgb_map_list = sorted(os.listdir(os.path.join(scan_path, 'color')), key=lambda s: int(s[:-4]))
        depth_map_list = sorted(os.listdir(os.path.join(scan_path, 'depth')), key=lambda s: int(s[:-4]))
    except Exception as e:
        print(f"错误: 无法读取场景文件列表 {scan_path}: {e}")
        return

    # 加载相机位姿
    poses = []
    for path in POSE_txt_list:
        pose = load_matrix_from_txt(os.path.join(scan_path, 'pose', path))
        if pose is not None:
            poses.append(pose)
        else:
            print(f"警告: 无法加载位姿文件 {path}")
            poses.append(np.eye(4))  # 使用单位矩阵作为默认值
    
    # 应用轴对齐变换
    aligned_poses = [np.dot(axis_align_matrix, pose) for pose in poses]

    # 导出3D场景数据
    try:
        aligned_mesh_vertices, instance_labels, label_map, object_id_to_label, \
            aligned_bboxes, bbox_instance_labels = export_one_scan(scan_name)
    except Exception as e:
        print(f"错误: 无法导出场景 {scan_name} 的3D数据: {e}")
        return

    # 处理每一帧图像
    for frame_i, (rgb_map_name, depth_map_name, pose, aligned_pose) in enumerate(
        zip(rgb_map_list, depth_map_list, poses, aligned_poses)):
        
        assert frame_i * 20 == int(rgb_map_name[:-4]), f"帧索引不匹配: {frame_i * 20} vs {rgb_map_name}"
        
        # 间隔采样：每10帧处理一次 (interval=200) 
        # 如果要保持与25k相同数量，使用5 (interval=100)
        if frame_i % 10 != 0:
            continue

        try:
            # 读取深度图和彩色图
            depth_map = cv2.imread(os.path.join(scan_path, 'depth', depth_map_name), -1)
            color_map = cv2.imread(os.path.join(scan_path, 'color', rgb_map_name))
            color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            
            if depth_map is None or color_map is None:
                print(f"警告: 无法读取图像文件 {rgb_map_name} 或 {depth_map_name}")
                continue
                
        except Exception as e:
            print(f"错误: 读取图像失败 {rgb_map_name}: {e}")
            continue

        img_path = os.path.join(scan_path, 'color', rgb_map_name)
        
        # 使用SAM生成超点分割
        try:
            if not SAM_AVAILABLE:
                print("警告: SAM不可用，跳过超点生成")
                continue
                
            masks = mask_generator.generate(color_map)
            masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
            
            # 为每个像素分配组ID
            group_ids = np.full((color_map.shape[0], color_map.shape[1]), -1, dtype=int)
            num_masks = len(masks)
            group_counter = 0
            for i in range(num_masks):
                mask_now = masks[i]["segmentation"]
                group_ids[mask_now] = group_counter
                group_counter += 1
                
        except Exception as e:
            print(f"错误: SAM分割失败 {rgb_map_name}: {e}")
            continue

        # 将深度图转换为点云
        height, width = depth_map.shape    
        w_ind = np.arange(width)
        h_ind = np.arange(height)

        # 生成像素网格坐标
        ww_ind, hh_ind = np.meshgrid(w_ind, h_ind)
        ww_ind = ww_ind.reshape(-1)
        hh_ind = hh_ind.reshape(-1)
        depth_map = depth_map.reshape(-1)
        group_ids = group_ids.reshape(-1)
        color_map = color_map.reshape(-1, 3)

        # 过滤有效深度点 (>0.1m)
        valid = np.where(depth_map > 0.1)[0]
        ww_ind = ww_ind[valid]
        hh_ind = hh_ind[valid]
        depth_map = depth_map[valid]
        group_ids = group_ids[valid]
        rgb = color_map[valid]

        # 转换为3D坐标 (SV格式: 零中心化，下采样到20000点)
        aligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, unify_intrinsic, aligned_pose)
        if np.isnan(aligned_xyz).any():
            print(f"警告: 3D坐标包含NaN值，跳过帧 {frame_i}")
            continue
            
        unaligned_xyz = convert_from_uvd(ww_ind, hh_ind, depth_map, unify_intrinsic, pose)
        
        # 零中心化处理
        xyz_offset = np.mean(unaligned_xyz, axis=0)
        unaligned_xyz -= xyz_offset
        
        # 保存中心化的位姿
        pose_centered = pose.copy()
        pose_centered[:3, 3] -= xyz_offset
        
        pose_centered_dir = './pose_centered/' + scan_name
        if not os.path.exists(pose_centered_dir):
            os.makedirs(pose_centered_dir)
        np.save(os.path.join(pose_centered_dir, rgb_map_name.replace('.jpg', '.npy')), pose_centered)

        # 拼接所有数据并随机采样到20000点
        unaligned_xyz = np.concatenate([unaligned_xyz, rgb], axis=-1)  # [N, 6]: xyz + rgb
        xyz_all = np.concatenate([unaligned_xyz, aligned_xyz, group_ids.reshape(-1, 1)], axis=-1)  # [N, 10]
        xyz_all = random_sampling(xyz_all, 20000)
        
        # 分离数据
        unaligned_xyz, aligned_xyz, group_ids = xyz_all[:, :6], xyz_all[:, 6:9], xyz_all[:, 9]

        # 通过KNN从3D标注获取实例标签
        if not POINTOPS_AVAILABLE:
            print("警告: pointops不可用，无法进行KNN查询")
            continue
            
        try:
            target_coord = torch.tensor(aligned_xyz).cuda().contiguous().float()
            target_offset = torch.tensor(target_coord.shape[0]).cuda().float()
            source_coord = torch.tensor(aligned_mesh_vertices[:, :3]).cuda().contiguous().float()
            source_offset = torch.tensor(source_coord.shape[0]).cuda().float()
            
            indices, dis = pointops.knn_query(1, source_coord, source_offset, target_coord, target_offset)
            indices = indices.cpu().numpy()
            ins = instance_labels[indices.reshape(-1)].astype(np.uint32)
            
            # 距离过滤 (可选)
            # mask_dis = dis.reshape(-1).cpu().numpy() > 0.05
            # ins[mask_dis] = 0
            
            # 使用边界框进一步去噪
            ins, object_num = select_points_in_bbox(aligned_xyz, ins, aligned_bboxes, bbox_instance_labels)
            
        except Exception as e:
            print(f"错误: KNN查询失败: {e}")
            continue

        # 从实例标签生成语义标签
        sem = np.zeros_like(ins, dtype=np.uint32)
        for ins_ids in np.unique(ins):
            if ins_ids != 0:
                try:
                    sem[ins == ins_ids] = label_map[object_id_to_label[ins_ids]]
                except KeyError:
                    print(f"警告: 实例ID {ins_ids} 没有对应的语义标签")
        
        # 处理超点分割
        # 对于没有被SAM分割的点，使用KMeans聚类生成10个超点
        points_without_seg = unaligned_xyz[group_ids == -1]
        if len(points_without_seg) < 10:
            # 如果未分割点太少，直接分配到一个新的组
            other_ins = np.zeros(len(points_without_seg), dtype=np.int64) + group_ids.max() + 1
        else:
            # 使用KMeans生成10个聚类
            other_ins = KMeans(n_clusters=10, n_init=10).fit(points_without_seg).labels_ + group_ids.max() + 1
            
        group_ids[group_ids == -1] = other_ins
        
        # 重新标记组ID，确保连续性
        unique_ids = np.unique(group_ids)
        if group_ids.max() != len(unique_ids) - 1:
            new_group_ids = np.zeros_like(group_ids)
            for i, ids in enumerate(unique_ids):
                new_group_ids[group_ids == ids] = i
            group_ids = new_group_ids

        # 保存处理结果
        try:
            frame_suffix = "%s" % (20 * frame_i)
            np.save(os.path.join(TARGET_DIR, scan_name + "_%s_sp_label.npy" % frame_suffix), group_ids)
            np.save(os.path.join(TARGET_DIR, scan_name + "_%s_vert.npy" % frame_suffix), unaligned_xyz)
            np.save(os.path.join(TARGET_DIR, scan_name + "_%s_sem_label.npy" % frame_suffix), sem)
            np.save(os.path.join(TARGET_DIR, scan_name + "_%s_ins_label.npy" % frame_suffix), ins)
            np.save(os.path.join(TARGET_DIR, scan_name + "_%s_axis_align_matrix.npy" % frame_suffix), axis_align_matrix)
            
            print(f"  成功处理帧 {frame_i}: {rgb_map_name}")
            
        except Exception as e:
            print(f"错误: 保存数据失败 {rgb_map_name}: {e}")
            continue


def make_split(mask_generator, path_dict, split="train"):
    """
    处理指定数据分割（train/val）的所有场景
    
    Args:
        mask_generator: SAM自动掩码生成器
        path_dict: 路径配置字典
        split: 数据分割类型 ("train" 或 "val")
    """
    TARGET_DIR = path_dict["TARGET_DIR_PREFIX"]
    path_dict["TARGET_DIR"] = TARGET_DIR
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    try:
        f = open("meta_data/scannetv2_%s.txt" % (split))
        scan_name_list = sorted(f.readlines())
        f.close()
    except Exception as e:
        print(f"错误: 无法读取数据分割文件 meta_data/scannetv2_{split}.txt: {e}")
        return

    print(f"开始处理 {split} 分割，共 {len(scan_name_list)} 个场景")
    
    for scan_name_index, scan_name in enumerate(tqdm(scan_name_list)):
        cur_parameter = {}
        cur_parameter["scan_name_index"] = scan_name_index
        cur_parameter["scan_name"] = scan_name
        cur_parameter["path_dict"] = path_dict
        cur_parameter["scan_num"] = len(scan_name_list)
        
        try:
            process_cur_scan(cur_parameter, mask_generator)
        except Exception as e:
            print(f"错误: 处理场景 {scan_name.strip()} 失败: {e}")
            continue


def main():
    """
    主函数：配置路径并处理ScanNet数据
    
    数据处理流程:
    1. 配置数据路径
    2. 初始化SAM模型
    3. 分别处理train和val数据分割
    4. 生成2D-3D对应的实例和语义标签
    """
    # 数据路径配置
    DATA_PATH = "./2D"                              # RGB-D序列数据路径
    TARGET_DIR_PREFIX = "./scannet_sv_instance_data"  # 输出数据路径
    INS_DATA_PATH = "./2D"                          # 实例数据路径 (与2D数据路径相同)
    AXIS_ALIGN_MATRIX_PATH = "./3D"                 # 轴对齐矩阵路径

    path_dict = {
        "DATA_PATH": DATA_PATH,
        "TARGET_DIR_PREFIX": TARGET_DIR_PREFIX,
        "INS_DATA_PATH": INS_DATA_PATH,
        "AXIS_ALIGN_MATRIX_PATH": AXIS_ALIGN_MATRIX_PATH       
    }

    # 检查必要路径是否存在
    for key, path in path_dict.items():
        if not os.path.exists(path):
            print(f"警告: 路径 {key} = {path} 不存在")

    # 要处理的数据分割
    splits = ["train", "val"]

    # 初始化SAM模型
    try:
        if not SAM_AVAILABLE:
            raise ImportError("SAM模块不可用")
            
        # SAM模型权重路径 - 请根据实际情况修改
        sam_checkpoint = "../sam_vit_h_4b8939.pth"
        if not os.path.exists(sam_checkpoint):
            print(f"错误: SAM权重文件不存在: {sam_checkpoint}")
            print("请下载SAM权重文件并放置在正确位置")
            return
            
        mask_generator = SamAutomaticMaskGenerator(
            build_sam(checkpoint=sam_checkpoint).to(device="cuda")
        )
        print("SAM模型初始化成功")
        
    except Exception as e:
        print(f"错误: SAM模型初始化失败: {e}")
        print("请确保:")
        print("1. 已安装segment-anything: pip install segment-anything")
        print("2. SAM权重文件路径正确")
        print("3. CUDA可用")
        return
    
    # 处理各个数据分割
    for cur_split in splits:
        print(f"\n开始处理 {cur_split} 数据分割...")
        try:
            make_split(mask_generator, path_dict, cur_split)
            print(f"{cur_split} 数据分割处理完成")
        except Exception as e:
            print(f"错误: 处理 {cur_split} 数据分割失败: {e}")
            continue
    
    print("\n所有数据处理完成！")


if __name__ == "__main__":
    main()