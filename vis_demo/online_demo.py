from argparse import ArgumentParser  # 从 argparse 模块导入 ArgumentParser，用于解析命令行参数

import os  # 导入 os 模块，提供与操作系统交互的接口
from os import path as osp  # 从 os.path 模块导入路径操作并重命名为 osp，便于后续使用
import random  # 导入 random 模块，用于生成随机数（当前脚本未直接使用）
import numpy as np  # 导入 numpy 并简写为 np，进行数值计算
import torch  # 导入 PyTorch 深度学习框架
from typing import Optional
import mmengine  # 导入 mmengine，MM 引擎的通用工具
from mmdet3d.apis import init_model  # 从 mmdet3d.apis 导入 init_model，用于初始化检测模型
from mmdet3d.registry import DATASETS  # 导入 DATASETS 注册表，用于构建数据集
from mmengine.dataset import Compose, pseudo_collate  # 导入 Compose（拼接数据处理流程）和 pseudo_collate（伪批量打包函数）
from pathlib import Path
from tools.update_infos_to_v2 import get_empty_standard_data_info, clear_data_info_unused_keys
import open3d as o3d  # 导入 open3d 库，用于点云可视化
from PIL import Image  # 从 Pillow 库导入 Image，用于图像读取
from vis_demo.utils.vis_utils import vis_pointcloud,Vis_color  # 从自定义可视化工具中导入 vis_pointcloud 与 Vis_color

import sys  # 导入 sys 模块，用于修改 Python 运行时环境
current_path = os.path.abspath(__file__)  # 获取当前脚本绝对路径
sys.path.append(os.path.dirname(os.path.dirname(current_path)))  # 将项目根目录加入到系统路径，方便导入上级模块

class DataConverter(object):  # 定义数据转换器类，用于将原始数据转换为模型可用格式
    def __init__(self, root_path, cfg):  # 构造函数，接收数据根目录与配置文件
        # 收集候选数据根目录，优先使用 CLI 提供的 root_path
        candidates = []
        def _resolve_case(path: str) -> Optional[str]:
            if not path:
                return None
            abs_path = osp.abspath(path)
            if osp.isdir(abs_path):
                return abs_path
            parts = abs_path.split(osp.sep)
            current = osp.sep if abs_path.startswith(osp.sep) else parts[0]
            start_idx = 1 if abs_path.startswith(osp.sep) else 0
            if start_idx == 0:
                if not osp.isdir(current):
                    return None
            for part in parts[start_idx:]:
                if not part:
                    continue
                try:
                    entries = os.listdir(current)
                except FileNotFoundError:
                    return None
                match = None
                for entry in entries:
                    if entry.lower() == part.lower():
                        match = entry
                        break
                if match is None:
                    return None
                current = osp.join(current, match)
            return current if osp.isdir(current) else None

        def _add(path):
            resolved = _resolve_case(path)
            if resolved and resolved not in candidates:
                candidates.append(resolved)

        _add(root_path)
        _add(getattr(cfg, 'data_root', None))
        # 使用数据集仅为了调用 `parse_data_info`，因此即使是验证集也足够
        # 使用数据集仅为了调用 `parse_data_info`，因此即使是验证集也足够
        self.dataset = DATASETS.build(cfg.val_dataloader.dataset)  # 通过注册表构建数据集实例
        _add(getattr(self.dataset, 'data_root', None))

        self.root_dir = None
        for path in candidates:
            if path and osp.isdir(path):
                self.root_dir = osp.abspath(path)
                break
        if self.root_dir is None:
            raise FileNotFoundError(
                f"无法找到可用的数据根目录，请检查 --data_root 或配置中的 data_root，候选: {candidates}")

        if hasattr(self.dataset, 'data_root'):
            self.dataset.data_root = self.root_dir

        self.split_dir = self.root_dir

    @staticmethod
    def _sort_key(fname: str, scene_idx: str) -> int:
        """提取文件名中的数字后缀用于排序；若匹配不到，回退为0。
        支持 '7.bin' 或 'sceneXXXX_YY_7.bin' 两种命名。
        """
        base = osp.splitext(osp.basename(fname))[0]
        if base.startswith(scene_idx + '_') and '_' in base:
            tail = base.split('_')[-1]
        else:
            tail = base
        return int(tail) if tail.isdigit() else 0

    def _gather_scene_files(self, base_dir: str, scene_idx: str, expect_ext: str) -> list:
        """同时兼容两种目录结构：
        1) 分目录：base_dir/<scene_idx>/<k>.ext
        2) 扁平化：base_dir/<scene_idx>_<k>.ext
        返回相对 base_dir 的文件名列表（仅文件名，用于后续路径拼接）。
        """
        subdir = osp.join(base_dir, scene_idx)
        files = []
        if osp.isdir(subdir):
            # 结构1：分目录
            for f in os.listdir(subdir):
                if f.endswith(expect_ext):
                    files.append(f)
            files.sort(key=lambda x: self._sort_key(x, scene_idx))
            return [osp.join(scene_idx, f) for f in files]
        # 结构2：扁平化：在 base_dir 下匹配前缀
        try:
            for f in os.listdir(base_dir):
                if f.startswith(scene_idx + '_') and f.endswith(expect_ext):
                    files.append(f)
        except FileNotFoundError:
            pass
        files.sort(key=lambda x: self._sort_key(x, scene_idx))
        return files
    
    def get_axis_align_matrix(self, idx):  # 读取坐标对齐矩阵（最小实现：恒等变换）
        # 按你的要求：当前阶段不使用该量，保持最小可运行实现
        return np.eye(4, dtype=np.float32)

    def process_single_scene(self, sample_idx):  # 处理单个场景，生成符合 pipeline 的数据字典（SV 标准 v2 格式）
        pc_info = {'num_features': 6, 'lidar_idx': sample_idx}
        # 收集点云块（优先 .bin，回退 .npy）
        pts_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'points'), sample_idx, '.bin')
        if len(pts_rel_files) == 0:
            pts_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'points'), sample_idx, '.npy')
        if len(pts_rel_files) == 0:
            raise FileNotFoundError(f"未找到场景 {sample_idx} 的点云块，请检查数据根目录 {self.root_dir}/points")
        first_rel = pts_rel_files[0]
        first_base = osp.basename(first_rel)
        # 对应的 mask/super points（若结构不一致，则尝试 .npy）
        inst_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'instance_mask'), sample_idx, osp.splitext(first_base)[1] or '.bin')
        if len(inst_rel_files) == 0:
            inst_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'instance_mask'), sample_idx, '.npy')
        sem_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'semantic_mask'), sample_idx, osp.splitext(first_base)[1] or '.bin')
        if len(sem_rel_files) == 0:
            sem_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'semantic_mask'), sample_idx, '.npy')
        super_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'super_points'), sample_idx, osp.splitext(first_base)[1] or '.bin')
        if len(super_rel_files) == 0:
            super_rel_files = self._gather_scene_files(osp.join(self.root_dir, 'super_points'), sample_idx, '.npy')

        inst_base = Path(inst_rel_files[0]).name if inst_rel_files else Path(first_base).name
        sem_base = Path(sem_rel_files[0]).name if sem_rel_files else Path(first_base).name
        super_base = Path(super_rel_files[0]).name if super_rel_files else Path(first_base).name
        # 推导单帧图像路径（仅用于可视化叠图）
        k = osp.splitext(first_base)[0]
        if k.startswith(sample_idx + '_') and '_' in k:
            k = k.split('_')[-1]
        img_rel = osp.join('2D', sample_idx, 'color', f'{k}.jpg')
        clip_rel = osp.join('clip_feat', sample_idx, f'{k}.pt')
        # 标准 v2 data_info
        data_info = get_empty_standard_data_info()
        data_info['lidar_points']['num_pts_feats'] = pc_info['num_features']
        # 仅放文件名，parse_data_info 会拼接 data_root / data_prefix
        base_name = Path(first_base).name
        data_info['lidar_points']['lidar_path'] = base_name
        data_info['pts_semantic_mask_path'] = sem_base
        data_info['pts_instance_mask_path'] = inst_base
        data_info['super_pts_path'] = super_base
        if img_rel is not None:
            data_info['img_path'] = img_rel
        if clip_rel is not None:
            data_info['clip_feat_path'] = clip_rel
            data_info['clip_feat_paths'] = [clip_rel]
        data_info, _ = clear_data_info_unused_keys(data_info)
        # 交给数据集做绝对路径拼接
        data_info = self.dataset.parse_data_info(data_info)
        # 兼容下游：补充列表形式的 img_paths
        if 'img_path' in data_info and 'img_paths' not in data_info:
            data_info['img_paths'] = [data_info['img_path']]
        if clip_rel is not None and 'clip_feat_paths' not in data_info:
            data_info['clip_feat_paths'] = [clip_rel]
        if clip_rel is not None and 'clip_feat_path' not in data_info:
            data_info['clip_feat_path'] = clip_rel
        return data_info
        
        
# ------------------------------ 推理函数 ----------------------------------

def inference_detector(model, scene_idx):  # 调用模型对单个场景进行推理
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        pcd (str): Point cloud files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg  # 获取模型配置
    
    # 构建数据处理 pipeline
    test_pipeline = Compose(cfg.test_pipeline)  # 使用 Compose 将 pipeline 串联起来
    # 构建数据转换器
    data_converter = DataConverter(root_path=cfg.data_root, cfg=cfg)  # 初始化数据转换器
    # 处理单个场景数据，得到数据字典
    data = data_converter.process_single_scene(scene_idx)
    img_paths = data['img_paths']  # 提取图像路径列表
    data = [test_pipeline(data)]  # 通过 pipeline 处理数据，并放入列表形成 batch（大小 1）
    collate_data = pseudo_collate(data)  # 使用伪批量打包函数，保持数据结构一致
    
    # 前向推理
    with torch.no_grad():  # 禁用梯度计算，加速并节省显存
        result = model.test_step(collate_data)  # 调用模型测试接口
    
    return result[0], data[0], img_paths  # 返回单场景结果、处理后数据及图像路径
        
# ------------------------------ 主函数入口 ----------------------------------

def main():  # 脚本入口函数
    parser = ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument('--scene_idx', default='scene0011_00', type=str, help='single scene index')  # 场景索引
    parser.add_argument('--config', type=str, help='Config file')  # 配置文件路径
    parser.add_argument('--checkpoint', type=str, help='Checkpoint file')  # 权重文件路径
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')  # 推理设备
    parser.add_argument('--use_vis', type=int, default="1")  # 是否启用可视化，1 表示启用
    args = parser.parse_args()  # 解析命令行参数
    # 初始化模型
    model = init_model(config=args.config, checkpoint=args.checkpoint, device=args.device)  # 根据配置与权重初始化模型
    model.map_to_rec_pcd = False  # 关闭映射至重建点云的功能（特定模型字段）
    # 推理单个场景
    result, data, img_paths = inference_detector(model=model, scene_idx=args.scene_idx)  # 获取推理结果与数据
    points = data['inputs']['points'][:,:,:3]  # 提取点坐标 (N,3)
    pred_ins_mask = result.pred_pts_seg.pts_instance_mask[0]  # 预测实例 mask（0/1）
    pred_ins_score = result.pred_pts_seg.instance_scores  # 每个实例置信分数

    # -------------------- 获取实例分割结果 --------------------
    pred_instance_masks_sorted = torch.Tensor(pred_ins_mask[pred_ins_score.argsort()])  # 按置信度排序
    pred_instance_masks_label = pred_instance_masks_sorted[0].long() - 1  # 初始化标签，背景为 -1
    for i in range(1, pred_instance_masks_sorted.shape[0]):  # 遍历剩余实例
        pred_instance_masks_label[pred_instance_masks_sorted[i].bool()] = i  # 将对应位置赋值为实例 id
        
    np.random.seed(0)  # 随机种子，保证颜色一致
    palette = np.random.random((max(pred_instance_masks_label) + 2, 3)) * 255  # 生成调色板
    palette[-1] = 200  # 背景颜色设为浅灰色
    
    pred_seg_color = palette[pred_instance_masks_label]  # 根据标签着色
    points_color = pred_seg_color.reshape(points.shape[0], points.shape[1], 3)  # 重塑为 (T, N, 3)
    
    # -------------------- 读取场景图像 --------------------
    scene_images = []  # 初始化图像列表
    for img_path in img_paths:
        scene_images.append(np.array(Image.open(img_path)))  # 读取并保存图像
    scene_images = np.array(scene_images)  # 转为 numpy 数组
        
    # -------------------- 可视化 --------------------
    vis_p = vis_pointcloud(args.use_vis)  # 创建点云可视化器
    vis_c = Vis_color(args.use_vis)  # 创建图像可视化器
    
    for i in range(len(scene_images)):  # 遍历时间序列帧
        vis_p.update(points[i], points_color[i])  # 更新点云显示
        vis_c.update(scene_images[i])  # 更新图像显示
    vis_p.run()  # 运行可视化窗口
    
    # 如果需要保存相机参数，可取消注释以下代码
    # param = vis_p.vis.get_view_control().convert_to_pinhole_camera_parameters()
    # o3d.io.write_pinhole_camera_parameters('temp.json', param)
    # vis_p.vis.destroy_window()
    
if __name__ == '__main__':  # 当脚本直接运行时执行
    main()  # 调用主函数
