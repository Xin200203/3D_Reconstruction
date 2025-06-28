from argparse import ArgumentParser  # 从 argparse 模块导入 ArgumentParser，用于解析命令行参数

import os  # 导入 os 模块，提供与操作系统交互的接口
from os import path as osp  # 从 os.path 模块导入路径操作并重命名为 osp，便于后续使用
import random  # 导入 random 模块，用于生成随机数（当前脚本未直接使用）
import numpy as np  # 导入 numpy 并简写为 np，进行数值计算
import torch  # 导入 PyTorch 深度学习框架
import mmengine  # 导入 mmengine，MM 引擎的通用工具
from mmdet3d.apis import init_model  # 从 mmdet3d.apis 导入 init_model，用于初始化检测模型
from mmdet3d.registry import DATASETS  # 导入 DATASETS 注册表，用于构建数据集
from mmengine.dataset import Compose, pseudo_collate  # 导入 Compose（拼接数据处理流程）和 pseudo_collate（伪批量打包函数）
import open3d as o3d  # 导入 open3d 库，用于点云可视化
from PIL import Image  # 从 Pillow 库导入 Image，用于图像读取
from utils.vis_utils import vis_pointcloud,Vis_color  # 从自定义可视化工具中导入 vis_pointcloud 与 Vis_color

import sys  # 导入 sys 模块，用于修改 Python 运行时环境
current_path = os.path.abspath(__file__)  # 获取当前脚本绝对路径
sys.path.append(os.path.dirname(os.path.dirname(current_path)))  # 将项目根目录加入到系统路径，方便导入上级模块

class DataConverter(object):  # 定义数据转换器类，用于将原始数据转换为模型可用格式
    def __init__(self, root_path, cfg):  # 构造函数，接收数据根目录与配置文件
        self.root_dir = root_path  # 保存数据根目录
        self.split_dir = osp.join(root_path)  # （预留）数据划分目录，此处等于 root_dir
        
        # 使用数据集仅为了调用 `parse_data_info`，因此即使是验证集也足够
        self.dataset = DATASETS.build(cfg.val_dataloader.dataset)  # 通过注册表构建数据集实例
    
    def get_axis_align_matrix(self, idx):  # 读取坐标对齐矩阵（ScanNet 专用）
        matrix_file = osp.join(self.root_dir, 'axis_align_matrix',  # 构造矩阵文件路径
                               f'{idx}.npy')
        mmengine.check_file_exist(matrix_file)  # 检查文件是否存在
        return np.load(matrix_file)  # 加载并返回矩阵

    def process_single_scene(self, sample_idx):  # 处理单个场景，生成符合 pipeline 的数据字典
        ## Data process
        info = dict()  # 初始化信息字典
        pc_info = {'num_features': 6, 'lidar_idx': sample_idx}  # 点云信息，6 维特征（XYZRGB）
        info['point_cloud'] = pc_info  # 填入信息字典
        files = os.listdir(osp.join(self.root_dir, 'points', sample_idx))  # 列出当前场景所有点云分片文件
        files.sort(key=lambda x: int(x.split('/')[-1][:-4]))  # 按文件名中的数字进行排序
        # 下面将各类文件路径保存到 info 中，供 pipeline 使用
        info['pts_paths'] = [osp.join('points', sample_idx, file) for file in files]  # 点云块路径
        info['super_pts_paths'] = [osp.join('super_points', sample_idx, file) for file in files]  # 超像素点云路径
        info['pts_instance_mask_paths'] = [osp.join('instance_mask', sample_idx, file) for file in files]  # 实例 mask 路径
        info['pts_semantic_mask_paths'] = [osp.join('semantic_mask', sample_idx, file) for file in files]  # 语义 mask 路径
        # 根据数据集不同，构造对应的图像路径及特殊信息
        if 'scannet' in self.root_dir:
            info['img_paths'] = [osp.join('2D', sample_idx, 'color', file.replace('bin','jpg')) for file in files]  # ScanNet 图像路径
            axis_align_matrix = self.get_axis_align_matrix(sample_idx)  # 读取坐标对齐矩阵
            info['axis_align_matrix'] = axis_align_matrix.tolist()  # 保存为 list，json 兼容
        elif '3RScan' in self.root_dir:
            info['img_paths'] = [osp.join('3RScan', sample_idx, 'sequence','frame-' + file.split('.')[0].zfill(6) + '.color.jpg') for file in files]  # 3RScan 图像路径
        elif 'scenenn' in self.root_dir:
            info['img_paths'] = [osp.join('SceneNN', sample_idx, 'image','image'+file.split('.')[0].zfill(5)+'.png') for file in files]  # SceneNN 图像路径

        ## Dataset process
        info = self.dataset.parse_data_info(info)  # 通过数据集的 parse_data_info 进一步处理
        return info  # 返回处理后的信息
        
        
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