from argparse import ArgumentParser  # 从 argparse 模块导入 ArgumentParser，用于解析命令行参数
import os  # 导入 os 模块，提供文件和目录操作
import time  # 导入 time 模块，用于计算耗时
import numpy as np  # 导入 numpy 并简写为 np，用于数值计算
import torch  # 导入 PyTorch 框架
import warnings  # 导入 warnings，用于警告信息提示
from mmdet3d.registry import MODELS  # 从 mmdet3d 的注册表中导入 MODELS，用于构建模型
from mmengine.registry import init_default_scope  # 导入默认作用域初始化函数
from mmengine.dataset import pseudo_collate  # 导入伪批量打包函数
from mmdet3d.structures import Det3DDataSample, PointData  # 导入 3D 数据样本与点云数据结构
from mmengine.config import Config  # 导入配置文件解析类
from mmengine.runner import load_checkpoint  # 导入权重加载函数

import sys  # 导入 sys 模块，用于修改环境变量
current_path = os.path.abspath(__file__)  # 当前脚本的绝对路径
sys.path.append(os.path.dirname(os.path.dirname(current_path)))  # 将项目根目录加入 Python 路径，方便跨目录导包
from vis_demo.utils.vis_utils import vis_pointcloud,Vis_color  # 导入自定义可视化模块
from vis_demo.utils.stream_data_utils import DataPreprocessor, StreamDataloader, StreamBotDataloader  # 导入数据预处理与数据加载器

# ------------------------- 过时的模型初始化函数（兼容旧代码） -------------------------

def init_model(config, checkpoint, device):  # 根据配置与权重初始化模型
    config = Config.fromfile(config)  # 读取配置文件
    init_default_scope(config.get('default_scope', 'mmdet3d'))  # 初始化默认作用域，确保 registry 正确
    model =  MODELS.build(config.model)  # 根据配置构建模型实例
    checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')  # 加载权重到 CPU
    model.cfg = config  # 将配置写回模型
    if device != 'cpu':  # 如果使用 GPU
        torch.cuda.set_device(device)  # 设置当前 GPU 设备
    else:
        warnings.warn('Don\'t suggest using CPU device. '  # 警告：不建议使用 CPU
                    'Some functions are not supported for now.')
    model.to(device)  # 将模型迁移到指定设备
    model.eval()  # 设置为评估模式
    return model  # 返回模型实例

# ------------------------- 离线推理函数，用于单次流式数据 -------------------------

def inference_detector(model, args):  # 使用模型对数据根目录下的序列进行推理
    """Inference point cloud with the detector.

    Args:
        model (nn.Module): The loaded detector.
        args (argparse.Namespace): The arguments containing the data root etc.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    cfg = model.cfg  # 读取模型配置
    intrinsic = np.loadtxt(os.path.join(args.data_root, 'intrinsic.txt'))  # 读取相机内参
    dataloader = StreamDataloader(args.data_root, interval=args.interval)  # 创建数据加载器，按 interval 取帧
        
    # 构建数据预处理器
    ckpt_path = os.path.join(os.path.dirname(os.path.dirname(current_path)), 'data', 'FastSAM-x.pt')  # FastSAM 权重路径
    data_preprocessor = DataPreprocessor(cfg, ckpt_path, intrinsic=intrinsic)  # 初始化预处理器
    all_images = []  # 保存所有彩色图
    all_points = []  # 保存所有点云
    # 逐帧处理整个序列
    while True:
        frame_i, color_map, depth_map, pose, end_flag = dataloader.next()  # 读取下一帧
        if end_flag: break  # 序列结束则退出循环
        group_ids, pts = data_preprocessor.process_single_frame(frame_i, color_map, depth_map, pose)  # 预处理当前帧，得到超像素分组与点云
        points = torch.from_numpy(pts).float()  # 转为 torch tensor 浮点型
        sp_pts_mask = torch.from_numpy(group_ids).long().to(args.device)  # 将超像素分组 id 转为 tensor 并迁移到设备
        input_dict = {'points':points.to(args.device)}  # 组装输入字典
        data_sample = Det3DDataSample()  # 创建数据样本对象
        gt_pts_seg = PointData()  # 创建点云数据结构
        gt_pts_seg['sp_pts_mask'] = sp_pts_mask  # 添加超像素 mask
        data_sample.gt_pts_seg = gt_pts_seg  # 写入样本
        data = [dict(inputs=input_dict, data_samples=data_sample)]  # 包装为 batch，大小 1
        collate_data = pseudo_collate(data)  # 伪打包
        
        # 关闭梯度，开始推理
        with torch.no_grad():
            result = model.test_step(collate_data)  # 调用模型 test_step 接口
        all_images.append(color_map)  # 保存彩色图
        all_points.append(points[:,:3])  # 保存点坐标
    return result[0], all_images, all_points  # 返回最终结果与序列数据

# ----------------------------------------------------------------------------
#                               流式演示主类
# ----------------------------------------------------------------------------

class StreamDemo:
    def __init__(self, args):  # 初始化函数，保存参数并完成准备工作
        self.args = args  # 保存命令行参数
        self.model = self.init_model()  # 初始化模型
        self.model.map_to_rec_pcd = False  # 关闭映射到重建点云的功能
        np.random.seed(0)  # 设置随机种子，保证调色板一致
        self.palette = np.random.random((256, 3)) * 255  # 生成 256 种随机颜色
        self.palette[-1] = 200  # 背景颜色浅灰
        self.palette = self.palette.astype(int)  # 转为 int
        self.device = args.device  # 保存设备信息
        self.online_vis = args.online_vis  # 是否在线可视化分割结果
        self.max_frames = args.max_frames  # 最多处理帧数
        self.intrinsic = np.loadtxt(os.path.join(args.data_root, 'intrinsic.txt'))  # 读取相机内参
        self.dataloader = StreamDataloader(args.data_root, args.interval)  # 创建数据加载器
        
        ckpt_path = os.path.join(os.path.dirname(os.path.dirname(current_path)), 'data', 'FastSAM-x.pt')  # FastSAM 权重路径
        self.data_preprocessor = DataPreprocessor(self.model.cfg, ckpt_path, intrinsic=self.intrinsic)  # 初始化预处理器
        
        self.former_points = np.zeros((0, 3), dtype=np.float32)  # 保存历史点云，初始为空
    
    # ------------------------ 模型初始化 ------------------------
    def init_model(self):
        args = self.args  # 取出参数
        config = Config.fromfile(args.config)  # 读取配置文件
        init_default_scope(config.get('default_scope', 'mmdet3d'))  # 初始化默认作用域
        model =  MODELS.build(config.model)  # 构建模型
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')  # 加载权重
        model.cfg = config  # 写回配置
        if args.device != 'cpu':
            torch.cuda.set_device(args.device)  # 设置当前 GPU
        else:
            warnings.warn('Don\'t suggest using CPU device. '  # 给出警告
                        'Some functions are not supported for now.')
        model.to(args.device)  # 模型迁移到设备
        model.eval()  # 评估模式
        return model  # 返回模型
    
    # ------------------------ 可视化接口 ------------------------
    def vis(self, cur_points, cur_points_color, cur_image, end_flag=False):
        if not hasattr(self, 'vis_p'):  # 首次调用时初始化可视化器
            self.vis_p = vis_pointcloud(self.args.use_vis)  # 点云可视化器
            self.vis_c = Vis_color(self.args.use_vis)  # 图像可视化器
        if end_flag:  # 终止标志，运行可视化窗口
            self.vis_p.run()
            # 如需保存相机参数，可取消以下注释
            # if self.vis_p.use_vis:
            #     param = self.vis_p.vis.get_view_control().convert_to_pinhole_camera_parameters()
            #     o3d.io.write_pinhole_camera_parameters('temp.json', param)
            #     self.vis_p.vis.destroy_window()
        self.vis_p.update(cur_points, cur_points_color)  # 更新点云
        self.vis_c.update(cur_image)  # 更新彩色图
    
    # ------------------------ mask → 颜色 ------------------------
    def mask_to_color(self, pred_ins_mask, order=None):
        if order is None:  # 若无排序，按最大值投票
            idx_mask = np.where(np.any(pred_ins_mask, axis=0), np.argmax(pred_ins_mask, axis=0), -1)
        else:  # 按置信度排序后遮盖（后覆盖前）
            idx_mask = -np.ones(pred_ins_mask.shape[1], dtype=int)
            for i in order:
                idx_mask[np.where(pred_ins_mask[i])] = i
        points_color = self.palette[idx_mask]  # 根据 idx 查调色板
        return points_color  # 返回颜色矩阵
    
    # ------------------------ 处理单帧 ------------------------
    def run_single_frame(self, color_map, depth_map, pose, intrinsic=None):
        group_ids, pts = self.data_preprocessor.process_single_frame(color_map, depth_map, pose, intrinsic=intrinsic)  # 预处理单帧
        points = torch.from_numpy(pts).float()  # 转为 tensor
        sp_pts_mask = torch.from_numpy(group_ids).long().to(self.device)  # mask tensor
        input_dict = {'points':points.to(self.device)}  # 输入字典
        data_sample = Det3DDataSample()
        gt_pts_seg = PointData()
        gt_pts_seg['sp_pts_mask'] = sp_pts_mask
        data_sample.gt_pts_seg = gt_pts_seg
        data = [dict(inputs=input_dict, data_samples=data_sample)]
        collate_data = pseudo_collate(data)
        with torch.no_grad():
            result = self.model.test_step(collate_data)  # 推理
        pred_ins_mask = result[0].pred_pts_seg.pts_instance_mask[0]  # 实例 mask
        pred_ins_score = result[0].pred_pts_seg.instance_scores  # 置信度
        order = pred_ins_score.argsort()  # 排序
        points_color = self.mask_to_color(pred_ins_mask, order)  # 上色
        all_points = np.concatenate([self.former_points, pts[:,:3]], axis=0)  # 拼接新旧点云
        self.former_points = all_points  # 更新历史
        return all_points, points_color, pred_ins_mask  # 返回
    
    # ------------------------ 主循环 ------------------------
    def run(self):
        all_images = []  # 存储所有彩色图
        all_points = []  # 存储所有点云
        all_points_color = []  # 存储所有点云颜色
        
        time0 = time.time()  # 起始时间
        while True:
            frame_i, color_map, depth_map, pose, end_flag = self.dataloader.next()  # 获取下一帧
            end_flag = end_flag or (frame_i >= self.args.max_frames)  # 判断结束条件
            if end_flag:
                self.vis(None, None, None, True)  # 强制刷新可视化并退出
                break
            group_ids, pts = self.data_preprocessor.process_single_frame(color_map, depth_map, pose)  # 预处理帧
            points = torch.from_numpy(pts).float()
            sp_pts_mask = torch.from_numpy(group_ids).long().to(self.device)
            input_dict = {'points':points.to(self.device)}
            data_sample = Det3DDataSample()
            gt_pts_seg = PointData()
            gt_pts_seg['sp_pts_mask'] = sp_pts_mask
            data_sample.gt_pts_seg = gt_pts_seg
            data = [dict(inputs=input_dict, data_samples=data_sample)]
            collate_data = pseudo_collate(data)
            with torch.no_grad():
                result = self.model.test_step(collate_data)
            all_images.append(color_map)  # 保存彩色图
            if self.online_vis:  # 在线可视化分割结果
                pred_ins_mask = result[0].pred_pts_seg.pts_instance_mask[0]
                pred_ins_score = result[0].pred_pts_seg.instance_scores
                order = pred_ins_score.argsort()
                points_color = self.mask_to_color(pred_ins_mask, order)
                whole_points = np.concatenate([self.former_points, pts[:,:3]], axis=0)
                self.former_points = whole_points
                
                all_points.append(whole_points)
                all_points_color.append(points_color)
                self.vis(whole_points, points_color, color_map)
            else:
                all_points.append(points[:,:3])  # 仅保存点坐标
                
        total_time = time.time() - time0  # 计算总用时
        print(f"Total Time: {total_time:.2f}")  # 打印总时间
        print(f"Frame Number: {len(all_images)}")  # 打印帧数
        print(f"FPS: {(len(all_images) / total_time):.2f}")  # 打印帧率
        
        images = np.array(all_images)  # 转 numpy
        if not self.online_vis:  # 离线可视化
            points = torch.stack(all_points)
            pred_ins_mask = result[0].pred_pts_seg.pts_instance_mask[0]
            pred_ins_score = result[0].pred_pts_seg.instance_scores
            pred_ins_masks_sorted = pred_ins_mask[pred_ins_score.argsort()]
            points_color = self.mask_to_color(pred_ins_masks_sorted).reshape(points.shape[0], points.shape[1], 3)
            for i in range(len(all_images)):
                self.vis(points[i], points_color[i], images[i])   
        else:
            points = all_points[-1]
            points_color = all_points_color[-1]
        save_dir = os.path.join(self.args.save_dir, self.args.data_root.split('/')[-1])  # 构造保存目录
        os.makedirs(save_dir, exist_ok=True)  # 创建目录
        np.save(os.path.join(save_dir, 'images.npy'), images)  # 保存图像
        np.save(os.path.join(save_dir, 'points.npy'), points)  # 保存点云
        np.save(os.path.join(save_dir, 'points_color.npy'), points_color)   # 保存颜色

        # 若需要将点云与颜色拼接保存，可参考以下注释代码
        # point_cloud = np.concatenate([points, points_color], axis=-1)
        # np.savetxt(os.path.join(save_dir, 'point_cloud.npy'), point_cloud.reshape(-1, 6))
    
# ------------------------ CLI 入口 ------------------------

def main():
    parser = ArgumentParser(add_help=True)  # 创建解析器
    # 输入输出相关参数
    parser.add_argument('--data_root', type=str, default=None, help='Data root')  # 数据根目录
    parser.add_argument('--save_dir', type=str, default='./vis_demo/results', help='Output directory')  # 输出目录
    parser.add_argument('--interval', type=int, default='1', help='Frame processing interval (process every Nth frame)')  # 处理间隔
    parser.add_argument('--max_frames', type=int, default=10000, help='Max frame number to process')  # 最大帧数
    # 模型相关参数
    parser.add_argument('--config', type=str, default='configs/ESAM-E_CA/ESAM-E_online_stream.py', help='Config file')  # 配置文件
    parser.add_argument('--checkpoint', type=str, default='work_dirs/ESAM-E_online_scannet200_CA/epoch_128.pth', help='Checkpoint file')  # 权重
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')  # 设备
    # 可视化相关参数
    parser.add_argument('--use_vis', type=int, default="1", help="Whether to enable visualization, set to 1 to enable")  # 是否可视化
    parser.add_argument('--online_vis', action='store_true', help="Whether to visualize segmentation results online, store true")  # 是否在线可视化
    args = parser.parse_args()  # 解析参数
    
    assert args.data_root is not None, "The input data root must be specified"  # 检查数据根目录
          
    demo = StreamDemo(args)  # 创建演示对象
    demo.run()  # 运行演示

if __name__ == '__main__':  # 脚本入口
    main()  # 运行主函数
