# Copied from mmdet3d/models/data_preprocessors/data_preprocessor.py
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.registry import MODELS
import torch
import numpy as np


@MODELS.register_module()
class Det3DDataPreprocessor_(Det3DDataPreprocessor):
    """
    We add only this 2 lines:
    if 'elastic_coords' in inputs:
        batch_inputs['elastic_coords'] = inputs['elastic_coords']
    """
    def simple_process(self, data, training=False):
        """Perform normalization, padding and bgr2rgb conversion for img data
        based on ``BaseDataPreprocessor``, and voxelize point cloud if `voxel`
        is set to be True.

        Args:
            data (dict): Data sampled from dataloader.
            training (bool): Whether to enable training time augmentation.
                Defaults to False.

        Returns:
            dict: Data in the same format as the model input.
        """
        # 提取核心数据
        inputs = data.get('inputs', {})
        
        if 'img' in data['inputs']:
            batch_pad_shape = self._get_pad_shape(data)

        data = self.collate_data(data)
        inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        if 'points' in inputs:
            batch_inputs['points'] = inputs['points']

            if self.voxel:
                voxel_dict = self.voxelize(inputs['points'], data_samples)
                batch_inputs['voxels'] = voxel_dict

        if 'elastic_coords' in inputs:
            batch_inputs['elastic_coords'] = inputs['elastic_coords']

        if 'clip_pix' in inputs:
            batch_inputs['clip_pix'] = inputs['clip_pix']
        if 'clip_global' in inputs:
            batch_inputs['clip_global'] = inputs['clip_global']

        if 'imgs' in inputs:
            imgs = inputs['imgs']
            
            # 统一处理各种图像格式
            tensor_imgs = []
            
            if isinstance(imgs, list) and len(imgs) > 0:
                # 检查是否是tuple格式（Pack3DDetInputs_的处理结果）
                if len(imgs) == 1 and isinstance(imgs[0], tuple):
                    # 展开tuple中的图像
                    tuple_imgs = imgs[0]
                    
                    for i, img in enumerate(tuple_imgs):
                        if isinstance(img, torch.Tensor):
                            # 确保tensor是正确的格式 (C, H, W)
                            if img.dim() == 3 and img.shape[0] in [1, 3]:
                                tensor_imgs.append(img)
                    
                    # 处理cam_info，确保长度匹配
                    if 'cam_info' in inputs:
                        cam_info = inputs['cam_info']
                        if isinstance(cam_info, list) and len(cam_info) == 1:
                            # 复制cam_info以匹配图像数量
                            batch_inputs['cam_info'] = [cam_info[0] for _ in range(len(tensor_imgs))]
                        else:
                            batch_inputs['cam_info'] = cam_info
                
                else:
                    # 处理其他格式的图像列表
                    for i, img in enumerate(imgs):
                        processed_img = self._process_single_image(img, i)
                        if processed_img is not None:
                            tensor_imgs.append(processed_img)
                    
                    # 处理cam_info
                    if 'cam_info' in inputs:
                        batch_inputs['cam_info'] = inputs['cam_info']
            
            # 验证最终的图像列表
            if len(tensor_imgs) == 0:
                raise ValueError("No valid images found after preprocessing")
            
            batch_inputs['imgs'] = tensor_imgs

        elif 'img' in inputs:
            # 原来的 img 处理逻辑
            batch_pad_shape = self._get_pad_shape(data)
            imgs = inputs['img']

            if data_samples is not None:
                # NOTE the batched image size information may be useful, e.g.
                # in DETR, this is needed for the construction of masks, which
                # is then used for the transformer_head.
                batch_input_shape = tuple(imgs[0].size()[-2:])
                for data_sample, pad_shape in zip(data_samples,
                                                  batch_pad_shape):
                    data_sample.set_metainfo({
                        'batch_input_shape': batch_input_shape,
                        'pad_shape': pad_shape
                    })

                if hasattr(self, 'boxtype2tensor') and self.boxtype2tensor:
                    from mmdet.models.utils.misc import \
                        samplelist_boxtype2tensor
                    samplelist_boxtype2tensor(data_samples)
                elif hasattr(self, 'boxlist2tensor') and self.boxlist2tensor:
                    # 某些版本的 mmdet 并未实现 `samplelist_boxlist2tensor`，
                    # 这里使用 try–except 保持向后兼容，避免静态分析报错。
                    try:
                        from mmdet.models.utils.misc import samplelist_boxlist2tensor  # type: ignore
                    except ImportError:  # fallback to same impl as boxtype2tensor
                        from mmdet.models.utils.misc import samplelist_boxtype2tensor as samplelist_boxlist2tensor  # type: ignore
                    samplelist_boxlist2tensor(data_samples)
                if self.pad_mask:
                    self.pad_gt_masks(data_samples)

                if self.pad_seg:
                    self.pad_gt_sem_seg(data_samples)

            if training and self.batch_augments is not None:
                for batch_aug in self.batch_augments:
                    imgs, data_samples = batch_aug(imgs, data_samples)
            batch_inputs['img'] = imgs
        
        if 'img_paths' in inputs:
            img_paths = []
            for batch_idx in range(len(inputs['img_paths'][0])):
                batch_paths = [paths[batch_idx] for paths in inputs['img_paths']]
                img_paths.append(batch_paths)
            batch_inputs['img_paths'] = img_paths
        
        if 'img_path' in inputs:
            batch_inputs['img_path'] = inputs['img_path']
        return {'inputs': batch_inputs, 'data_samples': data_samples}

    def _process_single_image(self, img, idx):
        """处理单个图像，支持多种输入格式"""
        try:
            # 处理tuple类型 - 这是Pack3DDetInputs_处理后的格式
            if isinstance(img, tuple):
                # tuple通常是(tensor, dtype, shape)格式，我们需要第一个元素
                if len(img) > 0 and isinstance(img[0], torch.Tensor):
                    return img[0]
                else:
                    # 尝试重构
                    if len(img) >= 3:  # (data, dtype, shape)
                        try:
                            reconstructed = torch.tensor(img[0], dtype=img[1])
                            if len(img) > 2:
                                reconstructed = reconstructed.view(img[2])
                            return reconstructed
                        except Exception:
                            pass
            
            elif isinstance(img, torch.Tensor):
                return img
            
            elif hasattr(img, 'size'):
                # 可能是PIL图像或numpy数组，转换为tensor
                if hasattr(img, 'convert'):  # PIL Image
                    img_array = np.array(img.convert('RGB'))
                    return torch.from_numpy(img_array).permute(2, 0, 1).float()
                else:  # numpy array
                    if len(img.shape) == 3 and img.shape[-1] == 3:  # (H,W,C)
                        return torch.from_numpy(img).permute(2, 0, 1).float()
                    else:  # (C,H,W)
                        return torch.from_numpy(img).float()
            
            else:
                # 其他情况，尝试转换
                if hasattr(img, 'dtype') and hasattr(img, 'shape'):
                    return img.float() if hasattr(img, 'float') else img
                elif hasattr(img, '__array__'):
                    try:
                        arr = np.array(img)
                        return torch.from_numpy(arr).float()
                    except Exception:
                        pass
                        
        except Exception:
            pass
            
        return None