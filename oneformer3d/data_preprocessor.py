# Copied from mmdet3d/models/data_preprocessors/data_preprocessor.py
import torch
import numpy as np
import os
from mmdet3d.models.data_preprocessors.data_preprocessor import \
    Det3DDataPreprocessor
from mmdet3d.registry import MODELS


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

        # 当同时存在单帧 `img` 和多帧 `imgs` 时，优先走单帧分支（ESAM+在线 DINO 的典型配置），
        # 避免误触发多帧 BiFusion 路径导致 `No valid images found` 等错误。
        if 'img' in inputs and 'imgs' in inputs:
            inputs = dict(inputs)
            inputs.pop('imgs')
            # 同步更新 data['inputs']，保证后续 _get_pad_shape 等看到一致视图
            data = dict(data)
            data['inputs'] = inputs
        
        # NOTE: `_get_pad_shape` expects batched tensor/list inputs.
        # For online DINO pipeline, `data` may still be a raw dict (e.g. img as tuple),
        # so defer padding shape computation until after `collate_data`.

        if isinstance(data, dict) and 'inputs' in data and 'data_samples' in data:
            data = self.cast_data(data)
            inputs, data_samples = data['inputs'], data['data_samples']
        else:
            data = self.collate_data(data)
            inputs, data_samples = data['inputs'], data['data_samples']
        batch_inputs = dict()

        # 调试：观察经过 collate 后的 inputs 键，便于确认 cam_info 是否存在
        if os.environ.get('DEBUG_CAMINFO_INPUTS') == '1':
            print(f"[Det3DDataPreprocessor_] inputs keys after collate: {list(inputs.keys())}")

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
                    
                    # 处理cam_info，保持batch中每个样本的独立性
                    if 'cam_info' in inputs:
                        cam_info = inputs['cam_info']
                        
                        # 🔧 关键修复：将clip_pix添加到cam_info中供BiFusion使用
                        if 'clip_pix' in inputs:
                            clip_pix = inputs['clip_pix']
                            # 确保cam_info是列表格式
                            if isinstance(cam_info, list):
                                # 对每个样本的cam_info添加clip_pix
                                for i, cam_meta in enumerate(cam_info):
                                    if isinstance(cam_meta, dict):
                                        # 单帧：直接添加clip_pix (应该是单个tensor)
                                        if torch.is_tensor(clip_pix):
                                            cam_meta['clip_pix'] = clip_pix
                                        elif isinstance(clip_pix, list) and len(clip_pix) > i:
                                            # 如果clip_pix是列表，取对应的tensor
                                            cam_meta['clip_pix'] = clip_pix[i] if i < len(clip_pix) else clip_pix[0]
                                        else:
                                            cam_meta['clip_pix'] = clip_pix
                            elif isinstance(cam_info, dict):
                                # 单个样本的情况
                                if torch.is_tensor(clip_pix):
                                    cam_info['clip_pix'] = clip_pix
                                elif isinstance(clip_pix, list) and len(clip_pix) > 0:
                                    cam_info['clip_pix'] = clip_pix[0]
                                else:
                                    cam_info['clip_pix'] = clip_pix
                                cam_info = [cam_info]  # 转换为列表
                                
                        batch_inputs['cam_info'] = cam_info
                        if os.environ.get('BIFUSION_DEBUG_CAMINFO'):
                            print(f"[Det3DDataPreprocessor_] tuple imgs cam_info len={len(cam_info)}")
                
                else:
                    # 处理其他格式的图像列表
                    for i, img in enumerate(imgs):
                        processed_img = self._process_single_image(img, i)
                        if processed_img is not None:
                            tensor_imgs.append(processed_img)
                    
                    # 处理cam_info
                    if 'cam_info' in inputs:
                        cam_info = inputs['cam_info']
                        
                        # 🔧 关键修复：将clip_pix添加到cam_info中供BiFusion使用
                        if 'clip_pix' in inputs:
                            clip_pix = inputs['clip_pix']
                            if isinstance(cam_info, list):
                                for i, cam_meta in enumerate(cam_info):
                                    if isinstance(cam_meta, dict):
                                        cam_meta['clip_pix'] = clip_pix
                            elif isinstance(cam_info, dict):
                                cam_info['clip_pix'] = clip_pix
                                
                        batch_inputs['cam_info'] = cam_info
                        if os.environ.get('BIFUSION_DEBUG_CAMINFO'):
                            print(f"[Det3DDataPreprocessor_] list imgs cam_info len={len(cam_info)}")
            
            # 验证最终的图像列表
            # 对于 BiFusion 多帧路径，如果没有成功解析任何图像，静默跳过，
            # 让后续的 `img` 或无图像路径接管，避免直接抛出异常。
            if len(tensor_imgs) > 0:
                batch_inputs['imgs'] = tensor_imgs
                if os.environ.get('BIFUSION_DEBUG_CAMINFO'):
                    print(f"[Det3DDataPreprocessor_] batch imgs={len(tensor_imgs)}")

        if 'img' in inputs:
            # 原来的 img 处理逻辑（先保证类型可被 _get_pad_shape 识别）
            imgs = inputs['img']
            if isinstance(imgs, tuple):
                imgs = list(imgs)
            if isinstance(imgs, list) and len(imgs) > 0 and not torch.is_tensor(imgs[0]):
                proc_imgs = []
                for i, im in enumerate(imgs):
                    proc = self._process_single_image(im, i)
                    if proc is not None:
                        proc_imgs.append(proc)
                if len(proc_imgs) > 0:
                    imgs = proc_imgs
            pad_ready = torch.is_tensor(imgs) or (isinstance(imgs, list) and len(imgs) > 0 and torch.is_tensor(imgs[0]))
            if pad_ready:
                try:
                    batch_pad_shape = self._get_pad_shape({'inputs': {'img': imgs}})
                except Exception as e:
                    batch_pad_shape = None
                    if os.environ.get('DEBUG_CAMINFO_INPUTS') == '1':
                        print(f"[Det3DDataPreprocessor_] pad_shape failed: {e} (img type={type(imgs)})")
            else:
                batch_pad_shape = None
                if os.environ.get('DEBUG_CAMINFO_INPUTS') == '1':
                    print(f"[Det3DDataPreprocessor_] skip pad_shape for img type={type(imgs)}")

            if data_samples is not None and batch_pad_shape is not None:
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

        # 关键修复：即使没有 imgs，也要把 cam_info 透传给模型（用于 DINO 投影等）
        if 'cam_info' in inputs and 'cam_info' not in batch_inputs:
            batch_inputs['cam_info'] = inputs['cam_info']
            if os.environ.get('DEBUG_CAMINFO_INPUTS') == '1':
                print(f"[Det3DDataPreprocessor_] propagate cam_info without imgs, len={len(inputs['cam_info']) if isinstance(inputs['cam_info'], list) else 1}")

        if os.environ.get('DEBUG_CAMINFO_INPUTS') == '1':
            print(f"[Det3DDataPreprocessor_] batch_inputs keys: {list(batch_inputs.keys())}")

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
