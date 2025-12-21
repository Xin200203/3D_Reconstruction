from mmengine.dataset import default_collate
from mmengine.registry import FUNCTIONS


def _normalize_cam_info_batch(cam_info):
    # cam_info can be dict (already batched) or list-of-one wrapper
    if isinstance(cam_info, list) and len(cam_info) == 1:
        cam_info = cam_info[0]

    if not isinstance(cam_info, dict):
        return cam_info

    # intrinsics: ensure Tensor(B,4)
    intr = cam_info.get('intrinsics', None)
    if intr is not None:
        if isinstance(intr, list) and len(intr) == 1:
            intr = intr[0]
        if hasattr(intr, 'reshape'):
            intr = intr.reshape(-1, 4)
        cam_info['intrinsics'] = intr

    img_size = cam_info.get('img_size_dino', None)
    if img_size is not None:
        if isinstance(img_size, list) and len(img_size) == 1:
            img_size = img_size[0]
        if hasattr(img_size, 'reshape'):
            img_size = img_size.reshape(-1, 2)
        cam_info['img_size_dino'] = img_size

    return cam_info


@FUNCTIONS.register_module()
def esam_collate(batch):
    data = default_collate(batch)
    inputs = data.get('inputs', None)
    if isinstance(inputs, dict) and 'cam_info' in inputs:
        inputs['cam_info'] = _normalize_cam_info_batch(inputs['cam_info'])
        data['inputs'] = inputs
    return data
