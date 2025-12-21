import torch
import torch.nn.functional as F
from typing import Optional

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from tools.extract_dinov2_features import DINOv2FeatureExtractor


@MODELS.register_module()
class DINOv2Backbone(BaseModule):
    """Thin wrapper around DINOv2 for online 2D feature extraction.

    - 接收已经做完 2D 数据增强、resize 到固定分辨率 (420×560) 的图像张量；
    - 使用冻结的 DINOv2 ViT 提取 patch token 特征；
    - 输出特征图形状为 (B, C, H_p, W_p)，例如 (B, 1024, 30, 40)；
    - 不对输入图像做任何几何变换（不 resize / flip），仅做归一化与 pad 到 patch size 的倍数。
    """

    def __init__(self,
                 arch: str = 'dinov2_vitl14',
                 checkpoint: Optional[str] = None,
                 device: str = 'cuda') -> None:
        super().__init__()
        self.arch = arch
        self.device = device
        self.checkpoint = checkpoint

        # 使用已有的特征提取器加载 DINOv2 模型（包括本地权重逻辑）
        extractor = DINOv2FeatureExtractor(
            arch=self.arch,
            device=self.device,
            dtype=torch.float32,  # 在内部再根据权重 dtype 调整
            checkpoint=self.checkpoint)
        self.model = extractor.model
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        # DINOv2 归一化参数（ImageNet 标准）
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False)
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False)

        # 对应 ViT-14 的 patch size，后续用于 pad 到整数倍
        self.patch_size: int = 14

    @torch.no_grad()
    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            imgs (Tensor): (B, 3, H, W) 输入图像，已经在 pipeline 中 resize
                到固定分辨率 (例如 420×560)，且包含所有 2D 数据增强。

        Returns:
            Tensor: (B, C, H_p, W_p) 的 DINOv2 patch 特征图。
        """
        assert imgs.dim() == 4 and imgs.size(1) == 3, \
            f'Expect imgs in (B,3,H,W), got shape={tuple(imgs.shape)}'

        B, _, H, W = imgs.shape
        device = next(self.model.parameters()).device
        weight_dtype = next(self.model.parameters()).dtype

        # 将输入移动到与 DINO 模型相同的 device / dtype
        x = imgs.to(device=device, dtype=torch.float32)

        # 归一化到 [0,1]
        # 训练数据通常是 0-255，这里统一除以 255；对 0-1 的输入不会造成问题（仅缩放）。
        x = x / 255.0

        # 使用 ImageNet 归一化参数
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        # pad 到 patch_size 的倍数（只在下 / 右填充，避免几何失真）
        pad_h = (self.patch_size - H % self.patch_size) % self.patch_size
        pad_w = (self.patch_size - W % self.patch_size) % self.patch_size
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='constant', value=0.0)

        _, _, H_pad, W_pad = x.shape
        x = x.to(dtype=weight_dtype)

        # 调用 DINOv2 的 forward_features，获得 patch tokens
        feats = self.model.forward_features(x)  # type: ignore[call-arg]

        if isinstance(feats, dict):
            # 优先使用标准键；不同版本 DINO 可能有差异
            if 'x_norm_patchtokens' in feats:
                tokens = feats['x_norm_patchtokens']
            elif 'x_prenorm' in feats:
                tokens = feats['x_prenorm']
            else:
                tokens = None
            h = feats.get('h', None)
            w = feats.get('w', None)
            if isinstance(h, torch.Tensor):
                h = int(h.item())
            if isinstance(w, torch.Tensor):
                w = int(w.item())
        else:
            # 退化情况：直接将输出当作 tokens 处理
            tokens, h, w = feats, None, None

        if tokens is None:
            raise RuntimeError('DINOv2Backbone: forward_features did not return patch tokens.')

        # tokens: (B, L, C)
        B2, L, C = tokens.shape
        assert B2 == B, f'Batch size mismatch: B_model={B2}, B_input={B}'

        if h is None or w is None:
            # 若未提供网格尺寸，则根据 pad 后的尺寸与 patch_size 推断
            h = H_pad // self.patch_size
            w = W_pad // self.patch_size

        # 理论上 h*w 应等于 L；如果不等则作为兜底按 sqrt(L) 处理
        if h * w > L:
            hw = int(L ** 0.5)
            h = w = hw

        tokens = tokens[:, :h * w, :]  # (B, h*w, C)
        feat_maps = tokens.transpose(1, 2).contiguous().view(B, C, h, w)

        # 轻量调试：记录最近一次的特征图尺寸，便于在 notebook 中查看
        self._last_feat_shape = feat_maps.shape  # type: ignore[attr-defined]

        return feat_maps
