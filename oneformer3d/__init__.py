from .oneformer3d import (
    ScanNetOneFormer3D, ScanNet200OneFormer3D, S3DISOneFormer3D, ScanNet200OneFormer3D_Online)
from .mixformer3d import ScanNet200MixFormer3D, ScanNet200MixFormer3D_Online
from .geo_aware_pool import GeoAwarePooling
from .instance_merge import ins_merge_mat, ins_cat, ins_merge, OnlineMerge, GTMerge
from .merge_head import MergeHead
from .merge_criterion import ScanNetMergeCriterion_Seal
from .multilevel_memory import MultilevelMemory
from .mink_unet import Res16UNet34C
from .query_decoder import ScanNetQueryDecoder, S3DISQueryDecoder
from .unified_criterion import (
    ScanNetUnifiedCriterion, ScanNetMixedCriterion, S3DISUnifiedCriterion)
from .semantic_criterion import (
    ScanNetSemanticCriterion, S3DISSemanticCriterion)
from .instance_criterion import (
    InstanceCriterion, MixedInstanceCriterion, QueryClassificationCost, MaskBCECost,
    MaskDiceCost, HungarianMatcher, SparseMatcher)
from .loading import LoadAnnotations3D_, NormalizePointsColor_
from .formatting import Pack3DDetInputs_
# Enhanced loss functions for BiFusion optimization
from .bife_clip_loss import ClipConsCriterion, LegacyClipConsCriterion
from .auxiliary_loss import SpatialConsistencyLoss, NoViewSupervisionLoss
from .transforms_3d import (
    ElasticTransfrom, AddSuperPointAnnotations, SwapChairAndFloor)
from .data_preprocessor import Det3DDataPreprocessor_
from .unified_metric import UnifiedSegMetric
from .scannet_dataset import ScanNetSegDataset_, ScanNet200SegDataset_, ScanNet200SegMVDataset_
from .time_divided_transformer import TimeDividedTransformer
from .bi_fusion_encoder import BiFusionEncoder
from .bife_clip_loss import ClipConsCriterion
# 将所有在 mmdet3d.registry.MODELS 中注册的模块同步到 mmengine.registry.MODELS，
# 以便 mmengine 的 BaseModel 能够直接查询到自定义组件（如 Det3DDataPreprocessor_）。
from mmdet3d.registry import MODELS as _M3D_MODELS  # noqa: E402
from mmengine.registry import MODELS as _MM_MODELS  # noqa: E402
for _name, _module in _M3D_MODELS.module_dict.items():
    if _name not in _MM_MODELS.module_dict:
        # 使用强制注册，避免与已有同名冲突时报错
        _MM_MODELS.register_module(name=_name, force=True)(_module)