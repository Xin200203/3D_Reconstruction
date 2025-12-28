"""Semantic-focused fine-tuning config (single-frame, online DINO injection).

Based on: configs/ESAM_CA/ESAM_sv_scannet200_CA_dino.py

Goal:
- Keep the same backbone + MixQueryDecoder (semantic head lives in decoder).
- Disable instance losses (semantic-only optimization) for fastest iteration.
- Switch evaluator / best checkpoint key to semantic mIoU.
"""

_base_ = ["./ESAM_sv_scannet200_CA_dino.py"]

# Disable instance losses (keep forward path intact, but no gradients from inst task).
model = dict(
    criterion=dict(
        inst_criterion=dict(
            loss_weight=[0.0, 0.0, 0.0, 0.0, 0.0],
        )
    )
)

# Semantic-only evaluator (mIoU / mAcc / aAcc).
# Metric will fall back to runner-injected dataset_meta for classes/ignore_index.
val_evaluator = dict(type="SemanticSegMetric")
test_evaluator = val_evaluator

# Save best by semantic metric.
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        save_best=["miou"],
        rule="greater",
    )
)

# Finetune schedule (typical: 10~20 epochs with smaller LR).
optim_wrapper = dict(
    optimizer=dict(lr=2e-5),
)
param_scheduler = dict(type="PolyLR", begin=0, end=20, power=0.9)
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=20, val_interval=5)
