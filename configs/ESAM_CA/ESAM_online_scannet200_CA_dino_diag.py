# Diagnostic-oriented online config: enable per-instance DINO/3D features + geom diagnostics.
# This file is intentionally minimal and only overrides test-time knobs.

_base_ = [
    './ESAM_online_scannet200_CA_dino.py',
]

# Make DINO/3D/geom diagnostics produce all matrices needed for analysis.
model = dict(
    # Keep online DINO extraction path; enable extra debug only when needed.
    # You can also set env var DINO_DEBUG=1 for extra prints.
    dino_debug=False,
    test_cfg=dict(
        track_features=dict(
            # Instance-level DINO features are pooled from per-point DINO cache.
            dino=dict(
                enable=True,
                pool='geoaware_sp',
                out_dim=256,
                normalize=True,
                use_valid_mask=True,
                min_valid_points=30,
            ),
            # Instance-level 3D pooled features (from superpoint features).
            feat3d=dict(
                enable=True,
                normalize=True,
                min_valid_points=30,
            ),
        ),
        track_geom_diag=dict(
            enable=True,
            # IMPORTANT: keep geom diagnostics comparable across runs.
            # 0.12m is substantially more robust for ScanNet-MV fused point clouds.
            voxel_size=0.12,
            diag_dino=True,
            diag_feat3d=True,
        ),
    ),
)
