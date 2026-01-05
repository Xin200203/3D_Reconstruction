#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
FORCE="${FORCE:-0}"

CONFIG="configs/ESAM_CA/ESAM_online_scannet200_CA.py"
CKPT="/home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/ESAM_CA_online_epoch_128.pth"
DATA_ROOT="/home/nebula/xxy/dataset/data/scannet200-mv_fast"
ANN_FILE="/home/nebula/xxy/dataset/data/scannet200-mv_fast/scannet200_mv_oneformer3d_infos_val.pkl"

BASE_OUT_ROOT="work_dirs/ESAM_online_scannet200_CA_mv_fast_ab"

common_cfg_opts=(
  "val_dataloader.dataset.data_root=${DATA_ROOT}"
  "val_dataloader.dataset.ann_file=${ANN_FILE}"
  "test_dataloader.dataset.data_root=${DATA_ROOT}"
  "test_dataloader.dataset.ann_file=${ANN_FILE}"
  "model.test_cfg.topk_insts=100"
  "model.test_cfg.inscat_topk_insts=100"
  "model.test_cfg.inst_score_thr=0.3"
  "model.test_cfg.sp_score_thr=0.4"
  "model.test_cfg.pan_score_thr=0.5"
  "model.test_cfg.npoint_thr=100"
  "model.test_cfg.nms=True"
  "model.test_cfg.matrix_nms_kernel=linear"
  "model.test_cfg.obj_normalization=True"
  "model.test_cfg.merge_type=learnable_online"

  # Monitoring (A/B/C panels)
  "model.test_cfg.online_monitor.enable=True"
  "test_evaluator.online_monitor.enable=True"
  "test_evaluator.online_monitor.out_dir=online_monitor"
  "model.test_cfg.baseline_stats.enable=True"
  "model.test_cfg.baseline_stats.pre_pool=after_nms"
  "test_evaluator.baseline_stats.enable=True"
  "test_evaluator.baseline_stats.out_dir=baseline_stats"

  # Step0 diagnostics (record-only, no behavior change)
  "model.test_cfg.geom_merge.stats.enable=True"
  "model.test_cfg.geom_merge.stats.record_union_metrics=True"
  "model.test_cfg.geom_merge.stats.record_gt_delta_iou=True"
  "model.test_cfg.geom_merge.stats.record_semantic=True"
  "model.test_cfg.geom_merge.stats.max_events_per_frame=50"

  # Enable geom_merge across the series
  "model.test_cfg.geom_merge.enable=True"
  "model.test_cfg.geom_merge.max_num=100"
  "model.test_cfg.geom_merge.sort_by=scores"
  "model.test_cfg.geom_merge.prefer_by=scores"
)

run_exp() {
  local exp_name="$1"
  shift
  local work_dir="${BASE_OUT_ROOT}/${exp_name}"

  mkdir -p "${work_dir}"

  if [[ "${FORCE}" != "1" && -f "${work_dir}/online_monitor/online_monitor_summary.json" && -f "${work_dir}/baseline_stats/baseline_stats_summary.json" ]]; then
    echo "[skip] ${exp_name} already has summaries: ${work_dir}"
    return 0
  fi

  echo "[run] ${exp_name} -> ${work_dir}"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
  python tools/test.py "${CONFIG}" "${CKPT}" \
    --cat-agnostic \
    --work-dir "${work_dir}" \
    --cfg-options \
    "${common_cfg_opts[@]}" \
    "$@"
}

# E1: duplicate-only (keep-best)
run_exp "E1_keepbest_dup_only_topk100_thr0p30_sp0p40" \
  "model.test_cfg.geom_merge.mode=keep_best" \
  "model.test_cfg.geom_merge.semantic_veto.enable=False" \
  "model.test_cfg.geom_merge.duplicate_criteria.use_point_iou_refine=False"

# E2: E1 + point IoU refine + stricter geometry
run_exp "E2_keepbest_refine_geomstrict_topk100_thr0p30_sp0p40" \
  "model.test_cfg.geom_merge.mode=keep_best" \
  "model.test_cfg.geom_merge.duplicate_criteria.iou_box_thr=0.85" \
  "model.test_cfg.geom_merge.duplicate_criteria.center_norm_thr=0.15" \
  "model.test_cfg.geom_merge.duplicate_criteria.size_ratio_min=0.50" \
  "model.test_cfg.geom_merge.duplicate_criteria.use_point_iou_refine=True" \
  "model.test_cfg.geom_merge.duplicate_criteria.iou_pts_thr=0.80" \
  "model.test_cfg.geom_merge.semantic_veto.enable=False"

# E3: E2 + semantic veto (conservative veto)
run_exp "E3_keepbest_refine_semveto_topk100_thr0p30_sp0p40" \
  "model.test_cfg.geom_merge.mode=keep_best" \
  "model.test_cfg.geom_merge.duplicate_criteria.iou_box_thr=0.85" \
  "model.test_cfg.geom_merge.duplicate_criteria.center_norm_thr=0.15" \
  "model.test_cfg.geom_merge.duplicate_criteria.size_ratio_min=0.50" \
  "model.test_cfg.geom_merge.duplicate_criteria.use_point_iou_refine=True" \
  "model.test_cfg.geom_merge.duplicate_criteria.iou_pts_thr=0.80" \
  "model.test_cfg.geom_merge.semantic_veto.enable=True" \
  "model.test_cfg.geom_merge.semantic_veto.use_semantic_label_gate=True" \
  "model.test_cfg.geom_merge.semantic_veto.conf_thr=0.60" \
  "model.test_cfg.geom_merge.semantic_veto.use_query_cos_veto=True" \
  "model.test_cfg.geom_merge.semantic_veto.query_cos_thr=0.20"

# E4: E3 + controlled union (fragment absorb with expansion constraints)
run_exp "E4_controlled_union_topk100_thr0p30_sp0p40" \
  "model.test_cfg.geom_merge.mode=controlled_union" \
  "model.test_cfg.geom_merge.duplicate_criteria.iou_box_thr=0.85" \
  "model.test_cfg.geom_merge.duplicate_criteria.center_norm_thr=0.15" \
  "model.test_cfg.geom_merge.duplicate_criteria.size_ratio_min=0.50" \
  "model.test_cfg.geom_merge.duplicate_criteria.use_point_iou_refine=True" \
  "model.test_cfg.geom_merge.duplicate_criteria.iou_pts_thr=0.80" \
  "model.test_cfg.geom_merge.semantic_veto.enable=True" \
  "model.test_cfg.geom_merge.semantic_veto.use_semantic_label_gate=True" \
  "model.test_cfg.geom_merge.semantic_veto.conf_thr=0.60" \
  "model.test_cfg.geom_merge.semantic_veto.use_query_cos_veto=True" \
  "model.test_cfg.geom_merge.semantic_veto.query_cos_thr=0.20" \
  "model.test_cfg.geom_merge.controlled_union.small_ratio_max=0.30" \
  "model.test_cfg.geom_merge.controlled_union.cov_thr=0.90" \
  "model.test_cfg.geom_merge.controlled_union.expansion_ratio_max=1.20" \
  "model.test_cfg.geom_merge.controlled_union.bbox_expand_ratio_max=1.50"

# E5: keep-best + many-to-one absorb (cross-frame supplement)
run_exp "E5_keepbest_many2one_absorb_topk100_thr0p30_sp0p40" \
  "model.test_cfg.geom_merge.mode=keep_best" \
  "model.test_cfg.geom_merge.semantic_veto.enable=False" \
  "model.test_cfg.geom_merge.duplicate_criteria.use_point_iou_refine=False" \
  "model.test_cfg.many_to_one_absorb.enable=True" \
  "model.test_cfg.many_to_one_absorb.only_to_matched=True" \
  "model.test_cfg.many_to_one_absorb.iou_thr=0.50" \
  "model.test_cfg.many_to_one_absorb.score_thr=0.00" \
  "model.test_cfg.many_to_one_absorb.min_mask_points=20" \
  "model.test_cfg.many_to_one_absorb.max_mask_points=500" \
  "model.test_cfg.many_to_one_absorb.max_absorb_per_frame=0" \
  "model.test_cfg.many_to_one_absorb.update_score=max"

# E6: one-to-many supporters association (coverage-based), suppress-birth only (no union)
run_exp "E6_supporters_cov_topk100_thr0p30_sp0p40" \
  "model.test_cfg.geom_merge.mode=keep_best" \
  "model.test_cfg.geom_merge.semantic_veto.enable=False" \
  "model.test_cfg.geom_merge.duplicate_criteria.use_point_iou_refine=False" \
  "model.test_cfg.many_to_one_absorb.enable=False" \
  "model.test_cfg.one_to_many_support.enable=True" \
  "model.test_cfg.one_to_many_support.only_to_matched=True" \
  "model.test_cfg.one_to_many_support.cov_thr=0.85" \
  "model.test_cfg.one_to_many_support.small_ratio_max=0.30" \
  "model.test_cfg.one_to_many_support.center_norm_thr=0.25" \
  "model.test_cfg.one_to_many_support.score_thr=0.00" \
  "model.test_cfg.one_to_many_support.min_mask_points=20" \
  "model.test_cfg.one_to_many_support.max_mask_points=500" \
  "model.test_cfg.one_to_many_support.max_support_per_frame=0"

echo "[done] All experiments finished."
