#!/usr/bin/env bash
set -euo pipefail

# Stage 2: candidate budget sweeps for single-frame (SV) evaluation.
# - Runs in category-agnostic (object) mode via --cat-agnostic.
# - Writes each experiment to its own work_dir to avoid overwrites.
#
# Usage:
#   bash tools/run_stage2_single_frame_ablation.sh \
#     /home/nebula/xxy/3D_Reconstruction/work_dirs/ESAM_sv_scannet200_CA_dino/epoch_35.pth
#
# Optional env overrides:
#   CONFIG=...          (default: configs/ESAM_CA/ESAM_sv_scannet200_CA_dino.py)
#   ROOT_WORK=...       (default: work_dirs/ESAM_sv_scannet200_CA_dino_1225)
#   SP_THR=0.45         (default: 0.45; chosen from Stage 1 sweep)
#   TOPK_LIST="100 50 30 20"
#   FIX_TOPK_FOR_INST=50
#   INST_THR_LIST="0.00 0.05 0.10 0.20 0.25"

CKPT="${1:-}"
if [[ -z "${CKPT}" ]]; then
  echo "Missing checkpoint path."
  echo "Example:"
  echo "  bash tools/run_stage2_single_frame_ablation.sh /abs/path/to/epoch_xx.pth"
  exit 2
fi
if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}"
  exit 2
fi

CONFIG="${CONFIG:-configs/ESAM_CA/ESAM_sv_scannet200_CA_dino.py}"
ROOT_WORK="${ROOT_WORK:-work_dirs/ESAM_sv_scannet200_CA_dino_1225}"
SP_THR="${SP_THR:-0.45}"
TOPK_LIST="${TOPK_LIST:-100 50 30 20}"
FIX_TOPK_FOR_INST="${FIX_TOPK_FOR_INST:-50}"
INST_THR_LIST="${INST_THR_LIST:-0.00 0.05 0.10 0.20 0.25}"

tag_sp="${SP_THR/./p}"

echo "[Stage2] config=${CONFIG}"
echo "[Stage2] ckpt=${CKPT}"
echo "[Stage2] root_work=${ROOT_WORK}"
echo "[Stage2] sp_score_thr=${SP_THR}"

mkdir -p "${ROOT_WORK}"

echo ""
echo "== Stage 2.1: topk_insts sweep (fixed inst_score_thr=0.0) =="
for topk in ${TOPK_LIST}; do
  exp="ESAM_sv_scannet200_CA_dino_eval_stage2_topk${topk}_sp${tag_sp}"
  work_dir="${ROOT_WORK}/${exp}"
  echo ""
  echo "[RUN] ${exp}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/test.py \
    "${CONFIG}" \
    "${CKPT}" \
    --cat-agnostic \
    --work-dir "${work_dir}" \
    --cfg-options \
      model.test_cfg.sp_score_thr="${SP_THR}" \
      model.test_cfg.topk_insts="${topk}" \
      model.test_cfg.inst_score_thr=0.0
done

echo ""
echo "== Stage 2.2: inst_score_thr sweep (fixed topk_insts=${FIX_TOPK_FOR_INST}) =="
for inst_thr in ${INST_THR_LIST}; do
  tag_thr="${inst_thr/./p}"
  exp="ESAM_sv_scannet200_CA_dino_eval_stage2_inst${tag_thr}_topk${FIX_TOPK_FOR_INST}_sp${tag_sp}"
  work_dir="${ROOT_WORK}/${exp}"
  echo ""
  echo "[RUN] ${exp}"
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python tools/test.py \
    "${CONFIG}" \
    "${CKPT}" \
    --cat-agnostic \
    --work-dir "${work_dir}" \
    --cfg-options \
      model.test_cfg.sp_score_thr="${SP_THR}" \
      model.test_cfg.topk_insts="${FIX_TOPK_FOR_INST}" \
      model.test_cfg.inst_score_thr="${inst_thr}"
done

echo ""
echo "[Stage2] Done."
echo "Each experiment writes diagnostics to: <work_dir>/instance_diagnostics/"

