# 12/27 Online 3D Reconstruction Experiment Plan

## 1. Experiment Objectives
1.  **Validation**: Verify if YOLO+FastSAM proposals reduce 3D repetition (`hit_ge2`) compared to FastSAM-only.
2.  **Analysis**: Collect "Online Health Metrics" to diagnose the online merging process.

## 2. Metrics Definition
| Category | Metric Name | Description | Implementation Source |
| :--- | :--- | :--- | :--- |
| **Supply** | `det_final_to_merge` | Number of instances provided by 2D detector per frame | `OnlineMerge.merge` input size |
| **Event** | `matched` | Number of successful merges with existing memory | `OnlineMerge` Hungarian match count |
| **Event** | `birth` | Number of new instances created | `OnlineMerge` unmatched count |
| **Map** | `mem_size` | Total number of instances in memory | `OnlineMerge.cur_scores` size |
| **Map** | `topk_drop` | Number of instances dropped due to memory limit | `OnlineMerge` topk truncation diff |
| **Continuity** | `GT_matched@0.1` | Percentage of GT instances matched by current memory (IoU > 0.1) | `MixFormer3D` loop (GT vs Memory slice) |

## 3. Implementation Plan
### Phase 1: Code Modification
1.  **`oneformer3d/instance_merge.py`**:
    *   Update `OnlineMerge.merge` signature to return a `stats` dictionary.
    *   Calculate `supply`, `matched`, `birth`, `drop`, `mem_size` inside `merge`.
2.  **`oneformer3d/mixformer3d.py`**:
    *   Add `collect_online_metrics` switch in config handling.
    *   In `predict` loop:
        *   Receive `stats` from `merge`.
        *   Calculate `GT_matched@0.1` using `batch_data_samples` (GT) and `online_merger.cur_masks` (Memory).
        *   Aggregate metrics per frame.
    *   Save metrics to `online_metrics.json` in `work_dirs`.

### Phase 2: Experiments (Parallel)
| Exp ID | 2D Detector | Merge Strategy | Note |
| :--- | :--- | :--- | :--- |
| **Exp 1** | FastSAM (Baseline) | Online (Default) | Baseline for repetition |
| **Exp 2** | YOLO+FastSAM | Online (Default) | Main hypothesis validation |
| **Exp 3** | FastSAM | Online (Strict IoU) | Ablation (if needed) |
| **Exp 4** | YOLO+FastSAM | Online (Strict IoU) | Ablation (if needed) |

## 4. Robustness & Safety
*   **Config Switch**: `test_cfg.collect_online_metrics = True/False` (Default False).
*   **Output Path**: Ensure `work_dirs` is unique for each experiment to avoid overwriting.
*   **Error Handling**: Graceful degradation if GT is missing or dimensions mismatch.
