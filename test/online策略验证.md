# Online 策略验证（ESAM / ScanNet200-MV / CA）

目标：把视角从“单帧诊断”拉回到 **online 场景级精度**，用最小 A/B 实验回答：
1) 重复候选（同一 GT 多 pred）在 online 中到底是主害处还是可吸收冗余？
2) 去冗余策略是否能降低 birth / 地图膨胀，并提升场景级 AP/PQ？
3) 当前 online baseline 的强剪（topk=20, inst_thr=0.25）是否在剪穿供给，导致断链/碎片化？

补充（基于本次实际问题的修订）：在进入 A/B 之前，必须先保证 OnlineMerge 的关联逻辑是“健康的”。
历史问题表现为 `matched≈0 / birth≈det_to_merge / mem_size_full 爆炸`，此时任何去冗余/供给策略结论都不可信。

---

## 1. 代码路径对齐（必须读懂）

### 1.1 每帧输出 -> online merge
- 每帧推理循环：`oneformer3d/mixformer3d.py:2031-2069`（`ScanNet200MixFormer3D_Online.predict()`）
- 单帧实例后处理：`oneformer3d/mixformer3d.py:2234`（`predict_by_feat_instance()`）
- 跨帧融合（online merge）：`oneformer3d/instance_merge.py:219`（`OnlineMerge.merge()`）

### 1.2 online baseline 的关键旋钮（当前默认）
来自 `configs/ESAM_CA/ESAM_online_scannet200_CA.py:79-91`（与官方 ESAM baseline 一致）：
- per-frame：`topk_insts=20`, `inst_score_thr=0.25`, `sp_score_thr=0.4`, `npoint_thr=100`, `nms=True`
- memory：`inscat_topk_insts=100`
- merge：`merge_type='learnable_online'`（实际上 OnlineMerge 默认走 `query_feats dot * bbox_iou`）

---

## 2. 我们已有的单帧硬证据（用于指导 online 假设）

来自 `test/1225单帧实验结果表.md`：
- `best-IoU-rank` 长尾：好候选经常 rank > 50
- 说明小 topK / 强 inst_thr 容易剪穿（红线上升）
- 重复高（hit_ge2 高），但 copy-suppress/refill 能压重复且不抬红线（在单帧里成立）

因此 online baseline 的 `topk_insts=20` 与 `inst_score_thr=0.25` 是潜在矛盾点：可能把“可续命候选”在进入 OnlineMerge 前就剪掉。

注意：上面这条“强剪假设”对应的就是官方 baseline 配置（`topk_insts=20 / inst_score_thr=0.25`），
因此“是否剪穿供给”的验证需要通过 **放宽供给**（例如增大 topk 或降低阈值）来做对照。

---

## 3. 现有代码已支持的监控（无需改代码）

### 3.1 最终场景级指标
- `UnifiedSegMetric`：输出 `all_ap`, `all_ap_50%`, `all_ap_25%` 等

### 3.2 可选：实例错误诊断（scene-level）
- `UnifiedSegMetric` 支持 `diagnostics.enable=True`，可输出：
  - purity/coverage 散点、duplicate rate、best-IoU 直方图、oracle@K/best-rank（若提供可用的 score_source）
- 注意：这只能诊断“最终 map”，不能诊断 online 过程（birth/matched/mem_size、track continuity）。

---

## 4. 本轮必须补齐的监控（否则无法做因果判断）

### 4.1 online 行为统计（过程量）
建议每帧/每 scene 统计并落盘：
- `det_to_merge`：送进 OnlineMerge 的 det 数（after postproc）
- `matched`：Hungarian 匹配到已有 track 的数
- `birth`：unmatched 新生 track 数
- `mem_size_full`：merge 后 memory track 数（未截断）
- `mem_size_kept`：用于输出的 topK memory 数（按 `inscat_topk_insts` 截断）
- `topk_drop`：memory 截断丢弃数（按 inscat_topk_insts）

这些量在 `OnlineMerge.merge()` 中都有天然落点：
- `row_ind/col_ind` -> matched
- `no_merge_masks` -> birth
- `kept_ins` -> topk_drop
现已实现（可按开关启用），输出为 JSON：
- 模型侧开关：`model.test_cfg.online_monitor.enable=True`（控制是否在 `pred_pts_seg` 中携带统计）
- evaluator 侧落盘：`test_evaluator.online_monitor.enable=True` + `test_evaluator.online_monitor.out_dir=<dir>`
  - 若 `out_dir` 为相对路径，会自动解析到 `--work-dir` 下，避免不同实验互相覆盖
- 输出文件：
  - `online_monitor/online_monitor.json`（每 scene 的 per-frame 统计）
  - `online_monitor/online_monitor_summary.json`（全数据集汇总：均值/中位数/p90/p95）

### 4.1.1 健康门槛（本轮新增：A/B 前置验收）
跑任何 A/B 前，先跑一次 O0-baseline，并检查 `online_monitor_summary.json`：
- `match_rate.mean`：建议 ≥ 0.70（本次修复后约 0.82）
- `birth_rate.mean`：建议 ≤ 0.30（本次修复后约 0.17）
- `mem_size_full.mean`：建议是几十级（本次修复后约 62；若变成几百通常表示断链）
- `topk_drop.mean`：应为小数到几十（本次修复后约 6；若上百表示地图膨胀+截断损失巨大）

### 4.2 track 连续性（ID switch / fragmentation / 断链）
当前 OnlineMerge 没有 track_id 与 det_to_track 映射输出，因此 evaluator 无法计算：
- ID switch（GT 对应的 track_id 变化次数）
- fragmentation（同一 GT 被分成多少段 track）
- 断链（GT 可见帧中未被任何 track 命中）

后续实现建议（最小改动）：
- 在 OnlineMerge 内维护 `track_ids`（单调递增），并输出每帧 `det_to_track_id`
- evaluator 侧用 per-frame mask IoU 将 GT→track_id 串起来，统计 switch/frag/miss（CA 下可忽略类别）

---

## 5. 最小 A/B 实验矩阵（先用最终指标裁决）

### 5.1 实验目标
只跑 3 个点，回答：
- 去冗余是否真的降低 birth/地图膨胀并提升 scene-level AP？
- 还是只是单帧好看，online 可吸收/无收益？

### 5.2 三个实验点（建议严格控制变量）

**共同固定：**
- 模型权重固定（预训练 online 模型）
- CA 模式评测口径固定
- evaluator 固定（UnifiedSegMetric）
- 建议额外开启 scene-level diagnostics（用于解释“最终重复/粘连/边界”）

#### O0-baseline：官方 baseline（必须先跑）
- 使用 `configs/ESAM_CA/ESAM_online_scannet200_CA.py` 当前默认（与 `/home/nebula/xxy/ESAM/configs/ESAM_CA/ESAM_online_scannet200_CA.py` 一致）
- `topk_insts=20`, `inst_score_thr=0.25`，copy_suppress 关闭
- 目的：确保链路健康、复现官方（ESAM 42.2 / 63.7 / 79.6）

#### O1-supply：放宽供给（验证“剪穿供给”）
- 在 O0-baseline 上仅改：
  - `topk_insts=100`（放宽候选池）
  - `inst_score_thr=0.25`（保持阈值不变，严格只改供给）
- 目的：验证“强剪是否剪穿续命候选”，并用 online 过程量解释（matched/birth/mem/topk_drop 的变化）

#### O2：Stage7 v2（τ=0.90 去冗余）
目标：在不靠更强 score 剪枝的情况下减少纯拷贝冗余
- 建议做成“更大候选池 + 固定输出预算 K”
  - `topk_insts = 100`（candidate pool）
  - `copy_suppress.max_num = 20`（输出预算，保证 online 输入规模不变）
  - `copy_suppress.iou_thr = 0.90`
  - `copy_suppress.refill=True`
  - `copy_suppress.allow_replace=True`
  - `copy_suppress.sort_by=scores` / `prefer_by=scores`
- 注：严格 A/B 时保持 `inst_score_thr=0.25` 与 O1-supply 一致，仅打开 copy_suppress。

#### O2b：保守去冗余（τ=0.95，可选）
- 与 O2 相同，但 `copy_suppress.iou_thr = 0.95`
- 目的：降低误伤互补候选的风险，作为“保守点”

---

## 6. 运行与结果保存规范

### 6.1 work-dir 规范
建议统一放到二级目录（每个实验一个 work-dir）：
- `work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/<EXP_NAME>/`

推荐命名（固定前缀 + 实验点 + 关键超参）：
- `O0_baseline_topk20_thr0p25`
- `O1_supply_topk100_thr0p25`
- `O2_copysuppress_tau0p90_K20`
- `O2b_copysuppress_tau0p95_K20`

### 6.2 开启 online 行为监控（推荐）
建议每次 online A/B 都开启（否则无法解释 birth/matched/mem 行为）：
- `--cfg-options model.test_cfg.online_monitor.enable=True test_evaluator.online_monitor.enable=True`
- 建议同时指定 `out_dir`（相对路径即可；`tools/test.py` 会强制解析到 `--work-dir` 下，避免不同实验互相覆盖）：
  - `test_evaluator.online_monitor.out_dir=online_monitor`

### 6.2 （可选）开启 scene-level instance diagnostics（解释最终结果）
- `test_evaluator.diagnostics.enable=True`
- `test_evaluator.diagnostics.out_dir=<work_dir 下的独立子目录>`
- `test_evaluator.diagnostics.score_source=instance_scores`（online 输出里目前没有 instance_select_scores）

---

## 7. 预期现象与判读

### 7.1 若重复是 online 主瓶颈（强成立）
- O1/O2：scene-level AP 上升或 map 重复下降
- 并且（需要补的 online 行为统计）：
  - `birth ↓`、`matched ↑`、`mem_size 更稳`、`topk_drop ↓/不变`

### 7.2 若重复可被 online 吸收（弱成立）
- O1/O2：scene-level AP 变化不显著
- 或去冗余导致 recall 降、AP 下降（说明冗余实际上在帮续命/遮挡恢复）

### 7.3 若当前主矛盾是“强剪剪穿”
- O0 可能 AP 很低/断链严重
- 当放宽供给（降低 inst_thr / 增大 topk）后 AP 会显著改善，但 birth 可能变大
- 这时去冗余应作为“配套”而不是单独动作：先把供给拉回来，再压歧义与伪新生

---

## 8. 可直接复现的命令（本轮新增）

统一约定：
- 在仓库根目录运行：`/home/nebula/xxy/3D_Reconstruction`
- CKPT：`/home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/ESAM_CA_online_epoch_128.pth`
- Config：`configs/ESAM_CA/ESAM_online_scannet200_CA.py`
- CA 评测口径：必须带 `--cat-agnostic`
- 每个实验一个 `--work-dir`，online_monitor 落到 `work_dir/online_monitor/`

通用开关（建议每次都带）：
- `--cfg-options model.test_cfg.online_monitor.enable=True test_evaluator.online_monitor.enable=True test_evaluator.online_monitor.out_dir=online_monitor`

### O0-baseline（官方 baseline）
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python tools/test.py configs/ESAM_CA/ESAM_online_scannet200_CA.py \
  /home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/ESAM_CA_online_epoch_128.pth \
  --cat-agnostic \
  --work-dir work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O0_baseline_topk20_thr0p25 \
  --cfg-options \
    model.test_cfg.online_monitor.enable=True \
    test_evaluator.online_monitor.enable=True \
    test_evaluator.online_monitor.out_dir=online_monitor
```

### O1-supply（放宽供给：topk=100）
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python tools/test.py configs/ESAM_CA/ESAM_online_scannet200_CA.py \
  /home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/ESAM_CA_online_epoch_128.pth \
  --cat-agnostic \
  --work-dir work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O1_supply_topk100_thr0p25 \
  --cfg-options \
    model.test_cfg.topk_insts=100 \
    model.test_cfg.inst_score_thr=0.25 \
    model.test_cfg.online_monitor.enable=True \
    test_evaluator.online_monitor.enable=True \
    test_evaluator.online_monitor.out_dir=online_monitor
```

### O2（copy-suppress τ=0.90, K=20）
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python tools/test.py configs/ESAM_CA/ESAM_online_scannet200_CA.py \
  /home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/ESAM_CA_online_epoch_128.pth \
  --cat-agnostic \
  --work-dir work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O2_copysuppress_tau0p90_K20 \
  --cfg-options \
    model.test_cfg.topk_insts=100 \
    model.test_cfg.inst_score_thr=0.25 \
    model.test_cfg.copy_suppress.enable=True \
    model.test_cfg.copy_suppress.iou_thr=0.90 \
    model.test_cfg.copy_suppress.max_num=20 \
    model.test_cfg.copy_suppress.allow_replace=True \
    model.test_cfg.copy_suppress.refill=True \
    model.test_cfg.copy_suppress.sort_by=scores \
    model.test_cfg.copy_suppress.prefer_by=scores \
    model.test_cfg.online_monitor.enable=True \
    test_evaluator.online_monitor.enable=True \
    test_evaluator.online_monitor.out_dir=online_monitor
```

### O2b（copy-suppress τ=0.95, K=20，可选）
```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=$(pwd):$PYTHONPATH \
python tools/test.py configs/ESAM_CA/ESAM_online_scannet200_CA.py \
  /home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/ESAM_CA_online_epoch_128.pth \
  --cat-agnostic \
  --work-dir work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O2b_copysuppress_tau0p95_K20 \
  --cfg-options \
    model.test_cfg.topk_insts=100 \
    model.test_cfg.inst_score_thr=0.25 \
    model.test_cfg.copy_suppress.enable=True \
    model.test_cfg.copy_suppress.iou_thr=0.95 \
    model.test_cfg.copy_suppress.max_num=20 \
    model.test_cfg.copy_suppress.allow_replace=True \
    model.test_cfg.copy_suppress.refill=True \
    model.test_cfg.copy_suppress.sort_by=scores \
    model.test_cfg.copy_suppress.prefer_by=scores \
    model.test_cfg.online_monitor.enable=True \
    test_evaluator.online_monitor.enable=True \
    test_evaluator.online_monitor.out_dir=online_monitor
```

---

## 9. 实验结果记录（已跑完：O0/O1/O2/O2b）

统一口径：
- dataset：ScanNet200-MV val（312 scenes / 13430 frames）
- eval：CA（`--cat-agnostic`）
- online_monitor：读取各自 work-dir 下 `online_monitor/online_monitor_summary.json`

Work-dir：
- O0：`work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O0_baseline_topk20_thr0p25`
- O1：`work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O1_supply_topk100_thr0p25`
- O2：`work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O2_copysuppress_tau0p90_K20`
- O2b：`work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/O2b_copysuppress_tau0p95_K20`

### 9.1 汇总表（最终场景级 + online 行为）

|Exp|AP|AP50|AP25|det_to_merge mean|det_to_merge p95|birth mean|mem_size_full mean|mem_size_full p95|topk_drop mean|topk_drop p95|
|-:|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|O0|0.4135|0.6300|0.7886|9.243|17|1.784|62.934|143|6.002|43|
|O1|0.4145|0.6317|0.7894|9.817|20|1.908|67.687|162|8.860|62|
|O2|0.4148|0.6316|0.7886|9.573|20|1.851|65.610|152|7.438|52|
|O2b|0.4148|0.6315|0.7886|9.574|20|1.851|65.615|152|7.438|52|

备注：
- O2 vs O2b 的 online 行为几乎完全一致（说明在当前工作点上 `copy_suppress.iou_thr` 从 0.90→0.95 基本不影响系统行为与最终精度）。
- 若日志里出现 `[UnifiedSegMetric] online monitor saved to: ...` 指向异常路径，以 `work_dir/online_monitor/` 内的 JSON 为准。

### 9.2 结论（基于上述 4 点的“可裁决”结论）

1) **最终 scene-level AP 基本持平**：O1/O2/O2b 相对 O0 只有 ~0.001 量级变化，不足以说明“去冗余/增供给”已显著提升在线重建精度上限。

2) **O1（topk 20→100）没有真正“拉回供给”**：`det_to_merge mean` 仅 9.243→9.817，说明 online 输入规模主要仍由 `inst_score_thr=0.25` 与 `npoint_thr=100` 限制，而不是 `topk_insts`。

3) **copy-suppress 当前主要在少数拥挤帧触发**：均值 `det_to_merge` 远小于 K=20，导致 τ 的改变难以产生系统级差异；其更现实的价值是“资源/膨胀风险控制”，而非直接抬 AP。

4) **memory 截断风险仍在**：O1 的 `mem_size_full/topk_drop` 明显高于 O0；O2/O2b 比 O1 有所缓解但仍未彻底压住（`mem_size_full p95` 仍在 152 左右）。

---

## 10. 下一步（推荐的最小 online 因果验证）

为了真正验证“强剪是否剪穿 / 去冗余是否能压住 birth 与膨胀”，下一步应优先做：

- C1：仅放宽 `inst_score_thr`（例如 0.25→0.05 或 0.0），其余保持 O0（可以先不改 topk）
  - 目标：让 `det_to_merge` 真正上升，再观察 `matched/birth/mem_size_full/topk_drop` 的联动。
- C2：在 C1 基础上加 `copy_suppress.max_num=20`（保持在线输入预算不爆）
  - 目标：在“供给回来了”的前提下检验去冗余是否能压 `birth`、稳 `mem_size_full`，并最终带来 scene-level AP/PQ 改善。
