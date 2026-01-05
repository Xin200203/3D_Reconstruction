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

### 4.3 baseline_stats：逐帧“过滤杀掉什么 + 同帧重复”统计（本轮新增）

目的：回答三个可证伪问题（baseline-only，不改策略）：
1) `det_to_merge` 为何不大：到底被 `nms / inst_score_thr / npoint_thr / copy_suppress` 哪个闸门控制？
2) 过滤掉的是垃圾还是“有用但被杀”的关键碎片（`killed_useful@0.5`）？
3) 过滤前后的同帧重复（GT 视角 `hit_ge2@0.5/hit0@0.5`）是否真的被压下，代价是什么？

实现落点（与代码一致）：
- 统计在 `predict_by_feat_instance()` 内完成（逐帧点云空间），并在 `ScanNet200MixFormer3D_Online.predict()` 汇总 GT birth/mem 与 OnlineMerge 行为量。

开关与输出：
- 模型侧：`model.test_cfg.baseline_stats.enable=True`
  - 建议同时设置 `model.test_cfg.baseline_stats.record_online=True`（默认 True）：即使不开 `online_monitor`，也会让 `OnlineMerge` 生成 `last_stats`，从而在 baseline_stats 里写入 `mem_size_full/topk_drop/...` 并计算 `inflation`。
  - 可选参数（建议先用默认）：`gt_vis_npoint=100`, `iou_thr=0.5`, `iou_lo_thr=0.1`, `pre_pool=after_nms`
- evaluator 侧落盘：`test_evaluator.baseline_stats.enable=True`
  - `test_evaluator.baseline_stats.out_dir=baseline_stats`（相对路径自动落在 `--work-dir` 下）
- 输出：
  - `baseline_stats/baseline_stats.json`（raw，按 scene→frames）
  - `baseline_stats/baseline_stats_summary.json`（汇总：stage counts/drops、killed_useful、dup pre/post、GT birth/mem、inflation）

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

## 5.3 更新后的主线计划（新增：单帧几何合并 → 阈值拆分 → many-to-one 吸收）

结论同步（基于你们已跑完的 O0/O1/O2/O2b 与 baseline_stats 证据）：
- O0 的 `after_topk≈20`，候选池太小，在这种设置下讨论“单帧合并”会被“候选不足”掩盖。
- O1 虽然 `topk=100`，但关键瓶颈是 `inst_score_thr` 在 **inst_thr 阶段大量砍掉 useful 候选**，所以“放宽 topk 但不改变选择/合并位置”很可能没有收益。

因此：单帧合并必须绑定 O1（扩大候选池）并且**插在 `inst_score_thr` 之前**，才有机会把“被阈值砍掉但有用的碎片/拷贝信息”转化为可保留的更完整候选。

### 实验点总览（沿用已完成实验作为对照基座）
- **E0 = O0**：`topk=20 + inst_thr=0.25`（已完成）
- **E0’ = O1**：`topk=100 + inst_thr=0.25`（已完成）
- **E0’’ = O2/O2b**：`topk=100 + inst_thr=0.25 + copy_suppress(K=20, τ=0.90/0.95)`（已完成）
- **E1（新增，关键）= O1 + 单帧几何合并再过滤（merge→K=20）**
- **E2（新增，第二关键）= E1 + assoc/birth 阈值拆分（assoc宽，birth严）**
- **E3（新增，第三关键）= E2 + two-stage many-to-one absorb**
- **E4（可选）= E3 + delayed confirm（tentative/confirm）**

> 推荐推进顺序：E1 →（若有效）E2 →（若 inflation 仍高/scene_dup@0.1 仍高）E3 →（若 birth 污染仍明显）E4。

| 实验点 | 以谁为底 | 核心改动（只写差异） | 主要要裁决的问题 | 必看指标（P0→P2） |
|---|---|---|---|---|
| E0(O0) | - | `topk=20, inst_thr=0.25` | 官方 baseline 行为基线 | scene AP；online_monitor/baseline_stats 全套 |
| E0'(O1) | E0 | `topk=100` | 放宽候选池是否能改善在线供给 | `killed_useful@0.5(inst_thr)`、`det_to_merge_hit0@0.5`、`inflation` |
| E0''(O2/O2b) | E0' | copy_suppress(K=20, τ=0.90/0.95) | 仅去“纯拷贝”是否影响 online | `birth_rate/match_rate`、`inflation`、scene AP |
| E1 | E0' | **单帧几何合并（inst_thr 前）→ 截断 K=20** | 合并能否把 useful 信息“救回”并降低漏检 | `scene_dup_iou05.hit0`、`det_to_merge_hit0@0.5`、`killed_useful@0.5(inst_thr)` |
| E2 | E1 | assoc/birth 阈值拆分（assoc宽、birth严） | 供给↑ 是否能被吸收而不 birth 爆 | `inflation`、`birth.mean`、`match_rate` |
| E3 | E2 | two-stage many-to-one absorb | 解决碎片化/同物多 track 是否有效 | `inflation`、`scene_dup_iou01.hit_ge2`、`topk_drop.p95` |
| E4 | E3 | delayed confirm | 处理一次性碎片/误检，进一步降污染 | `birth.mean`、`inflation`、scene AP |

---

## 5.4 E1：单帧几何合并再过滤（几何一致性优先）

### 为什么 E1 要以 O1 为底（而不是 O0）
- O0 的 pre_pool 近似 top20，互补碎片/低分但有用候选往往不在池里，合并“无材料”。
- E1 的目标是：**topk 放宽拿到碎片 → 合并成更完整候选 → 再让 inst_thr 去砍**，从而降低 `det_to_merge` 与 scene-level 的漏检。

### 插入位置（必须）
- 单帧后处理 `predict_by_feat_instance()` 中：`mask_pred = sigmoid(mask) > sp_score_thr` 之后、`score_mask = scores > inst_score_thr` 之前。
- 目的：让被 `inst_score_thr` 原本会砍掉的候选也能参与合并，从而“被救回”。

### 合并判据（E1 先做 duplicate merge：高一致性、低误并）
建议先用 bbox/几何为主（score 不可信、特征未必稳）：
- `IoU_bbox >= τ_iou_dup`（建议起步 **0.7**，更像“重复/拷贝”而不是“补全碎片”）
- `center_dist / diag <= τ_c`（建议 **0.25**）
- `min(size_ratio, 1/size_ratio) >= τ_s`（建议 **0.25**）
- 可选精筛：仅对通过 bbox gate 的 pair 计算点集 IoU，要求 `IoU_pts >= τ_pts`（如 0.6）

### 合并方式（避免把强候选拉低）
- mask：union（OR）
- score：`max(score)`（不要平均）
- query/feat：可选（先不做以减少变量；或 score 加权平均）
- **合并后再截断到固定预算 `K=20`**（保证送入 OnlineMerge 的规模不爆）

### E1 成功判据（必须能裁决“值不值”）
以 baseline_stats / online 行为量 / scene-level 指标三层同时判读：
- **供给红线（必须不变或改善）**
  - `det_to_merge_hit0@0.5` 下降（更少漏检进入融合）
  - `scene_dup_iou05.hit0` 下降（最终漏检下降）
- **不制造新重复（必须不恶化）**
  - `det_to_merge_hit_ge2@0.5` 不显著上升
- **解释因果（强建议）**
  - `killed_useful@0.5(inst_thr)` 下降，或至少“killed_useful 仍高但 scene_hit0 下降”（说明 useful 被并入保留候选）
  - online `inflation = mem_size_full / gt_mem` 不恶化（否则可能引入更多碎片 track）

---

## 5.5 E2：assoc/birth 阈值拆分（只让低分候选参与关联，不放任 birth）

目标：允许更多候选参与 **update existing track**，但 birth 仍严格，避免伪新生污染。
- `assoc_thr`：低（0~0.05），用于“参与匹配/更新”的资格
- `birth_thr`：保持严格（0.25 或更高），用于“允许新生”的资格
- npoint 也建议拆分（与上面一致的逻辑）：`npoint_assoc` 低、`npoint_birth` 高（默认仍可保持 100 做 birth）

核心观察：
- `det_to_merge.mean` ↑（预期）
- `match_rate` ↑（预期）
- `birth.mean` 不应爆（核心约束）
- `inflation` 应下降或至少不恶化（否则供给进来了但没被吸收，碎片化会更严重）

---

## 5.6 E3：two-stage many-to-one absorb（先 1-1，再吸收碎片）

动机：解决“一个 GT 对多个 pred/碎片”的跨帧累积碎片化问题（inflation 高、scene_dup@0.1 高）。
- Stage-1：仍用 Hungarian 做 1-to-1 主匹配（保持稳定）
- Stage-2：对 unmatched det，如果满足“碎片判据”（几何一致性/高重叠/包含关系），允许 many-to-one 吸收到已匹配/已存在 track（只做 mask/几何补全，弱更新特征）

主要看：
- `inflation` 明显下降（many-to-one 的首要目标）
- `scene_dup_iou01.hit_ge2` 下降（宽松阈值下的重复覆盖下降）
- `scene_dup_iou05.hit0` 不上升（不要误并导致漏检）
- `topk_drop.p95` 不变坏（否则内存压力更大）

---

## 5.7 E4（可选）：delayed confirm（tentative/confirm）

仅当 E2/E3 下 birth 仍显著污染（inflation 下不去、topk_drop 上升、scene_dup@0.1 仍高）时再引入：
- unmatched 先进入 tentative，需要连续多帧/多次命中才转正
- 目标：处理“一次性碎片/误检”，进一步降低 map 污染


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

### 6.3 开启 baseline_stats（本轮统计实验）
建议先在小子集 scene 上跑通（避免全量过慢），再跑全量：
- `--cfg-options model.test_cfg.baseline_stats.enable=True test_evaluator.baseline_stats.enable=True`
- 建议指定输出目录：
  - `test_evaluator.baseline_stats.out_dir=baseline_stats`

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

---

## 11. 后续最小轮次实验（E1–E5，优先灭 @0.1 红旗）

目标：在**最少轮次**下把因果链钉死：先止血（灭 `hit_ge2@0.1` 红旗、AP50 止跌），再逐步开放补全能力，最后把真正的补全迁移到跨帧 many-to-one。

### 11.1 公共设置（所有实验保持一致）

- Config：`configs/ESAM_CA/ESAM_online_scannet200_CA.py`
- CKPT：`/home/nebula/xxy/3D_Reconstruction/work_dirs/tmp/ESAM_CA_online_epoch_128.pth`
- 数据（MV-fast）：
  - `data_root=/home/nebula/xxy/dataset/data/scannet200-mv_fast`
  - `ann_file=/home/nebula/xxy/dataset/data/scannet200-mv_fast/scannet200_mv_oneformer3d_infos_val.pkl`
- 评测口径：必须带 `--cat-agnostic`
- 统一开启监控（每轮都要产出 A/B/C 面板）：
  - `model.test_cfg.online_monitor.enable=True`
  - `test_evaluator.online_monitor.enable=True`, `test_evaluator.online_monitor.out_dir=online_monitor`
  - `model.test_cfg.baseline_stats.enable=True`, `model.test_cfg.baseline_stats.pre_pool=after_nms`
  - `test_evaluator.baseline_stats.enable=True`, `test_evaluator.baseline_stats.out_dir=baseline_stats`
- 统一启用 Step0 证据（只加记录，不改策略）：
  - `model.test_cfg.geom_merge.stats.enable=True`
  - `model.test_cfg.geom_merge.stats.record_union_metrics=True`
  - `model.test_cfg.geom_merge.stats.record_gt_delta_iou=True`
  - `model.test_cfg.geom_merge.stats.record_semantic=True`（为 Step3/4 的语义门控做证据）
  - `model.test_cfg.geom_merge.stats.max_events_per_frame=50`（控制额外开销）

### 11.2 实验序列（按最少轮次拿最大信息量）

只改 `geom_merge`/`many_to_one_absorb` 的关键旋钮，其它全部保持一致：

1) **E1 duplicate-only（keep-best）**
   - 目的：先止血，验证“union 外扩/糊化”是否主因。
   - 配置差异：
     - `model.test_cfg.geom_merge.enable=True`
     - `model.test_cfg.geom_merge.mode=keep_best`

2) **E2 = E1 + 点集 IoU refine + 更严格几何门**
   - 目的：解决“几何近邻误并”（不是 union 的那部分）。
   - 配置差异（起步建议）：
     - `model.test_cfg.geom_merge.duplicate_criteria.iou_box_thr=0.85`
     - `model.test_cfg.geom_merge.duplicate_criteria.center_norm_thr=0.15`
     - `model.test_cfg.geom_merge.duplicate_criteria.size_ratio_min=0.50`
     - `model.test_cfg.geom_merge.duplicate_criteria.use_point_iou_refine=True`
     - `model.test_cfg.geom_merge.duplicate_criteria.iou_pts_thr=0.80`

3) **E3 = E2 + 语义 veto（保守否决，不主导）**
   - 目的：进一步压制“近邻异物误并”（椅子/桌子等）。
   - 配置差异（建议先做语义 label gate + query cosine veto）：
     - `model.test_cfg.geom_merge.semantic_veto.enable=True`
     - `model.test_cfg.geom_merge.semantic_veto.use_semantic_label_gate=True`
     - `model.test_cfg.geom_merge.semantic_veto.conf_thr=0.60`
     - `model.test_cfg.geom_merge.semantic_veto.use_query_cos_veto=True`
     - `model.test_cfg.geom_merge.semantic_veto.query_cos_thr=0.20`

4) **E4 = E3 + 受控 union（碎片吸收 + 外扩约束）**
   - 目的：在红旗被压住后，才尝试补全，且必须对外扩设硬约束。
   - 配置差异：
     - `model.test_cfg.geom_merge.mode=controlled_union`
     - `model.test_cfg.geom_merge.controlled_union.small_ratio_max=0.30`
     - `model.test_cfg.geom_merge.controlled_union.cov_thr=0.90`
     - `model.test_cfg.geom_merge.controlled_union.expansion_ratio_max=1.20`
     - `model.test_cfg.geom_merge.controlled_union.bbox_expand_ratio_max=1.50`

5) **E5 固定 E1=保守版，把补全放进 E3 many-to-one（跨帧吸收）**
   - 目的：把“补全”迁移到跨帧一致性上，降低误并风险，验证长程碎片化是否改善。
   - 配置差异：
     - `model.test_cfg.geom_merge.mode=keep_best`（保守供给）
     - `model.test_cfg.many_to_one_absorb.enable=True`
     - `model.test_cfg.many_to_one_absorb.only_to_matched=True`
     - `model.test_cfg.many_to_one_absorb.iou_thr=0.50`
     - `model.test_cfg.many_to_one_absorb.score_thr=0.00`
     - `model.test_cfg.many_to_one_absorb.min_mask_points=20`
     - `model.test_cfg.many_to_one_absorb.max_mask_points=500`
     - `model.test_cfg.many_to_one_absorb.max_absorb_per_frame=0`（0=不限制）

### 11.3 每轮必看 A/B/C 面板（优先级）

- **B（红旗必须先灭）**：`hit_ge2@0.1`
  - `baseline_stats/baseline_stats_summary.json`：
    - `dup_gt_iou01.det_to_merge_hit_ge2`
    - `scene_dup_iou01.hit_ge2`
  - 同时看 Step0 的因果证据：
    - `geom_merge_event.expansion_ratio_*`
    - `geom_merge_event.bbox_expand_ratio_*`
    - `geom_merge_event.delta_best_iou_*`

- **AP50（必须止跌）**：测试日志中的 `all_ap_50%`

- **A（别把供给全牺牲）**：`killed_useful@0.5(inst_thr)`
  - `baseline_stats/baseline_stats_summary.json`：
    - `killed_useful_iou05.inst_thr.any` / `iou` / `cov`

- **C（最后再优化，不要优先级倒置）**：`inflation/topk_drop`
  - `baseline_stats/baseline_stats_summary.json`：`inflation`
  - `online_monitor/online_monitor_summary.json`：`topk_drop`

### 11.4 自动跑实验（无需中途手动）

已提供脚本：`test/run_online_geommerge_series.sh`

```bash
cd /home/nebula/xxy/3D_Reconstruction
bash test/run_online_geommerge_series.sh
```

每个实验输出目录独立，统一写到：
`work_dirs/ESAM_online_scannet200_CA_mv_fast_ab/<EXP_NAME>/`

---

## 12. E6：one-to-many supporters（coverage-based，suppress-birth only）

动机：E1–E5 结果表明 `hit_ge2@0.1` 的主要来源更像“碎片级多命中”，而不是纯拷贝 duplicate；同时 E5 的 many-to-one absorb 用 `IoU>=0.5` 几乎不触发。E6 的目标是做一个最小因果验证：**碎片能否被“解释为已有 track 的 supporter”，从而抑制 birth / inflation / scene-level dup**，且不引入 union 外扩风险。

### 12.1 策略定义（只抑制 birth，不更新 mask）

- Stage-1：保持原 Hungarian 1-to-1 主匹配
- Stage-2：对 unmatched det，寻找最佳 track，并满足碎片 gate 后：
  - 标记为 `supporter`（允许 many-to-one：一个 track 可接收多个 supporter）
  - **禁止 birth**（det 不再创建新 track）
  - 不做 mask union，不更新 query/score（只做归属解释）

碎片 gate（起步建议）：
- `size_ratio = |det|/|track_cur_frame| <= 0.30`
- `coverage(det→track_cur_frame) = |det∩track|/|det| >= 0.85`
- 可选安全阀：`center_norm <= 0.25`（bbox 中心距离 / track bbox 对角线）

### 12.2 开关与参数（命令行 cfg-options）

- `model.test_cfg.one_to_many_support.enable=True`
- `model.test_cfg.one_to_many_support.only_to_matched=True`
- `model.test_cfg.one_to_many_support.cov_thr=0.85`
- `model.test_cfg.one_to_many_support.small_ratio_max=0.30`
- `model.test_cfg.one_to_many_support.center_norm_thr=0.25`
- `model.test_cfg.one_to_many_support.min_mask_points=20`
- `model.test_cfg.one_to_many_support.max_mask_points=500`
- `model.test_cfg.one_to_many_support.max_support_per_frame=0`（0=不限制）

### 12.3 主要判据（以 online/scene-level 为主）

E6 不改单帧 det 生成，因此 `det_to_merge_hit_ge2@0.1` 可能变化有限；重点看：
- `online_monitor_summary.json`：`birth.mean`↓、`mem_size_full.p95`↓、`topk_drop.p95`↓（长期压力缓解）
- `baseline_stats_summary.json`：`inflation.p95`↓、`scene_dup_iou01.hit_ge2`↓（地图级重复红旗缓解）
- AP/AP50：不要求立刻显著上升，但不应出现明显“断崖式下降”
