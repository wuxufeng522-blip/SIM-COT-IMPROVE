# SIM-CoT 全链强冲突实验追踪表

**日期：** 2026-08-21

**规格：** `docs/superpowers/specs/2026-08-21-simcot-full-conflict-noise-design.md`

**状态说明：** `READY` 为下一可执行项；`TODO` 为上游通过后执行；`CONDITIONAL` 仅由门解锁；`BLOCKED` 表示门失败；`DONE` 表示产物和哈希均已记录。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Artifact | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| FC001 | M0 | 固化配置与上游哈希 | config/provenance | all | `provenance.json` | MUST | READY | 首个实施任务 |
| FC002 | M0 | 精确五步资格与防泄漏审计 | exact-five selector | source train | 候选数、排除数、交集=0 | MUST | TODO | 需排除所有旧冻结分区 |
| FC003 | M0 | 冻结训练/确认/生成/备用清单 | split builder | train/confirm | 512/1024/256/128、SHA-256 | MUST | TODO | 结果前不可变 |
| FC004 | M0 | 生成固定提示任务 | prompt manifest | generation pool | 4×64 配额、JSONL hash | MUST | TODO | 当前 Codex 为生成器 |
| FC010 | M1 | 12 条小批量生成 | Codex full-conflict | 4 families × 3 | 原始 JSONL | MUST | TODO | 不进入训练 |
| FC011 | M1 | 小批量自动和二次可读性门 | validator | 12 samples | 12/12、四类通过 | MUST | TODO | 失败则停止并修订规格 |
| FC020 | M2 | 生成 256 条正式候选 | Codex full-conflict | tier0+tier1 | attempts、raw JSONL | MUST | TODO | 每题最多三次 |
| FC021 | M2 | 正式数据审计与冻结 | validator/reserve allocator | 256 accepted | `data_audit.json`、schedule hash | MUST | TODO | 不足 256 则停止 |
| FC030 | M3 | 新增与相关回归测试 | pytest | code | 全部测试通过 | MUST | TODO | 不清理用户已有改动 |
| FC031 | M3 | 官方 all-one loss parity | full-conflict loss | sanity | abs diff ≤ 1e-6 | MUST | TODO | 失败不得训练 |
| FC032 | M3 | 四臂 2-update GPU sanity | AO/Clean/Local25/Full25 | train sanity | finite、labels parity、≤7.4 GB | MUST | TODO | 固定微批次 seed |
| FC033 | M3 | 干净/全链梯度通路审计 | gradient audit | fixed audit set | norms、cosines、12 layers | MUST | TODO | 机制诊断，不代替 EM |
| FC100 | M4 | 25% 训练 | answer_only, seed 20260809 | train512 | checkpoint、loss | MUST | TODO | 64 updates |
| FC101 | M4 | 25% 训练 | clean_aux1, seed 20260809 | train512 | checkpoint、loss | MUST | TODO | 64 updates |
| FC102 | M4 | 25% 训练 | local_causal_25, seed 20260809 | train512 | checkpoint、loss | MUST | TODO | 128 题×3 步 |
| FC103 | M4 | 25% 训练 | full_conflict_25, seed 20260809 | train512 | checkpoint、loss | MUST | TODO | 同 128 题×5 步 |
| FC110 | M4 | 25% 训练 | answer_only, seed 20260810 | train512 | checkpoint、loss | MUST | TODO | 独立初始化 |
| FC111 | M4 | 25% 训练 | clean_aux1, seed 20260810 | train512 | checkpoint、loss | MUST | TODO | 独立初始化 |
| FC112 | M4 | 25% 训练 | local_causal_25, seed 20260810 | train512 | checkpoint、loss | MUST | TODO | 同一冻结处理 |
| FC113 | M4 | 25% 训练 | full_conflict_25, seed 20260810 | train512 | checkpoint、loss | MUST | TODO | 同一冻结处理 |
| FC120 | M4 | 25% 训练 | answer_only, seed 20260811 | train512 | checkpoint、loss | MUST | TODO | 独立初始化 |
| FC121 | M4 | 25% 训练 | clean_aux1, seed 20260811 | train512 | checkpoint、loss | MUST | TODO | 独立初始化 |
| FC122 | M4 | 25% 训练 | local_causal_25, seed 20260811 | train512 | checkpoint、loss | MUST | TODO | 同一冻结处理 |
| FC123 | M4 | 25% 训练 | full_conflict_25, seed 20260811 | train512 | checkpoint、loss | MUST | TODO | 同一冻结处理 |
| FC130 | M4 | 25% 确认评估 | answer_only, seed 20260809 | confirm1024 | EM/NLL/predictions | MUST | TODO | 训练全部完成后统一评估 |
| FC131 | M4 | 25% 确认评估 | clean_aux1, seed 20260809 | confirm1024 | EM/NLL/predictions | MUST | TODO | 可断点续评 |
| FC132 | M4 | 25% 确认评估 | local_causal_25, seed 20260809 | confirm1024 | EM/NLL/predictions | MUST | TODO | 官方答案字段 |
| FC133 | M4 | 25% 确认评估 | full_conflict_25, seed 20260809 | confirm1024 | EM/NLL/predictions | MUST | TODO | 官方答案字段 |
| FC134 | M4 | 25% 确认评估 | answer_only, seed 20260810 | confirm1024 | EM/NLL/predictions | MUST | TODO | 可断点续评 |
| FC135 | M4 | 25% 确认评估 | clean_aux1, seed 20260810 | confirm1024 | EM/NLL/predictions | MUST | TODO | 可断点续评 |
| FC136 | M4 | 25% 确认评估 | local_causal_25, seed 20260810 | confirm1024 | EM/NLL/predictions | MUST | TODO | 官方答案字段 |
| FC137 | M4 | 25% 确认评估 | full_conflict_25, seed 20260810 | confirm1024 | EM/NLL/predictions | MUST | TODO | 官方答案字段 |
| FC138 | M4 | 25% 确认评估 | answer_only, seed 20260811 | confirm1024 | EM/NLL/predictions | MUST | TODO | 可断点续评 |
| FC139 | M4 | 25% 确认评估 | clean_aux1, seed 20260811 | confirm1024 | EM/NLL/predictions | MUST | TODO | 可断点续评 |
| FC140 | M4 | 25% 确认评估 | local_causal_25, seed 20260811 | confirm1024 | EM/NLL/predictions | MUST | TODO | 官方答案字段 |
| FC141 | M4 | 25% 确认评估 | full_conflict_25, seed 20260811 | confirm1024 | EM/NLL/predictions | MUST | TODO | 官方答案字段 |
| FC142 | M4 | 训练对偏好诊断 | all 25% checkpoints | frozen 256 pairs | corrupt-clean NLL | MUST | TODO | 仅训练内描述性指标 |
| FC300 | M5 | 25% 主门 | gate analyzer | all 25% results | C1/C2、McNemar、bootstrap | MUST | TODO | PASS 后跳过 M6 |
| FC400 | M6 | 条件训练 | full_conflict_50, seed 20260809 | train512 | checkpoint、loss | CONDITIONAL | CONDITIONAL | 仅 FC300 FAIL 解锁 |
| FC401 | M6 | 条件训练 | full_conflict_50, seed 20260810 | train512 | checkpoint、loss | CONDITIONAL | CONDITIONAL | 仅 FC300 FAIL 解锁 |
| FC402 | M6 | 条件训练 | full_conflict_50, seed 20260811 | train512 | checkpoint、loss | CONDITIONAL | CONDITIONAL | 仅 FC300 FAIL 解锁 |
| FC410 | M6 | 条件评估 | full_conflict_50, seed 20260809 | confirm1024 | EM/NLL/predictions | CONDITIONAL | CONDITIONAL | 不与 Local25 作同覆盖率比较 |
| FC411 | M6 | 条件评估 | full_conflict_50, seed 20260810 | confirm1024 | EM/NLL/predictions | CONDITIONAL | CONDITIONAL | 可断点续评 |
| FC412 | M6 | 条件评估 | full_conflict_50, seed 20260811 | confirm1024 | EM/NLL/predictions | CONDITIONAL | CONDITIONAL | 可断点续评 |
| FC420 | M6 | 50% 条件门 | gate analyzer | all 50% results | C3 verdict | CONDITIONAL | CONDITIONAL | 无论结果如何不再加码 |
| FC500 | M7 | 汇总报告 | reporter | all completed | JSON、MD、tables | MUST | TODO | 明确允许/禁止结论 |
| FC501 | M7 | 下一步决定 | state machine | final | PROCEED/HIGH_DENSITY/STOP | MUST | TODO | 官方测试始终封存 |

## Immediate Queue

1. `FC001`：创建配置与 provenance 契约。
2. `FC002`：实现精确五步资格、防泄漏和候选池审计。
3. `FC003`：冻结 512/1,024/256/128 清单并记录哈希。

在 `FC011` 小批量门通过前，不生成全部 256 条；在 `FC032` sanity 通过前，不启动任何正式 GPU 训练。
