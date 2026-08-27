# 错误抵消步骤噪声控制实验追踪表

**日期：** 2026-08-27

**规格：** `docs/superpowers/specs/2026-08-27-error-cancellation-step-noise-design.md`

**实施计划：** `refine-logs/2026-08-27-error-cancellation-gsm8k-EXPERIMENT_PLAN.md`

**总体状态：** `PLAN_READY` — 尚未实施，等待启动授权。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Artifact | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| EC001 | M0 | 新建隔离配置 | v9 config | all | config/schema | MUST | TODO | 不覆盖 v8 |
| EC002 | M0 | 数据构造 fixture | pytest | synthetic | format/arithmetic/dependency | MUST | TODO | 先测试后实现 |
| EC003 | M0 | 损失与臂映射 fixture | pytest | synthetic | L_correct/lambda0/weights | MUST | TODO | 逐 token 归一化 |
| EC004 | M0 | 旧实验回归 | pytest | existing | regression status | MUST | TODO | 不改变旧损失行为 |
| EC010 | M1 | 选择 512 题 | selector | GSM8K train | IDs/test overlap | MUST | TODO | 五步可解析 |
| EC011 | M1 | 保留原始 Clean | freezer | train512 | byte equality | MUST | TODO | 禁止改写扩写 |
| EC012 | M1 | 构造局部错误/冗余 | constructor | train512 | false/cancel/redundancy | MUST | TODO | 相同槽位 |
| EC013 | M1 | 构造广域错误/冗余 | constructor | train512 | propagation/cancel | MUST | TODO | 至少一个 Clean 后续步骤 |
| EC014 | M1 | 全量自动验证 | validator | 2,560 | format/math/answer/token | MUST | TODO | 不合格题替换 |
| EC015 | M1 | 20 套人工语义审计 | audit | stratified | 100 trajectories | MUST | TODO | 局部/广域均覆盖 |
| EC016 | M1 | 冻结数据和 schedule | manifest | train512 | SHA-256/nested masks | MUST | TODO | 训练后不可变 |
| EC020 | M2 | L_correct 等价性 | audit | fixtures/train sample | exact parity | MUST | TODO | 变体不进入答案分支 |
| EC021 | M2 | lambda=0 更新等价 | audit | train sample | params/grad hash | MUST | TODO | 必须完全相同 |
| EC022 | M2 | 九臂 2-update sanity | all stage-1 arms | sanity | finite/loss/grad | MUST | TODO | <=7.4GB |
| EC023 | M2 | 最长样本显存门 | worst cases | sanity | peak reserved | MUST | TODO | 不改科学参数 |
| EC024 | M2 | 完整 Clean 64-update 显存门 | Clean s20260811 | train512 | checkpoint/VRAM | MUST | TODO | 与正式日程相同 |
| EC025 | M2 | 原始 checkpoint 官方评估 | base checkpoint | test1319 | Acc0/predictions | MUST | TODO | 只运行一次 |
| EC026 | M2 | 冻结门控状态机 | pipeline | all | state/hash | MUST | TODO | 安全续跑 |
| EC100 | M3 | Clean 训练 | C s20260809 | train512 | checkpoint/loss | MUST | TODO | 64 updates |
| EC101 | M3 | Clean 评估 | C s20260809 | test1319 | accuracy/predictions | MUST | TODO | 官方 EM |
| EC102 | M3 | Clean 训练 | C s20260810 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| EC103 | M3 | Clean 评估 | C s20260810 | test1319 | accuracy/predictions | MUST | TODO | 同一题序 |
| EC104 | M3 | Clean 训练 | C s20260811 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| EC105 | M3 | Clean 门判定 | C all seeds | test1319 | relative gate | MUST | TODO | 失败则全部停止 |
| EC110 | M4 | 第一阶段训练/评估 | RL25 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 依赖 EC105 PASS |
| EC111 | M4 | 第一阶段训练/评估 | RL50 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 嵌套覆盖 |
| EC112 | M4 | 第一阶段训练/评估 | EL25 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 错误等权 |
| EC113 | M4 | 第一阶段训练/评估 | EL50 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 错误等权 |
| EC114 | M4 | 第一阶段训练/评估 | RW25 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 匹配冗余 |
| EC115 | M4 | 第一阶段训练/评估 | RW50 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 主对照 |
| EC116 | M4 | 第一阶段训练/评估 | EW25 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 广域错误 |
| EC117 | M4 | 第一阶段训练/评估 | EW50 all seeds | train512/test1319 | 3 ckpts + predictions | MUST | BLOCKED | 主处理 |
| EC200 | M5 | 主伤害统计 | RW50 vs EW50 | paired test | mean/seed/CI | MUST | BLOCKED | 唯一主比较 |
| EC201 | M5 | 剂量与范围统计 | other matched pairs | paired test | harm/CI | MUST | BLOCKED | 支持性结果 |
| EC202 | M5 | 冗余开销与总伤害 | C/R/E | paired test | overhead/total harm | MUST | BLOCKED | 不替代主比较 |
| EC203 | M5 | 第一阶段判定 | analyzer | all | PASS/FAIL/severe | MUST | BLOCKED | FAIL 则停加权 |
| EC300 | M6 | 加权映射与梯度审计 | EW50/RW50 w01 | sanity | masks/grad/finite | MUST-IF-GATED | BLOCKED | 依赖 EC203 PASS |
| EC301 | M6 | 加权训练/评估 | EW50-w01 all seeds | train512/test1319 | 3 ckpts + predictions | MUST-IF-GATED | BLOCKED | 错误位置0.1 |
| EC302 | M6 | 掩码训练/评估 | RW50-w01 all seeds | train512/test1319 | 3 ckpts + predictions | MUST-IF-GATED | BLOCKED | 对应位置0.1 |
| EC303 | M6 | 选择性恢复统计 | weighted primary | paired test | R_error/R_redundant/R_selective | MUST-IF-GATED | BLOCKED | 扣除剂量效应 |
| EC304 | M6 | 恢复门判定 | analyzer | all seeds | CI/recovery fraction | MUST-IF-GATED | BLOCKED | FAIL 则停扩展 |
| EC400 | M7 | 条件式扩展 | wide25 w01 pair | train/test | selective recovery | NICE-IF-GATED | BLOCKED | 依赖 EC304 PASS |
| EC401 | M7 | 条件式扩展 | local50 w01 pair | train/test | selective recovery | NICE-IF-GATED | BLOCKED | 依赖 EC304 PASS |
| EC402 | M7 | 条件式扩展 | local25 w01 pair | train/test | selective recovery | NICE-IF-GATED | BLOCKED | 依赖 EC304 PASS |
| EC408 | M7 | 失败分析与审计汇总 | all artifacts | all | diagnostics/hash | MUST | BLOCKED | 不新增主张 |
| EC409 | M7 | 最终中文报告 | reporter | all | JSON/MD/table | MUST | BLOCKED | 明确半合成边界 |

## Frozen Stage-1 Arm Matrix

`C`、`RL25`、`RL50`、`EL25`、`EL50`、`RW25`、`RW50`、`EW25`、`EW50`。

## First Three Actions After Launch Approval

1. EC001：建立 v9 隔离配置和输出路径。
2. EC002：先写严格格式、算术错误、传播和冗余 fixture。
3. EC003：先写 `L_correct`、`lambda=0` 与逐 token 损失测试。
