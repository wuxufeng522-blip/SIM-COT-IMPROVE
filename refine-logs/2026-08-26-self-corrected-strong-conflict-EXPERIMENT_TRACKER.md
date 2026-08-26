# 强错误—显式自纠—正确答案实验追踪表

**日期：** 2026-08-26

**规格：** `docs/superpowers/specs/2026-08-26-simcot-self-corrected-strong-conflict-factorial-design.md`

**总体状态：** `IN_PROGRESS` — 规格已批准，开始实施 M0。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Artifact | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| SC001 | M0 | 固化配置与路径 | config/provenance | all | config hash | MUST | READY | 不继承旧自然噪声数据门 |
| SC002 | M0 | 数据构造 fixture | pytest | fixtures | structure/error/recovery tests | MUST | TODO | 先测试后实现 |
| SC003 | M0 | 旧实验回归测试 | pytest | existing | regression status | MUST | TODO | 不破坏 full-conflict/PRM |
| SC010 | M1 | 选择并去重 512 题 | selector | MATH train | 512 IDs、test overlap=0 | MUST | TODO | 仅题目与答案为官方来源 |
| SC011 | M1 | 构造五步 Clean | constructor | train512 | grader、step structure | MUST | TODO | 缺失时由 Codex 构造 |
| SC012 | M1 | 构造四类含噪轨迹 | constructor | train512 | A1/A2/B1/B2 | MUST | TODO | 最终均自身答对 |
| SC013 | M1 | 全量验证 | validator | 2560 | answer/error/length/edit audits | MUST | TODO | 不合格重构或换题 |
| SC014 | M1 | 20 套五联组审计 | audit bundle | train sample | readable audit | MUST | TODO | 100 条轨迹 |
| SC015 | M1 | 冻结数据 | freezer | train512 | manifest/hash | MUST | TODO | 训练后不可改 |
| SC020 | M2 | 九臂 mapper 与权重 | experiment | fixture | answer parity/weight vectors | MUST | TODO | 归一化均值为1 |
| SC021 | M2 | loss/gradient 审计 | experiment | sanity | loss parity/relative grad | MUST | TODO | 错误步骤相对0.1 |
| SC022 | M2 | 九臂 2-update GPU sanity | all arms | sanity | finite/VRAM | MUST | TODO | <=7.4GB |
| SC023 | M2 | 冻结运行清单 | state machine | all | schedule/hash | MUST | TODO | 27次训练 |
| SC100 | M3 | 正式训练 | Clean s20260809 | train512 | checkpoint/loss | MUST | TODO | 64 updates |
| SC101 | M3 | 正式训练 | Solution-N1-EQ s20260809 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| SC102 | M3 | 正式训练 | Solution-N1-W01 s20260809 | train512 | checkpoint/loss | MUST | TODO | 相对权重0.1 |
| SC103 | M3 | 正式训练 | Solution-N2-EQ s20260809 | train512 | checkpoint/loss | MUST | TODO | 两连续错误 |
| SC104 | M3 | 正式训练 | Solution-N2-W01 s20260809 | train512 | checkpoint/loss | MUST | TODO | 两连续错误 |
| SC105 | M3 | 正式训练 | Misread-N1-EQ s20260809 | train512 | checkpoint/loss | MUST | TODO | 题意误读 |
| SC106 | M3 | 正式训练 | Misread-N1-W01 s20260809 | train512 | checkpoint/loss | MUST | TODO | 相对权重0.1 |
| SC107 | M3 | 正式训练 | Misread-N2-EQ s20260809 | train512 | checkpoint/loss | MUST | TODO | 两连续错误 |
| SC108 | M3 | 正式训练 | Misread-N2-W01 s20260809 | train512 | checkpoint/loss | MUST | TODO | 相对权重0.1 |
| SC110 | M3 | 正式训练九臂 | all arms s20260810 | train512 | 9 checkpoints/loss | MUST | TODO | 与首种子同日程 |
| SC120 | M3 | 正式训练九臂 | all arms s20260811 | train512 | 9 checkpoints/loss | MUST | TODO | 与首种子同日程 |
| SC200 | M4 | 500题评估九臂 | all arms s20260809 | test500 | predictions/accuracy/EM | MUST | TODO | 无测试噪声 |
| SC210 | M4 | 500题评估九臂 | all arms s20260810 | test500 | predictions/accuracy/EM | MUST | TODO | 同一题序 |
| SC220 | M4 | 500题评估九臂 | all arms s20260811 | test500 | predictions/accuracy/EM | MUST | TODO | 同一题序 |
| SC300 | M5 | 四项伤害统计 | analyzer | all predictions | harm/CI/dose/family | MUST | TODO | 分层配对bootstrap |
| SC301 | M5 | 四项恢复统计 | analyzer | all predictions | recovery/ratio/CI | MUST | TODO | 仅伤害>0解释比例 |
| SC302 | M5 | 46条PRM补充审计 | supplement | PRM Noise-1 | repair/coverage | NICE | TODO | 不阻塞主结果 |
| SC303 | M5 | 中文报告与边界 | reporter | all artifacts | JSON/MD/hash | MUST | TODO | 不外推自然发生率 |

## Frozen Arm Matrix

`clean`、`solution_n1_equal`、`solution_n1_w01`、`solution_n2_equal`、`solution_n2_w01`、`misread_n1_equal`、`misread_n1_w01`、`misread_n2_equal`、`misread_n2_w01`。

## Next Action

执行 SC001–SC003：新增配置、数据 fixture 和最小实现，运行新旧测试。

