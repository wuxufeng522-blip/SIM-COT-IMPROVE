# PRM800K 自然步骤噪声伤害实验追踪表

**日期：** 2026-08-26

**规格：** `docs/superpowers/specs/2026-08-26-prm800k-natural-noise-damage-design.md`

**总体状态：** `BLOCKED` — 严格三联组 `0/512`，按预注册数据门停止；未启动 GPU 训练。

**状态说明：** `READY` 为下一可执行项；`TODO` 等待上游；`BLOCKED` 为门失败；`DONE` 为产物和哈希已记录。

| Run ID | Milestone | Purpose | System / Variant | Split | Metrics / Artifact | Priority | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| PN001 | M0 | 固化配置、路径与上游哈希 | config/provenance | all | `provenance.json` | MUST | DONE | 来源哈希已核验 |
| PN002 | M0 | 获取并校验官方PRM800K | downloader/validator | raw | 文件、行数、非LFS指针、SHA-256 | MUST | DONE | 98,731条；官方SHA一致 |
| PN003 | M0 | 数据解析与grader fixture | pytest | fixtures | parser/grader tests | MUST | DONE | 14项定向测试通过 |
| PN010 | M1 | 重建自然chosen轨迹 | trajectory rebuilder | PRM train | 轨迹数、排除原因 | MUST | DONE | 最终合格537，全部Clean |
| PN011 | M1 | 严格同题同批次三联审计 | triplet builder | PRM train | Clean/N1/N2、triplet count | MUST | DONE | 三联组0；N1/N2均无合格轨迹 |
| PN012 | M1 | 冻结训练/开发清单 | deterministic freezer | train/dev | 512 triples、256 clean、零重叠、hash | MUST | BLOCKED | `INSUFFICIENT_STRICT_TRIPLETS` |
| PN020 | M2 | 自由文本五步映射与单测 | arm mapper | sanity | target/answer parity | MUST | TODO | 原始步骤不改写 |
| PN021 | M2 | 官方loss parity与零梯度测试 | loss audit | sanity | all-one delta、AO aux grad=0 | MUST | TODO | 失败不得训练 |
| PN022 | M2 | 四组2-update GPU sanity | AO/Clean/N1/N2 | sanity | finite loss/grad、显存 | MUST | TODO | 红线7.4GB |
| PN023 | M2 | 冻结配置、清单与分析代码 | freeze audit | all | SHA-256 manifest | MUST | TODO | 后续不得改处理 |
| PN100 | M3 | 正式训练 | answer_only seed 20260809 | train512 | checkpoint/loss | MUST | TODO | 64 updates |
| PN101 | M3 | 正式训练 | clean_aux seed 20260809 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| PN102 | M3 | 正式训练 | natural_noise_1 seed 20260809 | train512 | checkpoint/loss | MUST | TODO | 20%步骤错误 |
| PN103 | M3 | 正式训练 | natural_noise_2 seed 20260809 | train512 | checkpoint/loss | MUST | TODO | 40%步骤错误 |
| PN110 | M3 | 正式训练 | answer_only seed 20260810 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| PN111 | M3 | 正式训练 | clean_aux seed 20260810 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| PN112 | M3 | 正式训练 | natural_noise_1 seed 20260810 | train512 | checkpoint/loss | MUST | TODO | 同冻结三联 |
| PN113 | M3 | 正式训练 | natural_noise_2 seed 20260810 | train512 | checkpoint/loss | MUST | TODO | 同冻结三联 |
| PN120 | M3 | 正式训练 | answer_only seed 20260811 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| PN121 | M3 | 正式训练 | clean_aux seed 20260811 | train512 | checkpoint/loss | MUST | TODO | 独立初始化 |
| PN122 | M3 | 正式训练 | natural_noise_1 seed 20260811 | train512 | checkpoint/loss | MUST | TODO | 同冻结三联 |
| PN123 | M3 | 正式训练 | natural_noise_2 seed 20260811 | train512 | checkpoint/loss | MUST | TODO | 同冻结三联 |
| PN130 | M3 | 工程开发评估 | all arms/seeds | dev256 | EM/NLL/clean-step NLL | MUST | TODO | 不作为停止门 |
| PN200 | M4 | 最终确认 | answer_only seed 20260809 | confirm500 | predictions/EM/NLL | MUST | TODO | 全checkpoint冻结后 |
| PN201 | M4 | 最终确认 | clean_aux seed 20260809 | confirm500 | predictions/EM/NLL | MUST | TODO | 官方grader |
| PN202 | M4 | 最终确认 | natural_noise_1 seed 20260809 | confirm500 | predictions/EM/NLL | MUST | TODO | 无测试噪声 |
| PN203 | M4 | 最终确认 | natural_noise_2 seed 20260809 | confirm500 | predictions/EM/NLL | MUST | TODO | 无测试噪声 |
| PN210 | M4 | 最终确认 | answer_only seed 20260810 | confirm500 | predictions/EM/NLL | MUST | TODO | 同一顺序 |
| PN211 | M4 | 最终确认 | clean_aux seed 20260810 | confirm500 | predictions/EM/NLL | MUST | TODO | 同一顺序 |
| PN212 | M4 | 最终确认 | natural_noise_1 seed 20260810 | confirm500 | predictions/EM/NLL | MUST | TODO | 无测试噪声 |
| PN213 | M4 | 最终确认 | natural_noise_2 seed 20260810 | confirm500 | predictions/EM/NLL | MUST | TODO | 无测试噪声 |
| PN220 | M4 | 最终确认 | answer_only seed 20260811 | confirm500 | predictions/EM/NLL | MUST | TODO | 同一顺序 |
| PN221 | M4 | 最终确认 | clean_aux seed 20260811 | confirm500 | predictions/EM/NLL | MUST | TODO | 同一顺序 |
| PN222 | M4 | 最终确认 | natural_noise_1 seed 20260811 | confirm500 | predictions/EM/NLL | MUST | TODO | 无测试噪声 |
| PN223 | M4 | 最终确认 | natural_noise_2 seed 20260811 | confirm500 | predictions/EM/NLL | MUST | TODO | 无测试噪声 |
| PN300 | M5 | 三项配对统计 | analyzer | all predictions | D1/D2/Ddose、McNemar、bootstrap | MUST | TODO | 按题重采样 |
| PN301 | M5 | 预注册结论 | gate reporter | all results | verdict、FLOOR flag | MUST | TODO | 互斥优先级 |
| PN302 | M5 | 中文总结与边界 | reporter | all artifacts | JSON/MD/hash | MUST | TODO | 不外推自然流行率 |

## Gate Outcome

- Phase 1严格结构：Clean 45，Noise-1 0，Noise-2 0。
- Phase 2严格结构：Clean 602，Noise-1 46，Noise-2 0；46条Noise-1最终答案全部错误。
- 最终严格合格：Clean 537，Noise-1 0，Noise-2 0。
- 训练与后续评估任务因上游数据门失败而不执行；新设计需重新批准。
