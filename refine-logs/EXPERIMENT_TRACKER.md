# Experiment Tracker：可靠性门控 SIM-CoT

状态仅使用：`TODO`、`RUNNING`、`PASS`、`FAIL`、`BLOCKED`、`CUT`。只有达到对应预注册门槛才能标记 `PASS`。

| Run ID | Milestone | Purpose | System / Variant | Split / Dataset | Metrics / Artifact | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R001 | M0 | 固定依赖和输入 | official repo/data/model/checkpoint | N/A | commit、revision、SHA-256、环境清单 | MUST | PASS | repo `d1d56af`；checkpoint `99b3f918…`；RTX 4060 8GB |
| R002 | M0 | 核验核心官方数值 | official checkpoint | GSM8K-Aug test | accuracy、样本数、解析失败数 | MUST | PASS | 586/1319=44.43%；相对 44.8% 为 -0.37 pp |
| R003 | M0 | 完成官方四任务复现 | official checkpoint | GSM-Hard/MultiArith/SVAMP | accuracy | MUST | PASS | 9.48% / 89.83% / 40.60%；目标 9.3% / 90.8% / 40.7%；MultiArith 距下界 0.03 pp |
| R004 | M0 | 最大样本 20-step 预检 | standard SIM-CoT | official train sample | loss、peak reserved、step 对齐 | MUST | PASS | 末窗 loss 0.00238 < 首窗 0.78261；5.52 GB；10 latent 对齐 |
| R005 | M0 | checkpoint 恢复验证 | standard SIM-CoT | smoke subset | reload 一致性、继续训练 loss、吞吐 | MUST | PASS | 恢复后真实更新 1 次；二次 reload Δ=0；5.14 GB；约 36 s（含载入/保存） |
| R010 | M1 | 本地基础模型方向 | Coconut seed 0 | fixed short-budget train / 4 tests | macro accuracy、per-task accuracy | MUST | TODO | 与 R011 同预算 |
| R011 | M1 | 本地 SIM-CoT 方向 | standard SIM-CoT seed 0 | same as R010 | macro accuracy、step NLL | MUST | TODO | 目标比 R010 +2.0 pp |
| R012 | M1 | 公平性审计 | R010 vs R011 | logs/configs | updates、sampler、optimizer、batch parity | MUST | TODO | 不一致则比较作废 |
| R020 | M2 | 无偏流行率抽样 | audit sampler | official train | ≥2,000 steps、split hash | MUST | TODO | 抽样前不筛异常 |
| R021 | M2 | 自动结构检查 | rule audit | prevalence audit | coverage、invalid candidates | MUST | TODO | 不自动决定 Utility |
| R022 | M2 | 人工复核 | human labels | prevalence audit | V/U 标签、分歧率、CI | MUST | TODO | 建议 10% 双标 |
| R023 | M2 | 构建自然检测封存集 | matched audit builder | official train held-out | ≥100 low / ≥100 clean、hash | MUST | TODO | 尽量 50 invalid + 50 low-U |
| R030 | M3 | 构建可靠性训练集 | five-family generator | question split 60/20/20 | family counts、leakage checks | MUST | TODO | 先切题再生成 |
| R031 | M3 | LOFO 数值族 | V×U head | held-out numeric | V/U/composite AUC | MUST | TODO | 未见题目 |
| R032 | M3 | LOFO 运算符族 | V×U head | held-out operator | V/U/composite AUC | MUST | TODO | 未见题目 |
| R033 | M3 | LOFO 依赖族 | V×U head | held-out dependency | V/U/composite AUC | MUST | TODO | 未见题目 |
| R034 | M3 | LOFO 无关族 | V×U head | held-out irrelevant | V/U/composite AUC | MUST | TODO | Utility 关键 |
| R035 | M3 | LOFO 冗余族 | V×U head | held-out redundant | V/U/composite AUC | MUST | TODO | Utility 关键 |
| R036 | M3 | 训练最终可靠性头 | all five families | head train/validation | checkpoint、temperature、Brier/ECE | MUST | TODO | 结构/超参已冻结 |
| R037 | M3 | 自然封存评测 | frozen final head | natural teacher audit | composite/V/U AUC | MUST | TODO | 只打开一次 |
| R038 | M3 | 补偿错误压力测试 | frozen final head | sealed compensating family | Validity AUC | MUST | TODO | 目标 ≥0.70 |
| R039 | M3 | 捷径与双门槛审计 | frozen final head | all audits | G3-S/G3-N、length/digit/position/template correlation | MUST | TODO | G3-S 控制全部加权；G3-N 控制自然主张 |
| R040 | M4 | 官方步骤离线赋权 | frozen scorer | official train minus audits | v/u/r/w、hash、weight stats | MUST | TODO | 每题 mean(w)=1 |
| R041 | M4 | 主对照等权 | standard SIM-CoT seed 0 | official train / 4 tests | accuracy、clean step NLL | MUST | TODO | 先运行并固定 updates |
| R042 | M4 | 主方法加权 | V×U weighted seed 0 | same as R041 | accuracy、clean step NLL | MUST | TODO | 完成 updates=R041 |
| R043 | M4 | 单种子判定 | paired evaluation | 4 tests | macro Δ、GSM8K guard、NLL Δ | MUST | TODO | 失败则停止 M5 大训练 |
| R044 | M4 | 分支公平性审计 | R041 vs R042 | logs/configs/checkpoints | config diff、sample order、OOM path | MUST | TODO | 唯一主差异应为 w |
| R050 | M5 | 多种子等权 | standard SIM-CoT seeds 1,2 | official train / 4 tests | accuracy/NLL | MUST | TODO | 仅 R043 PASS 后 |
| R051 | M5 | 多种子加权 | V×U weighted seeds 1,2 | same as R050 | accuracy/NLL | MUST | TODO | 与 R050 配对 |
| R052 | M5 | 多种子统计 | paired aggregation | seeds 0,1,2 | mean/std/bootstrap CI | MUST | TODO | 主表输入 |
| R055 | M5 | Validity-only 消融 | weighted V only | primary setting | macro accuracy | MUST | TODO | 贡献隔离 |
| R056 | M5 | Utility-only 消融 | weighted U only | primary setting | macro accuracy | MUST | TODO | 贡献隔离 |
| R057 | M5 | RSR/RD 基线 | legacy offline weighting | primary setting | macro accuracy、detector AUC | MUST | TODO | 不解释为正确率 |
| R058 | M5 | 文本-only 必要性基线 | text-only MLP/logistic | detection + downstream | AUC、macro accuracy | MUST | TODO | 检验隐空间必要性 |
| R060 | M5 | 5% 受控污染 | equal/predicted/oracle | controlled train / clean tests | damage、recovery ratio | CONDITIONAL | TODO | 自然噪声 <5% 时 MUST |
| R061 | M5 | 10% 受控污染 | equal/predicted/oracle | controlled train / clean tests | damage、recovery ratio | CONDITIONAL | TODO | 同上 |
| R062 | M5 | 20% 受控污染 | equal/predicted/oracle | controlled train / clean tests | damage、recovery ratio | CONDITIONAL | TODO | 同上 |
| R065 | M5 | 权重下限敏感性 | floor variants | primary setting | macro accuracy、stability | NICE | TODO | 主结果后再做 |
| R066 | M5 | MLP 宽度敏感性 | head width variants | detection audit | AUC/params | NICE | TODO | 只为附录 |
| R070 | M6 | 生成主表 | reporting pipeline | saved JSON | Tables 1–5 | MUST | TODO | 不手抄数值 |
| R071 | M6 | 生成主图 | reporting pipeline | saved predictions | Figures 2–4 | MUST | TODO | 自然/受控分开 |
| R072 | M6 | 失败案例审计 | false positives/negatives | sealed predictions | error taxonomy | MUST | TODO | 不挑选只支持方法的案例 |
| R073 | M6 | 复现包检查 | clean environment | configs/manifests | one-command dry run | MUST | TODO | hash 全部可追溯 |
| R074 | M6 | 结论边界检查 | gate report | all results | claim wording | MUST | TODO | 负结果也按规则报告 |

## Oracle Weighting 40% Follow-up (2026-08-10)

| Run ID | Milestone | Purpose | Status | Notes |
| --- | --- | --- | --- | --- |
| O101 | Controlled-noise schedule | Add a second distinct contamination to every frozen 20% example | PASS | 2,048 examples; 40% step noise; schedule SHA `9098f67d...` |
| O102 | Sanity | Loss parity and one-update GPU gate | PASS | Delta `4.768e-7`; peak `5.139 GiB` |
| O111 | Formal training | 40% noisy equal weighting | PASS | 256 updates; peak `5.527 GiB` |
| O112 | Formal training | 40% oracle raw-0.1 | PASS | 256 updates; peak `5.527 GiB` |
| O113 | Formal training | 40% oracle normalized-0.1 | PASS | 256 updates; peak `5.527 GiB` |
| O120 | Full paired evaluation | Official 1,319-example test and causal analysis | PASS | Execution PASS; scientific verdict `INCONCLUSIVE_INSUFFICIENT_NOISE_DAMAGE` |

Report: `outputs/reliable_simcot/oracle_weighting_40pct/oracle_weighting_40pct_causal_report_2026-08-10.md`

## Immediate Queue

1. R010：同预算本地 Coconut 短训练
2. R011：同预算本地标准 SIM-CoT 短训练
3. R012：两分支公平性审计

M0 已通过。R010–R012 可启动，但必须先冻结相同训练预算与 OOM 回退规则。
