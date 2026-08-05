# Experiment Tracker：可靠性门控 SIM-CoT

状态仅使用：`TODO`、`RUNNING`、`PASS`、`FAIL`、`BLOCKED`、`CUT`。只有达到对应预注册门槛才能标记 `PASS`。

| Run ID | Milestone | Purpose | System / Variant | Split / Dataset | Metrics / Artifact | Priority | Status | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R001 | M0 | 固定依赖和输入 | official repo/data/model/checkpoint | N/A | commit、revision、SHA-256、环境清单 | MUST | PASS | revision 与 SHA-256 已冻结 |
| R002 | M0 | 核验核心官方数值 | official checkpoint | GSM8K-Aug test | accuracy、样本数、解析失败数 | MUST | PASS | 44.43%，目标 44.8%±1.0 pp |
| R003 | M0 | 完成官方四任务复现 | official checkpoint | GSM-Hard/MultiArith/SVAMP | accuracy | MUST | PASS | 9.48% / 89.83% / 40.60%，均在 ±1.0 pp |
| R004 | M0 | 最大样本 20-step 预检 | standard SIM-CoT | official train sample | loss、peak reserved、step 对齐 | MUST | PASS | v3 完成 20/20 更新；5.5234 GB；首/末窗口 loss 1.025684/0.000213449；重载差值 0 |
| R005 | M0 | checkpoint 恢复验证 | standard SIM-CoT | smoke subset | reload 一致性、继续训练 loss、吞吐 | MUST | PASS | v3 续训后 loss 4.371e-7；第二次重载差值 0；5.1406 GB；checkpoint SHA 已保存 |
| R009 | M1-pre | checkpoint-24 来源审计 | official history + third-party Coconut checkpoint-24 | GSM8K/full where eligible | accuracy、revision、SHA-256 | MUST | FAIL | 官方完整历史从未发布 checkpoint-24；三个可读第三方候选 GSM 为 33.13%、31.69%、34.34%，均未达 36.6%±1 pp；另一个候选 gated 401 |
| R010 | M1 | 本地基础模型方向 | Coconut seed 0 | fixed short-budget train / 4 tests | macro accuracy、per-task accuracy | MUST | BLOCKED | 唯一阻塞为无经验证且共同可用的官方 checkpoint-24；M0 单卡门槛已通过；未启动 |
| R011 | M1 | 本地 SIM-CoT 方向 | standard SIM-CoT seed 0 | same as R010 | macro accuracy、step NLL | MUST | BLOCKED | 与 R010 同步停止，避免不公平或不可归因比较 |
| R012 | M1 | 公平性审计 | R010 vs R011 | logs/configs | updates、sampler、optimizer、batch parity | MUST | BLOCKED | 两分支未启动；v2 schedule 已冻结并排除 R020 审计题 |
| R020 | M2 | 无偏流行率抽样 | audit sampler | official train | ≥2,000 steps、split hash | MUST | PASS | 800 个唯一问题簇、2,110 个真实步骤；抽样前未筛异常 |
| R021 | M2 | 自动结构检查 | rule audit | prevalence audit | coverage、invalid candidates | MUST | PASS | 2,102 checked-match、5 mismatch candidates、2 manual、1 empty；仅作分流 |
| R022 | M2 | 人工复核 | human labels | prevalence audit | V/U 标签、分歧率、CI | MUST | BLOCKED | 盲评模板、10%（211 行）双标和统计脚本已就绪，等待两位独立评审填写 |
| R023 | M2 | 构建自然检测封存集 | matched audit builder | official train held-out | ≥100 low / ≥100 clean、hash | MUST | TODO | 尽量 50 invalid + 50 low-U |
| R030 | M3 | 构建可靠性训练集 | five-family generator | question split 60/20/20 | family counts、leakage checks | MUST | PASS | v2：2,000 题严格 1,200/400/400；33,887 行；五个开发族齐全；排除 R020 的 800 题；封存族未打开 |
| R030-F | M3 | 冻结可靠性特征缓存 | official checkpoint-28 + auxiliary decoder | R030 all splits | 33,887 行、67 shards、hash、finite、peak memory | MUST | PASS | manifest SHA d4fbc90a…；67/67 分片 hash 通过；峰值 0.6523 GB；冻结源无梯度 |
| R031 | M3 | LOFO 数值族 | V×U head | held-out numeric | V/U/composite AUC | MUST | FAIL | Validity 0.5462；composite 0.5300；低于最差族 0.70 门槛 |
| R032 | M3 | LOFO 运算符族 | V×U head | held-out operator | V/U/composite AUC | MUST | FAIL | Validity 0.6164；composite 0.6959；低于 0.70 |
| R033 | M3 | LOFO 依赖族 | V×U head | held-out dependency | V/U/composite AUC | MUST | FAIL | Validity 0.5612；composite 0.5993；低于 0.70 |
| R034 | M3 | LOFO 无关族 | V×U head | held-out irrelevant | V/U/composite AUC | MUST | PASS | Utility 0.7071；composite 0.8629 |
| R035 | M3 | LOFO 冗余族 | V×U head | held-out redundant | V/U/composite AUC | MUST | PASS | Utility 0.7269；composite 0.7791；五折宏平均仍仅 0.6934 |
| R036 | M3 | 训练最终可靠性头 | all five families | head train/validation | checkpoint、temperature、Brier/ECE | MUST | CUT | LOFO 宏 0.6934<0.75 且最差 0.5300<0.70；按规则不启动 |
| R037 | M3 | 自然封存评测 | frozen final head | natural teacher audit | composite/V/U AUC | MUST | CUT | 无通过门槛的最终头；自然审计未打开作模型选择 |
| R038 | M3 | 补偿错误压力测试 | frozen final head | sealed compensating family | Validity AUC | MUST | CUT | 封存族保持未打开；不得用来补救 LOFO 失败 |
| R039 | M3 | 捷径与双门槛审计 | frozen final head | all audits | G3-S/G3-N、length/digit/position/template correlation | MUST | CUT | 无最终头；仅完成开发折失败诊断，不冒充最终捷径审计 |
| R040 | M4 | 官方步骤离线赋权 | frozen scorer | official train minus audits | v/u/r/w、hash、weight stats | MUST | CUT | 检测门槛失败，无合格 frozen scorer |
| R041 | M4 | 主对照等权 | standard SIM-CoT seed 0 | official train / 4 tests | accuracy、clean step NLL | MUST | CUT | 当前配对主实验因 R036 CUT 而不启动；M1 复现仍独立 BLOCKED |
| R042 | M4 | 主方法加权 | V×U weighted seed 0 | same as R041 | accuracy、clean step NLL | MUST | CUT | 检测门槛失败，不允许加权蒸馏 |
| R043 | M4 | 单种子判定 | paired evaluation | 4 tests | macro Δ、GSM8K guard、NLL Δ | MUST | CUT | R041/R042 未启动 |
| R044 | M4 | 分支公平性审计 | R041 vs R042 | logs/configs/checkpoints | config diff、sample order、OOM path | MUST | CUT | R041/R042 未启动 |
| R050 | M5 | 多种子等权 | standard SIM-CoT seeds 1,2 | official train / 4 tests | accuracy/NLL | MUST | CUT | R043 未执行且检测门槛失败 |
| R051 | M5 | 多种子加权 | V×U weighted seeds 1,2 | same as R050 | accuracy/NLL | MUST | CUT | 同上 |
| R052 | M5 | 多种子统计 | paired aggregation | seeds 0,1,2 | mean/std/bootstrap CI | MUST | CUT | 无配对训练结果 |
| R055 | M5 | Validity-only 消融 | weighted V only | primary setting | macro accuracy | MUST | CUT | 无通过检测门槛的最终头 |
| R056 | M5 | Utility-only 消融 | weighted U only | primary setting | macro accuracy | MUST | CUT | 同上 |
| R057 | M5 | RSR/RD 基线 | legacy offline weighting | primary setting | macro accuracy、detector AUC | MUST | CUT | 当前注册主实验已在检测门槛停止 |
| R058 | M5 | 文本-only 必要性基线 | text-only MLP/logistic | detection + downstream | AUC、macro accuracy | MUST | CUT | 不在看到 LOFO 失败后追加模型选择 |
| R060 | M5 | 5% 受控污染 | equal/predicted/oracle | controlled train / clean tests | damage、recovery ratio | CONDITIONAL | CUT | 当前 scorer 不合格 |
| R061 | M5 | 10% 受控污染 | equal/predicted/oracle | controlled train / clean tests | damage、recovery ratio | CONDITIONAL | CUT | 同上 |
| R062 | M5 | 20% 受控污染 | equal/predicted/oracle | controlled train / clean tests | damage、recovery ratio | CONDITIONAL | CUT | 同上 |
| R065 | M5 | 权重下限敏感性 | floor variants | primary setting | macro accuracy、stability | NICE | CUT | 无主结果 |
| R066 | M5 | MLP 宽度敏感性 | head width variants | detection audit | AUC/params | NICE | CUT | 不在同一封存设计上事后调宽度 |
| R070 | M6 | 生成主表 | reporting pipeline | saved JSON | Tables 1–5 | MUST | TODO | 不手抄数值 |
| R071 | M6 | 生成主图 | reporting pipeline | saved predictions | Figures 2–4 | MUST | TODO | 自然/受控分开 |
| R072 | M6 | 失败案例审计 | false positives/negatives | sealed predictions | error taxonomy | MUST | TODO | 不挑选只支持方法的案例 |
| R073 | M6 | 复现包检查 | clean environment | configs/manifests | one-command dry run | MUST | TODO | hash 全部可追溯 |
| R074 | M6 | 结论边界检查 | gate report | all results | claim wording | MUST | TODO | 负结果也按规则报告 |

## Immediate Queue

1. 完成 R070/R074 负结果报告与复现证据检查。
2. R022 等待两位独立评审填写盲评 CSV；在此之前不估计自然噪声率。
3. R010/R011 等待可验证的共同 checkpoint-24；M0 已不再阻塞。
4. 若开展第二版可靠性头，必须新建规格和 run IDs，不能在当前 LOFO audit 上事后调参。
