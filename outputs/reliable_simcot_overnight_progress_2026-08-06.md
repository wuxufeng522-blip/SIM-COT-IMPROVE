# 可靠性门控 SIM-CoT：夜间进度报告

日期：2026-08-06

硬件：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB

## 结论先行

目前已经证明：官方 SIM-CoT checkpoint 的四任务评测管线可信；修正步骤解析后的 20-step 训练、checkpoint 重载和续训可在 RTX 4060 8GB 上完成；自然教师步骤可以按题目无泄漏抽样、冻结、自动分流并制作为双人盲评包；五个开发污染族的数据与冻结特征管线也已实现。

首版可靠性头已经得到明确负结果：五折 LOFO 宏平均复合 ROC-AUC 为 `0.6934`，低于 `0.75`；最差族为 `0.5300`，低于 `0.70`。因此没有训练最终头、没有打开补偿性封存族、没有进入加权蒸馏，也不能宣称可靠性加权提高答案正确率。R010/R011 同预算训练另有独立阻塞：官方没有发布训练配置引用的 Coconut checkpoint-24；M0 单卡门槛本身已通过。

## 已完成结果

| 模块 | 结果 | 判定 |
| --- | --- | --- |
| 官方 checkpoint：GSM8K-Aug | 44.43%，论文 44.8% | PASS |
| 官方 checkpoint：GSM-Hard | 9.48%，论文 9.3% | PASS |
| 官方 checkpoint：MultiArith | 89.83%，论文 90.8% | PASS，位于 ±1 pp 下界内 |
| 官方 checkpoint：SVAMP | 40.60%，论文 40.7% | PASS |
| 第三方 Coconut checkpoint-24 初始化审计 | 33.13 / 6.90 / 79.33 / 37.10 | FAIL，3/4 偏离论文 Coconut 行 |
| R004-v3 修正格式 20-step | 20/20 更新；峰值 5.5234 GB；重载差值 0 | PASS |
| R005-v3 checkpoint 续训 | 续训 loss 4.371e-7；第二次重载差值 0；峰值 5.1406 GB | PASS |
| R020 自然审计抽样 | 800 个唯一问题簇、2,110 个真实步骤 | PASS |
| R021 自动结构分流 | 2,102 算术吻合；5 个不匹配候选；2 个需手查；1 个空步骤 | PASS，仅作分流 |
| R022 盲评准备 | 211 行双标重叠；A 1,156 行，B 1,165 行 | PREP PASS，等待人工标签 |
| R030 五族可靠性数据 | 2,000 题；33,887 行；1,200/400/400 题级切分 | PASS |
| R031–R035 五折 LOFO | 宏 AUC 0.6934；最差族 0.5300 | FAIL；停止最终头与加权训练 |
| 完整回归测试 | 55 passed | PASS；pytest cache 权限警告不影响结果 |

## 关键勘误：训练步骤解析

训练文本以 `####` 分隔答案。旧适配器用最后一个 `##` 切分，导致每条轨迹多出一个伪 `##` 步骤；官方 `preprocessing/gsm_icot.py` 使用第一个推理字段和最后一个答案字段。该问题不影响 R002/R003 的 checkpoint 答案评测：GSM8K 推理时只使用题目与最终答案，OOD 三项直接读取公开 JSON；但会影响 R004/R005 的辅助步骤监督语义。

因此旧 R004/R005 已降级为“显存与保存路径曾运行”的工程证据，不再称为精确官方训练格式复现。R004-v2 的两次无 traceback 退出后来定位为外层命令超时，失败记录仍保留。加入原子进度记录和可恢复检查后，R004-v3 完成 20/20 更新：前/末五次更新平均 loss `1.025684/0.000213449`，最终探针 loss `3.8306e-6`，峰值 reserved `5.5234 GB`，重载差值 `0`。R005-v3 从该 checkpoint 续训后探针 loss `4.3710e-7`，第二次重载差值 `0`。两项均 PASS。

## M1：同预算短训练准备情况

- R010 Coconut 与 R011 SIM-CoT 的训练器、BF16、梯度累积、optimizer/RNG 断点恢复、验证 NLL 与公平性字段已实现。
- 共同 v2 schedule 冻结为 8,192 个微批次，SHA-256：`4575fb7e942307ff6066aec6de248fb0479e90dc1915c5382c59bfca656cae05`。
- schedule 强制读取 R020 freeze manifest，并排除了随机候选中命中的 18 个审计题；任何审计题泄漏都会直接报错。
- R010/R011 未启动。官方模型仓库的完整 Git/Hugging Face 历史从第一版开始仅提供 checkpoint-28，训练配置虽引用 checkpoint-24，但任何 commit、tag 或 branch 都未包含该权重。
- 可读取的 checkpoint-24 候选均已按“先跑完整 GSM8K，未达 `36.6%±1 pp` 就停止 OOD”审计：Onlydrinkwater 为 `33.13%`（且已有四任务 `33.13/6.90/79.33/37.10`）；batra98 为 `31.69%`；mpilligua 为 `34.34%`。darpanaswal 候选返回 gated 401，记为不可访问，不记作性能失败。没有挑相邻 epoch 以避免在测试集上选 checkpoint。

## M2：自然教师步骤审计

### R020 无偏冻结抽样

- 总体：385,536 个唯一题目，源文件另有 84 条重复题目记录。
- 抽样：在排序后的题目 SHA-256 ID 上，以 seed `20260805` 做 800 题不放回均匀抽样；先抽题，再保留整条轨迹的全部步骤。
- 产物：2,110 个步骤，超过预注册的 2,000 步下限。
- 抽样前没有按异常、长度、数字或自动规则筛选。
- 审计行 SHA-256：`114e8716ad8b2789fab39a0d71878a835a29a215f294c329c0b0fecda53458ef`。

### R021 自动分流

初版规则把常见两位小数近似误标为算术错误。样本 ID 保持冻结不变，改进规则作为独立 R021 派生层保存，并记录有 25 行的标记发生变化。当前统计：

- 2,102 `checked_match`；
- 5 个 `arithmetic_mismatch_candidate`；其中人工查看可见 3 个是截断近似，2 个是明显算术错误；
- 2 个带 `~` 的近似表达式需手查；
- 1 个空步骤；
- 32 个最终步骤数值与答案不同的候选，很多涉及向上/向下取整或单位变化，不能自动判错；
- 4 个重复步骤候选，重复可能有语义作用，不能自动判 Utility=0。

这些数字不是自然噪声流行率。只有完成 R022 盲评和分歧裁决后，才能报告 invalid、valid-but-low-utility 与受影响轨迹比例及 95% CI。

### R022 双评审包

- 两份 CSV 不含自动候选标记、算术状态、最终答案匹配状态或源行号。
- 每条步骤恰有一个主评审；211 行（10%）被隐藏地分配给两位评审。
- Validity 必须为 0/1；Utility 仅在 Validity=1 时允许为 0/1。
- 汇总脚本会核验所有不可编辑字段，计算 Wilson 95% CI、Validity/Utility Cohen’s κ，并输出待裁决分歧。

## M3：可靠性检测可行性

### R030 五族数据

- 从排除 R020 审计题后的 384,058 个可用唯一题目中固定抽取 2,000 题，严格按题目切成 head-train/head-validation/head-audit `1,200/400/400`，三者没有题目重叠。
- 共生成 33,887 行：clean 5,146、等价正例 5,146、数值错误 5,146、运算符/关系错误 5,027、依赖/顺序错误 3,145、无关但正确 5,146、冗余重复 5,131。
- 标签分布为 invalid 13,318、valid-but-useless 10,277、valid-and-useful 10,292。Validity=0 时 Utility 保持未定义，不强行标成 0。
- 补偿性错误压力族 2,088 行已生成并写入 hash，但仍处于封存状态；正式头冻结前没有读取内容。

### 冻结特征缓存

- 使用官方 checkpoint-28 对每题只运行一次 latent forward；候选步骤由辅助解码器 teacher forcing 后做 mask-aware 平均池化。Student、辅助解码器和特征全冻结，只训练后续小头。
- 33,887 行、2,000 题全部完成，67 个分片独立复核为 0 个 hash 错误，行数求和恰为 33,887；无 NaN/Inf；峰值 reserved 显存 0.6523 GB。
- 特征 manifest SHA-256：`d4fbc90a595a34fae75d202daae000a29ca646e32d72205d63813f56e7975c59`；cache key：`2c47d753028ddd20e3e8253d99538ce79bbd5ea1ce2d29c212ae0a0d48f92fb4`。
- 首次全量运行在 11,800 行附近因 Windows 对进度 JSON 的短暂读锁触发 `WinError 5`；已增加原子写重试和分片校验续跑。最终从 11,776 行安全恢复，没有重复或丢失，原失败证据保留。

### 可靠性头预检

- 双投影/交互 MLP 共 459,906 个可训练参数，低于 500,000 上限；损失严格为两项 BCE 加两项固定权重 0.5 的同题 margin 排序损失。
- 单折 20-step 工程预检完成，峰值 reserved 显存 0.0664 GB，checkpoint 重载固定输入 logits 最大差值 0；55 个回归测试全部通过。
- 该预检只跑 numeric 留出族，未执行完整五折门槛；20 步后的留出复合 AUC 0.5058 只是风险信号，不能用于选择超参数。五折正式配置已在看到正式结果前冻结并启动。

### R031–R035 五折正式结果

| 完整留出族 | validation 复合 AUC | 留出 Validity AUC | 留出 Utility AUC | 留出复合 AUC | 判定 |
| --- | ---: | ---: | ---: | ---: | --- |
| 数值错误 | 0.9812 | 0.5462 | N/A（留出负例全为 invalid） | 0.5300 | FAIL |
| 运算符/关系错误 | 0.9640 | 0.6164 | N/A | 0.6959 | FAIL |
| 依赖/顺序错误 | 0.9714 | 0.5612 | N/A | 0.5993 | FAIL |
| 无关但正确 | 0.9538 | N/A（所有样本均 valid） | 0.7071 | 0.8629 | PASS |
| 冗余重复 | 0.9664 | N/A | 0.7269 | 0.7791 | PASS |

- 五折宏平均复合 ROC-AUC `0.693430`，门槛 `≥0.75`；最差族 `0.529960`，门槛 `≥0.70`。两项均失败。
- 等价改写相对 clean 的平均可靠性下降为 `-0.313522`，满足“不下降超过 0.10”的单侧门槛；负值表示等价改写反而被打得更高，并不挽救 LOFO 失败，也提示模型可能偏好某种表面表达。
- 所有折的 checkpoint 重载最大 logits 差值均为 `0`，峰值 reserved 显存均为 `0.0684 GB`。因此失败不是 OOM、checkpoint 损坏或训练未收敛导致。
- 原始结果：`outputs/reliable_simcot/r031_r035_lofo/lofo_metrics.json`，SHA-256 `796d02c65f32252a6ec86cacacb9b9c9edbdaba013c8d1de92176214489e0393`；配置 SHA-256 `67adda4cbd59debe14b0f9f9a6c9fe884375db423cf0ab8325d9df829ca9c6c1`；代码 commit `3b56f41cb7dc92f3bb3b87430170d617820970b6`。

### 失败诊断（只分析，不调参）

- 训练/validation 上的复合 AUC 均为 `0.9538–0.9812`，但完整留出一个污染机制后，三类有效性错误显著下降。这是典型的整类泛化失败：模型能识别已见模板，却没有形成稳定的“算术/依赖语义是否成立”表征。
- 同题成对方向仍有弱信号：clean 得分高于污染的比例，数值/运算符/依赖分别为 `66.0%/76.6%/66.0%`；但跨题绝对分数波动压过了这个差异，无法达到需要在真实数据上单独给步骤打分的绝对 AUC 门槛。
- 数值族 98.1%、运算符族 100% 的成对文本长度完全相同，说明这两族失败不能简单归因于“污染文本更长/更短”。不过复合分数与步骤位置在数值/运算符/依赖三折的 Spearman 相关分别约为 `-0.375/-0.235/-0.352`，显示题目/位置基线仍是需要控制的潜在捷径或方差来源。
- 按预注册规则，当前版本到此停止。下一版若继续，应先改变训练证据而不是调门槛：增加同一语义机制的多样表达与可执行一致性信号，并检验题内归一化或基于前缀的局部差分能否在“不依赖对照污染步骤”的条件下改善绝对可靠性。

## 当前证据对实验思想的含义

1. **方案工程上可实现，但首版核心检测假设未通过。** 数据冻结、无泄漏、双标签、特征缓存和五折 LOFO 都已完整运行；首版头对未见有效性错误族泛化不足，因此不能进入加权蒸馏。
2. **自然噪声并非凭空假设。** 随机样本中已看到空步骤与明显算术错误，但当前只能称为案例，不能据此估计总体比例。
3. **不能跳过人工语义审计。** 取整、单位换算、重复步骤和“算术正确但用错题目数值”都说明纯规则或相似度不足以定义可靠性。
4. **不能把第三方 checkpoint 当官方初始化。** 否则后续 SIM-CoT 优势或加权收益会被初始化差异混淆。

## 下一决策点

1. 两位评审独立填写 `reviewer_a_labeled.csv` 与 `reviewer_b_labeled.csv`，再运行 `scripts/compile_human_review.py`。
2. 若自然低可靠性比例 `<1%`，按预注册规则把论文主张降级为受控污染鲁棒性；若为 `1%–5%`，自然实验与 5/10/20% 受控曲线并行；若 `≥5%`，保留自然主实验。
3. 只有得到可验证的 checkpoint-24，才恢复 R010/R011；修正版 20-step/reload 已经通过，不再是阻塞项。
4. 当前注册版本已经因 LOFO 门槛失败而停止。若继续第二版，必须先新建规格、污染表达与 run IDs，不能在本次 head-audit 结果上反复调参后仍称其为封存检验。

## 主要产物

- `outputs/reliable_simcot/r020_natural_audit/freeze_manifest.json`
- `outputs/reliable_simcot/r020_natural_audit/audit_rows.jsonl`
- `outputs/reliable_simcot/r021_auto_triage/triage_manifest.json`
- `outputs/reliable_simcot/r022_blinded_review_v2/reviewer_a.csv`
- `outputs/reliable_simcot/r022_blinded_review_v2/reviewer_b.csv`
- `outputs/reliable_simcot/r022_blinded_review_guide.md`
- `outputs/reliable_simcot/r004_v3/metrics.json`
- `outputs/reliable_simcot/r005_v3/metrics.json`
- `outputs/reliable_simcot/r030_reliability_data_v2/manifest.json`
- `outputs/reliable_simcot/r031_feature_cache/feature_manifest_full.json`
- `outputs/reliable_simcot/r031_r035_lofo/lofo_metrics.json`

本报告不把自动候选数写成自然噪声率，不把失败的 checkpoint 来源写成官方权重，也不把 validation 高分或局部通过的污染族写成整体方法有效。补偿性错误封存族没有打开，加权训练没有启动。

## 2026-08-09：Oracle 步骤加权机制验证

在用户明确要求先验证“若能正确识别污染步，给它 0.1 权重是否有助于推理”后，新建了独立的 O001–O020 受控机制实验。该实验不复用失败的可靠性头，而用合成数据生成器记录的真实污染位置作为完美判别器，因此只检验加权机制，不检验检测器。

- 四组都从已验证的官方 checkpoint 28 起步，使用同一批 2,048 条训练样本、同一顺序、同一初始化和逐微批 dropout 种子。
- 每题前五步恰好污染一步；五类污染与五个位置形成 25 个联合单元，每格 81–82 条。
- 全 1 自定义步骤损失与官方实现的绝对误差仅 `4.768e-07`；58 项测试通过；四组峰值训练显存均为 5.527 GB。
- 完整 1,319 题官方测试集结果：clean `40.561%`，noisy_equal `39.803%`，raw 0.1 `41.092%`，normalized 0.1 `41.774%`。
- raw 0.1 相对 noisy_equal 提高 `+1.289 pp`（73 胜/56 负，95% CI `[-0.379,+2.957] pp`，McNemar `p=0.1587`）。
- normalized 0.1 相对 noisy_equal 提高 `+1.971 pp`（82 胜/56 负，95% CI `[+0.227,+3.715] pp`，McNemar `p=0.0329`）。
- noisy_equal 相对 clean 只下降 `0.758 pp`，95% CI 跨 0，低于事先冻结的 1 pp 损伤门槛。因此严格判定为 `INCONCLUSIVE_INSUFFICIENT_NOISE_DAMAGE`，不能宣称已经完全证明恢复噪声伤害。

该结果支持继续研究题内归一化的相对步骤权重，但仅有一个训练种子，且所有续训组仍低于未经续训的 checkpoint-28 基线 44.43%。下一步应把污染提高到每题 2/5 步，先验证 noisy_equal 的损伤门槛，再补三个训练种子。

详细报告：`outputs/reliable_simcot/oracle_weighting/oracle_weighting_causal_report_2026-08-09.md`；机器可读分析：`outputs/reliable_simcot/oracle_weighting/o020_causal_analysis.json`；实现 commit：`9d29f08`。
