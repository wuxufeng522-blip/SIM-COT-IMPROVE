# PRM800K 自然步骤噪声伤害实验实施计划

**Problem：** 当前证据只证明人工构造的高密度强冲突步骤能够伤害 SIM-CoT，尚未证明教师模型自然生成、最终答案正确但中间步骤被人工标错的轨迹会伤害学生模型。

**Method Thesis：** 使用 PRM800K 中同题、同生成批次的五步 Clean / Noise-1 / Noise-2 原始教师轨迹三联组，在答案监督完全相同的条件下训练 SIM-CoT，可直接估计一处和两处自然步骤错误的答案伤害与剂量关系。

**Date：** 2026-08-26

**批准规格：** `docs/superpowers/specs/2026-08-26-prm800k-natural-noise-damage-design.md`

**硬件：** NVIDIA RTX 4060 Laptop GPU，8 GB 显存

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：一处自然错误步骤会伤害 SIM-CoT | 这是研究可靠性加权的最低必要前提 | 3/3 种子 Clean > Noise-1，题目级配对 bootstrap 95% CI 不跨 0 | B1–B4 |
| C2：两处自然错误比一处错误伤害更大 | 证明步骤错误数量具有剂量杠杆，而非偶然轨迹差异 | 3/3 种子 Clean > Noise-1 > Noise-2，三项配对 CI 均不跨 0 | B1–B4 |
| A1：差异不是题目、答案、教师批次、步数或人工改写造成 | 自然噪声主张必须建立在严格三联配对上 | 同题同 generation、同五步、答案均正确、原始 completion 未改、确定性长度匹配 | B1, B2 |
| A2：结果不是实现、长度、随机日程或显存失败造成 | 防止把工程异常当成噪声伤害 | loss parity、arm parity、逐步索引、有限梯度、7.4 GB 红线、三种子独立初始化 | B2, B3 |

## Paper Storyline

- 主结果必须回答：在100%题目覆盖的富集条件下，一处和两处自然教师错误分别造成多大答案伤害。
- 数据证据必须先证明：错误步骤来自 PRM800K 原始模型 completion 和人工 `-1` 标签，没有人工构造或修复。
- 支持结果报告 `answer_only`，区分“噪声侵蚀干净 CoT 收益”与“含噪 CoT 比不用 CoT 更差”。
- 本轮不测试可靠性头、步骤权重、自然流行率、其他模型或其他数据集。
- 若严格三联组不足512，数据假设失败即为有效负结果，停止而不是放宽规则。

## Experiment Blocks

### Block 1：PRM800K 原始轨迹与严格三联数据门

- **Claim tested：** A1。
- **Why this block exists：** 没有足量、同题且答案正确的自然三联轨迹，后续训练无法归因于步骤错误数量。
- **Dataset / split / task：** PRM800K 官方训练分区；官方 MATH grader；排除 QC、screening、human completion、flagged 和评分0。
- **Compared systems：** Clean（5×`+1`）、Noise-1（1×`-1`）、Noise-2（2×`-1`）。
- **Metrics：** 原始记录数、答案正确数、恰好五步数、三种标签模式数量、同题同 generation 三联数量、排除原因、错误位置分布、长度极差。
- **Setup details：** 多候选按 token 长度极差最小选择；并列按原始记录 SHA-256；问题按带域 SHA-256 冻结。
- **Success criterion：** 至少512个问题不重复严格三联组，并另有256道五步全正开发题；分区零重叠；清单重复运行字节一致。
- **Failure interpretation：** 输出 `INSUFFICIENT_STRICT_TRIPLETS` 或 `INSUFFICIENT_CLEAN_DEV`，停止全部 GPU 工作。
- **Table / figure target：** 数据审计主表、标签位置与长度分布附录。
- **Priority：** MUST-RUN。

### Block 2：自然语言步骤映射与工程审计

- **Claim tested：** A2。
- **Why this block exists：** PRM800K 是自由文本步骤，必须证明五个 latent states 与五个原始步骤一一对应且官方答案监督不变。
- **Dataset / split / task：** 冻结三联组中的小型 sanity 子集。
- **Compared systems：** `answer_only`、`clean_aux`、`natural_noise_1`、`natural_noise_2`。
- **Metrics：** all-one loss parity、答案 token/label parity、辅助目标 SHA-256、索引映射、有限 loss/grad、辅助梯度零测试、GPU peak reserved memory。
- **Setup details：** 固定边界符；不改步骤文本；单更新 BF16；相同微批次与 dropout 日程。
- **Success criterion：** 全部工程门通过，显存不超过7.4 GB。
- **Failure interpretation：** 属于实现失败；不得进入正式训练或降低数据/显存规则。
- **Table / figure target：** 工程审计表。
- **Priority：** MUST-RUN。

### Block 3：四组三种子自然噪声训练

- **Claim tested：** C1、C2、A2。
- **Why this block exists：** 直接估计自然错误步骤对同一 SIM-CoT 学生的训练伤害。
- **Dataset / split / task：** 512道严格三联训练题；256道工程开发题只做端到端检查；不以准确率决定继续与否。
- **Compared systems：** `answer_only`、`clean_aux`、`natural_noise_1`、`natural_noise_2`。
- **Metrics：** 训练总损失、答案损失、辅助损失、梯度裁剪前范数、显存峰值、checkpoint SHA-256。
- **Setup details：** checkpoint 28；64 updates；accumulation 8；BF16；LR `1e-4`；WD `0.01`；clip `1.0`；`lambda_aux=1`；种子 `20260809/10/11`。
- **Success criterion：** 12次训练完整结束，配置/数据哈希一致，所有 checkpoint 独立从起始模型初始化。
- **Failure interpretation：** 可恢复工程错误可同配置续跑；跨配置或跨组 checkpoint 恢复禁止。
- **Table / figure target：** 训练完整性附表。
- **Priority：** MUST-RUN。

### Block 4：封存500题的配对伤害与剂量判定

- **Claim tested：** C1、C2。
- **Why this block exists：** 训练损失不能替代未见干净问题上的答案性能。
- **Dataset / split / task：** PRM800K 官方封存500道 MATH 题；测试输入无教师步骤和噪声。
- **Compared systems：** 四组、三个种子。
- **Metrics：** MATH grader EM/正确题数、answer NLL、三种子均值/标准差、精确 McNemar、10,000次题目级配对 bootstrap、开发集 clean-step NLL。
- **Setup details：** 所有 checkpoint 和分析代码冻结后一次性打开；bootstrap 按问题整体重采样所有组和种子。
- **Success criterion：** 严格按规格产生 `DOSE_DEPENDENT_HARM`、`TWO_ERROR_HARM_ONLY`、`ONE_ERROR_HARM_ONLY`、`UNSTABLE_HARM` 或 `NO_OBSERVED_HARM`；必要时加 `FLOOR_LIMITED`。
- **Failure interpretation：** 地板不是停止门；无下降且出现地板时不得解释为噪声无害。
- **Table / figure target：** 主结果表、三项配对效应图。
- **Priority：** MUST-RUN。

### Block 5：结果边界和后续决策

- **Claim tested：** 不新增正向主张；限制 C1/C2 外推。
- **Why this block exists：** PRM800K 轨迹经过主动学习采样，本实验又使用100%含噪覆盖，不能视为自然流行率实验。
- **Dataset / split / task：** 已冻结审计、训练、预测和分析产物。
- **Compared systems：** 仅已完成组。
- **Metrics：** verdict、地板标记、按位置/题型/难度的描述性分层、允许/禁止主张清单。
- **Success criterion：** JSON、Markdown 和逐题预测数值一致；明确写出“自然错误内容、人工富集比例”。
- **Failure interpretation：** 工程失败与科学负结果分开报告。
- **Table / figure target：** 中文总结和限制表。
- **Priority：** MUST-RUN。

## Implementation Work Packages

### WP1：配置、下载契约和不可变来源

**Create：**

- `configs/reliable_simcot/prm800k_natural_noise.json`
- `src/reliable_simcot/prm800k_data.py`
- `tests/test_prm800k_data.py`
- `scripts/run_prm800k_natural_noise.py`

**Responsibilities：**

- 定义官方 PRM800K commit/文件清单、缓存目录和 SHA-256；
- 读取 phase1/phase2 JSONL 与官方 MATH split/grader；
- 规范化问题 ID，重建 chosen completion，验证答案；
- 输出 provenance 与只读原始文件哈希。

**Acceptance：** 下载/解析可重复；不把 Git LFS 指针当数据；原始数据不被训练代码改写。

### WP2：严格三联审计与冻结

**Create：**

- `src/reliable_simcot/prm800k_triplets.py`
- `tests/test_prm800k_triplets.py`

**Responsibilities：**

- 实现五步、标签、flag、QC、human completion、answer grader 和上下文长度门；
- 同题同 generation 分组；
- 最小 token 极差与 SHA-256 并列规则；
- 冻结512三联训练题和256全正开发题；
- 输出逐条排除理由、标签位置和重叠审计。

**Acceptance：** fixture 覆盖每个排除码；打乱输入行序不改变冻结结果；不足样本返回规格定义状态。

### WP3：四臂训练映射与GPU sanity

**Create：**

- `src/reliable_simcot/prm800k_experiment.py`
- `tests/test_prm800k_experiment.py`

**Reuse：**

- `official_adapter.py` 的 checkpoint/latent 推理路径；
- `oracle_weighting.py` 或 `full_conflict_experiment.py` 的 grouped auxiliary loss、逐微批次随机日程和安全保存；
- `m1_training.py` 的原子 JSON 与哈希工具。

**Responsibilities：**

- 自由文本五步 tokenization 与固定边界符；
- arm mapper、答案逐字节 parity、辅助目标 parity；
- loss parity、辅助梯度零测试、2-update sanity；
- 三种子独立训练、checkpoint 与状态机。

**Acceptance：** 相关单测及回归测试通过；显存、有限性和映射门通过。

### WP4：500题评估、统计和报告

**Create：**

- `src/reliable_simcot/prm800k_evaluation.py`
- `tests/test_prm800k_evaluation.py`
- `scripts/run_prm800k_overnight.py`

**Responsibilities：**

- 500题答案生成、官方 grader、断点续评；
- answer NLL、开发集 clean-step NLL；
- McNemar、题目级配对 bootstrap 和三种子汇总；
- 互斥 verdict、`FLOOR_LIMITED` 和中文报告；
- 状态机禁止在训练完成前打开最终确认结果。

**Acceptance：** 合成 fixture 覆盖全部 verdict；固定 bootstrap seed 可重复；报告与 JSON 一致。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Main Risk |
|---|---|---|---|---|---|
| M0 | 来源与代码基线 | PN001–PN003 | 官方数据文件有效、现有测试通过 | CPU 0.5–2 h（含下载） | Git LFS/网络/数据格式 |
| M1 | 严格三联审计 | PN010–PN012 | >=512三联且>=256全正开发题 | CPU 0.5–2 h | 样本数量不足，最高风险 |
| M2 | 映射、损失和GPU门 | PN020–PN023 | parity、有限性、索引、<=7.4 GB | GPU 0.2–0.5 h | 自由文本过长/OOM |
| M3 | 四组三种子训练 | PN100–PN133 | 12个独立checkpoint完成 | GPU 4–8 h | MATH文本更长，耗时上升 |
| M4 | 封存500题评估 | PN200–PN233 | 12组预测完整、grader一致 | GPU 2–5 h | 自回归答案生成耗时 |
| M5 | 统计与报告 | PN300–PN302 | verdict/报告/哈希一致 | CPU 0.2–0.5 h | 地板导致结论受限 |

## Detailed Execution Sequence

### M0–M1：先证实数据假设

1. 新增 fixture 驱动的数据解析测试。
2. 获取官方 PRM800K 数据，检查非 LFS 指针、文件数、行数和哈希。
3. 重建 chosen trajectories，用官方 grader 验证最终答案。
4. 运行严格五步标签审计并形成三联候选。
5. 先输出数量审计，再由程序按冻结规则选择512/256。
6. 若不足，写报告并停止；不实现或运行GPU队列。

### M2：证明现有 SIM-CoT 能接受冻结自然步骤

1. 实现固定边界自由文本步骤映射。
2. 对同一三联组核对题目和答案张量逐字节一致。
3. 跑 all-one loss parity 和 `answer_only` 辅助梯度零测试。
4. 四组各跑2 updates，记录显存峰值。
5. 通过后冻结配置、清单和分析代码版本。

### M3：训练队列

按 seed 外层、arm 内层固定顺序：

1. `20260809`：answer_only → clean_aux → natural_noise_1 → natural_noise_2；
2. `20260810`：同顺序；
3. `20260811`：同顺序。

不在训练中途运行最终确认评估。每臂独立保存进度、checkpoint、输入清单哈希和训练指标。

### M4–M5：一次性确认与结论

1. 全部12个checkpoint完成并核验哈希；
2. 冻结评估脚本和统计 fixture；
3. 一次性运行500题确认评估；
4. 计算三项配对比较和预注册 verdict；
5. 输出机器 JSON、逐题 JSONL、中文 Markdown 和 tracker 状态。

## Compute and Data Budget

- **GPU：** 预计 6–13 GPU-hours；长文本可能使上界增加。
- **CPU：** 数据下载、解析和 grader 审计预计 1–4 小时。
- **磁盘：** PRM800K 原始数据、12个checkpoint与预测需要预留数 GB；实施前记录实际容量。
- **人工评估：** 无；完全使用官方人工标签与 grader。
- **最大瓶颈：** 同题同 generation 的严格五步0/1/2错误三联数量。

## Risks and Mitigations

- **严格三联不足512：** 按规格停止并报告，不自动改为跨批次、允许0或合并步骤。
- **当前GPT-2在MATH地板：** 不停止训练；结果加 `FLOOR_LIMITED`，无下降不解释为无害。
- **PRM800K active-learning采样偏差：** 明确称“自然错误内容的人工富集实验”，不估计自然流行率。
- **自由文本长度改变辅助loss剂量：** 使用同题长度极差最小三联并报告token长度；不事后删异常值。
- **答案正确但中间`-1`标签存在标注争议：** 排除 flagged/0，仅使用官方chosen标签；不重新裁决标签。
- **OOM：** 复用 batch=1/gradient accumulation 路径；超过7.4GB即停止，不改红线追结果。
- **确认集泄漏：** 训练状态机在12个checkpoint与分析代码冻结前禁止调用最终评估模式。

## Final Checklist

- [x] 主张与批准规格一致
- [x] 自然错误与人工富集比例明确区分
- [x] 数据不足门先于GPU工作
- [x] 题目、答案、步数和教师批次混杂已控制
- [x] 三种子与配对统计已定义
- [x] 地板效应不作为停止门
- [x] 可靠性头和步骤加权明确排除
- [x] 必跑与后续研究已分开
