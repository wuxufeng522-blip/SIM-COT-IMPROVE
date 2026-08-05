# Experiment Plan：可靠性门控 SIM-CoT

- **Problem**：官方教师生成的 SIM-CoT 训练步骤可能包含错误、无关或冗余监督；等权蒸馏无法区分这些步骤。
- **Method Thesis**：冻结的 SIM-CoT 隐状态与辅助解码语义经 Validity/Utility 双头形成步骤监督可靠性，离线归一化赋权后可减少低可靠性教师步骤的伤害。
- **Date**：2026-08-05
- **Hardware**：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB
- **Frozen design**：`docs/superpowers/specs/2026-08-05-reliability-gated-simcot-natural-noise-design.md`

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
| --- | --- | --- | --- |
| C1（主要）可靠性头学习到跨污染机制、跨题目的步骤可靠性，而非模板捷径 | 没有可靠检测，加权 loss 只是任意重标度 | 五轮 LOFO 复合 ROC-AUC 宏平均 ≥0.75、最差族 ≥0.70；自然封存审计 ≥0.75；补偿性错误 Validity AUC ≥0.70；等价改写下降 ≤0.10 | B2、B3、B5 |
| C2（主要）可靠性加权缓解教师训练步骤噪声，同时保持 SIM-CoT 的干净测试性能 | 这是下游方法价值所在 | 与等权 SIM-CoT 同预算、同初始化、同数据顺序；单种子四任务宏平均 ≥+1.0 pp 且 GSM8K-Aug 不下降超过 1.0 pp，随后 3 seeds 与置信区间确认 | B1、B4、B5 |
| Anti-claim：增益来自更多参数、总体 loss 缩小、合成模板或不公平训练预算 | 排除“已有模块简单拼接”的主要审稿质疑 | 冻结 scorer、样本内权重均值 1、相同更新数；Validity-only/Utility-only/RSR-RD/oracle 消融；题目级切分和整类留出 | B3、B4、B5 |

## Paper Storyline

### Main paper must prove

- 官方 SIM-CoT 基线可复现，比较管线可信；
- 官方教师步骤中自然低可靠性比例经过无偏审计，而非预设；
- Validity×Utility 在未见污染族和自然教师步骤上达到预注册检测门槛；
- 冻结离线加权在相同预算下优于等权 SIM-CoT；
- 受控污染曲线显示恢复的是噪声造成的损失，而非一般正则化偶然收益。

### Appendix can support

- 每个污染族的详细 confusion/error taxonomy；
- 校准曲线、Brier score、长度/数字/位置捷径相关性；
- 5%、10%、20% 污染的完整逐数据集结果；
- 显存、吞吐、缓存大小和单卡工程细节；
- 额外权重下限或 MLP 宽度敏感性，仅在主结果通过后运行。

### Experiments intentionally cut

- 测试集人工加噪；
- Student 与可靠性头端到端联合训练；
- 动态在线赋权；
- 大规模超参数搜索；
- 多个弱 baseline 堆叠；
- 在检测门槛失败后继续做下游性能主张。

## Experiment Blocks

### Block 1：官方 SIM-CoT 复现与公平训练底座

- **Claim tested**：后续差异来自步骤权重，而不是错误实现或不公平预算。
- **Why this block exists**：现有仓库是受控算术 RSR/RD PoC，不能直接充当官方 SIM-CoT 复现。
- **Dataset / split / task**：官方 GSM8K-Aug、GSM-Hard、MultiArith、SVAMP；官方 train/test 划分与答案提取规则。
- **Compared systems**：官方 checkpoint；本地 Coconut 短预算；本地标准 SIM-CoT 短预算。
- **Decisive metrics**：四任务 accuracy；官方 checkpoint 各任务误差；本地 SIM-CoT 相对 Coconut 的宏平均差。
- **Secondary metrics**：step NLL、峰值 reserved 显存、updates/hour、checkpoint reload 一致性。
- **Setup details**：固定官方仓库 revision、模型 revision 和 tokenizer；单卡 mixed precision、batch 1、梯度累积；20-step preflight；7.4 GB 显存红线。
- **Success criterion**：A0 官方 checkpoint 每任务在报告值 ±1.0 pp；smoke loss 有限且下降、可保存重载；A1 本地同预算 SIM-CoT 宏平均至少比 Coconut 高 2.0 pp。
- **Failure interpretation**：A0 失败表示评测/环境不可信；A1 失败表示本地预算或 SIM-CoT 训练未复现，均停止新方法解释。
- **Table / figure target**：Table 1“Baseline reproduction”；Appendix Table A1“single-GPU profile”。
- **Priority**：MUST-RUN。

### Block 2：官方教师自然噪声流行率审计

- **Claim tested**：官方教师训练步骤中是否有足量自然低可靠性，决定主张边界。
- **Why this block exists**：不能因为教师是 GPT-4 就假定无噪声，也不能为了实验价值夸大噪声。
- **Dataset / split / task**：从官方训练题目随机封存至少 2,000 个步骤；按题目 ID 排除出后续蒸馏训练。
- **Compared systems**：结构化表达式自动检查；双人/复核式人工标签；不比较模型性能。
- **Decisive metrics**：invalid 比例、valid-but-low-utility 比例、含至少一个低可靠性步骤的轨迹比例及 95% CI。
- **Secondary metrics**：自动规则覆盖率、人工分歧率、各错误类型频数。
- **Setup details**：先随机抽样再检查；记录抽样 seed 和数据 hash；Utility 只在 Validity=1 时标注。
- **Success criterion**：完成无偏估计并按 ≥5%、1%–5%、<1% 规则选择自然主实验、自然+受控或仅受控主张。
- **Failure interpretation**：自然噪声过少不是方法失败，但禁止声称解决了官方数据中的普遍自然噪声。
- **Table / figure target**：Table 2“Teacher-step reliability audit”；Figure 2“noise taxonomy”。
- **Priority**：MUST-RUN。

### Block 3：Validity/Utility 可靠性检测

- **Claim tested**：可靠性头能泛化到未见题目、未见污染族和自然教师步骤。
- **Why this block exists**：这是进入加权训练前的独立资格门槛。
- **Dataset / split / task**：题目级 60/20/20；五个开发污染族；五轮 LOFO；自然封存检测审计；永久封存补偿性错误族。
- **Compared systems**：Validity-only、Utility-only、Validity×Utility；RSR/RD 经验分数；可选的文本-only Logistic/MLP 简单基线。
- **Decisive metrics**：复合 ROC-AUC、Validity AUC、有效步骤内 Utility AUC、最差族 AUC。
- **Secondary metrics**：PR-AUC、Brier score、ECE、等价改写分数下降、表面捷径相关性。
- **Setup details**：冻结 Student 与辅助解码器；128 维双投影；512 维交互；小型共享 MLP；同题同位置 ranking；未经类别重采样的 validation 做温度校准；缓存特征。
- **Success criterion**：完全满足规格中的四个 AUC 门槛与等价改写门槛。
- **Failure interpretation**：宏平均通过而最差族失败说明模板外泛化不足；合成通过而自然失败说明域差异；Validity 通过而 Utility 失败说明“正确”可判但“有用”不可判。
- **Table / figure target**：Table 3“LOFO and sealed detection”；Figure 3“calibration and shortcut audit”。
- **Priority**：MUST-RUN。

### Block 4：自然教师步骤上的主下游对照

- **Claim tested**：预测可靠性权重能优于等权 SIM-CoT，并保留干净任务性能。
- **Why this block exists**：检测 AUC 不等于蒸馏价值，必须以答案性能验证。
- **Dataset / split / task**：同一官方 GSM8K-Aug 教师训练步骤；四个官方干净测试集。
- **Compared systems**：Coconut、标准等权 SIM-CoT、Validity×Utility 离线加权。
- **Decisive metrics**：四任务 accuracy 宏平均；GSM8K-Aug accuracy；多种子差值与 bootstrap CI。
- **Secondary metrics**：干净步骤 NLL、训练稳定性、样本权重分布、每种错误类型的平均权重。
- **Setup details**：同初始化、样本顺序、更新数、optimizer、schedule、有效 batch、截断和种子；答案 loss 永不加权；`w=K(0.1+0.9r)/Σ(0.1+0.9r)`。
- **Success criterion**：单种子筛选达到宏平均 +1.0 pp、GSM8K-Aug 不下降超过 1.0 pp、干净步骤 NLL 改善；再以 3 seeds 确认。
- **Failure interpretation**：AUC 高但 accuracy 无改善表示可检测信息不影响最终任务；NLL 改善但 accuracy 不变只能主张步骤拟合；训练不稳定需检查权重分布而非改门槛追结果。
- **Table / figure target**：Table 4“Main distillation result”。
- **Priority**：MUST-RUN。

### Block 5：因果曲线、消融与失败分析

- **Claim tested**：收益确实来自 Validity/Utility 可靠性，而非更多参数、较小梯度或单一污染模板。
- **Why this block exists**：直接回应“已有模块组合、学术价值低”的质疑。
- **Dataset / split / task**：自然审计集；5%、10%、20% 受控污染训练集；四个干净测试集。
- **Compared systems**：clean、noisy equal、Validity-only、Utility-only、Validity×Utility、RSR/RD、oracle；文本-only 简单检测器作为必要性对照。
- **Decisive metrics**：各污染率 accuracy 损失与恢复比例；宏平均差值；oracle gap。
- **Secondary metrics**：各族 AUC、权重分布、典型 false positive/negative。
- **Setup details**：主方法结构和阈值冻结后运行；受控污染与自然结果分表报告。
- **Success criterion**：预测加权在至少 10% 与 20% 污染下稳定优于 noisy equal，并恢复 oracle 可恢复损失的实质比例；Validity×Utility 优于单头或揭示明确互补边界。
- **Failure interpretation**：只在见过模板上有效说明泛化不足；单头等价于双头说明复杂性不必要，应简化论文主方法；文本-only 持平说明隐空间组件没有被证明必要。
- **Table / figure target**：Figure 4“corruption response curve”；Table 5“ablation and necessity”。
- **Priority**：MUST-RUN 的核心消融；额外宽度/权重下限敏感性为 NICE-TO-HAVE。

## Implementation Work Packages

新实现与现有 `src/rsr_rd_simcot/` 隔离，旧 PoC 只作为 RSR/RD baseline 参考，不在第一轮重构。

### WP0：环境、官方 revision 与配置冻结

**Add**：

- `configs/reliable_simcot/a0_checkpoint_eval.json`
- `configs/reliable_simcot/a0_single_gpu_smoke.json`
- `src/reliable_simcot/provenance.py`
- `tests/reliable_simcot/test_provenance.py`

**Actions**：

1. 将官方仓库检出到 `work/vendor/SIM-CoT/`，记录 Git commit，不提交第三方源码；
2. 下载模型、tokenizer、数据和 checkpoint 到 `work/cache/`，记录 revision 与 SHA-256；
3. 保存 Python、PyTorch、CUDA、driver 和 GPU 清单；
4. 写配置 schema，拒绝缺少 seed、revision 或输出目录的运行。

**Verification**：`pytest tests/reliable_simcot/test_provenance.py -q`。

### WP1：官方数据与评测适配器

**Add**：

- `src/reliable_simcot/official_adapter.py`
- `src/reliable_simcot/answer_normalization.py`
- `scripts/reproduce_official_simcot.py`
- `tests/reliable_simcot/test_official_adapter.py`
- `tests/reliable_simcot/test_answer_normalization.py`

**Actions**：

1. 读取官方 `question/steps/answer`，保持原始 ID 与顺序；
2. 为四数据集实现与官方一致的答案提取；
3. 加入样本数量、重复 ID、空步骤、截断和 hash 检查；
4. 评测官方 checkpoint；
5. 运行 20-step 训练、保存、重载与继续训练预检。

**Verification**：单元测试通过；`python scripts/reproduce_official_simcot.py --config configs/reliable_simcot/a0_checkpoint_eval.json` 生成逐任务指标和误差表。

### WP2：自然审计与无泄漏切分

**Add**：

- `src/reliable_simcot/splits.py`
- `src/reliable_simcot/audit.py`
- `scripts/audit_teacher_steps.py`
- `tests/reliable_simcot/test_splits.py`
- `tests/reliable_simcot/test_audit_rules.py`

**Actions**：

1. 按题目 ID 固定抽取至少 2,000 步流行率集；
2. 实现表达式、结果、依赖和最终答案支持检查；
3. 输出待人工复核 CSV/JSONL，但不自动猜测 Utility；
4. 合并复核标签并计算置信区间；
5. 生成题目匹配的自然检测审计集并从蒸馏 train IDs 排除。

**Verification**：split hash 稳定；同题不跨 split；封存集在 freeze flag 前不可由训练命令读取。

### WP3：多污染族与可靠性特征

**Add**：

- `src/reliable_simcot/corruptions.py`
- `src/reliable_simcot/labels.py`
- `src/reliable_simcot/features.py`
- `scripts/build_reliability_dataset.py`
- `scripts/cache_reliability_features.py`
- `tests/reliable_simcot/test_corruptions.py`
- `tests/reliable_simcot/test_features.py`

**Actions**：

1. 实现五个开发污染族、等价正样本和永久封存补偿性错误族；
2. 先切题目再生成变体；
3. 强制记录 `question_id/step_index/family/template_id/y_valid/y_utility`；
4. 提取 `z_{t-1},z_t` 与 teacher-forced token hidden states；
5. mask-aware mean pooling，并以输入/模型/config hash 命名缓存。

**Verification**：padding 不参与平均；每对样本同题同位置；无效样本的 Utility 为 null 而不是 0；重复生成缓存 hash 一致。

### WP4：可靠性头、LOFO 与一次性封存评测

**Add**：

- `src/reliable_simcot/reliability_head.py`
- `src/reliable_simcot/head_training.py`
- `src/reliable_simcot/calibration.py`
- `src/reliable_simcot/detection_metrics.py`
- `scripts/train_reliability_head.py`
- `scripts/evaluate_reliability_head.py`
- `tests/reliable_simcot/test_reliability_head.py`
- `tests/reliable_simcot/test_masked_losses.py`
- `tests/reliable_simcot/test_sealed_access.py`

**Actions**：

1. 先写 masked Utility loss、ranking loss 和冻结参数测试；
2. 实现双投影、交互 MLP 与双头；
3. 运行五轮 LOFO 并汇总 macro/worst-family；
4. 冻结结构与超参后，用五族训练最终头并在未重采样 validation 上校准；
5. 生成 freeze manifest 后，分别只打开一次自然审计和补偿性错误集。

**Verification**：只有 head 参数有梯度；所有预注册指标均从逐样本预测重算；门槛失败时下游命令返回非零并说明失败项。

### WP5：离线赋权与公平下游训练

**Add**：

- `src/reliable_simcot/weighting.py`
- `src/reliable_simcot/distillation.py`
- `scripts/score_teacher_steps.py`
- `scripts/train_reliable_simcot.py`
- `tests/reliable_simcot/test_weighting.py`
- `tests/reliable_simcot/test_branch_parity.py`

**Actions**：

1. 对排除审计题目的官方 train steps 离线生成 `v/u/r/w`；
2. 校验 `w` 有限、范围受 floor 约束、每题均值为 1；
3. 从同一初始化 checkpoint 和 sampler state 启动 equal/weighted；
4. 答案 loss 始终等权，仅 step loss 使用 `w`；
5. 同一 OOM 回退、相同完成更新数、完整保存 optimizer/scheduler/RNG 状态。

**Verification**：全 1 权重严格复现 equal loss；逐字段 parity 测试只允许权重路径不同；20-step 两分支均低于 7.4 GB。

### WP6：评测、统计与论文表格

**Add**：

- `src/reliable_simcot/statistics.py`
- `src/reliable_simcot/reporting.py`
- `scripts/evaluate_reliable_simcot.py`
- `scripts/build_paper_tables.py`
- `tests/reliable_simcot/test_statistics.py`

**Actions**：

1. 复用同一答案规范化器评测四任务；
2. 计算宏平均、逐任务差值、3-seed 汇总和 bootstrap CI；
3. 输出自然结果与受控污染结果的独立表格；
4. 自动生成预注册门槛判定，禁止手写成功标签；
5. 保存逐样本预测，支持复算所有图表。

**Verification**：固定 toy 数据的统计单测；表中数值与 JSON 逐项一致；结论由 gate 文件自动产生。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Main Risk |
| --- | --- | --- | --- | --- | --- |
| M0 首晚 | 官方数值复现与单卡工程预检 | R001–R005 | 四任务各 ±1.0 pp；20-step、重载、≤7.4 GB 全通过 | 约 4–10 GPU-h | 官方依赖或 checkpoint 与 Windows/8GB 不兼容 |
| M1 基线 | 本地短预算 Coconut 与标准 SIM-CoT | R010–R012 | SIM-CoT 宏平均 ≥ Coconut +2.0 pp | 约 16–30 GPU-h，先以实测吞吐重估 | 短预算不能呈现官方方向 |
| M2 审计 | 确定自然噪声比例与主张边界 | R020–R023 | ≥2,000 步无偏审计完成；标签一致性可接受 | 约 2–5 GPU-h + 12–25 人时 | 自然正例过少、Utility 分歧大 |
| M3 检测 | 证明整类与自然泛化 | R030–R039 | G3-S 合成门槛决定能否做受控实验；G3-N 自然门槛决定能否做自然主张 | 约 8–20 GPU-h | 模板捷径、自然域差异 |
| M4 单种子主结果 | 等权与预测加权首次公平对照 | R040–R044 | 宏平均 +1.0 pp、GSM8K 守门、NLL 改善 | 每分支约一晚；由 R005 吞吐重估 | 效应小于单种子方差 |
| M5 确认 | 3 seeds、因果曲线、核心消融 | R050–R069 | 置信区间与消融支持 C2/anti-claim | 约 50–120 GPU-h，分阶段停止 | 单卡总时长较大 |
| M6 论文化 | 图表、失败样例、复现包 | R070–R074 | 所有表可由原始 JSON 重建 | 约 1–3 GPU-h | 选择性报告 |

## Stop/Go Rules

1. M0 未过：只修复官方复现，不生成自然噪声或新方法结论。
2. M1 未过：允许继续做“检测可行性”研究，但不开展或解释下游加权主结果。
3. M2 自然低可靠性 <1%：主下游改为受控污染；官方自然结果仅作小样本观察。
4. G3-S（合成泛化门槛）要求 LOFO 宏平均/最差族、补偿性错误和等价改写全部通过；失败则停止所有加权蒸馏。
5. G3-N（自然主张门槛）额外要求自然教师审计复合 AUC ≥0.75 且样本充足；失败或样本不足时禁止 M4 的自然主张，但 G3-S 通过后仍可运行受控污染 R060–R062。
6. M4 单种子未达到预注册标准：停止三种子大规模确认，先报告负结果和错误分析。
7. 任一比较分支发生不同 OOM 回退或不同更新数：该比较作废并重跑。

## Compute and Data Budget

- **首晚承诺**：M0，约 4–10 GPU 小时；若下载占用过久，至少完成环境/数据/checkpoint hash 和最大样本 20-step preflight。
- **检测可行性阶段**：特征抽取约 6–16 GPU 小时，小头五轮 LOFO 约 2–4 GPU 小时；缓存后不重复跑大模型。
- **下游单种子**：equal 与 weighted 各自最多一晚，weighted 完成更新数必须追随 equal；精确小时数由 R005 的 examples/s 推导。
- **三种子确认**：仅在单种子成功后启动，预计再需 4–8 个夜间窗口。
- **磁盘预算**：在 M0 实测后封顶；优先缓存 pooled features，不长期保存所有层全部 token states。
- **人工预算**：至少 2,000 步流行率审计；先自动规则分流，再对 Utility 和不确定样本人工复核。建议 10% 重叠双标以估计一致性。
- **最大瓶颈**：不是小型可靠性头，而是官方模型特征抽取、下游多分支训练和人工 Utility 标注。

## Risks and Mitigations

- **自然噪声不足**：按预注册阈值降级主张，使用受控曲线提供因果证据，不伪装成自然结果。
- **可靠性头记住表面模板**：整类留出、题目级切分、长度/数字/位置匹配、等价正样本和封存补偿性错误共同约束。
- **平衡训练破坏概率校准**：只在未重采样 validation 上校准并报告 Brier/ECE；权重在题内再次归一化。
- **隐空间组件没有必要**：加入文本-only 强基线；若持平则简化方法和论文主张。
- **双头没有必要**：Validity-only/Utility-only 为确认性消融；若双头无增益，不保留装饰性复杂度。
- **8GB OOM**：20-step 最大样本预检；统一采用 batch 1、梯度累积、mixed precision、checkpointing；比较分支共享回退规则。
- **单种子偶然性**：筛选门槛后才花费 3 seeds；报告差值分布和 CI。
- **已有 PoC 改动被破坏**：新包与新脚本独立；不覆盖 `src/rsr_rd_simcot/` 和 `run_experiment.py`。
- **封存泄漏**：audit ID 单独加密/只读并由 freeze manifest 控制；任何解封后的方法改动都会触发新封存集。

## First Three Runs to Launch

1. **R001**：固定官方仓库、数据、模型与 checkpoint revision，生成 provenance manifest。
2. **R002**：评测官方 checkpoint 的 GSM8K-Aug，先验证答案提取和核心数值。
3. **R004**：在最长样本上运行 20-step 单卡训练 + checkpoint 重载预检，确认 7.4 GB 显存门槛。

R001–R004 通过后再补齐 R003 的其余三任务评测；这样首晚优先暴露实现和显存问题。

## Final Checklist

- [x] 主张不超过两个，且每个实验块均关联主张
- [x] Main paper、Appendix 与 Cut 已区分
- [x] 标准 SIM-CoT 是主要下游对照，Coconut 是次要基线
- [x] 自然噪声与受控噪声结论边界已固定
- [x] 检测门槛独立于下游训练
- [x] Validity、Utility、双头与简单 baseline 能隔离贡献
- [x] 相同预算和权重均值 1 排除总梯度混淆
- [x] 首晚、单种子、三种子运行顺序与停止规则明确
- [x] Nice-to-have 不阻塞 must-run
- [x] 8GB 显存与人工审计瓶颈已计入
