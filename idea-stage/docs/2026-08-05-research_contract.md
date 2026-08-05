# Research Contract：可靠性门控 SIM-CoT

- **状态**：ACTIVE
- **日期**：2026-08-05
- **冻结规格**：`docs/superpowers/specs/2026-08-05-reliability-gated-simcot-natural-noise-design.md`
- **执行计划**：`refine-logs/EXPERIMENT_PLAN.md`

## Problem Anchor

官方教师生成的 SIM-CoT 训练步骤可能包含逻辑错误、计算错误、无关步骤或冗余步骤。标准 SIM-CoT 对这些步骤等权监督，可能使 Student 学习低可靠性推理。

## Primary Claims

1. 冻结的 Validity/Utility 可靠性头能够跨题目、跨污染族识别低可靠性步骤，并在自然教师审计上通过预注册门槛。
2. 在相同数据、初始化、更新数和优化设置下，离线可靠性加权能优于标准等权 SIM-CoT，同时不明显损害干净 GSM8K-Aug 表现。

## Anti-Claims to Rule Out

- 增益仅来自新增参数；
- 增益仅来自总体步骤 loss 或梯度变小；
- 检测器只记住合成污染模板；
- 隐空间组件只是装饰，文本简单基线同样有效；
- 受控污染结果被误写成官方自然噪声结果。

## Frozen Evidence Gates

- A0：官方 checkpoint 四任务准确率分别在官方报告 ±1.0 pp；单卡 20-step、保存重载、latent 对齐与 reserved 显存 ≤7.4 GB 通过。
- A1：同预算本地标准 SIM-CoT 的四任务宏平均准确率至少比 Coconut 高 2.0 pp。
- G3-S：五轮 LOFO 复合 ROC-AUC 宏平均 ≥0.75、最差族 ≥0.70；补偿性错误 Validity AUC ≥0.70；等价改写下降 ≤0.10。
- G3-N：G3-S 通过，且自然教师审计复合 ROC-AUC ≥0.75、样本充足。
- M4：单种子宏平均准确率相对等权 SIM-CoT ≥+1.0 pp，GSM8K-Aug 下降不超过 1.0 pp，干净步骤 NLL 改善；通过后才做 3 seeds。

G3-S 只允许受控污染主张；G3-N 才允许官方自然教师噪声主张。

## Data and Evaluation Contract

- 教师训练步骤来自官方 SIM-CoT/GSM8K-Aug；
- 最终测试集保持官方干净测试，不人工加噪；
- 评测 ground truth 必须来自数据集标签，禁止用其他模型输出充当标签；
- 所有派生变体先按题目切分，禁止同题跨 split；
- 自然流行率审计、自然检测审计和补偿性错误压力集均按冻结规则隔离；
- 自然结果与受控污染结果分表、分主张报告。

## Fairness Contract

- equal 与 weighted 使用相同初始化 checkpoint、样本顺序、完成更新数、optimizer、scheduler、有效 batch、截断、种子和 OOM 回退；
- 答案 loss 永不加权；
- 步骤权重设置 0.1 floor，并在每题内部归一化为均值 1；
- 下游训练时 scorer 完全冻结且只使用离线分数。

## Compute Contract

- 硬件：RTX 4060 Laptop，8188 MiB；
- reserved 显存红线：7.4 GB；
- sanity first：M0 通过前不启动大训练；
- 首晚只承诺 R001–R005；
- 单种子主结果通过前不启动多种子确认。

## Change Control

任何检测架构、损失、门槛、数据切分或权重公式的修改都必须：

1. 在打开封存集前完成；或
2. 若封存集已打开，将旧封存结果降级为开发结果，并建立新的封存集。
