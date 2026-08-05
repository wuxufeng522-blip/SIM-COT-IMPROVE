# 可靠性门控 SIM-CoT：夜间进度报告

日期：2026-08-06

硬件：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB

## 结论先行

目前已经证明：官方 SIM-CoT checkpoint 的四任务评测管线可信；自然教师步骤可以按题目无泄漏抽样、冻结、自动分流并制作为双人盲评包；未来 equal/weighted 分支的样本顺序、哈希和审计题排除也可以工程化实现。

目前还没有证明：可靠性头能达到 ROC-AUC 门槛，或可靠性加权能提高答案正确率。原因不是得到负效果，而是两个前置门槛尚未满足：官方没有发布训练配置引用的 Coconut checkpoint-24；修正训练文本解析后的 20-step GPU 预检两次被系统无 traceback 终止。R010/R011 因此没有启动，避免产生不可归因结果。

## 已完成结果

| 模块 | 结果 | 判定 |
| --- | --- | --- |
| 官方 checkpoint：GSM8K-Aug | 44.43%，论文 44.8% | PASS |
| 官方 checkpoint：GSM-Hard | 9.48%，论文 9.3% | PASS |
| 官方 checkpoint：MultiArith | 89.83%，论文 90.8% | PASS，位于 ±1 pp 下界内 |
| 官方 checkpoint：SVAMP | 40.60%，论文 40.7% | PASS |
| 第三方 Coconut checkpoint-24 初始化审计 | 33.13 / 6.90 / 79.33 / 37.10 | FAIL，3/4 偏离论文 Coconut 行 |
| R020 自然审计抽样 | 800 个唯一问题簇、2,110 个真实步骤 | PASS |
| R021 自动结构分流 | 2,102 算术吻合；5 个不匹配候选；2 个需手查；1 个空步骤 | PASS，仅作分流 |
| R022 盲评准备 | 211 行双标重叠；A 1,156 行，B 1,165 行 | PREP PASS，等待人工标签 |
| 单元/契约测试 | 33 passed | PASS |

## 关键勘误：训练步骤解析

训练文本以 `####` 分隔答案。旧适配器用最后一个 `##` 切分，导致每条轨迹多出一个伪 `##` 步骤；官方 `preprocessing/gsm_icot.py` 使用第一个推理字段和最后一个答案字段。该问题不影响 R002/R003 的 checkpoint 答案评测：GSM8K 推理时只使用题目与最终答案，OOD 三项直接读取公开 JSON；但会影响 R004/R005 的辅助步骤监督语义。

因此旧 R004/R005 已降级为“显存与保存路径曾运行”的工程证据，不再称为精确官方训练格式复现。修正版 R004-v2 第一次完成 15/20 次有限更新，loss 从 2.487672 降至 0.002481 后进程无 traceback 退出；第二次在首个更新前退出。按两次调试上限停止，R005-v2 未启动。

## M1：同预算短训练准备情况

- R010 Coconut 与 R011 SIM-CoT 的训练器、BF16、梯度累积、optimizer/RNG 断点恢复、验证 NLL 与公平性字段已实现。
- 共同 v2 schedule 冻结为 8,192 个微批次，SHA-256：`4575fb7e942307ff6066aec6de248fb0479e90dc1915c5382c59bfca656cae05`。
- schedule 强制读取 R020 freeze manifest，并排除了随机候选中命中的 18 个审计题；任何审计题泄漏都会直接报错。
- R010/R011 未启动。官方模型仓库仅提供 checkpoint-28，而训练配置要求 Coconut checkpoint-24。候选第三方 checkpoint-24 已固定 revision 与 SHA-256，但四任务初始化审计失败，不能冒充论文权重。

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

## 当前证据对实验思想的含义

1. **方案可实现，但效果未证实。** 数据冻结、无泄漏、双标签、LOFO 前置条件和公平训练 schedule 都已经具备可执行实现。
2. **自然噪声并非凭空假设。** 随机样本中已看到空步骤与明显算术错误，但当前只能称为案例，不能据此估计总体比例。
3. **不能跳过人工语义审计。** 取整、单位换算、重复步骤和“算术正确但用错题目数值”都说明纯规则或相似度不足以定义可靠性。
4. **不能把第三方 checkpoint 当官方初始化。** 否则后续 SIM-CoT 优势或加权收益会被初始化差异混淆。

## 下一决策点

1. 两位评审独立填写 `reviewer_a_labeled.csv` 与 `reviewer_b_labeled.csv`，再运行 `scripts/compile_human_review.py`。
2. 若自然低可靠性比例 `<1%`，按预注册规则把论文主张降级为受控污染鲁棒性；若为 `1%–5%`，自然实验与 5/10/20% 受控曲线并行；若 `≥5%`，保留自然主实验。
3. 只有得到可验证的 checkpoint-24，或重新完成修正版 Coconut/SIM-CoT 共同初始化训练，并通过 20-step/reload 后，才恢复 R010/R011。
4. 可靠性头必须满足 LOFO 宏 AUC ≥0.75、最差族 ≥0.70、自然审计 AUC ≥0.75 等全部门槛，才允许进入加权蒸馏。

## 主要产物

- `outputs/reliable_simcot/r020_natural_audit/freeze_manifest.json`
- `outputs/reliable_simcot/r020_natural_audit/audit_rows.jsonl`
- `outputs/reliable_simcot/r021_auto_triage/triage_manifest.json`
- `outputs/reliable_simcot/r022_blinded_review_v2/reviewer_a.csv`
- `outputs/reliable_simcot/r022_blinded_review_v2/reviewer_b.csv`
- `outputs/reliable_simcot/r022_blinded_review_guide.md`
- `outputs/reliable_simcot/r004_v2/failure.json`
- `outputs/reliable_simcot/r005_v2/blocked.json`

本报告不把自动候选数写成自然噪声率，不把失败的 checkpoint 来源写成官方权重，也不把尚未运行的加权训练写成方法有效。
