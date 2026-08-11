# SIM-CoT 辅助步骤梯度杠杆夜间实验

**日期：** 2026-08-11

**状态：** 已完成；预注册 verdict 为 `GRADIENT_PATHWAY_PRESENT_BUT_ANSWER_ROBUST`

**结果报告：** `outputs/reliable_simcot/gradient_leverage/gradient_leverage_report_2026-08-12.md`

**硬件：** NVIDIA RTX 4060 Laptop GPU，8 GB 显存

## 1. 背景与目标

此前的 20%/40% 孤立噪声和 25%/50%/75% 因果传播噪声均未在答案 EM 上造成稳定伤害，但因果噪声使干净 step-token NLL 最多恶化 71.76%。因此，本实验不继续训练可靠性头，而先回答更基础的问题：辅助步骤损失的梯度是否真正具有改变最终答案通路的因果杠杆。

本实验回答三个相互区分的问题：

1. **梯度通路：** 干净或错误辅助步骤损失是否向基础语言模型各层提供非平凡梯度？
2. **剂量响应：** 将辅助损失系数 `lambda_aux` 从 1 放大到 3 后，因果错误步骤是否会稳定损害答案 EM？
3. **自我纠错：** 若 25% 因果噪声再次提高答案 EM，它是否同时增强模型对干净推理链而非错误链的概率偏好？

## 2. 数据冻结

训练继续使用已经冻结、已经审计的 `causal-pilot-train` 512 题和相同 25% 覆盖率：128 题包含一条三步因果错误链，总步骤污染率 15%。这样可以直接复核此前 25% 组的探索性提升。

确认评估使用原 `causal-formal-train` 中按既有冻结顺序取出的前 1,024 题。该分区此前没有参与训练、开发集校准或任何结果选择；评估始终使用官方原始干净步骤和答案字段。确认集一经写入 manifest 即不可更换。

官方测试集继续封存，不读取、不哈希、不评估。原 512 题 causal-dev 不再作为本轮主判定集。

## 3. 逐层梯度审计

从 25% 活跃污染层中固定抽取 18 题，每个“错误族 × 枢轴位置”单元恰好 2 题。在官方 checkpoint 28、`eval()` 模式下，对 GPT-2 的 12 个 transformer blocks 分别计算：

- `g_answer`：干净答案损失梯度；
- `g_clean_aux`：干净五步辅助损失梯度；
- `g_noisy_aux`：同题三步因果污染辅助损失梯度。

每层保存：

- `||g_clean_aux|| / ||g_answer||`；
- `||g_noisy_aux|| / ||g_answer||`；
- `cos(g_clean_aux, g_answer)`；
- `cos(g_noisy_aux, g_answer)`；
- `cos(g_clean_aux, g_noisy_aux)`；
- 错误梯度相对干净梯度在答案方向上的投影变化。

梯度通路判据：12 层中至少 6 层的中位 `||g_clean_aux|| / ||g_answer|| >= 0.1`，且至少 6 层的中位 `cos(g_clean_aux, g_noisy_aux) <= 0.95`。该门只证明梯度可达且噪声改变方向，不等同于下游伤害。

## 4. 训练臂与种子

所有组从同一官方 checkpoint 28 初始化，使用相同 512 题顺序、64 updates、梯度累积 8、BF16、学习率 `1e-4`、weight decay `0.01`、梯度裁剪 `1.0`。随机种子固定为 `20260809`、`20260810`、`20260811`。

| 训练臂 | 辅助目标 | `lambda_aux` | 作用 |
|---|---|---:|---|
| `answer_only` | 干净步骤但辅助梯度乘 0 | 0 | 测量答案损失单独训练 |
| `clean_aux1` | 干净步骤 | 1 | 标准干净 SIM-CoT 对照 |
| `causal_aux1` | 25% 三步因果污染 | 1 | 复核低噪声点估计 |
| `clean_aux3` | 干净步骤 | 3 | 控制辅助梯度整体放大 |
| `causal_aux3` | 25% 三步因果污染 | 3 | 检查错误梯度剂量响应 |

答案损失在全部组中保持官方干净监督。`lambda_aux=0` 仍执行相同前向计算，以保持训练代码路径一致。

## 5. 确认评估

每个 checkpoint 在同一 1,024 题确认集上报告：

- 官方答案 exact match；
- answer NLL；
- 干净 step-token NLL；
- 前 256 个确认样本上的错误链 step-token NLL；
- `corrupt_NLL - clean_NLL` 干净链偏好边际，正值表示更偏好干净链。

所有 EM ground truth 必须直接来自确认集官方答案字段。不得使用基线模型或其他模型输出作标签。

## 6. 预注册判据

### G-L：梯度通路

满足第 3 节的逐层梯度判据即记为 `GRADIENT_PATHWAY_PRESENT`，否则记为 `GRADIENT_PATHWAY_WEAK`。

### G-D：错误步骤具有答案杠杆

必须同时满足：

1. `mean(EM_clean_aux3 - EM_causal_aux3) >= 2.0 pp`；
2. 至少 2/3 个种子上 `EM_causal_aux3 < EM_clean_aux3`；
3. `lambda=3` 的平均伤害比 `lambda=1` 至少大 1.0 pp。

通过时才允许称“错误步骤监督在放大后具有可观测答案杠杆”。

### G-S：低噪声自我纠错

必须同时满足：

1. `mean(EM_causal_aux1 - EM_clean_aux1) >= 1.0 pp`；
2. 至少 2/3 个种子方向为正；
3. 至少 2/3 个种子中，`causal_aux1` 的干净链偏好边际不低于 `clean_aux1`；
4. 至少 2/3 个种子中，`causal_aux1` 的干净 step-token NLL 不高于 `clean_aux1`。

若只有 EM 点估计提高，而干净链偏好或干净步骤 NLL 恶化，则结论固定为 `REGULARIZATION_OR_ROBUSTNESS_NOT_SELF_CORRECTION`。

## 7. 停止与解释规则

- G-L 失败：辅助梯度通路太弱，当前架构不适合继续做步骤加权。
- G-L 通过但 G-D 失败：梯度可达但最终答案在该预算内鲁棒/解耦，步骤加权仍属低优先级。
- G-D 通过：可以另立规格恢复 oracle 加权实验，但不能用本轮确认集调权重。
- G-S 通过：只允许称“受控低噪声下存在多种子自我纠错证据”；不外推到自然教师噪声。
- 任一任务超过同类 pilot 实测时长的两倍或 reserved 显存超过 7.4 GB 时停止该任务并保留诊断。

## 8. 夜间预算

- 数据/代码/sanity：约 0.5 小时；
- 逐层梯度审计：约 0.2–0.5 小时；
- 15 个 64-update 训练任务：约 2.3 小时；
- 15 个 1,024 题确认评估：约 1.5–2 小时；
- 汇总与报告：约 0.2 小时。

总预算约 4–5 小时，符合单张 RTX 4060 8 GB 一晚运行约束。
