# SIM-COT-IMPROVE：实验路线、结果与代码索引

> 更新日期：2026-09-03
>
> 当前分支：`codex/rsr-rd-poc-design`
>
> 本文性质：内部实验总览和可追溯索引，不是独立第三方完整性审计。
## 1. 项目在研究什么

本项目当前聚焦一个核心问题：

> **当 Teacher 在复杂问题的困难推理步骤上可能产生错误监督时，能否利用 Student 当前隐状态中的 confidence 判断其不确定程度，并在 confidence 低于阈值时降低对应 Teacher step 的监督权重，从而减少错误显式推理对 latent reasoning 的干扰，优先提高最终答案准确率？**

当前方法主线只保留两个核心信号：

- **Surprisal**：用于从正确 Teacher trajectory 中定位 Student 更困难的推理步骤，并在这些位置构造可控噪声；
- **Hidden-state Confidence**：用于训练时判断当前步骤是否需要降低 Teacher supervision。

当前路线按 P0–P4 推进：

### P0：验证错误步骤监督是否有害 —— 已完成

通过受控含噪步骤实验确认：当错误步骤具有足够大的语义冲突、覆盖范围和训练强度时，SIM-CoT 的步骤级辅助监督可以通过共享隐状态影响最终答案表现。P0 的作用是建立后续“为什么需要选择性降低错误 Teacher supervision”的实验动机。

### P1：构造面向困难步骤的含噪训练数据

参考 RSR 中 surprisal 对 Student 不熟悉程度 / 步骤难度的刻画，不再随机选择任意步骤污染，而是优先选择高-surprisal reasoning steps，在这些困难位置构造可审计的错误 Teacher steps，同时保持最终答案标签正确，并尽量控制步骤位置、长度和训练预算。

### P2：从 Student 隐状态中提取 Step Confidence

参考 **Efficient Reasoning with Balanced Thinking (ReBalance)** 的 stepwise confidence 与 hidden-state confidence 表征，直接从 Student 当前 reasoning step 的隐藏状态中读取 confidence，并将其作为内部不确定性信号。

### P3：基于 Confidence 阈值进行选择性降权

设第 `s` 个步骤的 confidence 为 `c_s`，阈值为 `τ`。训练时采用最直接的阈值门控：

$$
w_s=
\begin{cases}
1, & c_s\ge \tau,\\
\beta, & c_s<\tau,
\end{cases}
\qquad 0<\beta<1.
$$

即：

```text
confidence >= τ
        ↓
保持原始 SIM-CoT step supervision
        ↓
w_s = 1

confidence < τ
        ↓
当前步骤不确定性较高
        ↓
降低对应 Teacher step supervision
        ↓
w_s = β < 1
```

最终答案损失始终保持正常监督，只调整步骤级辅助损失：

$$
\mathcal{L}
=
\mathcal{L}_{\text{answer}}
+
\lambda\sum_s w_s\mathcal{L}_{\text{step},s}.
$$

### P4：验证是否提高困难问题准确率并缓解 underthinking

P4 以最终答案准确率为第一指标，比较等权含噪 SIM-CoT 与 confidence-threshold weighting。重点观察性能提升是否集中在高-surprisal、低-confidence 的困难步骤和困难样本上。

当前对 underthinking 的关注不是“生成 token 是否更长”，而是：当 Student 在困难区域不确定时，减少错误 Teacher step 对 latent state 的强制牵引，是否能避免过早锁定错误推理方向，并保留 Student 隐空间继续完成推理的能力。

**当前暂不处理 test-time scaling、动态推理预算、动态停止与 overthinking。准确率是当前最优先指标。**

截至目前，工程路径已经跑通；P0 已完成，当前重点转向 P1–P4。历史实验结果及其证据边界在后文保持原样记录。

## 2. 必须先区分的三种模型起点

SIM-CoT 是一个可附加到隐式 CoT 方法上的训练模块。论文同时在 Coconut 和 CODI 上测试它，并非只有一个统一底座。

| 名称 | 含义 | 本项目中的用途 |
| --- | --- | --- |
| 纯 Coconut | 主要依靠最终答案监督和 latent curriculum，没有 SIM-CoT 步骤解码器训练 | v18–v22、v24 的起点 |
| Coconut + SIM-CoT | 在 Coconut 隐式推理上加入辅助解码器和步骤级损失 | 官方发布 `internlm/SIM_COT-GPT2-Coconut/checkpoint_28`；多数 v10–v17、v23 实验的起点 |
| CODI + SIM-CoT | CODI 教师—学生轨迹蒸馏再加入 SIM-CoT 步骤监督 | 论文中存在，但本仓库目前**没有完成该路线训练** |

论文与官方代码：
- [SIM-CoT 论文](https://arxiv.org/pdf/2509.20317)
- [SIM-CoT 官方仓库](https://github.com/InternLM/SIM-CoT)
### 当前 checkpoint 事实

- 官方发布并成功复现的是 `internlm/SIM_COT-GPT2-Coconut/checkpoint_28`，本地 GSM8K-Aug test 为 `44.43%`。
- 论文所需的纯 Coconut `checkpoint_24` 没有随作者仓库公开。审计过的第三方候选只有 `31.69%–34.34%`，没有复现论文 Coconut 的 `36.6%±1 pp`。
- 后续纯 Coconut 实验采用兼容的第三方 `checkpoint_33`，起始 test accuracy 为 `31.61%`。它不能冒充作者的 checkpoint-24。
## 3. 证据等级与阅读规则

| 等级 | 数据/评测 | 可以说明什么 | 不能说明什么 |
| --- | --- | --- | --- |
| A | 官方 GSM8K-Aug test 或公开 OOD test | 模型在公开测试集上的实际答案表现 | 多次反复查看后不能继续当作全新封存检验 |
| B | 从 GSM8K-Aug train 源冻结的确认集 | 机制、梯度方向和受控处理是否可能生效 | checkpoint 可能已经见过同源数据，不能当独立泛化结果 |
| C | 合成算术、人工污染、短 smoke | 工程可行性、损失实现、显存、初步机制 | 论文级准确率或自然噪声结论 |
| D | 数据门/语义门前被拒绝的候选 | 记录为什么没有训练，防止错误数据进入实验 | 任何模型效果结论 |

特别注意：`79.82%` 的 Clean 基线来自 1,024 条 GSM8K-Aug **训练源确认集**，不是官方 1,319 题 test。它与 `44.43%` 的官方 test 结果不可直接比较。v14–v17 的大幅伤害只能作为强机制压力测试证据。

## 4. 实验全景图
### 阶段 A：最初 RSR/RD 加权概念验证

**目的。** 先验证 RTX 4060 8GB 能否运行 GPT-2 Small、辅助解码器、离线步骤评分和等权/加权双分支；同时测试 RSR 与 RD 能否直接变成步骤权重。

**设计。** 1,200 条合成训练题、20% 轨迹污染、每条污染一个步骤；400 次预热更新，两个分支各 800 次更新；干净测试集 200 条。

**重要改进。** 原式 `u=-0.5z(log RSR)-0.5z(log RD)` 将部分高 surprisal 错误步骤误当成高信息步骤，AUC 只有 `0.3505`。因此探索性改为低尾保护：

```text
u = -0.5 * max(0, -z(log RSR)) - 0.5 * z(log RD)
```

**结果。** 修订后污染检测 AUC `0.7588`，污染/干净步骤平均权重 `0.6796/1.0231`；但等权/加权答案 EM 为 `4.5%/4.0%`，加权组干净步骤 NLL 反而恶化 `6.39%`。因此只证明工程可行，没有证明质量收益。
**代码与结果。** `run_experiment.py`、`src/rsr_rd_simcot/`、`outputs/overnight_poc/`。
### 阶段 B：官方 checkpoint 复现与 8GB 工程门（R001–R012）

**改进目的。** 初始 PoC 的 4% 准确率太低，必须先确认官方模型、数据、答案解析和单卡训练器是否可信。
| Run | 尝试 | 结果 | 状态 |
| --- | --- | --- | --- |
| R001 | 冻结官方源码、模型、数据 revision 与 SHA-256 | 来源清单完成 | PASS |
| R002 | 官方 checkpoint 的 GSM8K-Aug test | `44.43%`，论文 `44.8%` | PASS |
| R003 | GSM-Hard / MultiArith / SVAMP | `9.48% / 89.83% / 40.60%`，均在论文值 ±1 pp | PASS |
| R004-v3 | 最长样本 20-update SIM-CoT 训练、步骤对齐、保存重载 | 峰值 `5.52 GB`，loss 下降，重载差为 0 | PASS |
| R005-v3 | 从 checkpoint 继续训练并二次重载 | 峰值 `5.14 GB`，参数确实改变 | PASS |
| R009 | 寻找论文纯 Coconut checkpoint-24 | 第三方候选均未复现 `36.6%` | FAIL |
| R010–R012 | 公平的 Coconut vs SIM-CoT 短预算训练 | 因无共同可信 checkpoint-24 未启动 | BLOCKED |
这一阶段还发现官方 `Coconut/run.py` 导入了 explainable dataset 函数，但发布训练调用曾指向普通 latent dataset。后续本地 adapter 按论文意图显式启用了步骤监督，并保留这一实现差异记录。

主要入口：

- `scripts/record_r001_provenance.py`
- `scripts/evaluate_official_simcot.py`
- `scripts/evaluate_official_simcot_ood.py`
- `scripts/run_official_simcot_smoke.py`
- `scripts/run_official_simcot_resume_probe.py`
- `src/reliable_simcot/official_adapter.py`
### 阶段 C：自然步骤审计与学习式步骤判别尝试（R020–R039）

**改进目的。** 固定 RSR/RD 不能稳定充当步骤错误判别信号，因此曾尝试使用冻结特征和轻量 MLP 直接学习步骤质量判别。

**尝试。**

- R020：从官方训练源无偏抽取 800 个问题、2,110 个真实步骤。
- R021：只用规则进行结构分流，不把规则结果冒充人工真值。
- R022：生成双人盲评包；人工标签未完成，因此没有自然噪声率。
- R030：构造五个开发污染族和等价正例，2,000 题、33,887 行，按题目切分。
- R030-F：缓存官方 checkpoint-28 与辅助解码器的冻结特征。
- R031–R035：用双投影/交互 MLP 做学习式步骤判别，并进行整类留出（LOFO）测试。

**结果。** LOFO 宏平均复合 ROC-AUC `0.6934 < 0.75`，最差族 `0.5300 < 0.70`。无关族和冗余族能过门，但数值、运算符、依赖错误无法跨族泛化。按冻结规则，后续自然封存评测和下游加权训练全部 CUT。

**结论。** 当前 MLP 更像在识别见过的污染模板，没有学到稳定的跨污染族判别能力。不能使用 validation 高分掩盖 LOFO 失败。

主要入口：

- `scripts/audit_teacher_steps.py`
- `scripts/triage_teacher_steps.py`
- `scripts/prepare_blinded_review.py`
- `scripts/build_reliability_dataset.py`
- `scripts/cache_reliability_features.py`
- `scripts/train_reliability_head.py`
- `src/reliable_simcot/reliability_head.py`
### 阶段 D：完美标签下的 oracle 加权

**改进目的。** 将“能否检测错误”和“检测正确后降权是否有用”拆开。直接使用人工生成器知道的错误位置，避免检测器误差干扰。

| 实验 | 污染 | 关键结果 | 判定 |
| --- | --- | --- | --- |
| O001–O020 | 每题 1/5 步错误，2,048 条，单种子 | Clean `40.56%`；等权噪声 `39.80%`；raw-0.1 `41.09%`；mean-one 0.1 `41.77%` | 等权噪声仅伤害 `0.76 pp`，未达到预注册损伤门；oracle 恢复结论不充分 |
| O101–O120 | 每题 2/5 步错误，即 40% | Clean `40.56%`；等权噪声 `41.55%`；raw-0.1 `41.02%`；mean-one 0.1 `40.49%` | 噪声组没有先受伤，两种降权均未优于等权组 |

**结论。** `0.1` 不是已被证明正确的权重。只有先建立稳定伤害，才能解释“恢复比例”。
主要入口：`scripts/run_oracle_weighting.py`、`scripts/evaluate_oracle_weighting.py`、`src/reliable_simcot/oracle_weighting.py`。
### 阶段 E：因果传播错误与梯度杠杆

#### E1. 因果传播错误校准

**改进目的。** 早期局部数值替换可能只改变少量 token，因此构造一个直接错误并沿后续依赖传播两个错误后代，确保整条推理链真的与正确逻辑冲突。

在 512 条训练题、64 updates 下，25%/50%/75% 题目覆盖均未使 512 条开发集 EM 下降至少 2 pp；但干净步骤 NLL 随污染覆盖率单调恶化 `25.73% / 43.31% / 71.76%`。因此正式 oracle 恢复阶段没有启动。
#### E2. 梯度杠杆实验

**改进目的。** 验证错误步骤损失是否真的能穿过辅助解码器进入基础模型，而不是被隔离在辅助头里。

在新的 1,024 条训练源确认集、3 个种子上：

| 训练条件 | 平均 EM |
| --- | ---: |
| 仅答案监督 | `68.75%` |
| 正确步骤监督，lambda=1 | `74.54%` |
| 因果错误步骤监督，lambda=1 | `71.45%` |
| 正确步骤监督，lambda=3 | `75.26%` |
| 因果错误步骤监督，lambda=3 | `73.21%` |

lambda=1 时，正确步骤相对仅答案提高 `5.79 pp`；错误步骤相对正确步骤降低 `3.09 pp`，3/3 种子同方向。逐层梯度审计显示辅助损失到达 12/12 transformer block。

不过所有更新都触发 `max_grad_norm=1.0` 裁剪，lambda=3 没有形成真正三倍剂量；而且评测集来自训练源，不能外推为独立测试表现。
主要入口：`scripts/run_causal_experiment.py`、`scripts/run_gradient_leverage.py`、`src/reliable_simcot/causal_experiment.py`、`src/reliable_simcot/gradient_leverage.py`。
### 阶段 F：全链强冲突压力测试

**改进目的。** 将“只有局部 token 错误”升级为整条轨迹大面积、持续性语义冲突，并比较局部错误与全链错误。

| 训练条件 | 三种子平均 EM | 相对 Clean |
| --- | ---: | ---: |
| Clean 正确步骤 | `79.82%` | — |
| 25% 局部因果错误 | `79.49%` | `-0.33 pp` |
| 25% 全链强冲突 | `76.86%` | `-2.96 pp` |
| 50% 全链强冲突 | `74.61%` | `-5.21 pp` |

25% 没达到预注册 `5 pp` 主门；条件式 50% 达到 `5 pp`。这支持“错误监督只有在密度高、范围大、持续冲突时才稳定显现”的机制假设。

**重要边界。** 这里的确认集来自 GSM8K-Aug train 源，官方 test 未打开；强冲突由 Codex 受约束生成，不是自然教师噪声。
主要入口：`scripts/run_full_conflict.py`、`scripts/run_full_conflict_overnight.py`、`src/reliable_simcot/full_conflict_*`。
### 阶段 G：PRM800K 自然错误步骤尝试

**改进目的。** 用真实模型生成且带过程奖励标签的错误轨迹，替代人工污染；要求同题同 generation、最终答案正确、恰好五步，并形成 Clean / 1错 / 2错三联组。

**结果。** 审计 98,731 条记录后，严格三联组为 `0/512`。PRM800K 在首错后通常停止标注；找到的 46 条 Noise-1 五步轨迹最终答案全部错误。按数据门停止，没有启动训练。

**结论。** 这不是“自然噪声不存在”，只说明 PRM800K 的采集结构不支持当前严格对照。

主要入口：`scripts/run_prm800k_natural_noise.py`、`src/reliable_simcot/prm800k_data.py`。
### 阶段 H：答案最终正确、过程错误的自修复链（v1–v8）

**改进目的。** 保持最终答案正确，只改变步骤监督；构造前面严重错误、后面纠错回到正确答案的链，避免错误答案标签与错误步骤同时变化。
| 版本 | 改进或问题 | 结果 |
| --- | --- | --- |
| v1 | 初版 MATH 自修复链 | 句子可能重排，恢复步骤缺少必要中间关系；训练前拒绝 |
| v2 | 改善恢复推导 | 审计样本覆盖不全、部分恢复只有公式；训练前拒绝 |
| v3 | 修正覆盖与公式恢复 | 阶乘符号被错误切句，恢复仍不完整；训练前拒绝 |
| v4 | 修复切句和推导 | 错误步骤含明显元语言捷径；训练前取代 |
| v5 | 去除元语言并正式训练 | 第三种子显存超过 7.4GB，且状态机没有立即停止失败臂；结果废止 |
| v6 | 修复状态机和部分显存问题 | 长调度累积碎片导致 `7.5137GB`，超过冻结上限；停止 |
| v7 | 加入统一碎片缓解和完整调度显存门 | 27 臂训练完成、评估 19 臂后，因 MATH clean 地板效应按用户要求中止；无完整结论 |
| v8 | 将评测切回 GSM8K | Clean 三种子均值仅 `20.65% < 50%`；按门停止，未启动含噪臂 |
这一阶段的价值主要是暴露了数据构造、语义审计、显存碎片和跨数据集评测的风险；没有形成可用于论文的最终结果。

主要入口：`scripts/run_self_corrected_strong_conflict.py`、`scripts/run_self_corrected_v6_pipeline.py`、`src/reliable_simcot/self_corrected_*`。
### 阶段 I：GSM8K 错误抵消与严格控制变量（v9–v13）

**改进目的。** 将最终答案标签始终固定为正确，只比较 `L_step` 的正确、冗余或错误内容；同时控制污染覆盖率、步骤长度和局部/全链范围。

缩写：`C`=正确步骤；`RL/RW`=局部/全链冗余；`EL/EW`=局部/全链错误；`25/50`=处理题目覆盖率。
| 版本 | 改进 | 结果 |
| --- | --- | --- |
| v9 | 同时要求错误/冗余长度严格匹配 | 仅 `2/512` 可构造，数据门停止 |
| v10 | 放宽不必要的冗余长度约束，lr=`1e-4` | Clean `41.67%`，未过 clean 门；探索性主伤害 `+0.38 pp`，CI 跨 0 |
| v11 | 只把学习率降为 `5e-5` | Clean `42.89%` 通过门；EW50 相对匹配 RW50 的伤害约 `-0.03 pp`，即没有伤害 |
| v12 | 将错误升级为更严重依赖冲突 | 审计发现一次改了多个相同常数，不满足单边干预；训练前拒绝 |
| v13 | 改成严格单边严重错误，保留所有其他变量 | Clean `42.84%`；EW50 相对 RW50 的伤害 `-0.38 pp`，CI `[-1.31,+0.43]`；无伤害 |

这些是官方 1,319 题 test 上最严格的短预算控制结果：在 512 条、64 updates 下，没有检测到单边严重错误步骤的稳定伤害。
主要入口：`scripts/run_error_cancellation_pipeline.py`、`src/reliable_simcot/error_cancellation_*`。
### 阶段 J：顺序、冗余与“误打误撞正确”机制（v14–v17）

这组实验复用 `79.82%` 的训练源 Clean 确认集，只能作为机制诊断。

| 版本 | 处理 | 结果 |
| --- | --- | --- |
| v14 | 100% 样本的五个步骤全部倒序 | `79.82% → 57.26%`，伤害 `22.56 pp` |
| v15 | 50% 冗余、50% 严重冲突、50% 倒序 | 冗余 `-0.39 pp`；严重冲突 `-5.21 pp`；倒序 `-2.47 pp` |
| v16 | 中间严重错误，最后一步回到正确答案 | `79.82% → 74.12%`，伤害 `5.70 pp`；正确尾部没有救回 |
| v17 | 完全无关的错误链，最后仍给正确答案 | `79.82% → 74.15%`，伤害 `5.66 pp`；与同题错误链差异接近 0 |

这些结果说明步骤顺序和大范围错误语义都可能产生强梯度影响，但高基线和大伤害不能与官方 test 结果混写。
主要入口：`scripts/run_step_order_reversal_pipeline.py`、`scripts/run_high_baseline_content_order_pipeline.py`。
### 阶段 K：从纯 Coconut 开始扩大训练量（v18–v22）

**改进目的。** 避免从已经完成 SIM-CoT 训练的 checkpoint-28 开始，以免模型对短时错误监督过于稳定；同时将训练从 512 条扩大到 8k/16k/32k。

**共同限制。** 使用的是第三方兼容纯 Coconut checkpoint-33（起点 `31.61%`），不是作者未发布的 checkpoint-24；全部为单种子；仍采用本地单卡简化续训，不是论文的完整 curriculum/epoch 训练。
| 版本 | 尝试 | 官方 test 结果 |
| --- | --- | --- |
| v18 | 8,192 条跨题、长度匹配的五步语义冲突 | `31.61% → 28.43%`，表面下降 `3.18 pp` |
| v19 | 加入同规模正确步骤、冗余步骤与 0.1 权重控制 | 正确步骤 `29.04%`；错误等权 `28.43%`，仅比正确低 `0.61 pp`；两种 0.1 加权均未恢复 |
| v20 | 计划扩到 32,768 | 8k 精确参数哈希复现门失败，停止并保留记录 |
| v21 | 用户授权后从 v20 的 8k 模型/优化器继续到 32k | 8k/16k/32k 为 `26.61%/28.73%/27.52%`；属于事后探索性续跑 |
| v22 | 从同一纯 Coconut 起点独立训练 32k 官方正确步骤对照 | 8k/16k/32k 为 `27.07%/29.72%/28.28%`；分别比错误组高 `0.45/0.99/0.76 pp`，均不显著 |
**结论。** 扩大到 32k 后，错误组持续略差于正确组，但主要下降是两组共享的续训退化，无法把 `3–5 pp` 的总下降全部归因于错误步骤。

主要入口：`scripts/run_semantic_conflict_pilot.py`、`scripts/run_semantic_supervision_controls.py`、`scripts/run_semantic_conflict_scaling.py`、`src/reliable_simcot/semantic_conflict_*`。
### 阶段 L：关闭全局梯度裁剪（v23–v24）

**改进目的。** 检验 `max_grad_norm=1.0` 是否压平了错误步骤与正确步骤之间的更新差异。

| 版本 | 起点 | Clean | EW50 | `EW50-Clean` |
| --- | --- | ---: | ---: | ---: |
| v23 | 官方 Coconut + SIM-CoT checkpoint-28 | `43.24%` | `43.57%` | `+0.33 pp` |
| v24 | 第三方纯 Coconut checkpoint-33 | `31.54%` | `31.59%` | `+0.05 pp` |

两轮均为 512 条、64 updates、3 个种子，所有更新的裁剪前梯度范数都大于 1，但关闭裁剪后仍没有出现错误步骤伤害。v24 中 Clean 训练也几乎没有改变 `31.61%` 的起始模型，说明该训练剂量不足以形成灵敏测试。

主要入口：`scripts/run_no_clip_small_error_pilot.py`，配置分别为：
- `configs/reliable_simcot/error_cancellation_gsm8k_v23_no_clip_pilot.json`
- `configs/reliable_simcot/error_cancellation_gsm8k_v24_pure_coconut_no_clip.json`
## 5. 当前结论：哪些成立，哪些没有成立

### 已有较强证据

1. RTX 4060 8GB 可以运行 GPT-2 级 SIM-CoT 辅助监督、保存恢复和小规模多臂实验。
2. 官方 `SIM_COT-GPT2-Coconut/checkpoint_28` 的四任务答案评测可复现。
3. 辅助步骤损失的梯度能够进入基础模型全部 transformer block；它不是只训练一个与答案无关的孤立解码器。
4. 大范围、持续、密集的步骤冲突可以破坏训练源确认集表现；冗余文本本身的伤害远小于语义冲突。
5. 简单 RSR/RD 单调解释不可靠；首版学习式步骤判别也不能泛化到未见错误族。
### 尚未得到支持

1. 没有证据证明 `0.1` 是通用最佳错误步骤权重。
2. 没有在独立官方 test 上证明 512 条单边错误步骤会稳定造成至少 2 pp 伤害。
3. 没有证明自然教师噪声率或自然噪声的下游伤害。
4. 没有证明预测式可靠性加权能提高最终答案正确率。
5. 没有完成 CODI + SIM-CoT 的干净复现或含噪对照。
6. 没有完成论文规模（GSM8K-Aug 约 385k）的多 epoch、多种子训练。

### 当前最重要的解释

现有结果更符合下面的分层图景：

- 错误步骤监督确实有梯度通路，也可能伤害模型；
- 伤害大小依赖污染密度、错误范围、训练剂量、模型起点和评测集；
- 正确答案损失会显著缓冲少量错误步骤；
- 在短预算和官方 test 上，随机波动往往大于错误语义带来的净差异；
- 因此，在证明“自然错误步骤构成稳定且可恢复的伤害”之前，不应把此前的学习式步骤判别路线作为论文主贡献。

## 6. 下一阶段方法：Hidden-State Confidence 阈值式选择性监督

P0 已完成错误步骤伤害验证。下一阶段围绕 P1–P4 验证一个更直接的问题：

> **在高-surprisal 困难步骤上，当 Student 当前 hidden-state confidence 低于阈值时，降低对应 Teacher step supervision，能否减少错误显式监督对 latent reasoning 的干扰，并提高最终答案准确率？**

当前方法坚持 **accuracy-first**。暂不研究 test-time scaling、动态计算预算和 overthinking。

### 6.1 P1：基于 Surprisal 的困难步骤噪声构造

新一轮含噪数据不再随机选择任意步骤污染，而是参考 **Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment (RSR)** 中 surprisal 对 Student 不熟悉程度 / 信息难度的刻画，优先在当前 Student 真正困难的 reasoning steps 上构造噪声。

对 Teacher 正确轨迹中的第 `s` 个步骤 `S_s`，先使用 Student forward pass 计算 step-wise surprisal：

$$
\operatorname{Surp}(S_s)
=
-\frac{1}{|\mathcal T_s|}
\sum_{t\in\mathcal T_s}
\log p_{\theta}(y_t^T\mid x,y_{<t}^T).
$$

其中 `y_t^T` 为 Teacher step token。高 surprisal 表示该步骤对当前 Student 更不熟悉、学习难度更高。

这里必须区分 surprisal 与 confidence 的职责：

- **Surprisal 不作为最终步骤权重，也不直接判断 Teacher 是否正确；**
- surprisal 只负责定位困难步骤，使噪声更集中在接近模型能力边界的位置；
- 真正控制训练时 Teacher step supervision 的信号是 hidden-state confidence。

数据构造流程：

1. 从正确 Teacher CoT 中分割 reasoning steps；
2. 用当前 Student 计算每个 step 的 surprisal；
3. 按 Top-q 或分位数选择高-surprisal 困难步骤；
4. 在这些困难步骤上注入可审计的语义错误，例如数值、运算符或依赖关系错误；
5. 尽量固定问题、步骤位置、步骤数量、文本长度和训练预算；
6. 最终答案标签保持正确，使主要干预集中在 `L_step`；
7. 保存 clean/noisy 配对和真实污染位置，仅用于数据审计、检测分析和 oracle 上界。

这一设计的目的不是让噪声更随机，而是让实验更接近“Teacher 在能力边界附近的困难步骤更容易产生错误”的研究情形。

### 6.2 P2：从 Hidden State 提取 Step Confidence

Confidence 的设计参考 **Efficient Reasoning with Balanced Thinking (ReBalance)**：reasoning 过程中的 confidence 可以由模型内部状态表征，并能从 hidden representations 中读出。

对于第 `s` 个 reasoning step，直接从 Student 对应隐藏状态 `h_s^{(l)}` 中得到 step-level confidence：

$$
c_s = g(h_s^{(l)}),
$$

其中 `g(·)` 表示从 hidden state 到 confidence 的读出映射。实现时重点验证：

- 哪一层 hidden state 对 confidence 最敏感；
- reasoning step 应使用哪个代表位置的 hidden state；
- hidden-state confidence 与 logit-derived confidence 的相关性和 calibration；
- confidence 在 clean step 与被污染困难 step 上的分布差异。

这里的目标不是训练一个“错误步骤分类器”，而是得到一个能反映 **Student 当前内部不确定程度** 的 confidence 信号。

### 6.3 P3：Confidence 阈值式选择性降权

训练时不使用复杂连续门控，第一版直接采用阈值规则。

设：

- `c_s`：第 `s` 个 reasoning step 的 hidden-state confidence；
- `τ`：confidence 阈值；
- `β`：低 confidence 步骤保留的 Teacher supervision 权重，满足 `0 < β < 1`。

定义：

$$
w_s=
\begin{cases}
1, & c_s\ge \tau,\\
\beta, & c_s<\tau.
\end{cases}
$$

训练目标为：

$$
\mathcal{L}
=
\mathcal{L}_{\text{answer}}
+
\lambda\sum_{s=1}^{S}
w_s\mathcal{L}_{\text{step},s}.
$$

机制非常直接：

```text
Teacher reasoning step
        ↓
Student hidden state
        ↓
读取 confidence c_s
        ↓
      c_s < τ ?
      /       \
    Yes       No
     ↓         ↓
 w_s = β    w_s = 1
     ↓         ↓
弱化教师监督  正常教师监督
```

其中最终答案监督 `L_answer` 不做降权。

当前不把低 confidence 解释为“Teacher 一定错误”。更准确的解释是：

> 当 Student 在当前困难步骤上处于高不确定状态时，如果 Teacher step 本身存在错误，强制等权对齐的负迁移风险更高，因此只降低该位置的显式步骤监督强度。

第一轮实验优先使用简单固定的 `τ` 与 `β`，然后做参数敏感性分析。核心不是追求复杂 gate，而是验证这个最小机制能否稳定提高 accuracy。

### 6.4 P4：准确率优先，并观察 Underthinking 是否缓解

P4 的第一优先级是最终答案准确率 / EM。

核心比较：

$$
\Delta_{\text{ours}}
=
Acc_{\text{Confidence-Threshold}}
-
Acc_{\text{Noisy-Equal}}.
$$

希望看到：

$$
Acc_{\text{Noisy-Equal}}
<
Acc_{\text{Confidence-Threshold}}
\le
Acc_{\text{Clean}}.
$$

如果 confidence 阈值降权主要改善高-surprisal、低-confidence 困难样本，而普通样本基本不退化，就更支持当前机制解释。

本阶段对 underthinking 的定义不是“推理长度太短”，而是：

> Student 在困难步骤上过早被一个错误或不适配的显式 Teacher trajectory 拉向单一方向，使 latent representation 提前失去继续探索正确推理路径的空间。

因此当前只观察：

```text
高-surprisal 困难步骤
        ↓
Student confidence 低
        ↓
若仍等权强制学习 Teacher
        ↓
错误 Teacher step 更可能牵引 latent state
        ↓
premature commitment / underthinking

confidence < τ 时降权
        ↓
减少显式错误监督牵引
        ↓
保留 Student latent reasoning
        ↓
困难样本 accuracy ↑
```

本阶段**不增加 latent token 数、不改变 test-time compute、不做动态停止**。所以 underthinking 是否缓解主要通过困难样本准确率、步骤行为和 hidden-state 分析来间接验证。

### 6.5 下一轮实验分组

| 组别 | 数据 | Step supervision | 作用 |
| --- | --- | --- | --- |
| Clean SIM-CoT | 正确步骤 | 等权 `w_s=1` | 干净参考 |
| Noisy Equal | 高-surprisal 困难步骤注入噪声 | 等权 `w_s=1` | 含噪基线 |
| Noisy Fixed | 同一含噪数据 | 所有步骤统一固定降权 | 排除“只要减弱 auxiliary loss 就会变好” |
| Noisy Random | 同一含噪数据 | 随机选取同等比例步骤降权 | 排除随机减弱监督带来的收益 |
| **Noisy Confidence-Threshold** | 同一含噪数据 | **`c_s<τ` 时 `w_s=β`，否则 `w_s=1`** | **核心方法** |
| Noisy Oracle | 同一含噪数据 | 真实污染步骤降权 | 可恢复上界 / 诊断 |

研究重点不是再次证明错误步骤有害，而是在同一噪声条件下验证：

> **Confidence 阈值是否能比等权、固定降权和随机降权更稳定地恢复最终答案准确率。**

### 6.6 主要消融

优先做以下消融：

1. **困难步骤选择**：随机污染 vs 高-surprisal 污染；
2. **confidence 来源**：logit-derived confidence vs hidden-state confidence；
3. **hidden layer**：浅层 / 中层 / 深层；
4. **阈值 `τ`**：比较不同 confidence cutoff；
5. **低置信度权重 `β`**：例如 `0.1 / 0.3 / 0.5 / 0.7`；
6. **noise ratio / hard-step quantile**：验证不同噪声强度和困难步骤比例下的稳定性；
7. **oracle gap**：比较 confidence-threshold 与真实污染位置 oracle 的性能差距。

### 6.7 核心评价指标

**第一优先级：最终答案 Accuracy / EM。**

辅助指标：

- 高-surprisal 困难样本子集 accuracy；
- `c_s < τ` 的步骤比例；
- clean/noisy step 的 confidence 分布；
- confidence 与污染标签的 ROC-AUC / PR-AUC，仅作为诊断，不将 confidence 宣称为通用 correctness verifier；
- 不同 hidden layer 的 confidence 可读出性；
- clean/noisy step NLL；
- 多种子均值、标准差和置信区间。

成功标准首先看：

$$
Acc_{\text{Confidence-Threshold}}
>
Acc_{\text{Noisy-Equal}}.
$$

如果提升主要来自高-surprisal、低-confidence 困难样本，同时 clean / easy 样本没有明显退化，则进一步支持“降低困难区域错误 Teacher 牵引”的机制解释。

### 6.8 当前明确不做的部分

当前阶段暂缓：

- Test-time Scaling；
- 根据 confidence 动态增加 latent reasoning steps；
- Early Exit / 动态停止；
- Overthinking 检测与抑制；
- 以 token reduction 或推理速度作为主优化目标；
- RSR-RD 联合权重或其他多信号复杂门控。

当前主线只回答一个问题：

> **高-surprisal 困难步骤上的低 hidden-state confidence，能否作为“降低 Teacher step supervision”的内部触发信号，从而提高含噪 SIM-CoT 的最终答案准确率？**

### 6.9 方法参考

- **Efficient Reasoning with Balanced Thinking (ReBalance, ICLR 2026)**  
  https://arxiv.org/abs/2603.12372  
  本项目借鉴其 stepwise confidence 与 hidden-state confidence 表征思路，但不采用其 test-time steering、动态计算或 overthinking 控制。

- **Which Reasoning Trajectories Teach Students to Reason Better? A Simple Metric of Informative Alignment (RSR, ACL 2026)**  
  https://aclanthology.org/2026.acl-long.1950/  
  本项目只借鉴 surprisal 对 Student 不熟悉 / 困难步骤的刻画，用于定位困难步骤并构造噪声，不直接沿用 RSR 作为最终监督权重。

## 7. 仓库结构

```text
configs/reliable_simcot/   每轮冻结配置；先看这里确认模型起点和实验变量
docs/superpowers/specs/    主要实验规格与预注册门
refine-logs/               早期计划、追踪表和阶段报告
scripts/                   命令行入口和流水线编排
src/reliable_simcot/       官方适配、数据构造、训练、评测与统计核心代码
src/rsr_rd_simcot/         最初 RSR/RD PoC 实现
tests/                     回归测试
outputs/                   指标、报告、预测和审计记录；部分文件未被 Git 跟踪
work/                      下载的数据、生成 manifest、checkpoint；默认不上传 GitHub
logs/                      运行日志；默认不上传 GitHub
```
### 为什么 GitHub 看不到部分实验结果

大型 checkpoint、预测、manifest、日志和部分 `outputs/` 由 `.gitignore` 排除，因此 GitHub 主要保存代码、冻结配置和少量关键报告。README 中写出的数值来自本地现存 JSON/报告；仅有代码并不等于远端仓库包含全部原始实验数据。

如果要发布论文复现包，应另行生成一个小型 `results-index.json`，保存每轮状态、关键指标、原始文件 SHA-256 和可下载位置，而不是把 GB 级 checkpoint 直接塞进 Git。
## 8. 代码提交索引

| Commit | 内容 |
| --- | --- |
| `783cb78` | 官方 SIM-CoT checkpoint 评测与单卡复现 |
| `2b63fe7` / `3b56f41` | 自然步骤审计、可靠性数据、特征缓存与 LOFO 头 |
| `9d29f08` | 20% oracle 步骤加权 |
| `299a4b2` | 40% oracle 步骤加权 |
| `234d0f7` | 因果传播错误与伤害门 |
| `23b2b7b` | 梯度杠杆实验 |
| `7d4f5de` | 全链强冲突压力测试 |
| `0a58182` | PRM800K 自然噪声数据门 |
| `4caee70` | v9–v22 及相关代码的汇总快照 |
| `3bdc03a` | v23 关闭梯度裁剪实验 |
| `956a311` | v24 从纯 Coconut 开始的关闭裁剪实验 |
## 9. 运行约定

PowerShell 环境下统一使用：

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe <script> --config <config>
```

运行任何旧实验前先检查：

1. config 中的 `checkpoint_path` 与 `checkpoint_sha256`；
2. 起点究竟是纯 Coconut、Coconut + SIM-CoT，还是 CODI；
3. 评测集是官方 test、OOD test，还是训练源确认集；
4. `max_grad_norm`、`lambda_aux`、训练样本数、updates 与随机种子；
5. 该实验是确认性、探索性、事后续跑，还是已在数据门前被拒绝；
6. 是否复用了已经多次查看的 test set。
## 10. 结果引用原则

对外写作时，应坚持以下表述边界：

- 可以说：“在受控高密度强冲突下，步骤错误能够损害 SIM-CoT 的训练源确认集表现。”
- 可以说：“在官方 test 的多轮短预算实验中，单边严重错误没有产生稳定可检测伤害。”
- 可以说：“纯 Coconut 的 32k 错误组略差于正确组，但差异小且单种子不显著。”
- 不可以说：“自然教师噪声已经被证明严重伤害 SIM-CoT。”
- 不可以说：“可靠性加权已经提高正确率。”
- 不可以把 `79.82%` 写成官方 GSM8K test baseline。
- 不可以把第三方 Coconut checkpoint-33 写成作者 checkpoint-24。
- 不可以把当前实验写成 CODI + SIM-CoT；该路线尚未完成。
