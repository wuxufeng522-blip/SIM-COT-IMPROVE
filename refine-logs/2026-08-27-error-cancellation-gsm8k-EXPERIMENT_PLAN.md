# 错误抵消步骤噪声控制实验实施计划

**Problem：** 既有全链污染实验同时改变了过程正确性和 CoT 自身终点，后续 v8 又因冗长 Clean 构造破坏了 SIM-CoT 步骤对齐，因此尚未严格回答“最终答案监督相同且正确时，错误步骤语义本身是否造成伤害”。

**Method Thesis：** 在同题、同答案、同五步槽位、同覆盖集合和逐 token 归一化步骤损失下，用数学正确的匹配冗余轨迹控制额外计算，仅比较错误抵消轨迹与冗余轨迹，即可隔离错误步骤语义的增量伤害；伤害通过后，再用匹配位置降权控制辅助剂量，验证 oracle 0.1 加权的选择性恢复。

**Date：** 2026-08-27

**批准规格：** `docs/superpowers/specs/2026-08-27-error-cancellation-step-noise-design.md`

**硬件：** RTX 4060 Laptop GPU，8GB 显存

**新实验命名空间：** `error_cancellation_gsm8k_v9`

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：错误抵消步骤比位置和长度匹配的正确冗余步骤更伤害 SIM-CoT | 这是继续研究步骤可靠性加权的必要前提，并排除了“只是信息更多”的解释 | 主比较 `RW50-EW50` 平均至少 2pp、三种子同向、分层配对 bootstrap 95% CI 下界大于 0；达到 5pp 才称严重伤害 | B1–B3 |
| C2：oracle 0.1 降权产生选择性恢复，而非单纯减少辅助损失 | 证明步骤选择性加权本身有价值 | `R_selective>=1.5pp`、三种子同向、95% CI 下界大于 0、恢复比例至少 50% | B4 |
| Anti-claim：差异来自答案标签、步骤长度、覆盖集合、总辅助剂量或失效 Clean | 这些是此前实验的主要混杂 | 逐样本 `L_correct` 等价、匹配冗余、逐 token 归一化、嵌套覆盖、相对 Clean 门、冗余掩码对照全部通过 | B1–B4 |

## Paper Storyline

- **主文必须证明：** 在最终答案监督完全相同时，广域 50% 错误抵消是否相对匹配冗余产生可检测伤害。
- **主文条件式证明：** 若伤害成立，oracle 0.1 是否在扣除匹配掩码收益后仍产生选择性恢复。
- **主文支持结果：** 25%/50% 剂量和局部/广域传播范围是否改变伤害。
- **附录：** 构造失败类型、算术审计、位置分布、token 比例、梯度与显存记录。
- **明确删除：** 本轮不训练可靠性头、不估计自然教师错误率、不使用自然噪声措辞、不再复用 v8 冗长 Clean。
- **Frontier necessity：** 本实验是确定性机制验证，不需要前沿模型组件；引入 LLM 判别器会增加不必要混杂，故明确跳过。

## Experiment Blocks

### Block 1：五变体数据构造、算术验证与冻结

- **Claim tested：** Anti-claim。
- **Why this block exists：** 保证 Clean 不被改写，错误抵消和正确冗余只在同一槽位改变步骤语义。
- **Dataset / split / task：** 官方 GSM8K 训练源选择 512 个具有五个可解析计算步骤的唯一题目；官方 test 1,319 题只用于评估。
- **Compared systems：** Clean、局部错误抵消、广域错误抵消、局部匹配冗余、广域匹配冗余。
- **Metrics：** 512 套完整率、2,560 条轨迹格式通过率、错误等式复算、错误传播与抵消、冗余等式正确率、答案一致率、token 比例、训练测试重叠、人工审计结果。
- **Setup details：** 严格五个 `<<表达式=结果>>`；错误链包含两个计算器可证明的错误等式；广域链中间状态真实传播；冗余只用 `+0/-0/*1//1/+(q-q)`；Clean 不做长度填充。
- **Success criterion：** 512 套、2,560/2,560 格式与答案通过、错误/冗余算术标签全正确、测试重叠为 0、变体对 Clean 总 token 比 0.9–1.1、错误与匹配冗余差不超过 5%、20 套人工审计通过。
- **Failure interpretation：** 剔除不合格题并继续候选搜索；不得扩写 Clean、放宽格式或伪造抵消。
- **Table / figure target：** 附录数据审计表和四类轨迹示例。
- **Priority：** MUST-RUN。

### Block 2：损失隔离、运行清单与 Clean 有效性

- **Claim tested：** Anti-claim。
- **Why this block exists：** 防止再次把坏 Clean 或目标长度剂量误判为噪声伤害。
- **Dataset / split / task：** 冻结 512 题训练日程；官方 test。
- **Compared systems：** 原始 checkpoint、Clean 三种子；fixture 中的五种目标。
- **Metrics：** 逐样本 `L_correct`、`lambda=0` 梯度/更新等价性、逐 token 步骤损失、有限 loss/gradient、峰值显存、原始 checkpoint 与 Clean 准确率。
- **Setup details：** `L_total=L_correct+L_step`，`L_step=mean_step(mean_token(CE))`；所有第一阶段步骤权重为 1；答案分支不读取变体步骤文本。
- **Success criterion：** 初始 `L_correct` 和 `lambda=0` 参数更新等价；2-update sanity 通过；最长样本和完整 64-update Clean 显存不超过 7.4GB；`mean Clean >= Acc0-2pp` 且任一种子不低于 `Acc0-5pp`。
- **Failure interpretation：** 任一门失败均在非 Clean 训练前停止；只允许修复工程错误，不得修改科学阈值后跨配置续跑。
- **Table / figure target：** 主文实验有效性表；附录 loss/显存审计。
- **Priority：** MUST-RUN。

### Block 3：第一阶段错误语义伤害矩阵

- **Claim tested：** C1。
- **Why this block exists：** 通过匹配冗余直接估计错误语义，而不是错误轨迹相对 Clean 的混合效应。
- **Dataset / split / task：** 冻结 512 题训练；官方 GSM8K test 1,319 题。
- **Compared systems：** C、RL25、RL50、EL25、EL50、RW25、RW50、EW25、EW50。
- **Metrics：** 官方精确匹配准确率、三种子均值/标准差、逐题配对差异、10,000 次分层配对 bootstrap、冗余开销、总错误伤害、覆盖剂量和传播范围差异。
- **Setup details：** 三种子；每臂 64 updates；accumulation 8；BF16；LR `1e-4`；WD `0.01`；clip `1.0`；`latent_stage=5`；`c_thought=2`；同一 checkpoint 独立初始化。
- **Success criterion：** 主比较 `Acc(RW50)-Acc(EW50)>=2pp`、三种子同向、95% CI 下界大于 0；至少 5pp 才称严重伤害。
- **Failure interpretation：** 若主门失败，停止所有加权臂；结论为在本压力设置下未观察到可检测的增量语义伤害，不推出自然噪声无害。
- **Table / figure target：** 主文表 1（九组准确率）和图 1（覆盖率 × 传播范围伤害）。
- **Priority：** MUST-RUN。

### Block 4：条件式 0.1 加权与匹配掩码对照

- **Claim tested：** C2、Anti-claim。
- **Why this block exists：** 将选择性错误抑制与一般性降低辅助损失剂量分开。
- **Dataset / split / task：** 与 Block 3 完全相同。
- **Compared systems：** EW50-equal、EW50-w01、RW50-equal、RW50-w01；通过后扩展到 wide25、local50、local25 的错误/冗余成对加权组。
- **Metrics：** `R_error`、`R_redundant`、`R_selective`、`F_recovery`、逐种子方向、分层配对 bootstrap。
- **Setup details：** EW 中 `DIRECT_FALSE/ERROR_DESCENDANT/CANCEL_FALSE` 权重 0.1；RW 对应位置权重 0.1；其余为 1.0；不对权重向量重新归一化，冗余掩码组负责控制总剂量下降。
- **Success criterion：** `R_selective>=1.5pp`、三种子同向、95% CI 下界大于 0、恢复比例至少 50%。
- **Failure interpretation：** 若错误组表面提高但选择性净恢复失败，解释为降低辅助监督剂量，而不是噪声识别有效。
- **Table / figure target：** 主文表 2（条件式恢复）；扩展结果放附录。
- **Priority：** MUST-RUN IF GATED。

### Block 5：失败分析、梯度诊断与结论边界

- **Claim tested：** 不新增主张，用于解释失败模式。
- **Dataset / split / task：** 已冻结训练与逐题预测产物。
- **Compared systems：** 重点比较 RW50/EW50 及其加权组。
- **Metrics：** 错误位置、答案翻转、预测频率、步骤 NLL、预裁剪梯度范数、目标 token 数和显存。
- **Success criterion：** 所有主结果均可追溯到同一 manifest/schedule/checkpoint hash；报告明确半合成边界。
- **Failure interpretation：** 区分“无语义伤害”“Clean 失效”“冗余本身有害”“加权只降低剂量”和“统计功效不足”。
- **Table / figure target：** 失败分析附录和中文阶段报告。
- **Priority：** MUST-RUN（诊断最小集）；扩展可视化 NICE-TO-HAVE。

## Implementation Work Packages

### WP1：独立配置和确定性数据模块

**Create：**

- `configs/reliable_simcot/error_cancellation_gsm8k_v9.json`
- `src/reliable_simcot/error_cancellation_data.py`
- `tests/test_error_cancellation_data.py`

**Reuse：** 官方 GSM8K 适配器、哈希/原子写入工具和 v8 的官方 test 冻结方式；不得复用 v8 的 Clean 重写函数。

**Acceptance：** fixture 覆盖局部/广域错误、第二错误抵消、错误传播断裂、冗余恒等式、格式错误、答案变化、token 超限、覆盖集合嵌套和不可变冻结。

### WP2：逐 token 损失、臂映射和等价性审计

**Create：**

- `src/reliable_simcot/error_cancellation_experiment.py`
- `tests/test_error_cancellation_experiment.py`

**Reuse：** 官方模型加载、训练状态和显存统计；新损失不得更改旧实验模块的行为。

**Acceptance：** 九个第一阶段臂映射精确；答案路径与变体无关；`lambda=0` 更新相同；加权错误/冗余位置匹配；CPU 与 2-update GPU 门通过。

### WP3：官方 test 评估、门控统计和报告

**Create：**

- `src/reliable_simcot/error_cancellation_evaluation.py`
- `tests/test_error_cancellation_evaluation.py`
- `scripts/run_error_cancellation_pipeline.py`
- `scripts/run_error_cancellation_overnight.py`

**Reuse：** 当前 GSM8K 精确匹配抽取、逐题 JSONL、安全续跑和 bootstrap 实现。

**Acceptance：** fixture 覆盖 Clean 门、伤害门、严重伤害门、无伤害停止、选择性恢复、纯剂量恢复和条件式扩展；JSON 与 Markdown 数字一致。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Estimated Cost | Risk / Mitigation |
|---|---|---|---|---|---|
| M0 | 新模块、fixture 与旧回归 | EC001–EC004 | 新旧 CPU 测试通过 | CPU 0.3–0.8h | 隔离新模块，不修改旧损失语义 |
| M1 | 构造、审计并冻结 512 套五变体 | EC010–EC016 | 2,560/2,560 合格、20 套审计通过 | CPU 0.5–2h | 不合格题替换，不填充 Clean |
| M2 | 损失等价、GPU sanity、显存和原始基线 | EC020–EC026 | parity、有限性、<=7.4GB、Acc0 完整 | GPU 0.3–0.6h | 先门控再正式训练 |
| M3 | 三个 Clean 种子 | EC100–EC105 | 相对 Clean 门通过 | GPU 0.4–0.7h | 失败即停，不启动非 Clean |
| M4 | 其余八组 × 三种子训练与评估 | EC110–EC133 | 24 checkpoint + 24 评估完整 | GPU 3.5–4.5h | 同臂安全续跑，禁止跨臂恢复 |
| M5 | 第一阶段统计 | EC200–EC203 | C1 主门明确 PASS/FAIL | CPU 0.2–0.5h | 主比较预注册，其他为支持 |
| M6 | EW50/RW50 条件式 0.1 对照 | EC300–EC307 | C2 四项门明确 PASS/FAIL | GPU 0.8–1.2h | 必须扣除冗余掩码效应 |
| M7 | 条件式扩展与最终报告 | EC400–EC409 | 扩展完整或按门停止 | GPU 0–3h + CPU 0.3h | 不让附加运行阻塞核心结论 |

## Execution Rules

1. 新实验只写入 `error_cancellation_gsm8k_v9` 路径，不覆盖 v5–v8 或旧 full-conflict 记录。
2. 数据、manifest、覆盖集合和 schedule 必须在正式 GPU 训练前冻结。
3. 原始 checkpoint 只评估一次；所有训练臂均从该 checkpoint 独立初始化。
4. 同一 seed 的所有臂共享题序和微批次随机日程。
5. 状态机以 `seed:arm:phase` 记录；恢复只允许同配置同臂。
6. 峰值 reserved 显存超过 7.4GB、非有限损失、哈希变化或门控失败时立即停止。
7. 工程恢复不得修改数据、种子、覆盖率、损失、权重或统计门。
8. 官方 test 已在 v8 打开，本轮复用同一冻结版本，不根据 test 结果修改构造。
9. 第一阶段失败时不运行第二阶段；第二阶段主条件失败时不运行扩展。

## Compute and Data Budget

- **必须运行的 GPU 预算：** 约 5–7 GPU-hours，包括原始基线、Clean 门、第一阶段和条件式 EW50/RW50 加权主实验。
- **全部条件式扩展后的上限：** 约 8–10 GPU-hours。
- **CPU 数据与测试：** 约 1–3 小时。
- **数据产物：** 512 套五变体、2,560 条轨迹、20 套人工审计、固定覆盖集合与训练 schedule。
- **最大瓶颈：** 在不修改 Clean 的前提下，同时满足错误抵消真实性、冗余正确性和 token 匹配。

## Risks and Mitigations

- **第二个错误只是贴回正确数值：** 必须保留可解析表达式、错误状态引用和依赖图；人工审计检查抵消是否显著。
- **错误链过于人工：** 结论严格限制为受控半合成机制，不外推自然发生率。
- **冗余组本身改变推理结构：** 冗余只能使用中性运算并保持每个 Clean 状态不变。
- **长度仍影响损失：** 逐 token、逐步骤归一化；同时执行 10%/5% token 门。
- **Clean 再次崩溃：** 原始步骤逐字保留，并使用相对原始 checkpoint 的停止门。
- **0.1 组只是少训练：** 必须扣除对应 RW 掩码组的变化。
- **多重比较：** wide50 是唯一主比较，其余按预定层次解释。
- **三种子统计功效有限：** 同题配对 bootstrap 和实用效应门并用，完整报告各种子。

## Final Checklist

- [x] 主张不超过两项，且每个实验块直接服务主张
- [x] Clean、冗余、错误三者的因果变量已分离
- [x] 25%/50% 和局部/广域矩阵已冻结
- [x] 第一阶段伤害与第二阶段恢复明确门控
- [x] 原始 checkpoint 与相对 Clean 门纳入执行顺序
- [x] 0.1 的一般剂量效应由匹配冗余掩码控制
- [x] 单卡 8GB 显存门和同臂续跑规则明确
- [x] 自然教师噪声主张已明确排除
- [x] MUST-RUN 与条件式扩展已分离
