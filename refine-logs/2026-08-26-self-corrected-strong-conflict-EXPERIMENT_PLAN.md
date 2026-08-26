# 强错误—显式自纠—正确答案实验实施计划

**Problem：** 此前伤害实验没有同时满足“中间步骤错误、CoT 自身最终答案正确、教师式步骤监督”三个条件，无法直接证明真正错误步骤的监督伤害。

**Method Thesis：** 在同题同答案的五联组中，仅改变错误类型、错误剂量和 oracle 步骤权重，可以隔离“强错误但最终自纠答对”的中间监督对 SIM-CoT 的伤害及可恢复性。

**Date：** 2026-08-26

**批准规格：** `docs/superpowers/specs/2026-08-26-simcot-self-corrected-strong-conflict-factorial-design.md`

**硬件：** RTX 4060 8GB

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1：自身最终答对的强错误 CoT 仍会伤害 SIM-CoT | 这是继续研究步骤可靠性加权的必要前提 | 至少一个污染族/剂量的三种子平均伤害大于 0，至少 2/3 种子同向，95% 分层配对 bootstrap CI 不跨 0 | B1–B4 |
| C2：oracle 0.1 降权能恢复伤害 | 证明“正确识别后加权”本身具有价值 | 对应加权臂显著优于等权臂；实用门为恢复 >=1.5 pp 且恢复比例 >=50% | B2–B4 |
| A1：差异不是终点标签冲突造成 | 此前实验的核心混杂必须消除 | 2560 条轨迹全部由自身第五步推导出 grader 正确答案 | B1 |
| A2：差异不是长度、总辅助损失或工程恢复造成 | 确保权重效应可归因 | 长度比 0.9–1.1、均值归一化权重、同题日程、独立初始化、同臂续跑 | B1–B3 |

## Paper Storyline

- 主文必须证明：强错误解法和题意误读在 CoT 最终自纠答对时是否仍有训练伤害。
- 主文必须证明：一个和两个连续错误的剂量差异，以及 oracle 0.1 降权的恢复量。
- 附录报告：46 条 PRM800K 自然 Noise-1 的半合成修复审计、失败构造案例和长度/位置分布。
- 本轮明确不训练可靠性头，不估计自然噪声发生率，不把 Codex 构造轨迹称为自然教师噪声。

## Experiment Blocks

### Block 1：512 套五联组构造与冻结

- **Claim tested：** A1、A2。
- **Dataset / split / task：** 官方 MATH 训练题和答案；官方 MATH 500 题仅用于最终干净测试。
- **Compared systems：** Clean、错误解法 N1/N2、题意误读 N1/N2。
- **Metrics：** 合格五联组数量、错误数量/位置、grader 正确率、长度比、编辑距离、训练测试重叠、拒绝原因。
- **Setup details：** 缺少合格步骤时由 Codex 构造；N1 第 2 步错、第 3 步纠错；N2 第 2–3 步连续错、第 4 步纠错；第五步自身答对。
- **Success criterion：** 512 套完整五联组、2560/2560 答案正确、长度比 0.9–1.1、零测试重叠；20 套人工审计通过。
- **Failure interpretation：** 重新构造或换题，不因自然噪声缺失停止，也不放宽结构和正确性。
- **Priority：** MUST-RUN。

### Block 2：九臂损失、权重和 GPU 工程门

- **Claim tested：** A2。
- **Compared systems：** Clean；两污染族 × 两剂量 × 等权/0.1 加权。
- **Metrics：** 输入/答案 parity、权重向量、平均权重、初始 loss、指定步骤梯度比例、有限 loss/grad、峰值显存。
- **Setup details：** `L=L_answer+L_steps`；`L_steps=5*sum(w_i L_i)/sum(w_i)`；错误步骤相对权重 0.1。
- **Success criterion：** 九臂 2-update sanity 全通过，显存 <=7.4GB；错误步骤相对梯度贡献符合 0.1。
- **Failure interpretation：** 仅修复工程错误，不改数据、权重、种子或训练规模。
- **Priority：** MUST-RUN。

### Block 3：九臂三种子完整训练

- **Claim tested：** C1、C2、A2。
- **Dataset / split / task：** 冻结 512 题；每臂 64 updates。
- **Compared systems：** 九臂完整析因矩阵。
- **Metrics：** 总 loss、答案 loss、步骤 loss、梯度范数、显存、checkpoint hash。
- **Setup details：** seeds `20260809/10/11`；BF16；accumulation 8；LR `1e-4`；WD `0.01`；clip `1.0`；`latent_stage=5`；`c_thought=2`。
- **Success criterion：** 27 个独立 checkpoint 完成，均从同一起始 checkpoint 初始化。
- **Failure interpretation：** 当前臂可安全续跑；禁止跨臂恢复。
- **Priority：** MUST-RUN。

### Block 4：干净 MATH 测试与分层配对统计

- **Claim tested：** C1、C2。
- **Dataset / split / task：** 未加噪的官方 MATH 500 题；测试时无标签和错误步骤。
- **Metrics：** 官方 grader 正确率、标准化 EM、伤害量、恢复量、恢复比例、剂量效应、污染族差异、10,000 次分层配对 bootstrap。
- **Success criterion：** 按批准规格报告统计门和实用门；全部九臂三种子完整公开。
- **Failure interpretation：** 低基线不停止，但标记统计功效限制；无伤害时不宣称加权必要。
- **Priority：** MUST-RUN。

### Block 5：真实性补充与结论边界

- **Claim tested：** 不新增主张，只检查构造模板是否过窄。
- **Dataset / split / task：** 46 条 PRM800K 自然 Noise-1 错答轨迹的显式纠错修复版本。
- **Metrics：** 修复成功数、错误位置、长度、纠错形式、与主实验错误族的覆盖关系。
- **Success criterion：** 与主结果分开报告，并明确这些修复轨迹仍是半合成数据。
- **Priority：** NICE-TO-HAVE；不得阻塞 B1–B4。

## Implementation Work Packages

### WP1：配置与五联组数据管线

**Create：**

- `configs/reliable_simcot/self_corrected_strong_conflict.json`
- `src/reliable_simcot/self_corrected_data.py`
- `tests/test_self_corrected_data.py`

**Reuse：** `prm800k_data.py` 的 MATH/grader/哈希读取，`full_conflict_generation.py` 和 `full_conflict_validation.py` 的构造记录与强冲突验证。

**Acceptance：** 合成 fixture 覆盖 Clean、两污染族、N1/N2、纠错、答案错误、长度超限、哈希去重和不可变冻结。

### WP2：九臂映射、归一化权重和训练

**Create：**

- `src/reliable_simcot/self_corrected_experiment.py`
- `tests/test_self_corrected_experiment.py`

**Reuse：** `full_conflict_experiment.py` 的官方 checkpoint 路径与状态保存，`oracle_weighting.py` 的 grouped auxiliary loss 和微批次随机日程。

**Acceptance：** 九臂名称/步骤目标/答案一致；N1/N2 原始权重向量和归一化 loss 精确；九臂 2-update GPU sanity 通过。

### WP3：评估、统计和报告

**Create：**

- `src/reliable_simcot/self_corrected_evaluation.py`
- `tests/test_self_corrected_evaluation.py`
- `scripts/run_self_corrected_strong_conflict.py`
- `scripts/run_self_corrected_overnight.py`

**Reuse：** `full_conflict_evaluation.py` 的生成、grader、bootstrap、逐题输出和报告结构。

**Acceptance：** fixture 覆盖伤害、恢复、剂量、无伤害和地板场景；固定 bootstrap seed 字节级可重复；JSON 与 Markdown 一致。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 配置、fixture 和回归基线 | SC001–SC003 | 新旧测试通过 | CPU 0.2–0.5 h | 旧代码接口差异 |
| M1 | 构造并冻结 512 五联组 | SC010–SC015 | 2560/2560 合格、20 套审计通过 | CPU/Codex 1–3 h | 复杂题构造失败率 |
| M2 | 九臂损失与 GPU 门 | SC020–SC023 | parity、梯度、有限性、<=7.4GB | GPU 0.2–0.5 h | 长文本/OOM |
| M3 | 九臂三种子训练 | SC100–SC128 | 27 个 checkpoint | GPU 4–6 h | 单臂失败/耗时 |
| M4 | 500 题评估 | SC200–SC228 | 27 份逐题预测完整 | GPU 2–5 h | 自回归生成耗时 |
| M5 | 统计与报告 | SC300–SC303 | JSON、表格、结论和边界一致 | CPU 0.2–0.5 h | 多比较和低基线 |

## Execution Rules

1. 先构造并冻结全部数据，再启动任何正式训练。
2. 九臂每个种子共享题目顺序和随机日程；不同种子独立初始化。
3. 状态机按 `seed:arm` 保存训练与评估完成状态，原子写 JSON。
4. 不覆盖原始构造记录；重试追加新记录并保存拒绝原因。
5. 工程恢复只能使用当前 `seed:arm` 检查点。
6. 数据、配置、种子、错误位置、覆盖率、0.1 权重和 `lambda=1` 在冻结后不可更改。
7. 500 题测试只含原始干净问题，不注入任何错误步骤。

## Compute and Data Budget

- **训练：** 预计 4–6 GPU-hours；27 臂完整评估预计再需 2–5 GPU-hours。
- **数据：** 2560 条五步轨迹、512 套构造记录和 20 套人工审计。
- **显存：** 峰值保留显存硬上限 7.4GB。
- **最大瓶颈：** 512 套强冲突、自纠和长度匹配同时通过验证。

## Risks and Mitigations

- **构造轨迹只是在末尾贴答案：** validator 必须验证纠错和恢复推导；不合格即重构。
- **人工模板过强或过窄：** 分开使用错误解法和题意误读，并保留 46 条 PRM 修复补充。
- **长度成为混杂：** 每个含噪版本限定为 Clean token 长度的 0.9–1.1 倍。
- **归一化导致“0.1”含义不清：** 报告原始相对权重和归一化后的有效权重；公式固定。
- **多臂偶然显著：** 使用三种子、共同题目分层配对 bootstrap，并完整报告全部对比。
- **地板效应：** 不作为停止门，但限制“无伤害”结论。
- **OOM 或进程失败：** 先修工程问题，同臂状态安全续跑；不改科学规格。

## Final Checklist

- [x] 两条主张与批准规格一致
- [x] 终点答案正确与答案标签正确已明确区分
- [x] 现成噪声不足不再阻止构造实验
- [x] 两污染族、两剂量、等权/0.1 加权完整覆盖
- [x] 三种子、配对统计和结论门已定义
- [x] 自然教师噪声与 Codex 构造噪声边界明确
- [x] 必跑与补充实验已分开

