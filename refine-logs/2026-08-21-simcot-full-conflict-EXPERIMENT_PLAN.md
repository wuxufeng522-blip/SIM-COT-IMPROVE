# SIM-CoT 全链强冲突实验实施计划

**Problem：** 既有 25% 题目覆盖率的三步因果污染只形成 15% 总步骤污染，并与干净轨迹共享大量 token，可能低估大面积错误显式监督对 SIM-CoT 的伤害。

**Method Thesis：** 在题目和官方答案始终正确的前提下，用 Codex 为同题生成五步全部冲突、内部算术自洽的反事实轨迹，可以直接检验显式步骤语义冲突是否具有足够强的答案因果杠杆。

**Date：** 2026-08-21

**批准规格：** `docs/superpowers/specs/2026-08-21-simcot-full-conflict-noise-design.md`

**硬件：** NVIDIA RTX 4060 Laptop GPU，8 GB 显存

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Blocks |
|---|---|---|---|
| C1（主张）25% 同题覆盖率下，五步全链冲突会稳定伤害答案正确率 | 若该门不成立，继续研究复杂可靠性头缺乏必要前提 | `Clean - Full25` 三种子平均至少 5 pp，3/3 种子同向 | B1, B2, B3 |
| C2（支持主张）全链冲突比旧三步局部因果错误伤害更大 | 直接检验原负结果是否来自监督冲突被局部正确 token 稀释 | `D_full25 - D_local25 >= 2 pp`，并且两组污染完全相同的 128 道题 | B1, B3 |
| C3（条件弱主张）若 25% 不足，50% 全链冲突可建立高密度压力伤害 | 区分“25% 处理太弱”和“答案通路对显式语义冲突普遍鲁棒” | 仅在 B3 失败后运行；`Clean - Full50 >= 5 pp` 且 3/3 同向 | B4 |
| A1（反主张）观察到的伤害只是答案标签错误、文本更长、问题不同或 lambda 放大造成 | 排除非语义混杂，防止把普通标签噪声误称为步骤对齐机制 | 官方答案逐字节相同；错误 token 长度为干净的 90%–110%；Local25/Full25 使用同题；固定 `lambda_aux=1` | B1, B2, B3 |

## Paper Storyline

- 主结果必须证明：在严格控制题目、答案、长度、训练顺序和覆盖题目后，全链错误是否达到预注册答案伤害门。
- 支持结果必须证明：处理确实进入辅助解码损失并改变基础模型梯度方向，而不是数据管线失效。
- 附录可报告：错误意图分布、拒绝原因、编辑距离、逐层梯度夹角、逐题翻转和代表性错误链。
- 本轮不把 Codex 生成器当作方法贡献；它只是构造强压力处理的数据工具，因此不强行设置“前沿模型必要性”主张。
- 简洁性由 `Local-causal-25` 对照承担：若局部与全链处理没有实质差异，就没有理由引入更复杂的可靠性系统。
- 明确删除：oracle `0.1/0`、可靠性头、自然教师噪声、官方测试集、`lambda>1`、覆盖率超过 50%、查看结果后修改错误链。

## Experiment Blocks

### Block 1：数据与强冲突处理完整性

- **Claim tested：** A1；为 C1/C2 提供可归因处理。
- **Why this block exists：** 若全链轨迹并非五步全部冲突、内部不自洽或答案字段被污染，后续 EM 差异无法解释。
- **Dataset / split / task：** 从官方训练源筛选全新、恰好五步且旧因果生成器可用的问题；冻结 512 训练题、1,024 确认题、256 生成主问题和 128 备用问题。
- **Compared systems：** 干净五步、旧三步局部因果链、Codex 五步全链错误。
- **Metrics：** 合格率、拒绝原因、5/5 步冲突率、5/5 结果冲突率、依赖连通率、归一化 token 编辑距离、长度比、答案碰撞率、分区重叠数。
- **Setup details：** 四类错误意图各 64 条；每题最多生成三次；失败后只能按冻结备用顺序替换。
- **Success criterion：** 12 条小批量先全部通过；随后得到恰好 256 条完整合格轨迹；零分区重叠、零答案字段变化、零未记录替换。
- **Failure interpretation：** 若备用池耗尽或某类无法稳定生成，说明 Codex 强错误构造不具备所需可控性，停止训练而不是放宽规则。
- **Table / figure target：** 主报告表 1（数据处理审计）；附录错误链案例表。
- **Priority：** MUST-RUN。

### Block 2：损失与梯度通路 sanity

- **Claim tested：** A1，并确认强错误目标确实能改变 SIM-CoT 的训练信号。
- **Why this block exists：** 排除权重映射、token 分组或辅助解码目标没有正确接入的实现错误。
- **Dataset / split / task：** 小批量 12 条加固定梯度审计子集；不读取确认集结果。
- **Compared systems：** `answer_only`、`clean_aux1`、`local_causal_25`、`full_conflict_25`。
- **Metrics：** 官方 all-one loss parity、每组总损失/答案损失/辅助损失、辅助梯度是否为零、逐层梯度范数比、干净/全链错误辅助梯度余弦、峰值 reserved 显存。
- **Setup details：** checkpoint 28；BF16；每组 2 updates sanity；固定逐微批次随机种子；显存红线 7.4 GB。
- **Success criterion：** all-one 绝对差不超过 `1e-6`；`answer_only` 辅助梯度为零；四组损失和梯度有限；全链错误与干净辅助梯度不完全相同；峰值显存不过线。
- **Failure interpretation：** 任一门失败都属于实现/处理失败，不允许进入答案实验。
- **Table / figure target：** 主报告工程审计表；附录逐层梯度图。
- **Priority：** MUST-RUN。

### Block 3：25% 四组三种子主实验

- **Claim tested：** C1、C2、A1。
- **Why this block exists：** 这是决定是否继续步骤加权方向的主伤害门。
- **Dataset / split / task：** 同一 512 道训练题；新的 1,024 道确认题；官方测试集不打开。
- **Compared systems：** `answer_only`、`clean_aux1`、`local_causal_25`、`full_conflict_25`。
- **Metrics：** 答案 EM（决定性）；answer NLL、clean step-token NLL、逐题胜负、精确 McNemar、10,000 次配对 bootstrap；256 个训练处理对上的错误链与干净链 NLL 差（仅描述性）。
- **Setup details：** 64 updates；梯度累积 8；`lambda_aux=1`；学习率 `1e-4`；weight decay `0.01`；裁剪 `1.0`；latent stage 5；每步 2 个 continuous-thought tokens；种子 `20260809/10/11`。
- **Success criterion：** `D_full25 >= 5 pp`；3/3 种子 `Clean > Full25`；`D_full25 - D_local25 >= 2 pp`；全部工程审计通过。
- **Failure interpretation：** 若 C1 或 C2 任一失败，25% 大面积冲突假设不成立；不得通过挑 seed 或 NLL 替代 EM 宣称成功。
- **Table / figure target：** 主表 2（四组 × 三种子 EM/NLL）；主图 1（每 seed 配对效应）。
- **Priority：** MUST-RUN。

### Block 4：条件式 50% 高密度压力实验

- **Claim tested：** C3。
- **Why this block exists：** 只在 25% 门失败时区分覆盖率不足与更普遍的答案鲁棒性。
- **Dataset / split / task：** 同一训练/确认分区；激活预先冻结的第二个 128 题层级。
- **Compared systems：** `clean_aux1` 已完成结果与新增 `full_conflict_50`；不重复 Clean，不将 Local25 当作同覆盖率对照。
- **Metrics：** 与 B3 相同。
- **Setup details：** 三种子，其他配置逐项与 B3 相同。
- **Success criterion：** `Clean - Full50` 三种子平均至少 5 pp，且 3/3 同向。
- **Failure interpretation：** 若失败，中止可靠性头/加权方向；不得继续提高覆盖率或修改处理。
- **Table / figure target：** 条件表 3；若未触发则报告 `NOT_RUN_BY_GATE`。
- **Priority：** CONDITIONAL MUST-RUN。

### Block 5：失败模式和结论边界

- **Claim tested：** 不新增正向主张；审计 C1/C2/C3 的适用边界。
- **Why this block exists：** 无论正负结果都必须解释处理是否被模型看到，以及结论为何不能外推到自然噪声。
- **Dataset / split / task：** 已完成训练、确认预测、梯度和生成审计记录。
- **Compared systems：** 按已触发的组进行配对诊断。
- **Metrics：** 四类错误意图的合格率与伤害描述、错误位置/长度/编辑距离分布、逐题翻转、EM 与 NLL 是否分离、跨 seed 范围。
- **Setup details：** 不新增训练，不按错误类别回头选择处理。
- **Success criterion：** 机器可读 gate verdict 与中文报告一致；每个允许/禁止结论都有对应证据。
- **Failure interpretation：** 报告生成或训练异常，不能把工程失败当作科学负结果。
- **Table / figure target：** 结论表、失败案例附录。
- **Priority：** MUST-RUN。

## Implementation Work Packages

### WP1：配置、选择器和防泄漏清单

**Create：**

- `configs/reliable_simcot/full_conflict.json`
- `src/reliable_simcot/full_conflict_data.py`
- `tests/test_full_conflict_data.py`

**Responsibilities：**

- 读取官方数据及所有既有冻结 manifest；
- 规范化问题 ID 并排除所有已使用分区；
- 筛选恰好五步、可解析、可构造旧因果链的候选；
- 使用固定种子和 SHA-256 优先级冻结 512/1,024/256/128 清单；
- 写入 eligibility、split、coverage-tier 和 provenance 哈希。

**Acceptance：** 相同输入重复运行产生字节一致 manifest；所有分区交集为空；512 个训练问题均满足恰好五步和 Local 对照资格。

### WP2：生成任务、原始记录和备用分配

**Create：**

- `src/reliable_simcot/full_conflict_generation.py`
- `tests/test_full_conflict_generation.py`

**Responsibilities：**

- 产生固定 JSON prompt manifest 和四类配额；
- 支持当前 Codex 分批写入原始 JSONL；
- 记录 attempt 1–3、拒绝原因和不可变原始输出；
- 按冻结备用顺序分配替代问题；
- 构建 12 条小批量门和 256 条正式接受清单。

**Acceptance：** 不允许跳号、覆盖原始输出、结果后换题或超过三次主问题重试。

### WP3：全链验证器

**Create：**

- `src/reliable_simcot/full_conflict_validation.py`
- `tests/test_full_conflict_validation.py`

**Reuse：**

- `src/reliable_simcot/causal_corruptions.py` 的算式解析、执行和依赖边数据结构；
- `src/reliable_simcot/official_adapter.py` 的官方样本解析。

**Responsibilities：**

- 强制五步、局部算术、结果冲突、连通依赖、错误终点；
- 计算 tokenizer 级编辑距离和长度比；
- 检查题目/答案/ID 完整性、身份泄露和上下文长度；
- 输出逐条 verdict、拒绝计数和汇总审计。

**Acceptance：** 为每个拒绝码提供正反单元测试；随机篡改答案或依赖必须被拒绝；已接受记录再次验证必须完全一致。

### WP4：训练臂和统一损失

**Create：**

- `src/reliable_simcot/full_conflict_experiment.py`
- `tests/test_full_conflict_experiment.py`
- `scripts/run_full_conflict.py`

**Reuse：**

- `src/reliable_simcot/oracle_weighting.py` 的 step tokenization 和 grouped auxiliary loss；
- `src/reliable_simcot/gradient_leverage.py` 的三种子训练、梯度审计和确认集评估模式；
- `src/reliable_simcot/single_gpu_smoke.py` 的编码与 tensorization；
- `src/reliable_simcot/official_adapter.py` 的 checkpoint 加载和答案评估。

**Responsibilities：**

- 实现四个 25% 臂和条件式 `full_conflict_50`；
- 保证相同问题、顺序、答案标签和逐微批次 dropout seed；
- 保存 checkpoint、训练曲线、梯度裁剪前范数、显存峰值和输入 manifest hash；
- 暴露 `prepare/gate-small/validate/sanity/audit/train/evaluate/analyze` CLI 模式。

**Acceptance：** all-one loss parity；`answer_only` 辅助梯度为零；臂间答案 labels 相同；Local25/Full25 活跃问题 ID 完全相同。

### WP5：条件式夜间编排、分析和报告

**Create：**

- `src/reliable_simcot/full_conflict_evaluation.py`
- `tests/test_full_conflict_evaluation.py`
- `scripts/run_full_conflict_overnight.py`

**Responsibilities：**

- 以可恢复状态机依序训练/评估 25% 四组；
- 计算 C1/C2 门并只在失败时打开 50% 队列；
- 支持预测 JSONL 断点续评，不允许训练断点跨配置恢复；
- 输出 McNemar、bootstrap、三种子汇总、机器 verdict 和中文报告；
- 记录 `official_test_opened=false`。

**Acceptance：** 用合成 metrics fixture 覆盖 25% PASS、25% FAIL/50% PASS、两级 FAIL 三种状态；固定 seed 的 bootstrap 可重复；报告数值与 JSON 一致。

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
|---|---|---|---|---|---|
| M0 | 固化配置与数据资格 | FC001–FC003 | 分区零重叠、候选充足、manifest 可复现 | CPU 0.5–1 h | 既有分区遗漏；用全 manifest 聚合审计 |
| M1 | 证明 Codex 生成器能稳定满足强冲突规则 | FC010–FC011 | 12/12 小批量通过且四类均可生成 | Codex 约 0.5–1 h | 生成过于模板化；先看分布而不改主门 |
| M2 | 冻结 256 条正式错误链 | FC020–FC021 | 恰好 256 条合格、哈希冻结、备用使用可追溯 | Codex/CPU 约 2–4 h | 合格率低；三次重试与 128 冻结备用 |
| M3 | 损失、梯度和单卡预检 | FC030–FC032 | 测试/parity/sanity/显存全部通过 | GPU 0.2–0.4 h | 8 GB OOM；保持现有 batch=1/accumulation 路径 |
| M4 | 25% 四组主实验 | FC100–FC143 | 完成 12 个训练和 12 个确认评估 | GPU 4–5 h | 运行中断；逐臂原子 checkpoint 与可恢复评估 |
| M5 | 25% 决策 | FC300 | 严格计算 C1/C2，不人工解释开门 | CPU <0.1 h | 看结果改门；配置中硬编码阈值与哈希 |
| M6 | 条件式 50% 压力实验 | FC400–FC423 | 仅 FC300 未通过时运行；完成 C3 判定 | GPU 1–1.5 h | 与 Local25 错误比较；报告层强制禁止该结论 |
| M7 | 报告和下一步决定 | FC500 | JSON、表格、中文结论一致 | CPU 0.2–0.5 h | 负结果被过度解释；自动允许/禁止结论模板 |

## Detailed Execution Sequence

### M0：Sanity stage——数据和配置

1. 新增测试，先证明旧代码没有全链处理路径。
2. 实现精确五步资格筛选和所有旧 manifest 排除。
3. 生成资格报告、冻结训练/确认/主生成/备用 ID，并记录数据与 checkpoint 哈希。
4. 生成固定 prompt manifest；此时不训练、不读取确认集预测。

**Stop：** 精确五步且旧因果对照合格的可用问题不足，或任意分区有重叠。

### M1–M2：Treatment stage——Codex 生成和验证

1. 每类生成 3 条小样，共 12 条。
2. 自动验证并进行去理由、打乱顺序的第二遍 Codex 可读性复核。
3. 小门通过后冻结 prompt 版本。
4. 以每批 8–16 题生成剩余候选，逐批验证并原子追加原始 JSONL。
5. 完成 256 条后停止生成，冻结 accepted manifest 和 SHA-256。

**Stop：** 12 条小门失败；备用池耗尽；任何接受记录无法重复验证。

### M3：Baseline stage——实现与 GPU 预检

1. 跑新增单元测试和既有相关回归测试。
2. 跑官方 all-one loss parity。
3. 四组各跑 2 updates sanity，核对损失拆分、梯度、裁剪和显存。
4. 跑固定小集梯度审计，确认 Full 与 Clean 梯度方向不同。

**Stop：** parity、有限性、辅助梯度、相同答案标签或显存任一失败。

### M4–M5：Main method stage——25% 主实验与决策

训练顺序按 seed 外层、arm 内层固定：

1. `20260809`：answer-only → clean → local25 → full25；
2. `20260810`：同顺序；
3. `20260811`：同顺序；
4. 所有训练完成后统一确认评估，避免中途结果影响后续运行；
5. 计算 FC300 伤害门。

如果 FC300 PASS，直接进入 M7；如果 FAIL，才解锁 M6。

### M6：Conditional decision stage——50% 压力

1. 训练三个 `full_conflict_50` checkpoint；
2. 在相同 1,024 题确认集评估；
3. 只计算 C3，不重新解释 C1/C2；
4. 无论通过与否都停止增加处理强度。

### M7：Polish stage——结果和边界

1. 生成主表、逐 seed 效应表、工程审计表和错误链案例；
2. 输出机器 verdict；
3. 输出中文报告和下一步状态：`PROCEED_TO_ORACLE_DESIGN`、`HIGH_DENSITY_ONLY` 或 `STOP_WEIGHTING_DIRECTION`；
4. 不打开官方测试集。

## Compute and Data Budget

- **25% 必跑 GPU：** 12 次训练 + 12 次确认评估，预计 4–5 小时。
- **50% 条件 GPU：** 3 次训练 + 3 次确认评估，预计额外 1–1.5 小时。
- **预检 GPU：** 约 0.2–0.4 小时。
- **最大 GPU 夜间预算：** 约 6.5 小时；异常任务超过同类已观测时间 2 倍时停止。
- **数据准备：** 512 训练、1,024 确认、256 合格全链错误、128 冻结备用。
- **Codex 生成预算：** 预计 2.5–5 小时，先于 GPU 夜间运行完成。
- **人工评估：** 不要求外部标注；当前 Codex 进行第二遍可读性复核，但不称为独立裁决。
- **最大瓶颈：** 在长度和编辑距离限制下生成 256 条五步连通、自洽且真正冲突的轨迹。

## Risks and Mitigations

- **风险：Codex 输出集中成少数表面模板。**
  - 缓解：四类意图配额、编辑距离分布报告、12 条小门；不按下游伤害挑类别。
- **风险：强错误链包含无法解释的常数或断裂依赖。**
  - 缓解：程序依赖图验证、明确拒绝码、三次上限和冻结备用顺序。
- **风险：错误链更长导致辅助损失自然增大。**
  - 缓解：逐题 token 长度控制在 90%–110%，报告实际长度比。
- **风险：全链错误组错误答案字段泄漏。**
  - 缓解：题目、答案、ID 字节级哈希；臂间 labels equality 测试。
- **风险：梯度裁剪掩盖名义损失剂量。**
  - 缓解：固定 `lambda_aux=1`，记录裁剪前范数，不做 lambda 剂量主张。
- **风险：确认集被反复用于决策。**
  - 缓解：冻结全新 1,024 题；只允许预注册 25%→50% 单次条件门；官方测试继续封存。
- **风险：三种子不足以支持普遍论文主张。**
  - 缓解：本轮定位为机制可行性门；即使通过，也只进入 oracle 设计，不直接声称自然噪声方法有效。
- **风险：工作区已有未提交改动。**
  - 缓解：所有实现提交只暂存本实验明确文件；不得批量清理或覆盖用户现有改动。

## Must-Run vs Nice-to-Have

### MUST-RUN

- B1 数据与 256 条强错误链审计；
- B2 loss parity、四臂 sanity、梯度通路；
- B3 25% 四组 × 三种子训练与确认；
- B5 机器 gate 和中文报告；
- B4 仅在 25% 失败后成为必须运行。

### NICE-TO-HAVE（不得延迟主门）

- 按四类错误意图展示描述性 EM/NLL 分布；
- 代表性 clean/local/full 三联案例；
- 编辑距离与辅助梯度夹角散点图；
- 逐层梯度热图。

### CUT

- 可靠性头训练；
- oracle 0.1/0 恢复；
- 自然教师噪声；
- 官方测试评估；
- 更多覆盖率或 lambda；
- 根据结果重新生成错误链。

## Planned Commands

以下命令在实现完成后使用，命令名和参数是本计划的接口契约：

```powershell
.\.venv\Scripts\python.exe scripts\run_full_conflict.py prepare --config configs\reliable_simcot\full_conflict.json
.\.venv\Scripts\python.exe scripts\run_full_conflict.py gate-small --config configs\reliable_simcot\full_conflict.json
.\.venv\Scripts\python.exe scripts\run_full_conflict.py validate --config configs\reliable_simcot\full_conflict.json
.\.venv\Scripts\python.exe -m pytest -q tests\test_full_conflict_data.py tests\test_full_conflict_generation.py tests\test_full_conflict_validation.py tests\test_full_conflict_experiment.py tests\test_full_conflict_evaluation.py
.\.venv\Scripts\python.exe scripts\run_full_conflict.py sanity --config configs\reliable_simcot\full_conflict.json
.\.venv\Scripts\python.exe scripts\run_full_conflict.py audit --config configs\reliable_simcot\full_conflict.json
.\.venv\Scripts\python.exe scripts\run_full_conflict_overnight.py --config configs\reliable_simcot\full_conflict.json
```

夜间脚本必须在生成数据已冻结且 M3 全部通过后才允许启动。

## Expected Artifacts

### Work artifacts

- `work/reliable_simcot/full_conflict/eligibility_manifest.json`
- `work/reliable_simcot/full_conflict/split_manifest.json`
- `work/reliable_simcot/full_conflict/prompt_manifest.jsonl`
- `work/reliable_simcot/full_conflict/raw_generations.jsonl`
- `work/reliable_simcot/full_conflict/accepted_chains.jsonl`
- `work/reliable_simcot/full_conflict/frozen_schedule.json`
- `work/reliable_simcot/full_conflict/confirm_1024.txt`

### Output artifacts

- `outputs/reliable_simcot/full_conflict/provenance.json`
- `outputs/reliable_simcot/full_conflict/small_batch_gate.json`
- `outputs/reliable_simcot/full_conflict/data_audit.json`
- `outputs/reliable_simcot/full_conflict/loss_parity.json`
- `outputs/reliable_simcot/full_conflict/sanity_gate.json`
- `outputs/reliable_simcot/full_conflict/gradient_audit.json`
- `outputs/reliable_simcot/full_conflict/analysis_25.json`
- `outputs/reliable_simcot/full_conflict/analysis_50.json`（条件式）
- `outputs/reliable_simcot/full_conflict/overnight_state.json`
- `outputs/reliable_simcot/full_conflict/full_conflict_report.md`
- `outputs/reliable_simcot/full_conflict/MANIFEST.md`

## Final Checklist

- [ ] C1、C2、C3 和反主张 A1 均映射到明确实验块
- [ ] 512/1,024/256/128 四份清单在结果前冻结
- [ ] 12 条小门先于 256 条正式生成
- [ ] 256 条全部通过五步、算术、依赖、冲突、长度和答案完整性规则
- [ ] Local25 与 Full25 使用相同 128 道题
- [ ] `answer_only`、Clean、Local25、Full25 共 12 次训练全部从 checkpoint 28 独立初始化
- [ ] 25% gate 在任何 50% 运行前计算并冻结
- [ ] 50% 仅在 25% 失败后启动
- [ ] 官方测试集未打开
- [ ] 机器 verdict、统计表和中文结论一致
- [ ] nice-to-have 不阻塞主实验
