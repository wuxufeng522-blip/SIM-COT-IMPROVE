# 40% 含噪步骤 Oracle 加权因果实验报告

**日期：** 2026-08-10  
**冻结规格：** `docs/superpowers/specs/2026-08-10-oracle-step-weighting-40pct-causal-test.md`  
**主分析：** `o120_causal_analysis.json`  
**严格判定：** `INCONCLUSIVE_INSUFFICIENT_NOISE_DAMAGE`

## 结论先行

本轮没有证明“正确识别噪声并赋予 0.1 权重”可以提高答案正确率。

原因首先不是检测器不准——本轮直接使用了完美标签；而是 40% 合成步骤污染并没有伤害等权 SIM-CoT。等权含噪组反而比干净组高 `0.99` 个百分点，但差异不显著。因此，实验缺少“先被噪声伤害、再由加权恢复”的必要前提。

在这个前提未成立的情况下，两种 oracle 加权相对等权组的点估计都更低：raw-0.1 低 `0.53` 个百分点，mean-one 归一化 0.1 低 `1.06` 个百分点；两者的 95% 置信区间均跨过 0。这个结果至少说明：在当前污染生成器、训练预算和单一种子下，知道哪些步骤是合成污染并将其降权，并不会自动带来收益。

## 实验完整性

- 2,048 条训练样本与 20% 父实验逐题、逐序完全相同。
- 父实验的第一处污染逐字段继承；每题新增第二处污染。
- 两处污染在每题内使用不同位置和不同污染族。
- 污染率严格为 `2/5 = 40%`。
- 五个污染族总数为 `818–820`；五个位置为 `819–820`；25 个族×位置单元为 `163–164`。
- 61 个不可直接生成的第二污染请求通过成对交换修复，未改变总体分布。
- 日程重复生成哈希完全一致：`9098f67d06920001028c0839c7b2e1b3d10b51d15d31b4fd3a7d755a450ff33b`。
- all-one 自定义损失与官方损失绝对差为 `4.768×10⁻⁷`，低于 `1×10⁻⁴` 门槛。
- 三组正式训练都从 checkpoint 28 的同一哈希开始，使用同一日程、种子、优化器和 256 次更新。
- 三组峰值 reserved 显存均为 `5.527 GiB`，低于 `7.4 GiB` 门槛。
- 三个正式 checkpoint 的文件 SHA-256 均与训练记录匹配。
- 评估使用官方测试集 answer 字段作为 ground truth；四组 1,319 条题目索引和标签逐项配对一致。

技术执行全部 PASS；这不等于研究假设通过。

## 官方测试集结果

| 组别 | 正确数 / 1319 | EM | 相对等权含噪组 |
|---|---:|---:|---:|
| 干净参考（复用 O010） | 535 | 40.561% | −0.986 pp |
| 40% 噪声，等权（O111） | 548 | 41.547% | 基准 |
| 40% 噪声，oracle raw-0.1（O112） | 541 | 41.016% | −0.531 pp |
| 40% 噪声，oracle normalized-0.1（O113） | 534 | 40.485% | −1.061 pp |

raw 权重是三个干净步骤 `1.0`、两个污染步骤 `0.1`，平均权重 `0.64`。normalized 权重将同样的 10:1 相对比例缩放为干净 `1.5625`、污染 `0.15625`，每题平均权重严格为 `1.0`。

## 配对统计

| 比较 | 差值 | 候选胜 / 负 | McNemar p | 配对 bootstrap 95% CI |
|---|---:|---:|---:|---:|
| 等权含噪 − 干净 | +0.986 pp | 76 / 63 | 0.3088 | [−0.758, +2.729] pp |
| raw-0.1 − 等权含噪 | −0.531 pp | 60 / 67 | 0.5946 | [−2.199, +1.137] pp |
| normalized-0.1 − 等权含噪 | −1.061 pp | 56 / 70 | 0.2467 | [−2.729, +0.607] pp |

冻结门槛要求噪声至少造成 `1.0 pp` 损害；实测噪声损害为 `−0.986 pp`，方向相反，因此 recovery ratio 不可定义，严格结果只能判为信息不足。

## 干净验证集 NLL（32 题，描述性）

| 组别 | Answer NLL | Step-token NLL |
|---|---:|---:|
| 干净参考 | 1.9760 | 1.0520 |
| 40% 噪声，等权 | **1.7363** | **1.0089** |
| 40% 噪声，raw-0.1 | 1.9164 | 1.0154 |
| 40% 噪声，normalized-0.1 | 1.9103 | 1.0159 |

小验证集上的 NLL 也没有显示等权污染造成伤害；等权含噪组反而最低。该结果只作机制诊断，不能替代 1,319 题主指标。

## 20% 到 40% 的嵌套剂量审计（探索性）

40% 日程是在 20% 日程上逐题增加第二处污染，因此可以做同题剂量比较：

| 组别 | 20% EM | 40% EM | 40% − 20% | 95% CI | p |
|---|---:|---:|---:|---:|---:|
| 等权 | 39.803% | 41.547% | +1.744 pp | [+0.152, +3.412] | 0.0487 |
| raw-0.1 | 41.092% | 41.016% | −0.076 pp | [−1.744, +1.592] | 1.0000 |
| normalized-0.1 | 41.774% | 40.485% | −1.289 pp | [−2.881, +0.303] | 0.1285 |

这是看到主结果后的探索性分析，未经多重比较校正且只有一个种子，不能据此声称“增加噪声能提高准确率”。它的用途是指出：当前合成污染强度与下游伤害不是单调关系，继续简单提高相同噪声比例缺乏科学依据。

## 为什么 40% “噪声”没有造成伤害

最可能的解释不是标签泄漏，而是当前污染定义混合了“错误”和“低效”：`irrelevant_but_correct` 与 `redundant_repeat` 不一定提供错误算术监督，可能起到额外语言/结构正则化作用。即使是无效步骤，也只污染辅助步骤解码目标；答案监督始终干净，模型可能在 256 次更新内吸收或忽略这些扰动。

另一方面，oracle 权重会同时减弱所有被标为污染的辅助信号。如果其中一部分虽低效却仍提供可学习结构，统一降到 0.1 也可能丢失有益正则化。这解释了为什么“标签判定正确”不等于“降权一定有用”。

## 对研究主线的影响

1. 可靠性头是否能识别污染，和识别后降权是否改善推理，是两个独立问题。本轮只检验第二个问题，而且没有得到正向证据。
2. 在 oracle 机制尚未通过前，不应投入大量算力训练最终可靠性头并宣称下游收益。
3. 也不应直接把相同污染继续提高到 60% 或 80%；当前数据表明数量增加未构成更强的有害处理。
4. 更合理的下一步是先在开发/验证集上构建“有害性校准”污染：保留多污染族，但把真正计算冲突、依赖错序、错误结果向后传播的连贯错误链，与无关/冗余的低效步骤分开报告。先冻结一个能稳定造成至少 2 pp 损害的处理，再在未触碰的测试集上比较 equal、oracle-raw 与 oracle-normalized。

## 复现命令

```powershell
$env:PYTHONPATH='src'
.\.venv\Scripts\python.exe scripts\run_oracle_weighting.py schedule --config configs\reliable_simcot\oracle_weighting_40pct.json
.\.venv\Scripts\python.exe scripts\run_oracle_weighting.py sanity --config configs\reliable_simcot\oracle_weighting_40pct.json
.\.venv\Scripts\python.exe scripts\run_oracle_weighting.py train --config configs\reliable_simcot\oracle_weighting_40pct.json --arm noisy_equal
.\.venv\Scripts\python.exe scripts\run_oracle_weighting.py train --config configs\reliable_simcot\oracle_weighting_40pct.json --arm oracle_raw_0.1
.\.venv\Scripts\python.exe scripts\run_oracle_weighting.py train --config configs\reliable_simcot\oracle_weighting_40pct.json --arm oracle_normalized_0.1
.\.venv\Scripts\python.exe scripts\evaluate_oracle_weighting.py evaluate --config configs\reliable_simcot\oracle_weighting_40pct.json --arm noisy_equal --resume
.\.venv\Scripts\python.exe scripts\evaluate_oracle_weighting.py evaluate --config configs\reliable_simcot\oracle_weighting_40pct.json --arm oracle_raw_0.1 --resume
.\.venv\Scripts\python.exe scripts\evaluate_oracle_weighting.py evaluate --config configs\reliable_simcot\oracle_weighting_40pct.json --arm oracle_normalized_0.1 --resume
.\.venv\Scripts\python.exe scripts\evaluate_oracle_weighting.py analyze --config configs\reliable_simcot\oracle_weighting_40pct.json
```

三组正式训练耗时约 `108.98` 分钟，三组正式评估耗时约 `11.47` 分钟；加上日程和 sanity，总 GPU 时间约 2 小时。
