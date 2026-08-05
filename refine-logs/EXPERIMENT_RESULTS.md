# M0 实验结果：官方 SIM-CoT 复现与 RTX 4060 单卡预检

日期：2026-08-05

硬件：NVIDIA GeForce RTX 4060 Laptop GPU，8188 MiB
结论：**R001–R005 全部 PASS，允许进入 R010–R012 本地短预算基线阶段。**

## 1. 官方 checkpoint 四任务复现

| 任务 | 样本数 | 官方报告 | 本地复现 | 差值 | ±1.0 pp 门槛 |
| --- | ---: | ---: | ---: | ---: | --- |
| GSM8K-Aug | 1319 | 44.8% | 44.43%（586） | -0.37 pp | PASS |
| GSM-Hard | 1319 | 9.3% | 9.48%（125） | +0.18 pp | PASS |
| MultiArith | 600 | 90.8% | 89.83%（539） | -0.97 pp | PASS，接近下界 |
| SVAMP | 1000 | 40.7% | 40.60%（406） | -0.10 pp | PASS |

四个任务都使用数据集答案字段作为真值；OOD 数值题使用发布版 `CODI/test.py` 的最后数值抽取规则。逐题预测均已保存。

## 2. 单卡训练与恢复预检

R004 使用最长可训练样本、BF16 autocast、FP32 参数、batch 1、梯度累积 8，运行 20 次 optimizer update：

- 前 5 次更新平均 loss：0.782606；后 5 次：0.002383；
- 最终固定 probe loss：`2.320607e-6`；
- latent token 10 个，分为 5 组，步骤监督对齐通过；
- PyTorch 峰值 reserved 显存：5.52 GB，低于 7.4 GB 红线；
- 用时 276.19 秒，重复单样本吞吐约 260.69 updates/hour；
- checkpoint SHA-256：`b1e02e0f55b5b5b23447b46e89059eb2433c9617c6a1a23466dad8908568efdc`；
- 保存后重载 probe loss 绝对差：0。

R005 从该 checkpoint 重载后真实执行 1 次更新，再次保存和重载：

- 更新前 probe loss：`2.320607e-6`；更新后：`2.940496e-7`；
- 二次重载绝对差：0；
- 新 checkpoint SHA 与原 checkpoint 不同，证明参数发生更新；
- 峰值 reserved 显存：5.14 GB；完整探针用时 36.00 秒。

这证明 8GB 显卡能够执行本项目所需的 batch-1 SIM-CoT 辅助监督路径，但重复单样本 smoke 的吞吐不能直接当作全数据训练时长。R010/R011 必须通过真实短训练重新估时。

## 3. 复现审计与勘误

1. 早期规格误读了官方表格：`12.2` 是平均生成 token 数。正确的三个 OOD 目标是 GSM-Hard 9.3%、MultiArith 90.8%、SVAMP 40.7%。规格与台账已勘误。
2. 作者代码引用的私有格式化 OOD JSON 未发布。复现采用源码注释指定的公开数据源和 revision；MultiArith 需合并公开的 180 test 与 420 train 才对应论文的 600 题主表。只跑 180 test 会得到 92.78%，不能与 90.8% 主表直接比较。
3. 发布版 `Coconut/run.py` 虽导入 `get_cot_with_explainable_latent_dataset`，训练处却调用普通 `get_cot_latent_dataset`。R004/R005 按论文与配置意图启用辅助步骤监督；正式实验必须持续记录这一实现差异。
4. 官方入口依赖 NCCL、DDP/FSDP，在 Windows 单卡环境无法原样启动。本地 adapter 复用了官方 `coconut.py`、tokenizer、latent token 构造、checkpoint 和答案规则，并用完整官方数值反向核验一致性。
5. 17 个单元测试全部通过。仅有 pytest 无法写 `.pytest_cache` 的 Windows 权限警告，不影响测试与运行结果。

## 4. M0 判定

M0 的证据门槛全部满足：四任务误差均在 ±1.0 pp、20-step loss 有限且下降、步骤对齐、显存低于 7.4 GB、checkpoint 可恢复并继续训练。

下一阶段应先冻结 R010/R011 的相同更新数、样本顺序、optimizer、有效 batch 和 OOM 回退，再比较本地 Coconut 与标准 SIM-CoT。当前结果只证明“官方 checkpoint 与单卡工程路径可信”，还没有证明新可靠性加权方法有效。
