# RSR-RD Weighted SIM-CoT 一晚概念验证实施计划

## 目标

在 Windows + RTX 4060 Laptop 8 GB 环境中，实现并运行已批准的受控微型端到端实验，比较等权与离线 RSR-RD 加权的 SIM-CoT 训练。

## 实施步骤

### 1. 建立最小 CUDA 环境

- 创建工作区本地 `.venv`。
- 安装 CUDA 版 PyTorch 2.5.1、Transformers 4.46.2、NumPy、scikit-learn、tqdm 和 pytest。
- 验证 `torch.cuda.is_available()`、设备名称、CUDA 版本和一次小张量运算。
- 不安装官方仓库中 Linux 专用的 NCCL/Triton 全量依赖。

验收：Python 能在 RTX 4060 上执行 CUDA 运算。

### 2. 实现确定性数据管线

文件：

- `src/rsr_rd_simcot/data.py`
- `tests/test_data.py`

实现：

- 生成 2–4 步、整数中间结果的算术问题。
- 生成 1,200/200/200 划分。
- 对 20% 训练样本注入单步数值、无关或重复噪声。
- 保留干净步骤、观测步骤、噪声位置与类型。
- 输出 JSONL、数据摘要和 SHA-256。

验收：固定种子下数据完全可复现；验证/测试无噪声；训练噪声率为 20%。

### 3. 实现微型 SIM-CoT 模型

文件：

- `src/rsr_rd_simcot/model.py`
- `tests/test_model.py`

实现：

- 加载 GPT-2 Small Student 与辅助步骤解码器。
- 从问题前缀依次构造最多 4 个连续 latent。
- 使用 latent 前缀计算最终答案损失。
- 将各 latent 与对应显式步骤输入辅助解码器。
- 返回答案损失、逐步骤平均 token loss、latent 和组合损失。
- 支持等权与外部离线权重，并正确 mask padding 步骤。

验收：CPU 小配置形状测试通过；权重全 1 时与等权 reduction 一致；梯度能回传到 Student。

### 4. 实现预热、评分与权重

文件：

- `src/rsr_rd_simcot/scoring.py`
- `src/rsr_rd_simcot/training.py`
- `tests/test_scoring.py`

实现：

- 在干净显式 CoT 上预热 Student 400 updates。
- 从预热 checkpoint 冻结评分模型。
- 计算步骤级 rank-clipped RSR。
- 计算加入步骤前后的答案 NLL 与增量 RD。
- 使用 median/MAD、指数变换和样本内归一化生成权重。
- 检查有限性、均值为 1 和噪声检测 ROC-AUC。

验收：手工构造分数的方向测试通过；权重无 NaN/Inf；样本内均值误差小于 `1e-5`。

### 5. 实现公平的双分支训练

文件：

- `src/rsr_rd_simcot/config.py`
- `src/rsr_rd_simcot/training.py`
- `run_experiment.py`

实现：

- 从同一预热 checkpoint 初始化等权和加权分支。
- 重置相同随机种子与 DataLoader 顺序。
- FP16、batch 1、梯度累积 8、checkpointing、梯度裁剪。
- 每 200 updates 保存 checkpoint 与可恢复状态。
- 记录 JSONL 日志、显存、吞吐和运行时间。

验收：20-step preflight 两组均通过，峰值保留显存不超过 7.4 GB。

### 6. 运行评测和生成报告

文件：

- `src/rsr_rd_simcot/evaluation.py`
- `src/rsr_rd_simcot/reporting.py`

实现：

- 在干净测试集上计算答案 Exact Match、步骤 NLL/PPL、token accuracy、sequence EM 和 latent 距离。
- 计算污染步骤检测 ROC-AUC。
- 根据预注册规则生成“工程可行/初步有效/未达到判据”结论。
- 将配置、哈希、日志、指标和一页中文摘要写入 `outputs/overnight_poc/`。

验收：报告中的每项结论均能追溯到保存的 JSON/CSV 指标。

### 7. 执行策略

- 先运行单元测试和极小数据 smoke test。
- 再运行两组各 20-step preflight。
- 只有 preflight 全部通过才启动整夜任务。
- 达到 10 小时硬停止时，两组保持相同有效更新步数。
- 若环境或显存门槛失败，停止并形成阻塞报告，不伪造训练结果。

