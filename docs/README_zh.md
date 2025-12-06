# 📘 ConnectX 详细文档

本文档提供了关于 ConnectX 双智能体项目的详细信息，包括实现细节、配置指南和故障排除。

## 🎯 项目目标

- **对比范式**: 对比基于价值 (Rainbow DQN) 与基于策略 (AlphaZero) 的强化学习。
- **超人类表现**: 训练出超越标准 Minimax/Negamax 基准的智能体。
- **可复用框架**: 为未来的棋盘游戏 RL 项目创建一个模块化系统。

## 📁 项目结构

项目遵循模块化架构：

```
connectX/
├── 📂 agents/               # 智能体实现
│   ├── 📂 base/             # 共享工具 (配置, 工具)
│   ├── 📂 dqn/              # 基准 DQN 实现
│   ├── 📂 rainbow/          # Rainbow DQN (6 项改进)
│   └── 📂 alphazero/        # AlphaZero (MCTS + ResNet)
│
├── 📂 evaluation/           # 统一评估框架
│   ├── arena.py             # 头对头比赛引擎
│   ├── benchmark.py         # 标准对手套件
│   └── compare.py           # 对比和可视化
│
├── 📂 tools/                # 工具脚本
│   ├── prepare_submission.py # Kaggle 提交打包器
│   └── visualize.py         # 训练可视化
│
├── 📂 outputs/              # 训练产物
│   ├── checkpoints/         # 模型检查点
│   ├── logs/                # TensorBoard 日志
│   ├── models/              # 最终模型
│   └── plots/               # 生成的图表
│
├── 📂 docs/                 # 文档
└── 📂 submission/           # Kaggle 提交文件
```

## 🚀 扩展快速开始

### 1. 安装

```bash
pip install -r requirements.txt
```

### 2. 运行完整实验

```bash
# 完整训练流程
python run_experiment.py

# 快速测试模式 (快速验证)
python run_experiment.py --quick
```

### 3. 训练单个智能体

**Rainbow DQN:**

```bash
python -m agents.rainbow.train_rainbow
```

**AlphaZero:**

```bash
python -m agents.alphazero.train_alphazero
```

### 4. 评估和对比

```bash
# 运行基准测试套件
python -m evaluation.benchmark

# 生成对比报告
python -m evaluation.compare
```

### 5. 准备 Kaggle 提交

```bash
# Rainbow DQN
python tools/prepare_submission.py \
    --agent rainbow \
    --model-path outputs/models/rainbow/best_model.pth

# AlphaZero
python tools/prepare_submission.py \
    --agent alphazero \
    --model-path outputs/models/alphazero/best_model.pth
```

## 📊 主要特性与实现

### 🌈 Rainbow DQN

Rainbow 结合了原始 DQN 算法的六项扩展：

1.  **Double DQN**: 解耦选择与评估以减少高估偏差。
2.  **Prioritized Experience Replay (PER)**: 更频繁地采样重要的转换。
3.  **Dueling Networks**: 使用两个流（价值和优势）来估计 Q 值。
4.  **Multi-step Learning**: 使用 n 步回报更快地传播奖励。
5.  **Noisy Nets**: 向权重添加参数化噪声以获得更好的探索。
6.  **Distributional RL (C51)**: 对回报分布进行建模，而不仅仅是均值（可选）。

**配置 (`agents/rainbow/rainbow_config.py`):**

```python
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
GAMMA = 0.99
PER_ALPHA = 0.6
N_STEP = 3
```

### 🤖 AlphaZero

AlphaZero 使用一种广义迭代算法：

1.  **MCTS**: 基于当前策略使用蒙特卡洛树搜索进行前瞻规划。
2.  **Policy-Value Network**: 一个输出移动概率 ($p$) 和位置价值 ($v$) 的残差网络。
3.  **Self-Play**: 智能体与自己对弈以生成训练数据 $(s, \pi, z)$。
4.  **Symmetry**: 利用棋盘的水平对称性使训练数据加倍。

**配置 (`agents/alphazero/az_config.py`):**

```python
NUM_SIMULATIONS = 800
C_PUCT = 1.5
LEARNING_RATE = 0.01
NUM_SELFPLAY_GAMES = 500
```

## 🔬 评估框架

### 标准对手

基准测试套件针对以下对手测试智能体：

- **Random**: 基准 (ELO ~800)
- **Negamax (Depth 2)**: 弱前瞻 (ELO ~1200)
- **Negamax (Depth 4)**: 中等前瞻 (ELO ~1400)
- **Negamax (Depth 6)**: 强前瞻 (ELO ~1600)

### 指标

- **胜率 (Win Rate)**: 赢得比赛的百分比。
- **ELO 评分 (ELO Rating)**: 估计的相对技能水平。
- **决策时间 (Decision Time)**: 每次移动的平均时间。

## 📈 监控

使用 TensorBoard 监控训练进度：

```bash
tensorboard --logdir outputs/logs
```

**关注指标:**

- **Rainbow**: `loss`, `avg_q_value`, `epsilon` (如果不是 noisy net)。
- **AlphaZero**: `policy_loss`, `value_loss`, `total_loss`。

## 🐛 故障排除

### 常见问题

**问题: 训练太慢。**

- **修复**: 减小 `BATCH_SIZE`，使用 GPU，或减少 `NUM_SIMULATIONS` (对于 AlphaZero)。

**问题: 智能体下出无效移动。**

- **修复**: 确保在模型输出中正确应用了动作掩码。

**问题: Kaggle 提交超时。**

- **修复**: 对于 AlphaZero，减少推理时的 MCTS 模拟次数。对于 Rainbow，确保模型不要太深。

## 📚 参考资料

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- [Mastering the Game of Go without Human Knowledge (AlphaZero)](https://nature.com/articles/nature24270)
