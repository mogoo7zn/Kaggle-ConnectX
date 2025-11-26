# ConnectX Dual-Agent Project

双轨强化学习方法实现：Rainbow DQN 和 AlphaZero

## 🚀 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 快速测试
python run_experiment.py --quick

# 完整训练
python run_experiment.py
```

## 📁 项目结构

```
connectX/
├── agents/              # 所有智能体实现
│   ├── base/           # 共享基础组件
│   ├── dqn/            # 基础DQN
│   ├── rainbow/        # Rainbow DQN
│   └── alphazero/      # AlphaZero
├── evaluation/         # 评估框架
├── tools/              # 工具脚本
├── outputs/            # 训练输出
├── docs/               # 文档
└── tests/              # 测试
```

## 📚 文档

- [快速开始](docs/QUICKSTART.md)
- [详细文档](docs/README.md)
- [架构说明](docs/ARCHITECTURE.md)

## 🎯 主要特性

### Rainbow DQN
- ✅ Prioritized Experience Replay
- ✅ Dueling Architecture
- ✅ Noisy Nets
- ✅ Multi-step Learning
- ✅ Double DQN

### AlphaZero
- ✅ Monte Carlo Tree Search
- ✅ Policy-Value Network
- ✅ Self-Play Training
- ✅ Data Augmentation

## 🏃 使用示例

### 训练Rainbow DQN
```bash
python -m agents.rainbow.train_rainbow
```

### 训练AlphaZero
```bash
python -m agents.alphazero.train_alphazero
```

### 评估Agent
```bash
python -m evaluation.benchmark
```

## 📊 预期性能

- **Rainbow DQN**: vs Negamax(depth=6) ~50% 胜率
- **AlphaZero**: vs Negamax(depth=8) ~60% 胜率

## 📝 许可证

MIT License
