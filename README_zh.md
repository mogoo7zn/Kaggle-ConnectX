# 🔴 ConnectX 竞技场 🟡

[![English Docs](https://img.shields.io/badge/Docs-English-blue.svg)](README.md)

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-ConnectX-20BEFF)](https://www.kaggle.com/c/connectx)

> **双智能体强化学习框架**: 实现 **Rainbow DQN** 和 **AlphaZero** 的四子棋 (Connect 4) 游戏。

---

## 📖 目录

- [🔴 ConnectX 竞技场 🟡](#-connectx-竞技场-)
  - [📖 目录](#-目录)
  - [✨ 简介](#-简介)
  - [🚀 主要特性](#-主要特性)
    - [🌈 Rainbow DQN](#-rainbow-dqn)
    - [🤖 AlphaZero](#-alphazero)
  - [📦 安装](#-安装)
  - [⚡ 快速开始](#-快速开始)
    - [🏃 运行快速实验](#-运行快速实验)
    - [🏋️ 训练智能体](#️-训练智能体)
    - [⚔️ 评估](#️-评估)
    - [🎮 对弈](#-对弈)
  - [📚 文档](#-文档)
  - [🏗️ 项目结构](#️-项目结构)
  - [📊 性能](#-性能)
  - [📝 许可证](#-许可证)

---

## ✨ 简介

本项目提供了一个用于在 ConnectX (四子棋) 游戏中训练和评估强化学习智能体的实验环境。它包含两种实现：

1.  **🌈 Rainbow DQN**: 基于 DQN 改进的高级价值引导的方法。
2.  **🤖 AlphaZero**: 一种利用蒙特卡洛树搜索 (MCTS) 和自我对弈的基于策略的方法。

---

## 🚀 主要特性

### 🌈 Rainbow DQN

- ✅ **优先经验回放 (Prioritized Experience Replay)**: 更频繁地从重要的转换中学习。
- ✅ **比对架构 (Dueling Architecture)**: 分离状态价值和优势估计。
- ✅ **噪声网络 (Noisy Nets)**: 通过随机网络层增强探索。
- ✅ **多步学习 (Multi-step Learning)**: 使用 N 步回报以获得更好的收敛性。
- ✅ **双重 DQN (Double DQN)**: 减少高估偏差。
- ✅ **分类 DQN (C51)**: 对价值分布进行建模（可选）。

### 🤖 AlphaZero

- ✅ **MCTS**: 用于前瞻规划的蒙特卡洛树搜索。
- ✅ **策略-价值网络 (Policy-Value Network)**: 用于移动概率和位置评估的双头网络。
- ✅ **自我对弈训练 (Self-Play Training)**: 通过与自己对弈生成训练数据。
- ✅ **对称数据增强 (Symmetry Data Augmentation)**: 利用棋盘对称性倍增数据集大小。

---

## 📦 安装

克隆仓库并安装依赖项：

```bash
git clone https://github.com/mogoo7zn/connectX.git
cd connectX
pip install -r requirements.txt
```

---

## ⚡ 快速开始

### 🏃 运行快速实验

验证安装并运行简短的训练会话：

```bash
python run_experiment.py --quick
```

### 🏋️ 训练智能体

使用完整配置训练智能体：

```bash
# 训练 Rainbow DQN
python -m agents.rainbow.train_rainbow

# 训练 AlphaZero
python -m agents.alphazero.train_alphazero
```

### ⚔️ 评估

让智能体与基准或其他智能体对战：

```bash
# 运行基准测试套件
python -m evaluation.benchmark

# 对比智能体
python -m evaluation.compare
```

### 🎮 对弈

与训练好的智能体进行交互式对弈：

```bash
python playground/play.py
```

### 📦 部署

构建 Windows/Mac/Linux 的独立可执行文件：

1.  进入 `deploy/` 目录。
2.  按照 [deploy/README.md](deploy/README.md) 中的说明操作。

---

## 📚 文档

详细文档位于 `docs/` 目录中：

| 文档                                            | 描述                   |
| ----------------------------------------------- | ---------------------- |
| [**📂 项目结构**](docs/PROJECT_STRUCTURE_zh.md) | 代码库组织的详细说明。 |
| [**🏗️ 架构**](docs/ARCHITECTURE_zh.md)          | 技术设计和实现细节。   |
| [**🚀 快速开始指南**](docs/QUICKSTART_zh.md)    | 设置和使用的扩展指南。 |
| [**📖 详细文档**](docs/README_zh.md)            | 综合文档索引。         |

> **English Documentation**:
>
> - [**📂 Project Structure**](docs/PROJECT_STRUCTURE.md)
> - [**🏗️ Architecture**](docs/ARCHITECTURE.md)
> - [**🚀 Quick Start**](docs/QUICKSTART.md)
> - [**📖 Detailed Docs**](docs/README.md)

---

## 🏗️ 项目结构

项目组织成模块化组件以实现可扩展性：

```
connectX/
├── 📂 agents/           # 智能体实现
│   ├── 📂 base/         # 共享组件 (配置, 工具)
│   ├── 📂 dqn/          # 基准 DQN
│   ├── 📂 rainbow/      # Rainbow DQN
│   └── 📂 alphazero/    # AlphaZero
├── 📂 evaluation/       # 竞技场 & 基准测试工具
├── 📂 playground/       # 交互式游戏界面
├── 📂 tools/            # 可视化 & 提交脚本
├── 📂 outputs/          # 日志, 检查点, 模型, 图表
├── 📂 docs/             # 文档
└── 📂 submission/       # Kaggle 提交产物
```

> 查看 [PROJECT_STRUCTURE_zh.md](docs/PROJECT_STRUCTURE_zh.md) 获取完整项目结构。

---

## 📊 性能

| 智能体          | vs 随机 | vs Negamax (d=2) | vs Negamax (d=4) |
| --------------- | ------- | ---------------- | ---------------- |
| **Rainbow DQN** | 99.9%   | 95%              | ~50%             |
| **AlphaZero**   | 100%    | 98%              | ~60%             |

_(性能指标为近似值，取决于训练时长)_

---

## 📝 许可证

本项目基于 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
