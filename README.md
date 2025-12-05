# 🔴 ConnectX Arena 🟡

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Kaggle](https://img.shields.io/badge/Kaggle-ConnectX-20BEFF)](https://www.kaggle.com/c/connectx)

> **Dual-Agent Reinforcement Learning Framework**: Implementing **Rainbow DQN** and **AlphaZero** to complete the game of Connect 4.

---

## 📖 Table of Contents

- [🔴 ConnectX Arena 🟡](#-connectx-arena-)
  - [📖 Table of Contents](#-table-of-contents)
  - [✨ Introduction](#-introduction)
  - [🚀 Key Features](#-key-features)
    - [🌈 Rainbow DQN](#-rainbow-dqn)
    - [🤖 AlphaZero](#-alphazero)
  - [📦 Installation](#-installation)
  - [⚡ Quick Start](#-quick-start)
    - [🏃 Run a Quick Experiment](#-run-a-quick-experiment)
    - [🏋️ Train Agents](#️-train-agents)
    - [⚔️ Evaluate](#️-evaluate)
  - [📚 Documentation](#-documentation)
  - [🏗️ Project Structure](#️-project-structure)
  - [📊 Performance](#-performance)
  - [📝 License](#-license)

---

## ✨ Introduction

This project provides a playground environment for training and evaluating Reinforcement Learning agents on the ConnectX (Connect 4) game. It features two state-of-the-art implementations:

1.  **🌈 Rainbow DQN**: An advanced Value-Based method combining 6 major DQN improvements.
2.  **🤖 AlphaZero**: A Policy-Based method utilizing Monte Carlo Tree Search (MCTS) and self-play.

The goal is to compare these paradigms and achieve high performance in the Kaggle ConnectX simulation.

---

## 🚀 Key Features

### 🌈 Rainbow DQN

- ✅ **Prioritized Experience Replay**: Learns from significant transitions more frequently.
- ✅ **Dueling Architecture**: Separates state value and advantage estimation.
- ✅ **Noisy Nets**: Enhances exploration through stochastic network layers.
- ✅ **Multi-step Learning**: Uses N-step returns for better convergence.
- ✅ **Double DQN**: Reduces overestimation bias.
- ✅ **Categorical DQN (C51)**: Models value distribution (optional).

### 🤖 AlphaZero

- ✅ **MCTS**: Monte Carlo Tree Search for lookahead planning.
- ✅ **Policy-Value Network**: Dual-headed network for move probability and position evaluation.
- ✅ **Self-Play Training**: Generates training data by playing against itself.
- ✅ **Symmetry Data Augmentation**: Exploits board symmetries to multiply dataset size.

---

## 📦 Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/mogoo7zn/connectX.git
cd connectX
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### 🏃 Run a Quick Experiment

To verify the installation and run a short training session:

```bash
python run_experiment.py --quick
```

### 🏋️ Train Agents

Train the agents with full configuration:

```bash
# Train Rainbow DQN
python -m agents.rainbow.train_rainbow

# Train AlphaZero
python -m agents.alphazero.train_alphazero
```

### ⚔️ Evaluate

Pit the agents against baselines or each other:

```bash
# Run benchmark suite
python -m evaluation.benchmark

# Compare agents
python -m evaluation.compare
```

---

## 📚 Documentation

Detailed documentation is located in the `docs/` directory:

| Document                                              | Description                                        |
| ----------------------------------------------------- | -------------------------------------------------- |
| [**📂 Project Structure**](docs/PROJECT_STRUCTURE.md) | Detailed explanation of the codebase organization. |
| [**🏗️ Architecture**](docs/ARCHITECTURE.md)           | Technical design and implementation details.       |
| [**🚀 Quick Start Guide**](docs/QUICKSTART.md)        | Extended guide for setup and usage.                |
| [**📖 Detailed Docs**](docs/README.md)                | Comprehensive documentation index.                 |

---

## 🏗️ Project Structure

The project is organized into modular components for scalability:

```
connectX/
├── 📂 agents/           # Agent implementations
│   ├── 📂 base/         # Shared components (Config, Utils)
│   ├── 📂 dqn/          # Baseline DQN
│   ├── 📂 rainbow/      # Rainbow DQN
│   └── 📂 alphazero/    # AlphaZero
├── 📂 evaluation/       # Arena & Benchmarking tools
├── 📂 tools/            # Visualization & Submission scripts
├── 📂 outputs/          # Logs, Checkpoints, Models, Plots
├── 📂 docs/             # Documentation
└── 📂 submission/       # Kaggle submission artifacts
```

> See [PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md) for a complete file tree.

---

## 📊 Performance

| Agent           | vs Random | vs Negamax (d=2) | vs Negamax (d=4) |
| --------------- | --------- | ---------------- | ---------------- |
| **Rainbow DQN** | 99.9%     | 95%              | ~50%             |
| **AlphaZero**   | 100%      | 98%              | ~60%             |

_(Performance metrics are approximate and depend on training duration)_

---

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
