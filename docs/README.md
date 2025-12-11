# 📘 ConnectX Detailed Documentation

This document provides details about the ConnectX Dual-Agent Project, including implementation specifics, configuration guides, and troubleshooting.

## 🎯 Project Goals

- **Compare Paradigms**: Contrast Value-Based (Rainbow DQN) vs Policy-Based (AlphaZero) RL.
- **Better Performance**: Train agents that surpass standard Minimax/Negamax baselines.
- **Reusable Framework**: Create a modular system for future board game RL projects.

## 📁 Project Structure

The project follows a modular architecture:

```
connectX/
├── 📂 agents/               # Agent implementations
│   ├── 📂 base/             # Shared utilities (Config, Utils)
│   ├── 📂 dqn/              # Baseline DQN implementation
│   ├── 📂 rainbow/          # Rainbow DQN (6 improvements)
│   └── 📂 alphazero/        # AlphaZero (MCTS + ResNet)
│
├── 📂 evaluation/           # Unified Evaluation Framework
│   ├── arena.py             # Head-to-head match engine
│   ├── benchmark.py         # Standard opponent suite
│   └── compare.py           # Comparison and visualization
│
├── 📂 tools/                # Utility scripts
│   ├── prepare_submission.py # Kaggle submission packager
│   └── visualize.py         # Training visualization
│
├── 📂 outputs/              # Training artifacts
│   ├── checkpoints/         # Model checkpoints
│   ├── logs/                # TensorBoard logs
│   ├── models/              # Final models
│   └── plots/               # Generated charts
│
├── 📂 docs/                 # Documentation
└── 📂 submission/           # Kaggle submission files
```

## 🚀 Extended Quick Start

### 1. Installation

#### Automated Setup (Recommended)

We provide automated scripts for easy environment setup:

**Windows:**
```bash
scripts\setup_env.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

The scripts will automatically:
- Create a Python virtual environment
- Install all dependencies from `requirements.txt`
- Set up the environment ready for use

#### Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows: venv\Scripts\activate.bat
# Linux/Mac: source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Note**: The project requires Python 3.8+ and includes dependencies for:
- Deep Learning (PyTorch)
- Game Interface (Pygame)
- Visualization (Matplotlib)
- Monitoring (TensorBoard)

### 2. Run Full Experiment

```bash
# Full training pipeline
python run_experiment.py

# Quick test mode (fast verification)
python run_experiment.py --quick
```

### 3. Train Individual Agents

**Rainbow DQN:**

```bash
python -m agents.rainbow.train_rainbow
```

**AlphaZero:**

```bash
python -m agents.alphazero.train_alphazero
```

### 4. Evaluate and Compare

```bash
# Run benchmark suite
python -m evaluation.benchmark

# Generate comparison report
python -m evaluation.compare
```

### 5. Prepare Kaggle Submission

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

## 📊 Key Features & Implementation

### 🌈 Rainbow DQN

Rainbow combines six extensions to the original DQN algorithm:

1.  **Double DQN**: Decouples selection from evaluation to reduce overestimation bias.
2.  **Prioritized Experience Replay (PER)**: Samples important transitions more frequently.
3.  **Dueling Networks**: Uses two streams (Value and Advantage) to estimate Q-values.
4.  **Multi-step Learning**: Uses n-step returns to propagate rewards faster.
5.  **Noisy Nets**: Adds parametric noise to weights for better exploration.
6.  **Distributional RL (C51)**: Models the distribution of returns instead of just the mean (Optional).

**Configuration (`agents/rainbow/rainbow_config.py`):**

```python
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
GAMMA = 0.99
PER_ALPHA = 0.6
N_STEP = 3
```

### 🤖 AlphaZero

AlphaZero uses a generalized iteration algorithm:

1.  **MCTS**: Uses Monte Carlo Tree Search for lookahead planning based on the current policy.
2.  **Policy-Value Network**: A residual network that outputs move probabilities ($p$) and position value ($v$).
3.  **Self-Play**: The agent plays against itself to generate training data $(s, \pi, z)$.
4.  **Symmetry**: Exploits the board's horizontal symmetry to double the training data.

**Configuration (`agents/alphazero/az_config.py`):**

```python
NUM_SIMULATIONS = 800
C_PUCT = 1.5
LEARNING_RATE = 0.01
NUM_SELFPLAY_GAMES = 500
```

## 🔬 Evaluation Framework

### Standard Opponents

The benchmark suite tests agents against:

- **Random**: Baseline (ELO ~800)
- **Negamax (Depth 2)**: Weak lookahead (ELO ~1200)
- **Negamax (Depth 4)**: Medium lookahead (ELO ~1400)
- **Negamax (Depth 6)**: Strong lookahead (ELO ~1600)

### Metrics

- **Win Rate**: Percentage of games won.
- **ELO Rating**: Estimated relative skill level.
- **Decision Time**: Average time per move.

## 📈 Monitoring

Use TensorBoard to monitor training progress:

```bash
tensorboard --logdir outputs/logs
```

**Metrics to Watch:**

- **Rainbow**: `loss`, `avg_q_value`, `epsilon` (if not noisy).
- **AlphaZero**: `policy_loss`, `value_loss`, `total_loss`.

## 🐛 Troubleshooting

### Common Issues

**Problem: Training is too slow.**

- **Fix**: Reduce `BATCH_SIZE`, use a GPU, or reduce `NUM_SIMULATIONS` (for AlphaZero).

**Problem: Agent plays invalid moves.**

- **Fix**: Ensure the action mask is correctly applied in the model output.

**Problem: Kaggle Submission Timeout.**

- **Fix**: For AlphaZero, reduce MCTS simulations during inference. For Rainbow, ensure the model isn't too deep.

## 📚 References

- [Rainbow: Combining Improvements in Deep Reinforcement Learning](https://arxiv.org/abs/1710.02298)
- [Mastering the Game of Go without Human Knowledge (AlphaZero)](https://nature.com/articles/nature24270)
