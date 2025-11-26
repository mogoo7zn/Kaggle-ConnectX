# ConnectX Dual-Agent Project

This project implements two state-of-the-art reinforcement learning approaches for the Kaggle ConnectX competition:

1. **Rainbow DQN** - Advanced value-based RL with 6 major improvements
2. **AlphaZero** - MCTS + Deep Neural Networks with self-play

## 🎯 Project Goals

- Compare two fundamentally different RL paradigms
- Achieve superhuman performance on ConnectX
- Provide comprehensive evaluation and analysis tools

## 📁 Project Structure

```
connectX/
├── core/                   # Shared utilities
│   ├── config.py
│   ├── utils.py
│   ├── dqn_agent.py       # Original DQN (baseline)
│   └── replay_buffer.py
│
├── rainbow/               # Rainbow DQN Implementation
│   ├── rainbow_config.py
│   ├── rainbow_model.py   # Dueling + Noisy Nets
│   ├── prioritized_buffer.py
│   ├── rainbow_agent.py
│   └── train_rainbow.py
│
├── alphazero/            # AlphaZero Implementation
│   ├── az_config.py
│   ├── mcts.py           # Monte Carlo Tree Search
│   ├── az_model.py       # Policy-Value Network
│   ├── self_play.py      # Self-play engine
│   └── train_alphazero.py
│
├── evaluation/           # Unified Evaluation Framework
│   ├── arena.py          # Head-to-head matches
│   ├── benchmark.py      # Standard opponent suite
│   └── compare.py        # Comparison and visualization
│
├── experiments/          # Experimental results
├── submission/           # Kaggle submission files
├── tools/                # Utility scripts
└── run_full_experiment.py  # Main experimental pipeline
```

## 🚀 Quick Start

### 1. Installation

```bash
pip install torch numpy matplotlib tensorboard
```

### 2. Run Full Experiment

```bash
# Full training (may take hours/days)
python run_full_experiment.py

# Quick test mode
python run_full_experiment.py --quick
```

### 3. Train Individual Agents

**Rainbow DQN:**
```bash
cd rainbow
python train_rainbow.py
```

**AlphaZero:**
```bash
cd alphazero
python train_alphazero.py
```

### 4. Evaluate and Compare

```bash
# Run benchmark
python -m evaluation.benchmark

# Generate comparison report
python -m evaluation.compare
```

### 5. Prepare Kaggle Submission

```bash
# Rainbow DQN
python tools/prepare_kaggle_submission.py \
    --agent rainbow \
    --model-path rainbow/checkpoints/best_rainbow_*.pth \
    --output submission/rainbow_agent.py

# AlphaZero
python tools/prepare_kaggle_submission.py \
    --agent alphazero \
    --model-path alphazero/checkpoints/best_alphazero_*.pth \
    --output submission/alphazero_agent.py \
    --mcts-sims 100
```

## 📊 Key Features

### Rainbow DQN

✅ **Double DQN** - Reduces overestimation bias  
✅ **Prioritized Experience Replay** - Learns from important transitions  
✅ **Dueling Networks** - Separates value and advantage  
✅ **Multi-step Learning** - Better credit assignment (n=3)  
✅ **Noisy Nets** - Learnable exploration  
✅ **Distributional RL (C51)** - Optional full Rainbow

**Expected Performance:**
- vs Random: 95%+ win rate
- vs Negamax(depth=4): 70%+ win rate
- vs Negamax(depth=6): 50%+ win rate
- Training time: 2-3 days (single GPU)

### AlphaZero

✅ **Monte Carlo Tree Search** - Planning with UCB formula  
✅ **Policy-Value Network** - ResNet-style dual-head  
✅ **Self-Play** - Learns from playing against itself  
✅ **Data Augmentation** - Horizontal board flips  
✅ **Iterative Improvement** - Only keeps better models

**Expected Performance:**
- vs Random: 99%+ win rate
- vs Negamax(depth=6): 80%+ win rate
- vs Negamax(depth=8): 60%+ win rate
- Training time: 5-7 days (single GPU)

## 🔬 Evaluation Framework

### Standard Opponents

- **Random** - Baseline (ELO ~800)
- **Center Preference** - Simple heuristic (ELO ~1000)
- **Negamax Depth 4** - Minimax search (ELO ~1400)
- **Negamax Depth 6** - Stronger search (ELO ~1600)
- **Negamax Depth 8** - Very strong (ELO ~1800)

### Metrics

- Win/Loss/Draw rates
- Average decision time
- Average game length
- Estimated ELO rating

### Outputs

- JSON benchmark results
- Comparison charts (bar, radar, ELO)
- HTML interactive report

## ⚙️ Configuration

### Rainbow DQN (`rainbow/rainbow_config.py`)

```python
# Key hyperparameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 256
GAMMA = 0.99
PER_ALPHA = 0.6
N_STEP = 3
NUM_RES_BLOCKS = 10
```

### AlphaZero (`alphazero/az_config.py`)

```python
# Key hyperparameters
NUM_SIMULATIONS = 800
C_PUCT = 1.5
LEARNING_RATE = 0.01
NUM_SELFPLAY_GAMES = 500
NUM_RES_BLOCKS = 10
```

## 📈 Monitoring Training

### TensorBoard

```bash
# Rainbow
tensorboard --logdir rainbow/logs/runs

# AlphaZero
tensorboard --logdir alphazero/logs/runs
```

### Metrics to Watch

- **Loss**: Should decrease over time
- **Win Rate**: Should increase during evaluation
- **TD Error**: Indicates learning progress
- **Q Values**: Should stabilize
- **ELO Rating**: Tracks overall strength

## 🎮 Testing Agents

### Interactive Play

```python
from agents.rainbow.rainbow_agent import RainbowAgent
from evaluation.arena import Arena

agent = RainbowAgent()
agent.load_model('rainbow/checkpoints/best_rainbow.pth')

# Play against random
arena = Arena()
results = arena.play_match(
    create_agent_wrapper(agent, 'rainbow'),
    random_policy,
    num_games=100
)
```

### Benchmark

```python
from evaluation.benchmark import Benchmark

benchmark = Benchmark()
results = benchmark.run_benchmark(
    agent_fn,
    agent_name="MyAgent",
    games_per_opponent=100
)
```

## 🏆 Expected Results

### Rainbow DQN

| Opponent | Win Rate | ELO |
|----------|----------|-----|
| Random | 95% | ~1500 |
| Center | 85% | ~1500 |
| Negamax-4 | 70% | ~1600 |
| Negamax-6 | 50% | ~1650 |

### AlphaZero

| Opponent | Win Rate | ELO |
|----------|----------|-----|
| Random | 99% | ~1800 |
| Center | 95% | ~1800 |
| Negamax-4 | 90% | ~1850 |
| Negamax-6 | 80% | ~1900 |
| Negamax-8 | 60% | ~1950 |

## 🐛 Troubleshooting

### Training Issues

**Problem: Training very slow**
- Solution: Reduce episodes in config, use GPU, decrease batch size

**Problem: Model not improving**
- Solution: Check learning rate, increase epsilon decay, verify reward function

**Problem: Out of memory**
- Solution: Reduce batch size, decrease model size, use smaller replay buffer

### Submission Issues

**Problem: File too large**
- Solution: Use lighter model architecture, quantize weights

**Problem: Timeout on Kaggle**
- Solution: Reduce MCTS simulations, use faster inference, optimize code

## 📚 References

### Rainbow DQN
- [Rainbow Paper](https://arxiv.org/abs/1710.02298)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Dueling Networks](https://arxiv.org/abs/1511.06581)

### AlphaZero
- [AlphaZero Paper](https://arxiv.org/abs/1712.01815)
- [AlphaGo Zero](https://www.nature.com/articles/nature24270)
- [MCTS Tutorial](https://web.stanford.edu/class/cs234/slides/lecture13.pdf)

## 📝 License

MIT License - See LICENSE file for details

## 🤝 Contributing

Contributions welcome! Please see CONTRIBUTING.md for guidelines.

## 📧 Contact

For questions or issues, please open a GitHub issue.

---

**Happy Training! 🚀**

