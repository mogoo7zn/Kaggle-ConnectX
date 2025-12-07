# 🚀 Quick Start Guide

Quick start guide for ConnectX Dual-Agent Project

## ⚡ 5-Minute Quick Test

```bash
# 1. Setup environment using automated script (Recommended)
# Windows:
scripts\setup_env.bat

# Linux/Mac:
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# Or install dependencies manually
pip install -r requirements.txt

# 2. Activate virtual environment (if using automated script)
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 3. Quick test (approx. 5-10 minutes)
python run_full_experiment.py --quick

# 4. View results
ls experiments/comparison_*/comparison_report.html
```

## 📖 Detailed Steps

### Step 1: Environment Preparation

#### Method A: Automated Setup (Recommended)

**Windows:**

```bash
# Run automated script
scripts\setup_env.bat
```

**Linux/Mac:**

```bash
# Add execution permission and run
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

The script will automatically:

- Check Python version (requires 3.8+)
- Create virtual environment `venv/`
- Install all dependencies (including PyTorch, NumPy, Matplotlib, Pygame, TensorBoard, etc.)

#### Method B: Manual Setup

```bash
# Check Python version (requires 3.8+)
python --version

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# Install all dependencies
pip install -r requirements.txt

# Optional: CUDA support (GPU acceleration)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Choose Training Scheme

#### Scheme A: Quick Test Mode (Recommended for Beginners)

```bash
# Training time: approx. 10-20 minutes
# Purpose: Verify code works correctly
python run_full_experiment.py --quick
```

#### Scheme B: Train Rainbow DQN Only

```bash
# Training time: hours to 1 day
cd rainbow
python train_rainbow.py
```

#### Scheme C: Train AlphaZero Only

```bash
# Training time: 1-2 days
cd alphazero
python train_alphazero.py
```

#### Scheme D: Full Training (Requires GPU)

```bash
# Training time: 3-7 days
python run_full_experiment.py
```

### Step 3: Monitor Training Progress

```bash
# Start TensorBoard in a new terminal
tensorboard --logdir rainbow/logs/runs --logdir alphazero/logs/runs

# Visit http://localhost:6006
```

Watch the following metrics:

- **Loss**: Should decrease
- **Win Rate**: Should increase
- **Q Values**: Should stabilize
- **ELO Rating**: Should grow

### Step 4: Evaluate Models

```bash
# Run benchmark
python -m evaluation.benchmark

# Or use Python script
python << EOF
from evaluation.benchmark import Benchmark
from agents.rainbow.rainbow_agent import RainbowAgent
from evaluation.arena import create_agent_wrapper

# Load trained model
agent = RainbowAgent()
agent.load_model('rainbow/checkpoints/best_rainbow_*.pth')

# Run benchmark
benchmark = Benchmark()
results = benchmark.run_benchmark(
    create_agent_wrapper(agent, 'rainbow'),
    agent_name="My Rainbow Agent",
    games_per_opponent=50
)
EOF
```

### Step 5: Prepare Kaggle Submission

```bash
# Rainbow DQN Submission
python tools/prepare_kaggle_submission.py \
    --agent rainbow \
    --model-path rainbow/checkpoints/best_rainbow_full_20251125_*.pth \
    --output submission/my_rainbow_agent.py

# AlphaZero Submission
python tools/prepare_kaggle_submission.py \
    --agent alphazero \
    --model-path alphazero/checkpoints/best_alphazero_20251125_*.pth \
    --output submission/my_alphazero_agent.py \
    --mcts-sims 100
```

### Step 6: Local Test Submission File

```python
# Test Rainbow agent
from submission.my_rainbow_agent import agent

# Mock Kaggle observation
class Obs:
    def __init__(self):
        self.board = [0] * 42
        self.mark = 1

obs = Obs()
action = agent(obs, None)
print(f"Agent selected action: {action}")
```

## 🎮 Interactive Play (Optional)

```python
from evaluation.arena import Arena
from agents.rainbow.rainbow_agent import RainbowAgent
from evaluation.benchmark import StandardOpponents

# Load your agent
my_agent = RainbowAgent()
my_agent.load_model('rainbow/checkpoints/best_rainbow.pth')

# Create arena
arena = Arena()

# Play match
results = arena.play_match(
    agent1_fn=lambda b,m: my_agent.select_action(b, m, epsilon=0),
    agent2_fn=StandardOpponents.negamax_depth_4,
    num_games=10,
    agent1_name="My Agent",
    agent2_name="Negamax-4",
    verbose=True
)
```

## 📊 View Results

### 1. TensorBoard Visualization

```bash
tensorboard --logdir experiments/
```

### 2. HTML Report

Open in browser:

```
experiments/comparison_*/comparison_report.html
```

### 3. JSON Data

```python
import json

with open('experiments/rainbow_benchmark.json') as f:
    data = json.load(f)

print(f"Overall win rate: {data['overall']['overall_win_rate']:.1%}")
print(f"Estimated ELO: {data['overall']['estimated_elo']:.0f}")
```

## 🔧 FAQ

### Q: Training is too slow?

**A**: Several solutions:

```bash
# 1. Use quick mode
python run_full_experiment.py --quick

# 2. Reduce training episodes
# Edit rainbow/rainbow_config.py
SELF_PLAY_EPISODES = 1000  # Default 8000

# 3. Reduce MCTS simulations
# Edit alphazero/az_config.py
NUM_SIMULATIONS = 200  # Default 800
```

### Q: Out of memory?

**A**: Reduce buffer size:

```python
# rainbow/rainbow_config.py
REPLAY_BUFFER_SIZE = 100000  # Default 500000
BATCH_SIZE = 128  # Default 256

# alphazero/az_config.py
REPLAY_BUFFER_SIZE = 200000  # Default 500000
BATCH_SIZE = 256  # Default 512
```

### Q: How to use GPU?

**A**: PyTorch automatically detects GPU:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Q: How to load pre-trained model?

**A**:

```python
from agents.rainbow.rainbow_agent import RainbowAgent

agent = RainbowAgent()
agent.load_model('path/to/model.pth')
# Or
agent.load_checkpoint('path/to/checkpoint.pth')
```

### Q: How to resume interrupted training?

**A**:

```python
# Rainbow
from agents.rainbow.rainbow_agent import RainbowAgent
from rainbow.train_rainbow import RainbowTrainer

agent = RainbowAgent()
agent.load_checkpoint('rainbow/checkpoints/rainbow_ep5000.pth')

trainer = RainbowTrainer(agent)
trainer.train(num_episodes=3000)  # Continue training

# AlphaZero
from alphazero.train_alphazero import AlphaZeroTrainer

trainer = AlphaZeroTrainer()
trainer.load_checkpoint('alphazero/checkpoints/alphazero_iter50.pth')
trainer.train(max_iterations=50)  # Continue training
```

## 📝 Next Steps

After completing the quick start, you can:

1. **Tune Hyperparameters**

   - Modify `rainbow/rainbow_config.py`
   - Modify `alphazero/az_config.py`

2. **Experiment with Architectures**

   - Try deeper networks
   - Adjust ResBlock count
   - Test Distributional RL

3. **Add New Opponents**

   - Implement custom strategies
   - Add to benchmark suite

4. **Optimize Performance**

   - Use model quantization
   - Implement batched inference
   - Multi-GPU parallel training

5. **Submit to Kaggle**
   - Prepare submission file
   - Local test
   - Upload and evaluate

## 🎯 Recommended Learning Path

### Beginner

1. Run `--quick` mode to understand the flow
2. Read `DUAL_AGENT_README.md`
3. Study `rainbow/rainbow_agent.py` code
4. Try modifying simple parameters and re-train

### Intermediate

1. Fully train Rainbow DQN
2. Analyze TensorBoard logs
3. Implement custom evaluation metrics
4. Optimize hyperparameters

### Advanced

1. Fully train both agents
2. Implement distributed training
3. Add new RL algorithms
4. Participate in Kaggle competition

## 🆘 Get Help

- 📖 Full Documentation: `DUAL_AGENT_README.md`
- 💡 Implementation Details: `IMPLEMENTATION_SUMMARY.md`
- 🐛 Issue Report: GitHub Issues
- 💬 Discussion: GitHub Discussions

---

Feel free to check docs or open issues if you have questions!
