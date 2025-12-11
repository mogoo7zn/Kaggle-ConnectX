# 📁 ConnectX Project Structure

## 🎯 Design Principles

1. **Modularity**: Each component has clear responsibilities and is independently testable.
2. **Extensibility**: Easy to add new agent implementations.
3. **Standardization**: Follows Python package management best practices.
4. **Centralization**: Unified organization of outputs and documentation.

## 📂 Complete Directory Structure

```
connectX/
│
├── 📁 agents/                       # All agent implementations
│   ├── __init__.py                  # Package initialization
│   │
│   ├── 📁 base/                     # Shared base components
│   │   ├── __init__.py
│   │   ├── config.py                # Base configuration class
│   │   └── utils.py                 # Common utility functions
│   │
│   ├── 📁 dqn/                      # Basic DQN implementation
│   │   ├── __init__.py
│   │   ├── dqn_model.py             # DQN Neural Network
│   │   ├── dqn_agent.py             # DQN Agent
│   │   ├── replay_buffer.py         # Experience Replay
│   │   └── train_dqn.py             # Training script
│   │
│   ├── 📁 rainbow/                  # Rainbow DQN implementation
│   │   ├── __init__.py
│   │   ├── rainbow_config.py        # Rainbow configuration
│   │   ├── rainbow_model.py         # Dueling + Noisy Nets
│   │   ├── rainbow_agent.py         # Rainbow Agent
│   │   ├── prioritized_buffer.py    # Prioritized Experience Replay
│   │   ├── train_rainbow.py         # Training script
│   │   └── README.md                # Rainbow documentation
│   │
│   └── 📁 alphazero/                # AlphaZero implementation
│       ├── __init__.py
│       ├── az_config.py             # AlphaZero configuration
│       ├── az_model.py              # Policy-Value Network
│       ├── mcts.py                  # MCTS implementation
│       ├── self_play.py             # Self-play engine
│       ├── train_alphazero.py       # Training script
│       └── README.md                # AlphaZero documentation
│
├── 📁 evaluation/                   # Evaluation Framework
│   ├── __init__.py
│   ├── arena.py                     # Match arena
│   ├── benchmark.py                 # Benchmark suite
│   └── compare.py                   # Performance comparison
│
├── 📁 playground/                   # Interactive Game Interface
│   └── play.py                      # PyGame main program
│
├── 📁 scripts/                      # Automation Scripts
│   ├── setup_env.bat                # Windows setup script
│   └── setup_env.sh                 # Linux/Mac setup script
│
├── 📁 tools/                        # Utility Scripts
│   ├── __init__.py
│   ├── prepare_submission.py        # Kaggle submission preparation
│   ├── visualize.py                 # Visualization tools
│   └── README.md
│
├── 📁 outputs/                      # Training Outputs (Unified)
│   ├── __init__.py
│   ├── 📁 checkpoints/             # Training checkpoints
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   ├── 📁 logs/                    # Training logs
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   ├── 📁 models/                  # Final models
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   └── 📁 plots/                   # Training plots
│       ├── dqn/
│       ├── rainbow/
│       └── alphazero/
│
├── 📁 docs/                         # Documentation
│   ├── README.md                    # Detailed documentation
│   ├── QUICKSTART.md                # Quick start guide
│   ├── ARCHITECTURE.md              # Architecture description
│   └── REORGANIZATION.md            # Reorganization plan
│
├── 📁 tests/                        # Test Code
│   ├── __init__.py
│   ├── test_dqn.py                  # DQN tests
│   ├── test_rainbow.py              # Rainbow tests
│   ├── test_alphazero.py            # AlphaZero tests
│   └── test_evaluation.py           # Evaluation tests
│
├── 📁 experiments/                  # Experimental Results
│   ├── .gitkeep
│   └── README.md
│
├── 📁 submission/                   # Kaggle Submission Files
│   ├── dqn_agent.py
│   ├── rainbow_agent.py
│   ├── alphazero_agent.py
│   └── README.md
│
├── 📄 run_experiment.py            # Main Experiment Script
├── 📄 cleanup_old_files.py         # Cleanup Script
├── 📄 requirements.txt             # Dependencies
├── 📄 .gitignore                   # Git Ignore
├── 📄 LICENSE                      # License
├── 📄 README.md                    # Project Main README
├── 📄 REORGANIZATION_COMPLETE.md   # Reorganization Completion Note
└── 📄 PROJECT_STRUCTURE.md         # This File
```

## 🔍 Directory Description

### agents/ - Agent Implementations

**Role**: Contains implementations of all Reinforcement Learning agents.

**Subdirectories**:

- `base/`: Shared base components (config, utils).
- `dqn/`: Basic DQN implementation (baseline).
- `rainbow/`: Rainbow DQN (6 major improvements).
- `alphazero/`: AlphaZero (MCTS + Neural Network).

**Features**:

- Independent directory for each agent.
- Shared components in `base/`.
- Easy to add new agents.

### evaluation/ - Evaluation Framework

**Role**: Unified tool for agent evaluation and comparison.

**Components**:

- `arena.py`: Fair match platform.
- `benchmark.py`: Standardized performance testing.
- `compare.py`: Multi-agent comparison analysis.

**Features**:

- Agent-agnostic evaluation interface.
- Standardized performance metrics.
- Automatic comparison report generation.

### playground/ - Interactive Game Interface

**Role**: Provides a graphical interface to play against AI.

**Components**:

- `play.py`: PyGame-based interactive game program.

**Features**:

- Real-time gameplay.
- Visualized board.
- Supports loading trained models.

**Dependencies**: Requires `pygame` library (included in `requirements.txt`).

### scripts/ - Automation Scripts

**Role**: Provides convenient environment setup and automation tools.

**Components**:

- `setup_env.bat`: Windows environment setup script.
- `setup_env.sh`: Linux/Mac environment setup script.

**Functions**:

- Automatically creates Python virtual environment.
- Checks Python version.
- Installs all project dependencies.
- Provides clear installation feedback.

**Usage**:

```bash
# Windows
scripts\setup_env.bat

# Linux/Mac
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

### tools/ - Utility Scripts

**Role**: Tools for development and deployment assistance.

**Includes**:

- Kaggle submission preparation.
- Training visualization.
- Diagnostic tools.

### outputs/ - Training Outputs

**Role**: Unified management of all files generated during training.

**Structure**: Organized by agent type and output type.

- `checkpoints/`: Training checkpoints.
- `logs/`: TensorBoard logs.
- `models/`: Final trained models.
- `plots/`: Training curve plots.

**Advantages**:

- Centralized management.
- Easy to clean up.
- Convenient for backup.

### docs/ - Documentation

**Role**: Centralized management of all project documentation.

**Includes**:

- User guides.
- API documentation.
- Architecture description.
- Development documentation.

### tests/ - Tests

**Role**: Unit tests and integration tests.

**Organization**: Test files organized by module.

## 🚀 Usage

### Train Agent

```bash
# Using module mode
python -m agents.rainbow.train_rainbow
python -m agents.alphazero.train_alphazero

# Or run directly
python agents/rainbow/train_rainbow.py
python agents/alphazero/train_alphazero.py
```

### Run Full Experiment

```bash
# Quick test
python run_experiment.py --quick

# Full training
python run_experiment.py
```

### Evaluate Performance

```bash
# Benchmark test
python -m evaluation.benchmark

# Generate comparison report
python -m evaluation.compare
```

### Prepare Submission

```bash
python tools/prepare_submission.py \
    --agent rainbow \
    --model-path outputs/models/rainbow/best.pth
```

## 📦 Package Import Examples

```python
# Import base components
from agents.base.config import config
from agents.base.utils import encode_state, get_valid_moves

# Import specific agent
from agents.rainbow.rainbow_agent import RainbowAgent
from agents.alphazero.mcts import MCTS

# Import evaluation tools
from evaluation.arena import Arena
from evaluation.benchmark import Benchmark
```

## 🔄 Adding a New Agent

Standard process for adding a new agent:

```bash
# 1. Create directory
mkdir agents/new_agent

# 2. Create necessary files
touch agents/new_agent/__init__.py
touch agents/new_agent/new_agent_config.py
touch agents/new_agent/new_agent_model.py
touch agents/new_agent/new_agent_agent.py
touch agents/new_agent/train_new_agent.py

# 3. Inherit base components
# In code: from agents.base import config, utils

# 4. Add to evaluation
# Implement standard interface, can be directly evaluated by evaluation framework
```

## 🛠️ Maintenance Guide

### Clean Outputs

```bash
# Clean all training outputs
rm -rf outputs/checkpoints/*
rm -rf outputs/logs/*
rm -rf outputs/plots/*

# Keep latest models
# outputs/models/ recommended to manage manually
```

### Backup Important Files

```bash
# Backup checkpoints
cp -r outputs/checkpoints/ backup/checkpoints_$(date +%Y%m%d)/

# Backup best models
cp -r outputs/models/ backup/models_$(date +%Y%m%d)/
```

### Version Control

```bash
# Track source code only, ignore outputs
git add agents/ evaluation/ tools/ docs/
git add run_experiment.py README.md requirements.txt

# outputs/ should be in .gitignore
```

## 📊 File Statistics

- **Python Files**: ~35
- **Config Files**: 3
- **Doc Files**: 8
- **Test Files**: 4 (To be improved)
- **Total Lines of Code**: ~7,500 lines

## ✅ Quality Check

### Code Style

```bash
# Format using black
black agents/ evaluation/ tools/

# Check using flake8
flake8 agents/ evaluation/ tools/

# Type check using mypy
mypy agents/
```

### Run Tests

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_rainbow.py

# Generate coverage report
pytest --cov=agents tests/
```

## 🎯 Best Practices

1. **Modular Development**: Develop and test each component independently.
2. **Documentation First**: Write docs before code.
3. **Test Driven**: Key features covered by tests.
4. **Version Control**: Use semantic versioning.
5. **Continuous Integration**: Automated testing and deployment.

## 📚 Related Documentation

- [README.md](README.md) - Project Overview
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - Quick Start
- [docs/README.md](docs/README.md) - Detailed Documentation
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture Design
- [REORGANIZATION_COMPLETE.md](REORGANIZATION_COMPLETE.md) - Reorganization Note

---
