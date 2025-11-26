# AlphaZero Implementation

AlphaZero combines Monte Carlo Tree Search (MCTS) with deep neural networks through self-play reinforcement learning.

## Key Components

1. **MCTS** - Planning algorithm guided by neural network
2. **Policy-Value Network** - Predicts move probabilities and position evaluation
3. **Self-Play** - Generates training data by playing against itself
4. **Iterative Training** - Improves network and replaces old version if stronger

## Files

- `az_config.py` - Configuration parameters
- `mcts.py` - Monte Carlo Tree Search implementation
- `az_model.py` - ResNet-style policy-value network
- `az_agent.py` - AlphaZero agent combining MCTS + network
- `self_play.py` - Self-play data generation engine
- `train_alphazero.py` - Training loop

## Quick Start

```python
from alphazero import az_config
from alphazero.az_agent import AlphaZeroAgent

# Create agent
agent = AlphaZeroAgent()

# Train
# python train_alphazero.py
```

