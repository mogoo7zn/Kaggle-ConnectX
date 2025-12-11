# AlphaZero Implementation

AlphaZero combines Monte Carlo Tree Search (MCTS) with deep neural networks through self-play reinforcement learning.

## Key Components

1. **MCTS** - Planning algorithm guided by neural network
2. **Policy-Value Network** - Predicts move probabilities and position evaluation
3. **Self-Play** - Generates training data by playing against itself
4. **Iterative Training** - Improves network and replaces old version if stronger

## Files

- `az_config.py` - Configuration presets
- `az_model.py` - ResNet-style policy-value networks
- `mcts.py` - Monte Carlo Tree Search implementation
- `fast_board.py` - Bitboard-based ConnectX board
- `batched_inference.py` - Batched/Sync inference wrappers
- `self_play.py` - Parallel and simple self-play engines
- `train_alphazero.py` - Training loop using unified components

## Quick Start

```python
from alphazero import az_config
from alphazero.az_agent import AlphaZeroAgent

# Create agent
agent = AlphaZeroAgent()

# Train
# python train_alphazero.py
```

