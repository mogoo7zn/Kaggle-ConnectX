# Rainbow DQN Implementation

Rainbow DQN integrates six independent improvements to DQN:

1. **Double DQN** - Reduces overestimation bias
2. **Prioritized Experience Replay** - Samples important transitions more frequently  
3. **Dueling Networks** - Separates value and advantage streams
4. **Multi-step Learning** - Uses n-step returns for better credit assignment
5. **Noisy Nets** - Learnable exploration through network parameters
6. **Distributional RL (C51)** - Models value distribution instead of expectation

## Files

- `rainbow_config.py` - Configuration parameters
- `prioritized_buffer.py` - Prioritized experience replay buffer
- `rainbow_model.py` - Neural network with Dueling + Noisy layers
- `rainbow_agent.py` - Rainbow DQN agent with all components
- `train_rainbow.py` - Training script

## Quick Start

```python
from rainbow import rainbow_config
from agents.rainbow.rainbow_agent import RainbowAgent

# Create agent
agent = RainbowAgent()

# Train
# python train_rainbow.py
```

