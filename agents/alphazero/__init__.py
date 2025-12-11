"""AlphaZero public interface (unified version)."""

from agents.alphazero.az_config import (
    az_config,
    AlphaZeroConfig,
    BalancedConfig,
    FastConfig,
    StrongPlusConfig,
    UltraConfig,
)
from agents.alphazero.az_model import (
    create_alphazero_model,
    PolicyValueNetwork,
    DualHeadNetwork,
)
from agents.alphazero.mcts import MCTS

__all__ = [
    'MCTS',
    'az_config',
    'AlphaZeroConfig',
    'BalancedConfig',
    'FastConfig',
    'StrongPlusConfig',
    'UltraConfig',
    'create_alphazero_model',
    'PolicyValueNetwork',
    'DualHeadNetwork',
]
