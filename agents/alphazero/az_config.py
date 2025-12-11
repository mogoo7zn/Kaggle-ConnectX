"""
AlphaZero Configuration
Unified configuration for AlphaZero agent

Key Features:
1. Larger network (more residual blocks and filters)
2. More MCTS simulations
3. More self-play games
4. Longer training iterations
5. Optimized learning rate schedule
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class AlphaZeroConfig:
    """
    AlphaZero Configuration
    
    Design Goals:
    - Train a strong Connect Four AI
    - Trade some training speed for model quality when GPU is available
    - Sufficient MCTS search
    """
    
    # ============== Environment Parameters ==============
    ROWS = 6
    COLUMNS = 7
    INAROW = 4
    
    # ============== MCTS Parameters ==============
    # Key: More simulations = Stronger policy
    NUM_SIMULATIONS = 400       # Simulations during training
    NUM_SIMULATIONS_EVAL = 800  # Simulations during evaluation
    
    # Adaptive simulations
    SIMS_EARLY_GAME = 400   # Early game (first 6 moves) - Critical decisions
    SIMS_MID_GAME = 400     # Mid game (6-20 moves) - Standard play
    SIMS_LATE_GAME = 200    # Late game (after 20 moves) - Simpler positions
    
    C_PUCT = 1.5            # Exploration constant
    TEMPERATURE = 1.0       # Action selection temperature
    TEMP_THRESHOLD = 15     # Use greedy policy after 15 moves
    
    # Dirichlet noise (Exploration)
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25  # Noise weight
    
    # Virtual loss (Parallel MCTS)
    VIRTUAL_LOSS = 3
    
    # ============== Neural Network Architecture ==============
    HISTORY_LENGTH = 1      # ConnectX doesn't need history
    INPUT_CHANNELS = 3      # Player, Opponent, Valid moves
    
    # ResNet Architecture
    NUM_RES_BLOCKS = 8      # Number of residual blocks
    NUM_FILTERS = 128       # Number of filters
    
    # Head Networks
    POLICY_FILTERS = 32     # Policy head filters
    VALUE_FILTERS = 32      # Value head filters
    VALUE_HIDDEN = 128      # Value head hidden layer
    
    # Regularization
    DROPOUT = 0.1
    L2_REGULARIZATION = 1e-4
    
    # ============== Training Parameters ==============
    LEARNING_RATE = 0.01    # Initial learning rate (SGD)
    LR_MIN = 0.0001         # Minimum learning rate
    MOMENTUM = 0.9
    LR_SCHEDULE = "cosine"  # Cosine annealing
    
    BATCH_SIZE = 512        # Batch size
    TRAINING_EPOCHS = 10    # Epochs per iteration
    
    # ============== Self-Play Parameters ==============
    NUM_SELFPLAY_GAMES = 200    # Self-play games per iteration
    NUM_PARALLEL_GAMES = 16     # Parallel games
    
    # Replay Buffer
    REPLAY_BUFFER_SIZE = 500000  # Buffer size
    MIN_REPLAY_SIZE = 10000      # Minimum samples before training
    
    # ============== Batched Inference Parameters ==============
    MAX_BATCH_SIZE = 256
    MAX_WAIT_MS = 5.0
    USE_BATCHED_INFERENCE = True
    
    # ============== Evaluation Parameters ==============
    EVAL_GAMES = 50             # Evaluation games
    EVAL_WIN_RATE_THRESHOLD = 0.55  # Win rate threshold to replace model
    
    ARENA_OPPONENTS = ["random", "negamax"]
    ARENA_GAMES_PER_OPPONENT = 100
    
    # ============== Training Iterations ==============
    MAX_ITERATIONS = 500        # Maximum iterations
    EVAL_INTERVAL = 5           # Evaluation interval
    SAVE_INTERVAL = 10          # Save interval
    
    # ============== Data Augmentation ==============
    USE_AUGMENTATION = True     # Horizontal flip augmentation
    
    # ============== Paths ==============
    MODEL_DIR = "agents/alphazero/models"
    CHECKPOINT_DIR = "agents/alphazero/checkpoints"
    LOG_DIR = "agents/alphazero/logs"
    PLOT_DIR = "agents/alphazero/plots"
    SELFPLAY_DIR = "agents/alphazero/selfplay_data"
    
    # ============== Device ==============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============== Mixed Precision ==============
    USE_AMP = True
    
    # ============== ELO Tracking ==============
    INITIAL_ELO = 1500
    K_FACTOR = 32
    
    # ============== Performance Tuning ==============
    PIN_MEMORY = True
    NUM_WORKERS = 4
    USE_TORCH_COMPILE = False
    
    def get_adaptive_simulations(self, move_count: int) -> int:
        """Get adaptive simulation count based on game phase"""
        if move_count < 6:
            return self.SIMS_EARLY_GAME
        elif move_count < 20:
            return self.SIMS_MID_GAME
        else:
            return self.SIMS_LATE_GAME
    
    def __repr__(self):
        return (f"AlphaZeroConfig(sims={self.NUM_SIMULATIONS}, "
                f"res_blocks={self.NUM_RES_BLOCKS}, filters={self.NUM_FILTERS})")


# Preset: Ultra Strong (Requires powerful GPU)
class UltraConfig(AlphaZeroConfig):
    """Ultra strong configuration (Requires powerful GPU)"""
    NUM_SIMULATIONS = 800
    NUM_SIMULATIONS_EVAL = 1600
    NUM_RES_BLOCKS = 12
    NUM_FILTERS = 256
    NUM_SELFPLAY_GAMES = 500
    BATCH_SIZE = 1024
    MAX_ITERATIONS = 1000


# Preset: Strong+ (Recommended for Kaggle submission)
class StrongPlusConfig(AlphaZeroConfig):
    """
    Strong+ Configuration (Recommended for Kaggle submission)
    
    Design Goals:
    - Stronger training quality than standard config
    - Model file size < 100MB for Kaggle submission
    - Compensates for network size with more MCTS simulations
    """
    # Network Architecture - Balance size and capability
    NUM_RES_BLOCKS = 10         # +2 blocks vs standard
    NUM_FILTERS = 160           # +32 filters vs standard
    
    # Head Networks
    POLICY_FILTERS = 48
    VALUE_FILTERS = 48
    VALUE_HIDDEN = 192
    
    # MCTS Parameters
    NUM_SIMULATIONS = 600
    NUM_SIMULATIONS_EVAL = 1200
    SIMS_EARLY_GAME = 600
    SIMS_MID_GAME = 600
    SIMS_LATE_GAME = 300
    
    # Self-Play
    NUM_SELFPLAY_GAMES = 300
    NUM_PARALLEL_GAMES = 24
    
    # Training
    BATCH_SIZE = 768
    TRAINING_EPOCHS = 15
    MAX_ITERATIONS = 600
    
    # Buffer
    REPLAY_BUFFER_SIZE = 800000
    
    def __repr__(self):
        return (f"StrongPlusConfig(sims={self.NUM_SIMULATIONS}, "
                f"res_blocks={self.NUM_RES_BLOCKS}, filters={self.NUM_FILTERS}, "
                f"<100MB for Kaggle submission)")


# Preset: Balanced (Moderate resources)
class BalancedConfig(AlphaZeroConfig):
    """Balanced configuration"""
    NUM_SIMULATIONS = 200
    NUM_SIMULATIONS_EVAL = 400
    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 128
    NUM_SELFPLAY_GAMES = 150
    BATCH_SIZE = 256


# Preset: Fast (Fast training)
class FastConfig(AlphaZeroConfig):
    """Fast configuration"""
    NUM_SIMULATIONS = 150
    NUM_SIMULATIONS_EVAL = 300
    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 96
    NUM_SELFPLAY_GAMES = 100
    BATCH_SIZE = 256
    MAX_ITERATIONS = 300


# Create default configuration instance
az_config = AlphaZeroConfig()


def print_config_comparison():
    """Print configuration comparison"""
    
    print("=" * 70)
    print("AlphaZero Configuration Comparison")
    print("=" * 70)
    
    configs = {
        "Standard (Recommended)": az_config,
        "Strong+ (Submission)": StrongPlusConfig(),
        "Balanced": BalancedConfig(),
        "Fast": FastConfig(),
        "Ultra": UltraConfig(),
    }
    
    print(f"\n{'Name':<25} {'Sims':<12} {'Blocks':<10} {'Filters':<10} {'SelfPlay':<10}")
    print("-" * 70)
    
    for name, cfg in configs.items():
        print(f"{name:<25} {cfg.NUM_SIMULATIONS:<12} {cfg.NUM_RES_BLOCKS:<10} "
              f"{cfg.NUM_FILTERS:<10} {cfg.NUM_SELFPLAY_GAMES:<10}")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    print_config_comparison()
