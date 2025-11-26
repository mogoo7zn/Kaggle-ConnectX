"""
AlphaZero Configuration
Configuration for MCTS + Neural Network self-play training
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.base.config import DQNConfig


class AlphaZeroConfig:
    """Configuration class for AlphaZero hyperparameters"""
    
    # ============== Environment Parameters ==============
    ROWS = 6
    COLUMNS = 7
    INAROW = 4
    
    # ============== MCTS Parameters ==============
    # NUM_SIMULATIONS = 800  # Number of MCTS simulations per move
    NUM_SIMULATIONS = 100
    C_PUCT = 1.5  # Exploration constant for UCB formula
    TEMPERATURE = 1.0  # Temperature for action selection
    TEMP_THRESHOLD = 30  # Move number after which to use greedy policy (temp→0)
    
    # Dirichlet noise for exploration at root
    DIRICHLET_ALPHA = 0.3  # Dirichlet noise parameter (lower = more peaked)
    DIRICHLET_EPSILON = 0.25  # Weight of noise in root prior
    
    # Virtual loss for parallel MCTS
    VIRTUAL_LOSS = 3
    
    # ============== Neural Network Architecture ==============
    # Input: (batch, history_length * 3, 6, 7)
    HISTORY_LENGTH = 8  # Number of recent board states to stack
    INPUT_CHANNELS = HISTORY_LENGTH * 3  # 3 channels per state
    
    # ResNet architecture
    NUM_RES_BLOCKS = 10  # Number of residual blocks (5-20 typical)
    NUM_FILTERS = 128  # Number of filters in conv layers
    
    # Policy head
    POLICY_FILTERS = 32
    
    # Value head  
    VALUE_FILTERS = 32
    VALUE_HIDDEN = 256
    
    # Regularization
    DROPOUT = 0.1  # Light dropout
    L2_REGULARIZATION = 1e-4  # Weight decay
    
    # ============== Training Parameters ==============
    LEARNING_RATE = 0.01  # Higher LR with SGD+momentum
    MOMENTUM = 0.9
    LR_SCHEDULE = "step"  # Learning rate schedule
    LR_MILESTONES = [100000, 200000, 300000]  # Steps to decay LR
    LR_GAMMA = 0.1  # LR decay factor
    
    BATCH_SIZE = 512  # Large batch size for stable training
    TRAINING_EPOCHS = 10  # Epochs per iteration
    
    # ============== Self-Play Parameters ==============
    # NUM_SELFPLAY_GAMES = 500  # Games per iteration
    NUM_SELFPLAY_GAMES = 100
    NUM_PARALLEL_GAMES = 8  # Parallel self-play games
    
    # Replay buffer
    REPLAY_BUFFER_SIZE = 500000  # Keep recent games
    MIN_REPLAY_SIZE = 10000  # Minimum samples before training
    
    # ============== Evaluation Parameters ==============
    EVAL_GAMES = 40  # Games to evaluate new vs old model
    EVAL_WIN_RATE_THRESHOLD = 0.55  # Win rate needed to replace old model
    
    # Arena opponents
    ARENA_OPPONENTS = ["random", "negamax"]
    ARENA_GAMES_PER_OPPONENT = 100
    
    # ============== Training Iterations ==============
    MAX_ITERATIONS = 1000  # Maximum training iterations
    EVAL_INTERVAL = 5  # Evaluate every N iterations
    SAVE_INTERVAL = 10  # Save checkpoint every N iterations
    
    # ============== Data Augmentation ==============
    USE_AUGMENTATION = True  # Horizontal flip for data augmentation
    
    # ============== Saving and Logging ==============
    MODEL_DIR = "alphazero/models"
    CHECKPOINT_DIR = "alphazero/checkpoints"
    LOG_DIR = "alphazero/logs"
    PLOT_DIR = "alphazero/plots"
    SELFPLAY_DIR = "alphazero/selfplay_data"
    
    # ============== Device ==============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============== Rewards ==============
    # Terminal rewards
    REWARD_WIN = 1.0
    REWARD_LOSS = -1.0
    REWARD_DRAW = 0.0
    
    # ============== Mixed Precision Training ==============
    USE_AMP = True  # Automatic Mixed Precision for faster training
    
    # ============== ELO Tracking ==============
    INITIAL_ELO = 1500
    K_FACTOR = 32
    
    def __repr__(self):
        return (f"AlphaZeroConfig(num_sims={self.NUM_SIMULATIONS}, "
                f"c_puct={self.C_PUCT}, res_blocks={self.NUM_RES_BLOCKS}, "
                f"lr={self.LEARNING_RATE})")


# Create a default AlphaZero config instance
az_config = AlphaZeroConfig()


if __name__ == "__main__":
    # Test configuration
    print("AlphaZero Configuration")
    print("=" * 60)
    print(az_config)
    print(f"\nDevice: {az_config.DEVICE}")
    print(f"MCTS simulations: {az_config.NUM_SIMULATIONS}")
    print(f"ResNet blocks: {az_config.NUM_RES_BLOCKS}")
    print(f"Filters: {az_config.NUM_FILTERS}")
    print(f"Self-play games per iteration: {az_config.NUM_SELFPLAY_GAMES}")
    print(f"Batch size: {az_config.BATCH_SIZE}")

