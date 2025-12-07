"""
Rainbow DQN Configuration
Extends base DQN with advanced features:
- Prioritized Experience Replay (PER)
- Dueling Network Architecture
- Multi-step Learning
- Noisy Nets
- Distributional RL (C51) [Optional]
"""

import torch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.base.config import DQNConfig


class RainbowConfig(DQNConfig):
    """Configuration class for Rainbow DQN hyperparameters"""
    
    # ============== Rainbow Specific Parameters ==============
    
    # Prioritized Experience Replay (PER)
    USE_PER = True
    PER_ALPHA = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
    PER_BETA_START = 0.4  # Initial importance sampling weight
    PER_BETA_END = 1.0  # Final importance sampling weight (remove bias)
    PER_BETA_FRAMES = 100000  # Number of frames to anneal beta
    PER_EPSILON = 1e-6  # Small constant to avoid zero priority
    
    # Multi-step Learning
    USE_MULTI_STEP = True
    N_STEP = 3  # Number of steps for n-step returns
    
    # Noisy Networks
    USE_NOISY_NETS = True
    NOISY_SIGMA_INIT = 0.5  # Initial noise parameter
    
    # Distributional RL (C51)
    USE_DISTRIBUTIONAL = False  # Optional: Can be enabled for full Rainbow
    NUM_ATOMS = 51  # Number of discrete atoms in value distribution
    V_MIN = -10.0  # Minimum value
    V_MAX = 10.0  # Maximum value
    
    # Dueling Architecture (already in base, but emphasized here)
    USE_DUELING = True
    
    # ============== Network Architecture ==============
    # Inherited from DQNConfig but can be overridden
    CONV_CHANNELS = [128, 256, 256]  # Larger network for Rainbow
    FC_HIDDEN = 512
    DROPOUT = 0.2  # Less dropout with Noisy Nets
    
    # ============== Training Parameters ==============
    LEARNING_RATE = 1e-4  # Lower LR for stability
    MIN_LEARNING_RATE = 1e-5
    BATCH_SIZE = 256  # Larger batch for better gradient estimates
    GAMMA = 0.99
    TARGET_UPDATE_FREQ = 1000  # Less frequent updates
    
    # Learning rate schedule
    LR_SCHEDULE = "cosine"
    LR_T_MAX = 100000
    
    # Exploration (less important with Noisy Nets)
    EPSILON_START = 0.5  # Lower initial epsilon
    EPSILON_END = 0.01
    EPSILON_DECAY_STEPS = 30000  # Faster decay
    
    # Memory parameters
    REPLAY_BUFFER_SIZE = 500000  # Larger buffer for more diversity
    MIN_REPLAY_SIZE = 10000  # More samples before training
    TRAINING_STEPS_PER_EPISODE = 4
    GRAD_CLIP_NORM = 10.0
    
    # ============== Training Episodes ==============
    SELF_PLAY_EPISODES = 10000  # More episodes for Rainbow
    OPPONENT_EPISODES = 6000
    EVAL_INTERVAL = 200
    EVAL_GAMES = 100
    
    # ============== Reward Shaping ==============
    # Use same rewards as base DQN
    # (inherited from DQNConfig)
    
    # ============== Saving and Logging ==============
    MODEL_DIR = "agents/rainbow/models"
    CHECKPOINT_DIR = "agents/rainbow/checkpoints"
    LOG_DIR = "agents/rainbow/logs"
    PLOT_DIR = "agents/rainbow/plots"
    SAVE_INTERVAL = 1000
    
    # ============== Validation ==============
    VALIDATION_BOTS = ["random", "center", "negamax"]
    TOP_K_SNAPSHOTS = 5  # Keep more snapshots
    
    def __repr__(self):
        return (f"RainbowConfig(lr={self.LEARNING_RATE}, batch={self.BATCH_SIZE}, "
                f"gamma={self.GAMMA}, per_alpha={self.PER_ALPHA}, n_step={self.N_STEP})")


# Create a default Rainbow config instance
rainbow_config = RainbowConfig()


if __name__ == "__main__":
    # Test configuration
    print("Rainbow DQN Configuration")
    print("=" * 60)
    print(rainbow_config)
    print(f"\nDevice: {rainbow_config.DEVICE}")
    print(f"PER enabled: {rainbow_config.USE_PER}")
    print(f"Multi-step: {rainbow_config.N_STEP}")
    print(f"Noisy Nets: {rainbow_config.USE_NOISY_NETS}")
    print(f"Distributional: {rainbow_config.USE_DISTRIBUTIONAL}")
    print(f"Dueling: {rainbow_config.USE_DUELING}")

