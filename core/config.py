"""
Configuration file for DQN ConnectX Agent
Contains all hyperparameters and training settings
"""

import torch

class DQNConfig:
    """Configuration class for DQN hyperparameters"""
    
    # Environment parameters
    ROWS = 6
    COLUMNS = 7
    INAROW = 4
    
    # Neural network parameters
    INPUT_CHANNELS = 3  # current player, opponent, valid moves
    CONV_CHANNELS = [64, 128, 128]  # convolutional layer channels
    FC_HIDDEN = 256  # fully connected hidden layer size
    OUTPUT_SIZE = 7  # number of possible actions (columns)
    
    # Training parameters
    LEARNING_RATE = 5e-4
    MIN_LEARNING_RATE = 1e-5
    BATCH_SIZE = 128
    GAMMA = 0.99  # discount factor
    TARGET_UPDATE_FREQ = 500  # update target network every N steps

    # Learning rate schedule
    LR_SCHEDULE = "cosine"  # options: "cosine", "step", "none"
    LR_T_MAX = 75000  # steps for cosine annealing
    LR_STEP_INTERVAL = 20000
    LR_STEP_GAMMA = 0.85

    # Exploration parameters
    EPSILON_START = 1.0
    EPSILON_END = 0.01
    EPSILON_DECAY_STEPS = 50000  # decay epsilon over this many steps

    # Memory parameters
    REPLAY_BUFFER_SIZE = 300000
    MIN_REPLAY_SIZE = 5000  # minimum samples before training starts
    TRAINING_STEPS_PER_EPISODE = 4

    # Training episodes
    SELF_PLAY_EPISODES = 5000
    OPPONENT_EPISODES = 3000
    EVAL_INTERVAL = 100  # evaluate every N episodes
    EVAL_GAMES = 50  # number of games for evaluation
    
    # Reward shaping
    REWARD_WIN = 1.0
    REWARD_LOSS = -1.0
    REWARD_DRAW = 0.0
    REWARD_INVALID = -0.5
    REWARD_STEP = 0.0  # small negative reward per step (optional)
    
    # Saving and logging
    SAVE_INTERVAL = 500  # save checkpoint every N episodes
    MODEL_DIR = "models"
    CHECKPOINT_DIR = "training/checkpoints"
    LOG_DIR = "logs"
    PLOT_DIR = "plots"

    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Opponent types for training
    OPPONENTS = ["random", "negamax"]
    VALIDATION_BOTS = ["random", "center", "negamax"]
    TOP_K_SNAPSHOTS = 3
    
    def __repr__(self):
        return f"DQNConfig(lr={self.LEARNING_RATE}, batch={self.BATCH_SIZE}, gamma={self.GAMMA}, device={self.DEVICE})"


# Create a default config instance
config = DQNConfig()

