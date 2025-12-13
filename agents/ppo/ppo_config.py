"""
PPO configuration for ConnectX (minimal, single-process).
"""

import torch
import sys
import os

# Make base config importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from agents.base.config import config as base_config


class PPOConfig:
    # Environment
    ROWS = base_config.ROWS
    COLUMNS = base_config.COLUMNS
    INAROW = base_config.INAROW

    # Network
    INPUT_CHANNELS = 3
    CONV_CHANNELS = [64, 128]
    FC_HIDDEN = 256

    # PPO hyperparameters
    LR = 2.5e-4
    BATCH_SIZE = 256
    MINI_BATCHES = 4
    PPO_EPOCHS = 2
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_RANGE = 0.2
    VF_COEF = 0.5
    ENT_COEF = 0.01
    MAX_GRAD_NORM = 0.5

    # Rollout
    ROLLOUT_STEPS = 1024  # per iteration

    # Training
    TOTAL_UPDATES = 500  # keep small for quick runs

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


ppo_config = PPOConfig()


if __name__ == "__main__":
    print("PPO config:")
    print(vars(ppo_config))

