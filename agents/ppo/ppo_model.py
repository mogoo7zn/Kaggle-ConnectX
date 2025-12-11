"""
Actor-Critic model for PPO on ConnectX.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.ppo.ppo_config import ppo_config


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        c = ppo_config.CONV_CHANNELS
        self.conv1 = nn.Conv2d(ppo_config.INPUT_CHANNELS, c[0], 3, padding=1)
        self.conv2 = nn.Conv2d(c[0], c[1], 3, padding=1)
        conv_out = ppo_config.ROWS * ppo_config.COLUMNS * c[1]

        self.fc = nn.Linear(conv_out, ppo_config.FC_HIDDEN)

        # Policy and value heads
        self.policy = nn.Linear(ppo_config.FC_HIDDEN, ppo_config.COLUMNS)
        self.value = nn.Linear(ppo_config.FC_HIDDEN, 1)

    def forward(self, x):
        # x: (B, 3, 6, 7)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        logits = self.policy(x)
        value = self.value(x)
        return logits, value


def make_model():
    model = ActorCritic().to(ppo_config.DEVICE)
    return model


if __name__ == "__main__":
    model = make_model()
    total_params = sum(p.numel() for p in model.parameters())
    print("ActorCritic params:", total_params)

