"""
Train PPO agent with simple opponent pool (random + negamax).
This is a minimal, single-process trainer for ConnectX.
"""

import random
import numpy as np
import torch
from collections import deque

from agents.ppo.ppo_agent import PPOAgent
from agents.ppo.ppo_config import ppo_config
from agents.base.utils import get_negamax_move, get_valid_moves


def random_policy(board, mark):
    moves = get_valid_moves(board)
    return random.choice(moves) if moves else 0


def negamax_policy(board, mark):
    return get_negamax_move(board, mark, depth=3)


def train():
    agent = PPOAgent()
    opponents = [random_policy, negamax_policy]
    reward_log = deque(maxlen=50)

    for update in range(1, ppo_config.TOTAL_UPDATES + 1):
        opponent_fn = random.choice(opponents)
        batch = agent.generate_rollout(opponent_fn, ppo_config.ROLLOUT_STEPS)
        loss = agent.update(batch)

        # simple reward proxy: mean of returns
        reward_log.append(batch.returns.mean().item())

        if update % 10 == 0:
            avg_rew = np.mean(reward_log) if reward_log else 0.0
            print(f"Update {update}/{ppo_config.TOTAL_UPDATES} | loss {loss:.3f} | avg_ret {avg_rew:.3f}")

    # save model
    torch.save(agent.model.state_dict(), "ppo_model.pth")
    print("Saved ppo_model.pth")

    return agent


if __name__ == "__main__":
    train()

