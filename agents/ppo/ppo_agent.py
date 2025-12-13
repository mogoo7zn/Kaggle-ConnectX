"""
Minimal PPO agent for ConnectX.
Single-process, on-policy rollouts with masking invalid actions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

from agents.ppo.ppo_config import ppo_config
from agents.ppo.ppo_model import make_model
from agents.base.utils import encode_state, get_valid_moves, make_move, is_terminal


@dataclass
class RolloutBatch:
    states: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    values: torch.Tensor
    masks: torch.Tensor


class PPOAgent:
    def __init__(self):
        self.model = make_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=ppo_config.LR)

    # -------- acting -------- #
    def select_action(self, board: List[int], mark: int) -> Tuple[int, float, float]:
        state = encode_state(board, mark)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(ppo_config.DEVICE)
        logits, value = self.model(state_t)

        valid_moves = get_valid_moves(board)
        mask = torch.full_like(logits, float("-inf"))
        mask[0, valid_moves] = 0
        logits = logits + mask

        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    # -------- rollouts -------- #
    def generate_rollout(self, opponent_fn, rollout_steps: int):
        states = []
        actions = []
        rewards = []
        dones = []
        log_probs = []
        values = []
        masks = []

        # 轮流先手：跨 episode 在玩家1/2 间切换
        start_with_agent = getattr(self, "_start_with_agent", True)
        board = [0] * (ppo_config.ROWS * ppo_config.COLUMNS)
        agent_mark = 1 if start_with_agent else 2
        current_mark = 1  # player 1 总是先落，若 agent_mark=2 则对手先手

        for _ in range(rollout_steps):
            # reset when done
            done, winner = is_terminal(board)
            if done:
                board = [0] * (ppo_config.ROWS * ppo_config.COLUMNS)
                start_with_agent = not start_with_agent
                agent_mark = 1 if start_with_agent else 2
                current_mark = 1
                done = False
                winner = None

            if current_mark == agent_mark:
                action, logp, val = self.select_action(board, current_mark)
            else:
                action = opponent_fn(board, current_mark)
                # store dummy logp/val for opponent turns
                logp, val = 0.0, 0.0

            next_board = make_move(board, action, current_mark)
            done, winner = is_terminal(next_board)
            reward = 0.0
            if done:
                if winner == agent_mark:
                    reward = 1.0
                elif winner == 3 - agent_mark:
                    reward = -1.0
                else:
                    reward = 0.0

            # record only from player 1 perspective
            if current_mark == agent_mark:
                states.append(encode_state(board, current_mark))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(logp)
                values.append(val)
                masks.append(valid_action_mask(board))

            board = next_board
            current_mark = 3 - current_mark

        batch = self._process_rollout(states, actions, rewards, dones, log_probs, values, masks)
        # 持久化先手切换状态，下一次 generate_rollout 继续轮换
        self._start_with_agent = start_with_agent
        return batch

    def _process_rollout(self, states, actions, rewards, dones, log_probs, values, masks):
        states = torch.tensor(np.array(states), dtype=torch.float32, device=ppo_config.DEVICE)
        actions = torch.tensor(actions, dtype=torch.long, device=ppo_config.DEVICE)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=ppo_config.DEVICE)
        dones = torch.tensor(dones, dtype=torch.float32, device=ppo_config.DEVICE)
        log_probs = torch.tensor(log_probs, dtype=torch.float32, device=ppo_config.DEVICE)
        values = torch.tensor(values, dtype=torch.float32, device=ppo_config.DEVICE)
        masks_t = torch.tensor(np.array(masks), dtype=torch.float32, device=ppo_config.DEVICE)

        # GAE-Lambda
        returns = []
        advantages = []
        gae = 0.0
        next_value = 0.0
        for step in reversed(range(len(rewards))):
            mask = 1.0 - dones[step]
            delta = rewards[step] + ppo_config.GAMMA * next_value * mask - values[step]
            gae = delta + ppo_config.GAMMA * ppo_config.GAE_LAMBDA * mask * gae
            advantages.insert(0, gae)
            next_value = values[step]
            returns.insert(0, gae + values[step])

        returns = torch.stack(returns)
        advantages = torch.stack(advantages)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return RolloutBatch(
            states=states,
            actions=actions,
            log_probs=log_probs,
            returns=returns,
            advantages=advantages,
            values=values,
            masks=masks_t,
        )

    # -------- update -------- #
    def update(self, batch: RolloutBatch):
        total_loss = 0.0
        total_policy = 0.0
        total_value = 0.0
        total_entropy = 0.0
        total_kl = 0.0
        total_clip = 0.0
        steps = 0
        batch_size = batch.states.size(0)
        mini_batch_size = batch_size // ppo_config.MINI_BATCHES

        for _ in range(ppo_config.PPO_EPOCHS):
            indices = torch.randperm(batch_size, device=ppo_config.DEVICE)
            for start in range(0, batch_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]
                (
                    loss,
                    policy_loss,
                    value_loss,
                    entropy,
                    approx_kl,
                    clip_frac,
                ) = self._ppo_loss(batch, mb_idx)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), ppo_config.MAX_GRAD_NORM)
                self.optimizer.step()
                total_loss += loss.item()
                total_policy += policy_loss.item()
                total_value += value_loss.item()
                total_entropy += entropy.item()
                total_kl += approx_kl.item()
                total_clip += clip_frac.item()
                steps += 1

        # 返回平均指标，方便日志记录
        return {
            "loss": total_loss / max(1, steps),
            "policy_loss": total_policy / max(1, steps),
            "value_loss": total_value / max(1, steps),
            "entropy": total_entropy / max(1, steps),
            "approx_kl": total_kl / max(1, steps),
            "clip_frac": total_clip / max(1, steps),
        }

    def _ppo_loss(self, batch: RolloutBatch, idx):
        states = batch.states[idx]
        actions = batch.actions[idx]
        old_log_probs = batch.log_probs[idx]
        returns = batch.returns[idx]
        advantages = batch.advantages[idx]
        masks = batch.masks[idx]

        logits, values = self.model(states)
        # mask invalid actions for stability
        logits = logits + (masks + 1e-8).log()  # log(1) for valid, log(eps) for invalid

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        log_ratio = log_probs - old_log_probs
        ratio = torch.exp(log_ratio)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - ppo_config.CLIP_RANGE, 1.0 + ppo_config.CLIP_RANGE) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values.squeeze(-1), returns)

        loss = policy_loss + ppo_config.VF_COEF * value_loss - ppo_config.ENT_COEF * entropy

        # 监控指标
        approx_kl = 0.5 * (log_ratio ** 2).mean()
        clip_frac = (torch.abs(ratio - 1.0) > ppo_config.CLIP_RANGE).float().mean()

        return loss, policy_loss, value_loss, entropy, approx_kl, clip_frac


# -------- utility for masks -------- #
def valid_action_mask(board: List[int]) -> np.ndarray:
    mask = np.full(ppo_config.COLUMNS, 0.0, dtype=np.float32)
    for c in range(ppo_config.COLUMNS):
        if board[c] == 0:
            mask[c] = 1.0
    return mask

