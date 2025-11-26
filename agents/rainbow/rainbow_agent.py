"""
Rainbow DQN Agent
Integrates all Rainbow components:
- Prioritized Experience Replay
- Dueling Network + Noisy Nets
- Multi-step Learning
- Double DQN
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Optional, List, Dict, Tuple
from collections import deque
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.rainbow.rainbow_config import rainbow_config
from agents.rainbow.rainbow_model import create_rainbow_model, count_parameters
from agents.rainbow.prioritized_buffer import PrioritizedReplayBuffer
from agents.base.utils import encode_state, get_valid_moves, state_to_tensor


class RainbowAgent:
    """
    Rainbow DQN Agent with all improvements.
    
    Features:
    - Prioritized Experience Replay (PER)
    - Dueling Network Architecture
    - Noisy Nets for exploration
    - Multi-step Learning (n-step returns)
    - Double DQN
    - Optional: Distributional RL (C51)
    """
    
    def __init__(self, use_noisy: bool = None, use_distributional: bool = None):
        """
        Initialize Rainbow Agent.
        
        Args:
            use_noisy: Whether to use Noisy Nets (default from config)
            use_distributional: Whether to use C51 (default from config)
        """
        self.device = rainbow_config.DEVICE
        
        # Use config defaults if not specified
        if use_noisy is None:
            use_noisy = rainbow_config.USE_NOISY_NETS
        if use_distributional is None:
            use_distributional = rainbow_config.USE_DISTRIBUTIONAL
        
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.use_per = rainbow_config.USE_PER
        self.n_step = rainbow_config.N_STEP if rainbow_config.USE_MULTI_STEP else 1
        
        # Create policy and target networks
        self.policy_net = create_rainbow_model(use_noisy, use_distributional)
        self.target_net = create_rainbow_model(use_noisy, use_distributional)
        
        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=rainbow_config.LEARNING_RATE
        )
        
        # Learning rate scheduler
        self.scheduler = None
        if rainbow_config.LR_SCHEDULE == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=rainbow_config.LR_T_MAX,
                eta_min=rainbow_config.MIN_LEARNING_RATE
            )
        
        # Loss function
        self.criterion = nn.SmoothL1Loss(reduction='none')  # Huber loss, no reduction for PER
        
        # Replay buffer (Prioritized or standard)
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                capacity=rainbow_config.REPLAY_BUFFER_SIZE,
                alpha=rainbow_config.PER_ALPHA,
                beta_start=rainbow_config.PER_BETA_START,
                beta_frames=rainbow_config.PER_BETA_FRAMES,
                epsilon=rainbow_config.PER_EPSILON
            )
        else:
            # Fallback to standard replay buffer
            from agents.dqn.replay_buffer import ReplayBuffer
            self.memory = ReplayBuffer(rainbow_config.REPLAY_BUFFER_SIZE)
        
        # N-step buffer for multi-step learning
        if self.n_step > 1:
            self.n_step_buffer = deque(maxlen=self.n_step)
        
        # Training statistics
        self.steps_done = 0
        self.epsilon = rainbow_config.EPSILON_START
        self.losses = []
        
        print(f"Rainbow Agent initialized on {self.device}")
        print(f"Model parameters: {count_parameters(self.policy_net):,}")
        print(f"Noisy Nets: {use_noisy}")
        print(f"Distributional RL: {use_distributional}")
        print(f"PER: {self.use_per}")
        print(f"N-step: {self.n_step}")
    
    def select_action(self, board: List[int], mark: int,
                     epsilon: Optional[float] = None) -> int:
        """
        Select action using policy network.
        
        Args:
            board: Current board state
            mark: Current player's mark
            epsilon: Exploration rate (less important with Noisy Nets)
        
        Returns:
            Selected action
        """
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            return 0
        
        # Epsilon-greedy (less important with Noisy Nets)
        if epsilon is None:
            epsilon = self.epsilon if not self.use_noisy else 0.0
        
        if np.random.random() < epsilon:
            return np.random.choice(valid_moves)
        
        # Greedy action selection
        state = encode_state(board, mark)
        state_tensor = state_to_tensor(state, self.device).unsqueeze(0)
        
        with torch.no_grad():
            if self.use_distributional:
                q_values = self.policy_net.get_q_values(state_tensor)
            else:
                q_values = self.policy_net(state_tensor)
            
            # Mask invalid moves
            mask = torch.full_like(q_values, float('-inf'))
            mask[0, valid_moves] = 0
            q_values = q_values + mask
            
            action = q_values.argmax(dim=1).item()
        
        return action
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store transition in replay buffer.
        
        For multi-step learning, accumulates n-step returns before storing.
        
        Args:
            state: Current state
            action: Action taken
            reward: Immediate reward
            next_state: Next state
            done: Whether episode ended
        """
        if self.n_step > 1:
            # Add to n-step buffer
            self.n_step_buffer.append((state, action, reward, next_state, done))
            
            # Only store when we have n steps or episode ends
            if len(self.n_step_buffer) == self.n_step or done:
                # Compute n-step return
                n_step_return = self._compute_n_step_return()
                
                # Get first state and action
                first_state, first_action, _, _, _ = self.n_step_buffer[0]
                
                # Get last next_state and done
                last_next_state = next_state
                last_done = done
                
                # Store n-step transition
                self.memory.push(first_state, first_action, n_step_return,
                               last_next_state, last_done)
                
                # Clear buffer if episode ends
                if done:
                    self.n_step_buffer.clear()
        else:
            # Standard 1-step storage
            self.memory.push(state, action, reward, next_state, done)
    
    def _compute_n_step_return(self) -> float:
        """
        Compute n-step return from buffer.
        
        Returns:
            n-step discounted return
        """
        n_step_return = 0.0
        gamma = rainbow_config.GAMMA
        
        for i, (_, _, reward, _, done) in enumerate(self.n_step_buffer):
            n_step_return += (gamma ** i) * reward
            if done:
                break
        
        return n_step_return
    
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Perform one training step.
        
        Returns:
            Training statistics dict or None
        """
        # Check if enough samples
        if not self.memory.is_ready(rainbow_config.MIN_REPLAY_SIZE):
            return None
        
        # Sample batch
        if self.use_per:
            (states, actions, rewards, next_states, dones,
             indices, weights) = self.memory.sample(rainbow_config.BATCH_SIZE)
            weights = torch.from_numpy(weights).to(self.device)
        else:
            states, actions, rewards, next_states, dones = self.memory.sample(
                rainbow_config.BATCH_SIZE)
            weights = torch.ones(rainbow_config.BATCH_SIZE).to(self.device)
            indices = None
        
        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        
        # Reset noise for Noisy Nets
        if self.use_noisy:
            self.policy_net.reset_noise()
            self.target_net.reset_noise()
        
        # Compute loss based on algorithm type
        if self.use_distributional:
            loss, td_errors = self._distributional_loss(
                states, actions, rewards, next_states, dones, weights
            )
        else:
            loss, td_errors = self._dqn_loss(
                states, actions, rewards, next_states, dones, weights
            )
        
        # Validate loss
        if not torch.isfinite(loss):
            print("Warning: non-finite loss")
            return None
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=rainbow_config.GRAD_CLIP_NORM
        )
        
        self.optimizer.step()
        
        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Update priorities in PER
        if self.use_per and indices is not None:
            self.memory.update_priorities(indices, td_errors.detach().cpu().numpy())
        
        # Update statistics
        self.steps_done += 1
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Update epsilon (if not using Noisy Nets)
        if not self.use_noisy:
            self.update_epsilon()
        
        # Update target network
        if self.steps_done % rainbow_config.TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
        
        # Statistics
        with torch.no_grad():
            if self.use_distributional:
                q_values = self.policy_net.get_q_values(states)
            else:
                q_values = self.policy_net(states)
            
            stats = {
                'loss': loss_value,
                'td_error_mean': td_errors.abs().mean().item(),
                'td_error_max': td_errors.abs().max().item(),
                'grad_norm': float(grad_norm),
                'q_mean': q_values.mean().item(),
                'q_max': q_values.max().item(),
                'q_min': q_values.min().item(),
                'weights_mean': weights.mean().item() if self.use_per else 1.0,
            }
        
        return stats
    
    def _dqn_loss(self, states: torch.Tensor, actions: torch.Tensor,
                  rewards: torch.Tensor, next_states: torch.Tensor,
                  dones: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Double DQN loss.
        
        Returns:
            Tuple of (weighted_loss, td_errors)
        """
        # Current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use policy net to select action, target net to evaluate
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            
            # Compute n-step target
            gamma_n = rainbow_config.GAMMA ** self.n_step
            target_q_values = rewards + (1 - dones) * gamma_n * next_q_values
        
        # TD errors
        td_errors = target_q_values - current_q_values
        
        # Huber loss (element-wise)
        huber_loss = self.criterion(current_q_values, target_q_values)
        
        # Weighted loss for PER
        weighted_loss = (huber_loss * weights).mean()
        
        return weighted_loss, td_errors
    
    def _distributional_loss(self, states: torch.Tensor, actions: torch.Tensor,
                            rewards: torch.Tensor, next_states: torch.Tensor,
                            dones: torch.Tensor, weights: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute distributional RL (C51) loss.
        
        Returns:
            Tuple of (weighted_loss, td_errors)
        """
        # Get current distribution
        q_dist = self.policy_net(states)  # (batch, actions, atoms)
        actions_expanded = actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, q_dist.size(2))
        current_dist = q_dist.gather(1, actions_expanded).squeeze(1)  # (batch, atoms)
        
        # Get target distribution
        with torch.no_grad():
            # Double DQN for action selection
            next_q_values = self.policy_net.get_q_values(next_states)
            next_actions = next_q_values.argmax(dim=1)
            
            # Get target distribution
            next_dist = self.target_net(next_states)
            next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(1).expand(-1, -1, next_dist.size(2))
            next_dist = next_dist.gather(1, next_actions_expanded).squeeze(1)
            
            # Project distribution
            target_dist = self._project_distribution(
                next_dist, rewards, dones, self.policy_net.atoms
            )
        
        # Cross-entropy loss
        log_probs = torch.log(current_dist + 1e-8)
        ce_loss = -(target_dist * log_probs).sum(dim=1)
        
        # Weighted loss
        weighted_loss = (ce_loss * weights).mean()
        
        # Approximate TD errors for PER (using expected values)
        with torch.no_grad():
            current_q = (current_dist * self.policy_net.atoms.to(current_dist.device)).sum(dim=1)
            target_q = (target_dist * self.policy_net.atoms.to(target_dist.device)).sum(dim=1)
            td_errors = target_q - current_q
        
        return weighted_loss, td_errors
    
    def _project_distribution(self, next_dist: torch.Tensor, rewards: torch.Tensor,
                             dones: torch.Tensor, atoms: torch.Tensor) -> torch.Tensor:
        """
        Project Bellman update onto categorical distribution.
        
        Args:
            next_dist: Next state distribution
            rewards: Rewards
            dones: Done flags
            atoms: Atom values
        
        Returns:
            Projected target distribution
        """
        atoms = atoms.to(next_dist.device)
        batch_size = next_dist.size(0)
        
        # Compute Bellman update for each atom
        gamma_n = rainbow_config.GAMMA ** self.n_step
        tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma_n * atoms.unsqueeze(0)
        
        # Clip to support
        tz = tz.clamp(self.policy_net.v_min, self.policy_net.v_max)
        
        # Compute projection
        b = (tz - self.policy_net.v_min) / self.policy_net.delta_z
        l = b.floor().long()
        u = b.ceil().long()
        
        # Handle edge case
        l[(u > 0) * (l == u)] -= 1
        u[(l < (self.policy_net.num_atoms - 1)) * (l == u)] += 1
        
        # Distribute probability
        target_dist = torch.zeros_like(next_dist)
        offset = torch.linspace(0, (batch_size - 1) * self.policy_net.num_atoms,
                               batch_size).unsqueeze(1).expand(batch_size,
                               self.policy_net.num_atoms).long().to(next_dist.device)
        
        target_dist.view(-1).index_add_(0, (l + offset).view(-1),
                                        (next_dist * (u.float() - b)).view(-1))
        target_dist.view(-1).index_add_(0, (u + offset).view(-1),
                                        (next_dist * (b - l.float())).view(-1))
        
        return target_dist
    
    def update_target_network(self):
        """Copy weights from policy to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Update epsilon (linear decay)."""
        epsilon_range = rainbow_config.EPSILON_START - rainbow_config.EPSILON_END
        decay_fraction = min(1.0, self.steps_done / rainbow_config.EPSILON_DECAY_STEPS)
        self.epsilon = rainbow_config.EPSILON_START - epsilon_range * decay_fraction
    
    def save_checkpoint(self, filepath: str, episode: int, metrics: dict = None):
        """Save agent checkpoint."""
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'metrics': metrics or {},
            'config': {
                'use_noisy': self.use_noisy,
                'use_distributional': self.use_distributional,
                'use_per': self.use_per,
                'n_step': self.n_step,
            }
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> dict:
        """Load agent checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint.get('epsilon', 0.01)
        self.losses = checkpoint.get('losses', [])
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Episode: {checkpoint['episode']}, Steps: {self.steps_done}")
        
        return checkpoint
    
    def save_model(self, filepath: str):
        """Save model weights only."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load model weights."""
        state_dict = torch.load(filepath, map_location=self.device, weights_only=False)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()
        print(f"Model loaded from {filepath}")
    
    def get_statistics(self) -> dict:
        """Get training statistics."""
        return {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_loss_100': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'total_losses': len(self.losses)
        }


if __name__ == "__main__":
    # Test Rainbow Agent
    print("Testing Rainbow Agent...")
    print("=" * 60)
    
    # Test standard Rainbow
    agent = RainbowAgent(use_noisy=True, use_distributional=False)
    
    # Test action selection
    empty_board = [0] * (rainbow_config.ROWS * rainbow_config.COLUMNS)
    action = agent.select_action(empty_board, mark=1)
    print(f"\nSelected action: {action}")
    
    # Test storing transitions
    state = encode_state(empty_board, 1)
    next_state = encode_state(empty_board, 1)
    agent.store_transition(state, action, 0.5, next_state, False)
    print(f"Buffer size: {len(agent.memory)}")
    
    print("\n✓ Rainbow Agent test passed!")

