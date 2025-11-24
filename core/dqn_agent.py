"""
DQN Agent for ConnectX
Implements the DQN algorithm with policy and target networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from typing import Optional, List
from config import config
from dqn_model import create_model, count_parameters
from replay_buffer import ReplayBuffer
from utils import encode_state, get_valid_moves, state_to_tensor


class DQNAgent:
    """
    Deep Q-Network Agent with epsilon-greedy exploration.
    
    Features:
    - Policy network for action selection
    - Target network for stable training
    - Experience replay
    - Epsilon-greedy exploration with decay
    - GPU support
    """
    
    def __init__(self, model_type='standard', use_double_dqn=True):
        """
        Initialize DQN Agent.
        
        Args:
            model_type: Type of model ('standard' or 'dueling')
            use_double_dqn: Whether to use Double DQN algorithm
        """
        self.device = config.DEVICE
        self.use_double_dqn = use_double_dqn
        
        # Create policy and target networks
        self.policy_net = create_model(model_type)
        self.target_net = create_model(model_type)
        
        # Initialize target network with policy network weights
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is always in eval mode
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=config.LEARNING_RATE)

        # Optional learning rate scheduler
        self.scheduler = None
        if config.LR_SCHEDULE == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.LR_T_MAX,
                eta_min=config.MIN_LEARNING_RATE
            )
        elif config.LR_SCHEDULE == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.LR_STEP_INTERVAL,
                gamma=config.LR_STEP_GAMMA
            )

        # Loss function
        self.criterion = nn.SmoothL1Loss()  # Huber loss
        
        # Replay buffer
        self.memory = ReplayBuffer(config.REPLAY_BUFFER_SIZE)
        
        # Training statistics
        self.steps_done = 0
        self.epsilon = config.EPSILON_START
        self.losses = []
        
        print(f"DQN Agent initialized on {self.device}")
        print(f"Model parameters: {count_parameters(self.policy_net):,}")
        print(f"Use Double DQN: {use_double_dqn}")
    
    def select_action(self, board: List[int], mark: int, 
                     epsilon: Optional[float] = None) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            board: Current board state (flattened list)
            mark: Current player's mark (1 or 2)
            epsilon: Exploration rate (uses self.epsilon if None)
        
        Returns:
            Selected action (column index)
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        valid_moves = get_valid_moves(board)
        
        if not valid_moves:
            return 0  # Should not happen in valid game
        
        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            # Random action (exploration)
            return np.random.choice(valid_moves)
        else:
            # Greedy action (exploitation)
            state = encode_state(board, mark)
            state_tensor = state_to_tensor(state, self.device).unsqueeze(0)
            
            with torch.no_grad():
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
        Store a transition in replay memory.
        
        Args:
            state: Current state (3, rows, cols)
            action: Action taken
            reward: Reward received
            next_state: Next state (3, rows, cols)
            done: Whether episode ended
        """
        self.memory.push(state, action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step (sample batch and update network).
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        # Check if enough samples in memory
        if not self.memory.is_ready(config.MIN_REPLAY_SIZE):
            return None
        
        # Sample batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(config.BATCH_SIZE)
        
        # Convert to tensors
        states = torch.from_numpy(states).to(self.device)
        actions = torch.from_numpy(actions).to(self.device)
        rewards = torch.from_numpy(rewards).to(self.device)
        next_states = torch.from_numpy(next_states).to(self.device)
        dones = torch.from_numpy(dones).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute next Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use policy net to select action, target net to evaluate
                next_actions = self.policy_net(next_states).argmax(dim=1)
                next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            else:
                # Standard DQN: use target net for both selection and evaluation
                next_q_values = self.target_net(next_states).max(dim=1)[0]
            
            # Compute target Q values
            target_q_values = rewards + (1 - dones) * config.GAMMA * next_q_values
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Scheduler step
        if self.scheduler is not None:
            self.scheduler.step()

        # Update statistics
        self.steps_done += 1
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Update epsilon
        self.update_epsilon()
        
        # Update target network periodically
        if self.steps_done % config.TARGET_UPDATE_FREQ == 0:
            self.update_target_network()
        
        return loss_value
    
    def update_target_network(self):
        """Copy weights from policy network to target network."""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """Update epsilon using linear decay."""
        epsilon_range = config.EPSILON_START - config.EPSILON_END
        decay_fraction = min(1.0, self.steps_done / config.EPSILON_DECAY_STEPS)
        self.epsilon = config.EPSILON_START - epsilon_range * decay_fraction
    
    def save_checkpoint(self, filepath: str, episode: int, metrics: dict = None):
        """
        Save agent checkpoint.
        
        Args:
            filepath: Path to save checkpoint
            episode: Current episode number
            metrics: Dictionary of training metrics
        """
        checkpoint = {
            'episode': episode,
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
            'metrics': metrics or {}
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> dict:
        """
        Load agent checkpoint.
        
        Args:
            filepath: Path to checkpoint file
        
        Returns:
            Dictionary containing checkpoint metadata
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.steps_done = checkpoint['steps_done']
        self.epsilon = checkpoint['epsilon']
        self.losses = checkpoint.get('losses', [])
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Episode: {checkpoint['episode']}, Steps: {self.steps_done}, Epsilon: {self.epsilon:.4f}")
        
        return checkpoint
    
    def save_model(self, filepath: str):
        """
        Save only the model weights (for deployment).
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.policy_net.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """
        Load model weights.
        
        Args:
            filepath: Path to model file
        """
        state_dict = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(state_dict)
        self.policy_net.eval()
        print(f"Model loaded from {filepath}")
    
    def get_statistics(self) -> dict:
        """
        Get training statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            'steps_done': self.steps_done,
            'epsilon': self.epsilon,
            'buffer_size': len(self.memory),
            'avg_loss_100': np.mean(self.losses[-100:]) if self.losses else 0.0,
            'total_losses': len(self.losses)
        }
    
    def reset_statistics(self):
        """Reset training statistics."""
        self.losses = []


def evaluate_agent(agent: DQNAgent, opponent_agent, num_games: int = 100) -> dict:
    """
    Evaluate agent against an opponent.
    
    Args:
        agent: DQN agent to evaluate
        opponent_agent: Opponent agent function or DQNAgent
        num_games: Number of games to play
    
    Returns:
        Dictionary with win/loss/draw statistics
    """
    from kaggle_environments import make
    from utils import is_terminal
    
    wins = 0
    losses = 0
    draws = 0
    
    # Create agent function for evaluation
    def dqn_agent_func(observation, configuration):
        board = observation.board
        mark = observation.mark
        action = agent.select_action(board, mark, epsilon=0.0)  # Greedy policy
        return int(action)
    
    # Create environment
    env = make("connectx", debug=False)
    
    for game in range(num_games):
        # Alternate starting player
        if game % 2 == 0:
            agents = [dqn_agent_func, opponent_agent]
            agent_position = 0
        else:
            agents = [opponent_agent, dqn_agent_func]
            agent_position = 1
        
        # Run game
        env.reset()
        result = env.run(agents)
        
        # Check result
        final_state = result[-1]
        
        if final_state[agent_position]['status'] == 'DONE':
            if final_state[agent_position]['reward'] == 1:
                wins += 1
            elif final_state[agent_position]['reward'] == 0:
                draws += 1
            else:
                losses += 1
        else:
            losses += 1
    
    win_rate = wins / num_games
    loss_rate = losses / num_games
    draw_rate = draws / num_games
    
    return {
        'wins': wins,
        'losses': losses,
        'draws': draws,
        'win_rate': win_rate,
        'loss_rate': loss_rate,
        'draw_rate': draw_rate,
        'total_games': num_games
    }


if __name__ == "__main__":
    # Test the agent
    print("Testing DQN Agent...\n")
    
    agent = DQNAgent(model_type='standard', use_double_dqn=True)
    
    # Test action selection
    empty_board = [0] * (config.ROWS * config.COLUMNS)
    action = agent.select_action(empty_board, mark=1)
    print(f"\nSelected action on empty board: {action}")
    
    # Test storing transition
    state = encode_state(empty_board, 1)
    next_state = encode_state(empty_board, 1)
    agent.store_transition(state, action, 0.0, next_state, False)
    print(f"Stored transition. Buffer size: {len(agent.memory)}")
    
    print("\nAgent test completed!")

