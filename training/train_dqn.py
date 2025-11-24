"""
Training Pipeline for DQN ConnectX Agent
Supports self-play and opponent-based training with visualization
"""

import numpy as np
import random
import time
import os
from typing import List, Tuple, Optional
from collections import deque
from kaggle_environments import make

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from config import config
from dqn_agent import DQNAgent, evaluate_agent
from utils import (
    encode_state, get_valid_moves, is_terminal, 
    make_move, calculate_reward, create_directories,
    get_negamax_move
)
from visualize import plot_training_metrics, plot_win_rates, create_training_summary


class TrainingMetrics:
    """Track and store training metrics."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.loss_values = []
        self.epsilon_values = []
        self.eval_episodes = []
        
        # Running averages
        self.recent_rewards = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
    
    def add_episode(self, reward: float, length: int, epsilon: float):
        """Add episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)
        self.recent_rewards.append(reward)
        self.recent_lengths.append(length)
    
    def add_evaluation(self, episode: int, win_rate: float):
        """Add evaluation result."""
        self.eval_episodes.append(episode)
        self.win_rates.append(win_rate)
    
    def add_loss(self, loss: float):
        """Add training loss."""
        if loss is not None:
            self.loss_values.append(loss)
    
    def get_recent_avg_reward(self) -> float:
        """Get average reward over recent episodes."""
        return np.mean(self.recent_rewards) if self.recent_rewards else 0.0
    
    def get_recent_avg_length(self) -> float:
        """Get average episode length over recent episodes."""
        return np.mean(self.recent_lengths) if self.recent_lengths else 0.0
    
    def get_recent_avg_loss(self) -> float:
        """Get average loss over recent training steps."""
        return np.mean(self.loss_values[-100:]) if self.loss_values else 0.0


def play_self_play_episode(agent: DQNAgent) -> Tuple[float, int]:
    """
    Play one episode of self-play (agent vs itself).
    
    Args:
        agent: DQN agent
    
    Returns:
        Tuple of (total_reward, episode_length)
    """
    # Initialize board
    board = [0] * (config.ROWS * config.COLUMNS)
    current_mark = 1
    episode_length = 0
    total_reward = 0.0
    
    # Store transitions for both players
    transitions_p1 = []
    transitions_p2 = []
    
    while True:
        # Encode current state
        state = encode_state(board, current_mark)
        
        # Select action
        action = agent.select_action(board, current_mark)
        
        # Make move
        next_board = make_move(board, action, current_mark)
        
        # Check if game ended
        done, winner = is_terminal(next_board)
        
        # Calculate reward
        reward = calculate_reward(board, action, current_mark, next_board, done, winner)
        
        # Encode next state
        next_state = encode_state(next_board, current_mark)
        
        # Store transition for current player
        if current_mark == 1:
            transitions_p1.append((state, action, reward, next_state, done))
        else:
            transitions_p2.append((state, action, reward, next_state, done))
        
        episode_length += 1
        
        if done:
            # Assign final rewards
            if winner == 1:
                p1_final_reward = config.REWARD_WIN
                p2_final_reward = config.REWARD_LOSS
            elif winner == 2:
                p1_final_reward = config.REWARD_LOSS
                p2_final_reward = config.REWARD_WIN
            else:  # Draw
                p1_final_reward = config.REWARD_DRAW
                p2_final_reward = config.REWARD_DRAW
            
            # Update final rewards in transitions
            if transitions_p1:
                last_state, last_action, _, last_next_state, last_done = transitions_p1[-1]
                transitions_p1[-1] = (last_state, last_action, p1_final_reward, last_next_state, last_done)
            
            if transitions_p2:
                last_state, last_action, _, last_next_state, last_done = transitions_p2[-1]
                transitions_p2[-1] = (last_state, last_action, p2_final_reward, last_next_state, last_done)
            
            # Store all transitions
            for transition in transitions_p1 + transitions_p2:
                agent.store_transition(*transition)
            
            # Return reward from player 1's perspective
            total_reward = p1_final_reward
            break
        
        # Switch player
        board = next_board
        current_mark = 3 - current_mark
    
    return total_reward, episode_length


def play_opponent_episode(agent: DQNAgent, opponent_type: str) -> Tuple[float, int]:
    """
    Play one episode against an opponent.
    
    Args:
        agent: DQN agent
        opponent_type: Type of opponent ('random' or 'negamax')
    
    Returns:
        Tuple of (total_reward, episode_length)
    """
    board = [0] * (config.ROWS * config.COLUMNS)
    episode_length = 0
    
    # Agent is player 1, opponent is player 2
    agent_mark = 1
    opponent_mark = 2
    current_mark = 1
    
    transitions = []
    
    while True:
        if current_mark == agent_mark:
            # Agent's turn
            state = encode_state(board, agent_mark)
            action = agent.select_action(board, agent_mark)
            next_board = make_move(board, action, agent_mark)
            
            done, winner = is_terminal(next_board)
            reward = calculate_reward(board, action, agent_mark, next_board, done, winner)
            next_state = encode_state(next_board, agent_mark)
            
            transitions.append((state, action, reward, next_state, done))
            
            episode_length += 1
            
            if done:
                # Store all transitions
                for transition in transitions:
                    agent.store_transition(*transition)
                
                if winner == agent_mark:
                    return config.REWARD_WIN, episode_length
                elif winner == 0:
                    return config.REWARD_DRAW, episode_length
                else:
                    return config.REWARD_LOSS, episode_length
            
            board = next_board
        else:
            # Opponent's turn
            valid_moves = get_valid_moves(board)
            
            if not valid_moves:
                break
            
            if opponent_type == 'random':
                action = random.choice(valid_moves)
            elif opponent_type == 'negamax':
                action = get_negamax_move(board, opponent_mark)
            else:
                action = random.choice(valid_moves)
            
            board = make_move(board, action, opponent_mark)
            
            done, winner = is_terminal(board)
            
            if done:
                # Update last transition with final outcome
                if transitions:
                    last_state, last_action, _, last_next_state, _ = transitions[-1]
                    
                    if winner == agent_mark:
                        final_reward = config.REWARD_WIN
                    elif winner == 0:
                        final_reward = config.REWARD_DRAW
                    else:
                        final_reward = config.REWARD_LOSS
                    
                    transitions[-1] = (last_state, last_action, final_reward, last_next_state, True)
                
                # Store all transitions
                for transition in transitions:
                    agent.store_transition(*transition)
                
                if winner == agent_mark:
                    return config.REWARD_WIN, episode_length
                elif winner == 0:
                    return config.REWARD_DRAW, episode_length
                else:
                    return config.REWARD_LOSS, episode_length
        
        current_mark = 3 - current_mark
    
    return 0.0, episode_length


def train_agent(agent: DQNAgent, 
                mode: str = 'self_play',
                num_episodes: int = 5000,
                eval_interval: int = 100,
                save_interval: int = 500,
                opponent_type: str = 'random') -> TrainingMetrics:
    """
    Train the DQN agent.
    
    Args:
        agent: DQN agent to train
        mode: Training mode ('self_play' or 'opponent')
        num_episodes: Number of episodes to train
        eval_interval: Evaluate every N episodes
        save_interval: Save checkpoint every N episodes
        opponent_type: Type of opponent for opponent mode
    
    Returns:
        TrainingMetrics object with training history
    """
    metrics = TrainingMetrics()
    best_win_rate = 0.0
    
    print(f"\n{'='*60}")
    print(f"Starting training: {mode} mode")
    print(f"Episodes: {num_episodes}")
    print(f"Device: {config.DEVICE}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        
        # Play episode
        if mode == 'self_play':
            reward, length = play_self_play_episode(agent)
        else:  # opponent mode
            reward, length = play_opponent_episode(agent, opponent_type)
        
        # Train agent
        loss = agent.train_step()
        
        # Update metrics
        metrics.add_episode(reward, length, agent.epsilon)
        metrics.add_loss(loss)
        
        # Progress reporting
        if episode % 10 == 0:
            episode_time = time.time() - episode_start
            avg_reward = metrics.get_recent_avg_reward()
            avg_length = metrics.get_recent_avg_length()
            avg_loss = metrics.get_recent_avg_loss()
            
            print(f"Episode {episode:5d}/{num_episodes} | "
                  f"Reward: {reward:6.2f} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Length: {length:3d} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(agent.memory):6d} | "
                  f"Time: {episode_time:.2f}s")
        
        # Evaluation
        if episode % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at episode {episode}")
            print(f"{'='*60}")
            
            # Evaluate against random opponent
            eval_results = evaluate_agent(agent, "random", num_games=config.EVAL_GAMES)
            win_rate = eval_results['win_rate']
            
            print(f"Win rate vs random: {win_rate:.2%} "
                  f"({eval_results['wins']}W-{eval_results['losses']}L-{eval_results['draws']}D)")
            
            metrics.add_evaluation(episode, win_rate)
            
            # Save best model
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                model_path = os.path.join(config.MODEL_DIR, f'best_model_{mode}.pth')
                agent.save_model(model_path)
                print(f"New best model saved! Win rate: {win_rate:.2%}")
            
            print(f"{'='*60}\n")
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(config.MODEL_DIR, 
                                          f'checkpoint_{mode}_ep{episode}.pth')
            checkpoint_metrics = {
                'episode': episode,
                'mode': mode,
                'avg_reward': metrics.get_recent_avg_reward(),
                'best_win_rate': best_win_rate
            }
            agent.save_checkpoint(checkpoint_path, episode, checkpoint_metrics)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best win rate: {best_win_rate:.2%}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total steps: {agent.steps_done}")
    print(f"{'='*60}\n")
    
    # Save final checkpoint
    final_path = os.path.join(config.MODEL_DIR, f'final_model_{mode}.pth')
    agent.save_model(final_path)
    print(f"Final model saved to {final_path}")
    
    # Generate visualization
    plot_path = os.path.join(config.PLOT_DIR, f'training_metrics_{mode}.png')
    plot_training_metrics(metrics, save_path=plot_path, show=False)
    
    return metrics


def main():
    """Main training function."""
    # Create directories
    create_directories()
    
    # Create agent
    agent = DQNAgent(model_type='standard', use_double_dqn=True)
    
    # Phase 1: Self-play training
    print("\n" + "="*60)
    print("PHASE 1: SELF-PLAY TRAINING")
    print("="*60)
    
    self_play_metrics = train_agent(
        agent=agent,
        mode='self_play',
        num_episodes=config.SELF_PLAY_EPISODES,
        eval_interval=config.EVAL_INTERVAL,
        save_interval=config.SAVE_INTERVAL
    )
    
    # Phase 2: Fine-tuning against opponents
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING AGAINST OPPONENTS")
    print("="*60)
    
    # Train against random opponent
    print("\nTraining against RANDOM opponent...")
    random_metrics = train_agent(
        agent=agent,
        mode='opponent',
        num_episodes=config.OPPONENT_EPISODES // 2,
        eval_interval=config.EVAL_INTERVAL,
        save_interval=config.SAVE_INTERVAL,
        opponent_type='random'
    )
    
    # Train against negamax opponent
    print("\nTraining against NEGAMAX opponent...")
    negamax_metrics = train_agent(
        agent=agent,
        mode='opponent',
        num_episodes=config.OPPONENT_EPISODES // 2,
        eval_interval=config.EVAL_INTERVAL,
        save_interval=config.SAVE_INTERVAL,
        opponent_type='negamax'
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    print("\nEvaluating against RANDOM opponent...")
    random_eval = evaluate_agent(agent, "random", num_games=100)
    print(f"Win rate: {random_eval['win_rate']:.2%} "
          f"({random_eval['wins']}W-{random_eval['losses']}L-{random_eval['draws']}D)")
    
    print("\nEvaluating against NEGAMAX opponent...")
    negamax_eval = evaluate_agent(agent, "negamax", num_games=100)
    print(f"Win rate: {negamax_eval['win_rate']:.2%} "
          f"({negamax_eval['wins']}W-{negamax_eval['losses']}L-{negamax_eval['draws']}D)")
    
    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")
    
    # Combined win rates plot
    metrics_dict = {
        'Self-Play': self_play_metrics,
        'vs Random': random_metrics,
        'vs Negamax': negamax_metrics
    }
    
    win_rates_path = os.path.join(config.PLOT_DIR, 'win_rates_comparison.png')
    plot_win_rates(metrics_dict, save_path=win_rates_path, show=False)
    
    # Training summary
    summary_path = os.path.join(config.LOG_DIR, 'training_summary.txt')
    create_training_summary(agent, metrics_dict, summary_path)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Models saved in: {config.MODEL_DIR}")
    print(f"Plots saved in: {config.PLOT_DIR}")
    print(f"Logs saved in: {config.LOG_DIR}")
    print("="*60)
    
    return agent, self_play_metrics, random_metrics, negamax_metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run training
    agent, self_play_metrics, random_metrics, negamax_metrics = main()

