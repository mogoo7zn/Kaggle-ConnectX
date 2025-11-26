"""
Training Script for Rainbow DQN
Supports self-play and opponent-based training
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import random
import time
from datetime import datetime
from typing import Dict, Tuple
from collections import deque

import torch
from torch.utils.tensorboard import SummaryWriter

from agents.rainbow.rainbow_config import rainbow_config
from agents.rainbow.rainbow_agent import RainbowAgent
from agents.base.utils import (
    encode_state, get_valid_moves, is_terminal,
    make_move, calculate_reward, create_directories
)

# 导入绘图工具
from tools.plot_training import TrainingMetrics, plot_rainbow_training_metrics


class RainbowTrainer:
    """Trainer for Rainbow DQN agent."""
    
    def __init__(self, agent: RainbowAgent, run_name: str = None):
        """
        Initialize trainer.
        
        Args:
            agent: Rainbow agent to train
            run_name: Name for this training run
        """
        self.agent = agent
        self.run_name = run_name or f"rainbow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create directories
        os.makedirs(rainbow_config.MODEL_DIR, exist_ok=True)
        os.makedirs(rainbow_config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(rainbow_config.LOG_DIR, exist_ok=True)
        os.makedirs(rainbow_config.PLOT_DIR, exist_ok=True)
        
        # TensorBoard writer
        self.writer = SummaryWriter(os.path.join(rainbow_config.LOG_DIR, "runs", self.run_name))
        
        # Metrics - 使用 TrainingMetrics 类收集数据
        self.metrics = TrainingMetrics()
        
        # Recent stats for moving average (保持向后兼容)
        self.recent_rewards = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
    
    def play_self_play_episode(self) -> Tuple[float, int]:
        """
        Play one episode of self-play.
        
        Returns:
            Tuple of (total_reward, episode_length)
        """
        board = [0] * (rainbow_config.ROWS * rainbow_config.COLUMNS)
        current_mark = 1
        episode_length = 0
        
        transitions_p1 = []
        transitions_p2 = []
        
        while True:
            state = encode_state(board, current_mark)
            action = self.agent.select_action(board, current_mark)
            next_board = make_move(board, action, current_mark)
            done, winner = is_terminal(next_board)
            reward = calculate_reward(board, action, current_mark, next_board, done, winner)
            next_state = encode_state(next_board, current_mark)
            
            if current_mark == 1:
                transitions_p1.append((state, action, reward, next_state, done))
            else:
                transitions_p2.append((state, action, reward, next_state, done))
            
            episode_length += 1
            
            if done:
                # Assign final rewards
                if winner == 1:
                    final_reward_p1, final_reward_p2 = rainbow_config.REWARD_WIN, rainbow_config.REWARD_LOSS
                elif winner == 2:
                    final_reward_p1, final_reward_p2 = rainbow_config.REWARD_LOSS, rainbow_config.REWARD_WIN
                else:
                    final_reward_p1 = final_reward_p2 = rainbow_config.REWARD_DRAW
                
                # Update final rewards
                if transitions_p1:
                    s, a, _, ns, d = transitions_p1[-1]
                    transitions_p1[-1] = (s, a, final_reward_p1, ns, d)
                if transitions_p2:
                    s, a, _, ns, d = transitions_p2[-1]
                    transitions_p2[-1] = (s, a, final_reward_p2, ns, d)
                
                # Store all transitions
                for transition in transitions_p1 + transitions_p2:
                    self.agent.store_transition(*transition)
                
                return final_reward_p1, episode_length
            
            board = next_board
            current_mark = 3 - current_mark
    
    def play_opponent_episode(self, opponent_fn) -> Tuple[float, int]:
        """
        Play one episode against an opponent.
        
        Args:
            opponent_fn: Opponent policy function
        
        Returns:
            Tuple of (total_reward, episode_length)
        """
        board = [0] * (rainbow_config.ROWS * rainbow_config.COLUMNS)
        agent_mark = 1
        opponent_mark = 2
        current_mark = 1
        episode_length = 0
        transitions = []
        
        while True:
            if current_mark == agent_mark:
                state = encode_state(board, agent_mark)
                action = self.agent.select_action(board, agent_mark)
                next_board = make_move(board, action, agent_mark)
                done, winner = is_terminal(next_board)
                reward = calculate_reward(board, action, agent_mark, next_board, done, winner)
                next_state = encode_state(next_board, agent_mark)
                
                transitions.append((state, action, reward, next_state, done))
                episode_length += 1
                
                if done:
                    for transition in transitions:
                        self.agent.store_transition(*transition)
                    
                    if winner == agent_mark:
                        return rainbow_config.REWARD_WIN, episode_length
                    elif winner == 0:
                        return rainbow_config.REWARD_DRAW, episode_length
                    else:
                        return rainbow_config.REWARD_LOSS, episode_length
                
                board = next_board
            else:
                # Opponent's turn
                action = opponent_fn(board, opponent_mark)
                board = make_move(board, action, opponent_mark)
                done, winner = is_terminal(board)
                
                if done:
                    if transitions:
                        s, a, _, ns, _ = transitions[-1]
                        if winner == agent_mark:
                            final_reward = rainbow_config.REWARD_WIN
                        elif winner == 0:
                            final_reward = rainbow_config.REWARD_DRAW
                        else:
                            final_reward = rainbow_config.REWARD_LOSS
                        transitions[-1] = (s, a, final_reward, ns, True)
                    
                    for transition in transitions:
                        self.agent.store_transition(*transition)
                    
                    if winner == agent_mark:
                        return rainbow_config.REWARD_WIN, episode_length
                    elif winner == 0:
                        return rainbow_config.REWARD_DRAW, episode_length
                    else:
                        return rainbow_config.REWARD_LOSS, episode_length
            
            current_mark = 3 - current_mark
    
    def train(self, num_episodes: int, mode: str = 'self_play',
             opponent_fn=None, eval_interval: int = 200,
             save_interval: int = 500):
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train
            mode: Training mode ('self_play' or 'opponent')
            opponent_fn: Opponent function (if mode='opponent')
            eval_interval: Evaluate every N episodes
            save_interval: Save checkpoint every N episodes
        """
        print(f"\n{'='*60}")
        print(f"Starting Rainbow DQN Training: {mode} mode")
        print(f"Episodes: {num_episodes}")
        print(f"Device: {rainbow_config.DEVICE}")
        print(f"{'='*60}\n")
        
        best_win_rate = 0.0
        start_time = time.time()
        
        for episode in range(1, num_episodes + 1):
            # Play episode
            if mode == 'self_play':
                reward, length = self.play_self_play_episode()
            else:
                reward, length = self.play_opponent_episode(opponent_fn)
            
            # Train agent
            train_stats = []
            for _ in range(rainbow_config.TRAINING_STEPS_PER_EPISODE):
                stats = self.agent.train_step()
                if stats:
                    train_stats.append(stats)
            
            # Update metrics
            self.metrics.add_episode(reward, length, self.agent.epsilon)
            self.recent_rewards.append(reward)
            self.recent_lengths.append(length)
            
            # Log to TensorBoard
            self.writer.add_scalar('train/reward', reward, episode)
            self.writer.add_scalar('train/episode_length', length, episode)
            self.writer.add_scalar('train/epsilon', self.agent.epsilon, episode)
            
            if train_stats:
                avg_loss = np.mean([s['loss'] for s in train_stats])
                avg_td_error = np.mean([s['td_error_mean'] for s in train_stats])
                avg_q_mean = np.mean([s['q_mean'] for s in train_stats])
                
                # 记录损失到 metrics
                self.metrics.add_loss(avg_loss)
                
                self.writer.add_scalar('train/loss', avg_loss, episode)
                self.writer.add_scalar('train/td_error', avg_td_error, episode)
                self.writer.add_scalar('train/q_mean', avg_q_mean, episode)
            
            # Progress reporting
            if episode % 10 == 0:
                avg_reward = np.mean(self.recent_rewards)
                avg_length = np.mean(self.recent_lengths)
                print(f"Episode {episode:5d}/{num_episodes} | "
                      f"Reward: {reward:6.2f} | "
                      f"Avg Reward: {avg_reward:6.2f} | "
                      f"Length: {length:3d} | "
                      f"Epsilon: {self.agent.epsilon:.4f} | "
                      f"Buffer: {len(self.agent.memory):6d}")
            
            # Evaluation
            if episode % eval_interval == 0:
                win_rate = self.evaluate()
                self.metrics.add_evaluation(episode, win_rate)
                self.writer.add_scalar('eval/win_rate', win_rate, episode)
                
                print(f"\n{'='*60}")
                print(f"Evaluation at episode {episode}: Win rate = {win_rate:.2%}")
                print(f"{'='*60}\n")
                
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    model_path = os.path.join(rainbow_config.CHECKPOINT_DIR,
                                            f'best_rainbow_{self.run_name}.pth')
                    self.agent.save_model(model_path)
            
            # Save checkpoint
            if episode % save_interval == 0:
                checkpoint_path = os.path.join(rainbow_config.CHECKPOINT_DIR,
                                             f'rainbow_ep{episode}_{self.run_name}.pth')
                self.agent.save_checkpoint(checkpoint_path, episode,
                                          {'win_rate': best_win_rate})
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best win rate: {best_win_rate:.2%}")
        print(f"{'='*60}\n")
        
        # Save final model
        final_path = os.path.join(rainbow_config.MODEL_DIR,
                                  f'final_rainbow_{self.run_name}.pth')
        self.agent.save_model(final_path)
        
        # 生成训练图像
        plot_path = os.path.join(rainbow_config.PLOT_DIR, 
                                f'training_metrics_{self.run_name}.png')
        plot_rainbow_training_metrics(self.metrics, save_path=plot_path, show=False,
                                     title=f'Rainbow DQN Training Metrics - {self.run_name}')
        
        self.writer.close()
    
    def evaluate(self, num_games: int = 100) -> float:
        """
        Evaluate agent against standard opponents.
        
        Args:
            num_games: Number of games to play
        
        Returns:
            Overall win rate
        """
        from agents.base.utils import get_negamax_move
        
        def random_policy(board, mark):
            moves = get_valid_moves(board)
            return random.choice(moves) if moves else 0
        
        def negamax_policy(board, mark):
            return get_negamax_move(board, mark)
        
        # Evaluate against random
        wins_random = self._play_evaluation_games(random_policy, num_games // 2)
        
        # Evaluate against negamax
        wins_negamax = self._play_evaluation_games(negamax_policy, num_games // 2)
        
        total_wins = wins_random + wins_negamax
        win_rate = total_wins / num_games
        
        return win_rate
    
    def _play_evaluation_games(self, opponent_fn, num_games: int) -> int:
        """Play evaluation games and return number of wins."""
        wins = 0
        
        for game in range(num_games):
            board = [0] * (rainbow_config.ROWS * rainbow_config.COLUMNS)
            
            # Alternate starting player
            if game % 2 == 0:
                agent_mark, opponent_mark = 1, 2
            else:
                agent_mark, opponent_mark = 2, 1
            
            current_mark = 1
            
            while True:
                valid_moves = get_valid_moves(board)
                if not valid_moves:
                    break
                
                if current_mark == agent_mark:
                    action = self.agent.select_action(board, current_mark, epsilon=0.0)
                else:
                    action = opponent_fn(board, current_mark)
                
                if action not in valid_moves:
                    if current_mark == opponent_mark:
                        wins += 1
                    break
                
                board = make_move(board, action, current_mark)
                done, winner = is_terminal(board)
                
                if done:
                    if winner == agent_mark:
                        wins += 1
                    break
                
                current_mark = 3 - current_mark
        
        return wins


def main():
    """Main training function."""
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create agent
    print("Creating Rainbow DQN Agent...")
    agent = RainbowAgent(use_noisy=True, use_distributional=False)
    
    # Create trainer
    run_name = f"rainbow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = RainbowTrainer(agent, run_name=run_name)
    
    # Phase 1: Self-play training
    print("\n" + "="*60)
    print("PHASE 1: SELF-PLAY TRAINING")
    print("="*60)
    
    trainer.train(
        num_episodes=rainbow_config.SELF_PLAY_EPISODES,
        mode='self_play',
        eval_interval=rainbow_config.EVAL_INTERVAL,
        save_interval=rainbow_config.SAVE_INTERVAL
    )
    
    # Phase 2: Opponent training (optional)
    if rainbow_config.OPPONENT_EPISODES > 0:
        print("\n" + "="*60)
        print("PHASE 2: OPPONENT TRAINING")
        print("="*60)
        
        from agents.base.utils import get_negamax_move
        
        def negamax_opponent(board, mark):
            return get_negamax_move(board, mark)
        
        trainer.train(
            num_episodes=rainbow_config.OPPONENT_EPISODES,
            mode='opponent',
            opponent_fn=negamax_opponent,
            eval_interval=rainbow_config.EVAL_INTERVAL,
            save_interval=rainbow_config.SAVE_INTERVAL
        )
    
    print("\n🎉 Rainbow DQN training completed!")


if __name__ == "__main__":
    main()

