"""
通用训练可视化工具
支持 Rainbow DQN 和 AlphaZero 的训练指标可视化
"""

import matplotlib
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Optional, Dict, List
from collections import deque


class TrainingMetrics:
    """通用的训练指标收集类"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.eval_episodes = []
        self.loss_values = []
        self.epsilon_values = []
        self.policy_losses = []  # AlphaZero 专用
        self.value_losses = []   # AlphaZero 专用
        self.learning_rates = [] # AlphaZero 专用
        self.elo_ratings = []    # AlphaZero 专用
        
        # 运行平均值
        self.recent_rewards = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
    
    def add_episode(self, reward: float, length: int, epsilon: Optional[float] = None):
        """添加回合统计"""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        if epsilon is not None:
            self.epsilon_values.append(epsilon)
        self.recent_rewards.append(reward)
        self.recent_lengths.append(length)
    
    def add_evaluation(self, episode: int, win_rate: float):
        """添加评估结果"""
        self.eval_episodes.append(episode)
        self.win_rates.append(win_rate)
    
    def add_loss(self, loss: float):
        """添加训练损失"""
        if loss is not None:
            self.loss_values.append(loss)
    
    def add_alphazero_losses(self, policy_loss: float, value_loss: float, lr: Optional[float] = None):
        """添加 AlphaZero 的损失"""
        if policy_loss is not None:
            self.policy_losses.append(policy_loss)
        if value_loss is not None:
            self.value_losses.append(value_loss)
        if lr is not None:
            self.learning_rates.append(lr)
    
    def add_elo(self, elo: float):
        """添加 ELO 评分"""
        if elo is not None:
            self.elo_ratings.append(elo)
    
    def get_recent_avg_reward(self) -> float:
        """获取最近的平均奖励"""
        return np.mean(self.recent_rewards) if self.recent_rewards else 0.0


def plot_rainbow_training_metrics(metrics: TrainingMetrics, save_path: Optional[str] = None, 
                                   show: bool = False, title: str = "Rainbow DQN Training Metrics"):
    """
    绘制 Rainbow DQN 训练指标
    
    Args:
        metrics: TrainingMetrics 对象
        save_path: 保存路径
        show: 是否显示
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    if metrics.episode_rewards:
        episodes = range(1, len(metrics.episode_rewards) + 1)
        ax.plot(episodes, metrics.episode_rewards, alpha=0.3, color='blue', label='Raw')
        
        # 移动平均
        window = min(100, len(metrics.episode_rewards) // 10)
        if len(metrics.episode_rewards) >= window and window > 0:
            moving_avg = np.convolve(metrics.episode_rewards, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(metrics.episode_rewards) + 1), 
                   moving_avg, color='red', linewidth=2, label=f'{window}-episode MA')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 2. Episode Lengths
    ax = axes[0, 1]
    if metrics.episode_lengths:
        episodes = range(1, len(metrics.episode_lengths) + 1)
        ax.plot(episodes, metrics.episode_lengths, alpha=0.3, color='green', label='Raw')
        
        window = min(100, len(metrics.episode_lengths) // 10)
        if len(metrics.episode_lengths) >= window and window > 0:
            moving_avg = np.convolve(metrics.episode_lengths, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(metrics.episode_lengths) + 1), 
                   moving_avg, color='red', linewidth=2, label=f'{window}-episode MA')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Length')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Win Rate over Time
    ax = axes[0, 2]
    if metrics.win_rates and metrics.eval_episodes:
        ax.plot(metrics.eval_episodes, metrics.win_rates, 
               marker='o', linewidth=2, markersize=6, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Opponents')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax.legend()
    
    # 4. Training Loss
    ax = axes[1, 0]
    if metrics.loss_values:
        steps = range(1, len(metrics.loss_values) + 1)
        ax.plot(steps, metrics.loss_values, alpha=0.3, color='orange', label='Raw')
        
        window = min(100, len(metrics.loss_values) // 10)
        if len(metrics.loss_values) >= window and window > 0:
            moving_avg = np.convolve(metrics.loss_values, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(metrics.loss_values) + 1), 
                   moving_avg, color='red', linewidth=2, label=f'{window}-step MA')
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Epsilon Decay
    ax = axes[1, 1]
    if metrics.epsilon_values:
        episodes = range(1, len(metrics.epsilon_values) + 1)
        ax.plot(episodes, metrics.epsilon_values, color='brown', linewidth=2)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (Epsilon)')
        ax.grid(True, alpha=0.3)
    
    # 6. Reward Distribution
    ax = axes[1, 2]
    if metrics.episode_rewards:
        ax.hist(metrics.episode_rewards, bins=30, color='teal', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_alphazero_training_metrics(metrics: TrainingMetrics, save_path: Optional[str] = None,
                                    show: bool = False, title: str = "AlphaZero Training Metrics"):
    """
    绘制 AlphaZero 训练指标
    
    Args:
        metrics: TrainingMetrics 对象
        save_path: 保存路径
        show: 是否显示
        title: 图表标题
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # 1. Win Rate over Iterations
    ax = axes[0, 0]
    if metrics.win_rates and metrics.eval_episodes:
        ax.plot(metrics.eval_episodes, metrics.win_rates, 
               marker='o', linewidth=2, markersize=6, color='purple')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Opponents')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax.legend()
    
    # 2. Policy Loss
    ax = axes[0, 1]
    if metrics.policy_losses:
        iterations = range(1, len(metrics.policy_losses) + 1)
        ax.plot(iterations, metrics.policy_losses, alpha=0.3, color='blue', label='Raw')
        
        window = min(50, len(metrics.policy_losses) // 10)
        if len(metrics.policy_losses) >= window and window > 0:
            moving_avg = np.convolve(metrics.policy_losses, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(metrics.policy_losses) + 1), 
                   moving_avg, color='red', linewidth=2, label=f'{window}-iter MA')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Policy Loss')
        ax.set_title('Policy Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. Value Loss
    ax = axes[0, 2]
    if metrics.value_losses:
        iterations = range(1, len(metrics.value_losses) + 1)
        ax.plot(iterations, metrics.value_losses, alpha=0.3, color='green', label='Raw')
        
        window = min(50, len(metrics.value_losses) // 10)
        if len(metrics.value_losses) >= window and window > 0:
            moving_avg = np.convolve(metrics.value_losses, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(metrics.value_losses) + 1), 
                   moving_avg, color='red', linewidth=2, label=f'{window}-iter MA')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Value Loss')
        ax.set_title('Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 4. Total Loss
    ax = axes[1, 0]
    if metrics.policy_losses and metrics.value_losses:
        min_len = min(len(metrics.policy_losses), len(metrics.value_losses))
        total_losses = [metrics.policy_losses[i] + metrics.value_losses[i] 
                       for i in range(min_len)]
        iterations = range(1, len(total_losses) + 1)
        ax.plot(iterations, total_losses, alpha=0.3, color='orange', label='Raw')
        
        window = min(50, len(total_losses) // 10)
        if len(total_losses) >= window and window > 0:
            moving_avg = np.convolve(total_losses, 
                                    np.ones(window)/window, mode='valid')
            ax.plot(range(window, len(total_losses) + 1), 
                   moving_avg, color='red', linewidth=2, label=f'{window}-iter MA')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Total Loss')
        ax.set_title('Total Loss (Policy + Value)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 5. Learning Rate
    ax = axes[1, 1]
    if metrics.learning_rates:
        iterations = range(1, len(metrics.learning_rates) + 1)
        ax.plot(iterations, metrics.learning_rates, color='brown', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Learning Rate')
        ax.set_title('Learning Rate Schedule')
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # 6. ELO Rating
    ax = axes[1, 2]
    if metrics.elo_ratings:
        iterations = range(1, len(metrics.elo_ratings) + 1)
        ax.plot(iterations, metrics.elo_ratings, color='teal', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('ELO Rating')
        ax.set_title('ELO Rating Progression')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()

