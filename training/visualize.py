"""
Visualization utilities for DQN training
Generates plots for training metrics and performance analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))
from config import config


def plot_training_metrics(metrics, save_path: Optional[str] = None, show: bool = False):
    """
    Plot comprehensive training metrics.
    
    Args:
        metrics: TrainingMetrics object with training history
        save_path: Path to save plot (optional)
        show: Whether to display plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('DQN ConnectX Training Metrics', fontsize=16, fontweight='bold')
    
    # 1. Episode Rewards
    ax = axes[0, 0]
    if metrics.episode_rewards:
        episodes = range(1, len(metrics.episode_rewards) + 1)
        ax.plot(episodes, metrics.episode_rewards, alpha=0.3, color='blue', label='Raw')
        
        # Moving average
        window = 100
        if len(metrics.episode_rewards) >= window:
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
        
        # Moving average
        window = 100
        if len(metrics.episode_lengths) >= window:
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
    if metrics.win_rates:
        ax.plot(metrics.eval_episodes, metrics.win_rates, 
               marker='o', linewidth=2, markersize=6, color='purple')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Win Rate')
        ax.set_title('Win Rate vs Random Opponent')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        # Add horizontal line at 50%
        ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
        ax.legend()
    
    # 4. Training Loss
    ax = axes[1, 0]
    if metrics.loss_values:
        steps = range(1, len(metrics.loss_values) + 1)
        ax.plot(steps, metrics.loss_values, alpha=0.3, color='orange', label='Raw')
        
        # Moving average
        window = 100
        if len(metrics.loss_values) >= window:
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
    
    # Save plot
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Training plot saved to {save_path}")
    
    # Show plot
    if show:
        plt.show()
    else:
        plt.close()


def plot_win_rates(metrics_dict: dict, save_path: Optional[str] = None, show: bool = False):
    """
    Plot win rates comparison across different training phases.
    
    Args:
        metrics_dict: Dictionary mapping phase names to TrainingMetrics objects
        save_path: Path to save plot (optional)
        show: Whether to display plot
    """
    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'green', 'red', 'purple', 'orange']
    
    for idx, (phase_name, metrics) in enumerate(metrics_dict.items()):
        if metrics.win_rates:
            color = colors[idx % len(colors)]
            plt.plot(metrics.eval_episodes, metrics.win_rates, 
                    marker='o', linewidth=2, markersize=5, 
                    label=phase_name, color=color)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Win Rate', fontsize=12)
    plt.title('Win Rate Progression Across Training Phases', fontsize=14, fontweight='bold')
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Add horizontal line at 50%
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% baseline')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Win rates plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_q_value_heatmap(agent, board: List[int], mark: int, 
                        save_path: Optional[str] = None, show: bool = False):
    """
    Plot Q-value heatmap for a given board state.
    
    Args:
        agent: DQN agent
        board: Board state
        mark: Player mark
        save_path: Path to save plot (optional)
        show: Whether to display plot
    """
    import torch
    from utils import encode_state, state_to_tensor, get_valid_moves
    
    # Get Q-values
    state = encode_state(board, mark)
    state_tensor = state_to_tensor(state, agent.device).unsqueeze(0)
    
    with torch.no_grad():
        q_values = agent.policy_net(state_tensor).cpu().numpy()[0]
    
    valid_moves = get_valid_moves(board)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot board
    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)
    
    # Create color map for board
    cmap_board = plt.cm.colors.ListedColormap(['white', 'red', 'yellow'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap_board.N)
    
    im1 = ax1.imshow(board_2d, cmap=cmap_board, norm=norm)
    ax1.set_title('Current Board State', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.set_xticks(range(config.COLUMNS))
    ax1.set_yticks(range(config.ROWS))
    
    # Add grid
    for i in range(config.ROWS + 1):
        ax1.axhline(i - 0.5, color='black', linewidth=1)
    for i in range(config.COLUMNS + 1):
        ax1.axvline(i - 0.5, color='black', linewidth=1)
    
    # Plot Q-values
    colors = ['red' if i not in valid_moves else 'green' for i in range(config.COLUMNS)]
    bars = ax2.bar(range(config.COLUMNS), q_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_title('Q-Values by Column', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Column')
    ax2.set_ylabel('Q-Value')
    ax2.set_xticks(range(config.COLUMNS))
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, q_val) in enumerate(zip(bars, q_values)):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{q_val:.2f}',
                ha='center', va='bottom' if height >= 0 else 'top',
                fontsize=9)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='green', alpha=0.7, label='Valid Move'),
                      Patch(facecolor='red', alpha=0.7, label='Invalid Move')]
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Q-value heatmap saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def create_training_summary(agent, metrics_dict: dict, save_path: str):
    """
    Create a comprehensive training summary report.
    
    Args:
        agent: Trained DQN agent
        metrics_dict: Dictionary of training metrics
        save_path: Path to save summary text file
    """
    with open(save_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("DQN ConnectX Training Summary\n")
        f.write("="*70 + "\n\n")
        
        # Agent info
        f.write("Agent Configuration:\n")
        f.write(f"  Device: {config.DEVICE}\n")
        f.write(f"  Total training steps: {agent.steps_done}\n")
        f.write(f"  Final epsilon: {agent.epsilon:.4f}\n")
        f.write(f"  Buffer size: {len(agent.memory)}\n")
        f.write(f"  Model parameters: {sum(p.numel() for p in agent.policy_net.parameters()):,}\n")
        f.write("\n")
        
        # Hyperparameters
        f.write("Hyperparameters:\n")
        f.write(f"  Learning rate: {config.LEARNING_RATE}\n")
        f.write(f"  Batch size: {config.BATCH_SIZE}\n")
        f.write(f"  Gamma: {config.GAMMA}\n")
        f.write(f"  Target update frequency: {config.TARGET_UPDATE_FREQ}\n")
        f.write(f"  Replay buffer size: {config.REPLAY_BUFFER_SIZE}\n")
        f.write("\n")
        
        # Training phases summary
        f.write("Training Phases Summary:\n")
        for phase_name, metrics in metrics_dict.items():
            f.write(f"\n  {phase_name}:\n")
            f.write(f"    Episodes: {len(metrics.episode_rewards)}\n")
            
            if metrics.episode_rewards:
                f.write(f"    Average reward: {np.mean(metrics.episode_rewards):.3f}\n")
                f.write(f"    Final 100 avg reward: {metrics.get_recent_avg_reward():.3f}\n")
            
            if metrics.episode_lengths:
                f.write(f"    Average episode length: {np.mean(metrics.episode_lengths):.1f}\n")
            
            if metrics.win_rates:
                f.write(f"    Best win rate: {max(metrics.win_rates):.2%}\n")
                f.write(f"    Final win rate: {metrics.win_rates[-1]:.2%}\n")
        
        f.write("\n" + "="*70 + "\n")
    
    print(f"Training summary saved to {save_path}")


if __name__ == "__main__":
    # Test visualization with dummy data
    from train_dqn import TrainingMetrics
    
    # Create dummy metrics
    metrics = TrainingMetrics()
    
    for i in range(500):
        reward = np.random.randn() + (i / 100)  # Increasing trend
        length = np.random.randint(10, 42)
        epsilon = max(0.01, 1.0 - i / 500)
        
        metrics.add_episode(reward, length, epsilon)
        
        if i % 5 == 0:
            loss = np.random.rand() * 0.5
            metrics.add_loss(loss)
        
        if i % 50 == 0:
            win_rate = min(0.95, 0.1 + i / 600)
            metrics.add_evaluation(i, win_rate)
    
    # Generate plot
    plot_path = os.path.join(config.PLOT_DIR, 'test_metrics.png')
    plot_training_metrics(metrics, save_path=plot_path, show=False)
    
    print("Test visualization complete!")

