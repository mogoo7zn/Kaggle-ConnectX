"""
Training Script for AlphaZero
Iterative training loop: Self-play -> Train -> Evaluate -> Update
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import time
from datetime import datetime
from typing import Tuple, Dict
from torch.utils.tensorboard import SummaryWriter

from agents.alphazero.az_config import az_config
from agents.alphazero.az_model import create_alphazero_model, count_parameters
from agents.alphazero.mcts import MCTS
from agents.alphazero.self_play import SelfPlayEngine
from agents.base.utils import get_valid_moves, make_move, is_terminal

# 导入绘图工具
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from tools.plot_training import TrainingMetrics, plot_alphazero_training_metrics


class AlphaZeroTrainer:
    """
    Trainer for AlphaZero agent.
    
    Implements the iterative training loop:
    1. Generate self-play data
    2. Train neural network
    3. Evaluate new network vs old
    4. Update if improved
    """
    
    def __init__(self, network=None, run_name: str = None):
        """
        Initialize AlphaZero trainer.
        
        Args:
            network: Policy-value network (creates new if None)
            run_name: Name for this training run
        """
        self.run_name = run_name or f"alphazero_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.device = az_config.DEVICE
        
        # Create or use provided network
        if network is None:
            # Use light model for faster training
            self.network = create_alphazero_model('light')
        else:
            self.network = network
        
        self.network.to(self.device)
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=az_config.LEARNING_RATE,
            momentum=az_config.MOMENTUM,
            weight_decay=az_config.L2_REGULARIZATION
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=az_config.LR_MILESTONES,
            gamma=az_config.LR_GAMMA
        )
        
        # Mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if az_config.USE_AMP else None
        
        # Self-play engine
        self.selfplay_engine = SelfPlayEngine(self.network)
        
        # Create directories
        os.makedirs(az_config.MODEL_DIR, exist_ok=True)
        os.makedirs(az_config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(az_config.LOG_DIR, exist_ok=True)
        os.makedirs(az_config.PLOT_DIR, exist_ok=True)
        
        # TensorBoard
        self.writer = SummaryWriter(os.path.join(az_config.LOG_DIR, "runs", self.run_name))
        
        # Training statistics
        self.iteration = 0
        self.best_win_rate = 0.0
        self.elo_rating = az_config.INITIAL_ELO
        
        # Metrics - 使用 TrainingMetrics 类收集数据
        self.metrics = TrainingMetrics()
        
        print(f"AlphaZero Trainer initialized")
        print(f"Model parameters: {count_parameters(self.network):,}")
        print(f"Device: {self.device}")
    
    def train_network(self, num_epochs: int = None) -> Dict[str, float]:
        """
        Train network on current buffer data.
        
        Args:
            num_epochs: Number of training epochs (default from config)
        
        Returns:
            Dictionary of training statistics
        """
        if num_epochs is None:
            num_epochs = az_config.TRAINING_EPOCHS
        
        if not self.selfplay_engine.buffer.is_ready(az_config.MIN_REPLAY_SIZE):
            print(f"Not enough data in buffer ({len(self.selfplay_engine.buffer)} < {az_config.MIN_REPLAY_SIZE})")
            return {}
        
        self.network.train()
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(num_epochs):
            # Sample batch
            states, target_policies, target_values = self.selfplay_engine.get_training_batch(
                az_config.BATCH_SIZE
            )
            
            # Convert to tensors
            states = torch.from_numpy(states).float().to(self.device)
            target_policies = torch.from_numpy(target_policies).float().to(self.device)
            target_values = torch.from_numpy(target_values).float().unsqueeze(1).to(self.device)
            
            # Forward pass
            if az_config.USE_AMP:
                with torch.cuda.amp.autocast():
                    policy_logits, values = self.network(states)
                    policy_loss = self._policy_loss(policy_logits, target_policies)
                    value_loss = self._value_loss(values, target_values)
                    loss = policy_loss + value_loss
                
                # Backward pass with mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy_logits, values = self.network(states)
                policy_loss = self._policy_loss(policy_logits, target_policies)
                value_loss = self._value_loss(values, target_values)
                loss = policy_loss + value_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            # Statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_loss += loss.item()
            num_batches += 1
        
        # Update learning rate
        self.scheduler.step()
        
        # Return statistics
        stats = {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'total_loss': total_loss / num_batches,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
        
        return stats
    
    def _policy_loss(self, policy_logits: torch.Tensor,
                    target_policies: torch.Tensor) -> torch.Tensor:
        """
        Compute policy loss (cross-entropy).
        
        Args:
            policy_logits: Predicted policy logits
            target_policies: Target policy distributions from MCTS
        
        Returns:
            Policy loss
        """
        # Cross-entropy loss
        log_probs = F.log_softmax(policy_logits, dim=1)
        loss = -(target_policies * log_probs).sum(dim=1).mean()
        return loss
    
    def _value_loss(self, values: torch.Tensor,
                   target_values: torch.Tensor) -> torch.Tensor:
        """
        Compute value loss (MSE).
        
        Args:
            values: Predicted values
            target_values: Target values from game outcomes
        
        Returns:
            Value loss
        """
        return F.mse_loss(values, target_values)
    
    def evaluate_network(self, num_games: int = None) -> float:
        """
        Evaluate current network against baseline opponents.
        
        Args:
            num_games: Number of evaluation games (default from config)
        
        Returns:
            Win rate
        """
        if num_games is None:
            num_games = az_config.ARENA_GAMES_PER_OPPONENT
        
        self.network.eval()
        mcts = MCTS(self.network)
        
        total_wins = 0
        total_games = 0
        
        # Evaluate against random
        wins_random = self._evaluate_vs_opponent(mcts, self._random_policy, num_games // 2)
        total_wins += wins_random
        total_games += num_games // 2
        
        # Evaluate against greedy
        wins_greedy = self._evaluate_vs_opponent(mcts, self._greedy_policy, num_games // 2)
        total_wins += wins_greedy
        total_games += num_games // 2
        
        win_rate = total_wins / total_games if total_games > 0 else 0.0
        
        return win_rate
    
    def _evaluate_vs_opponent(self, mcts: MCTS, opponent_fn, num_games: int) -> int:
        """Evaluate MCTS agent vs opponent."""
        wins = 0
        
        for game in range(num_games):
            board = [0] * (az_config.ROWS * az_config.COLUMNS)
            
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
                    # Use MCTS with fewer simulations for speed
                    action = mcts.get_best_action(board, current_mark)
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
    
    def _random_policy(self, board, mark):
        """Random baseline policy."""
        moves = get_valid_moves(board)
        return random.choice(moves) if moves else 0
    
    def _greedy_policy(self, board, mark):
        """Greedy policy using network evaluation."""
        from agents.base.utils import encode_state
        import torch
        
        state = encode_state(board, mark)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, _ = self.network(state_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
        
        # Mask and select
        valid_moves = get_valid_moves(board)
        masked_probs = np.zeros_like(policy_probs)
        masked_probs[valid_moves] = policy_probs[valid_moves]
        
        if masked_probs.sum() > 0:
            return int(np.argmax(masked_probs))
        return random.choice(valid_moves) if valid_moves else 0
    
    def train(self, max_iterations: int = None):
        """
        Main training loop.
        
        Args:
            max_iterations: Maximum training iterations (default from config)
        """
        if max_iterations is None:
            max_iterations = az_config.MAX_ITERATIONS
        
        print(f"\n{'='*60}")
        print(f"Starting AlphaZero Training")
        print(f"Max iterations: {max_iterations}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for iteration in range(1, max_iterations + 1):
            self.iteration = iteration
            iter_start = time.time()
            
            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{max_iterations}")
            print(f"{'='*60}")
            
            # 1. Self-play
            print(f"\n[1/3] Generating self-play data...")
            num_examples = self.selfplay_engine.generate_self_play_data(
                num_games=az_config.NUM_SELFPLAY_GAMES
            )
            
            # 2. Train network
            print(f"\n[2/3] Training network...")
            train_stats = self.train_network()
            
            if train_stats:
                print(f"  Policy loss: {train_stats['policy_loss']:.4f}")
                print(f"  Value loss: {train_stats['value_loss']:.4f}")
                print(f"  Total loss: {train_stats['total_loss']:.4f}")
                print(f"  Learning rate: {train_stats['learning_rate']:.6f}")
                
                # 记录损失到 metrics
                self.metrics.add_alphazero_losses(
                    train_stats['policy_loss'],
                    train_stats['value_loss'],
                    train_stats['learning_rate']
                )
                
                # Log to TensorBoard
                self.writer.add_scalar('train/policy_loss', train_stats['policy_loss'], iteration)
                self.writer.add_scalar('train/value_loss', train_stats['value_loss'], iteration)
                self.writer.add_scalar('train/total_loss', train_stats['total_loss'], iteration)
                self.writer.add_scalar('train/learning_rate', train_stats['learning_rate'], iteration)
            
            # 3. Evaluate
            if iteration % az_config.EVAL_INTERVAL == 0:
                print(f"\n[3/3] Evaluating network...")
                win_rate = self.evaluate_network()
                print(f"  Win rate: {win_rate:.2%}")
                
                # 记录评估结果到 metrics
                self.metrics.add_evaluation(iteration, win_rate)
                self.writer.add_scalar('eval/win_rate', win_rate, iteration)
                
                # Save if improved
                if win_rate > self.best_win_rate:
                    self.best_win_rate = win_rate
                    model_path = os.path.join(az_config.CHECKPOINT_DIR,
                                            f'best_alphazero_{self.run_name}.pth')
                    self.save_model(model_path)
                    print(f"  ✓ New best model saved!")
                
                # Update ELO
                self.elo_rating = self._update_elo(self.elo_rating, win_rate)
                print(f"  ELO rating: {self.elo_rating:.0f}")
                self.metrics.add_elo(self.elo_rating)
                self.writer.add_scalar('eval/elo', self.elo_rating, iteration)
            
            # Save checkpoint
            if iteration % az_config.SAVE_INTERVAL == 0:
                checkpoint_path = os.path.join(az_config.CHECKPOINT_DIR,
                                             f'alphazero_iter{iteration}_{self.run_name}.pth')
                self.save_checkpoint(checkpoint_path, iteration)
            
            iter_time = time.time() - iter_start
            print(f"\nIteration time: {iter_time:.1f}s")
            print(f"Buffer size: {len(self.selfplay_engine.buffer)}")
        
        # Final statistics
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Total time: {total_time/3600:.2f} hours")
        print(f"Best win rate: {self.best_win_rate:.2%}")
        print(f"Final ELO: {self.elo_rating:.0f}")
        print(f"{'='*60}\n")
        
        # Save final model
        final_path = os.path.join(az_config.MODEL_DIR,
                                  f'final_alphazero_{self.run_name}.pth')
        self.save_model(final_path)
        
        # 生成训练图像
        plot_path = os.path.join(az_config.PLOT_DIR, 
                                f'training_metrics_{self.run_name}.png')
        plot_alphazero_training_metrics(self.metrics, save_path=plot_path, show=False,
                                       title=f'AlphaZero Training Metrics - {self.run_name}')
        
        self.writer.close()
    
    def _update_elo(self, current_elo: float, win_rate: float) -> float:
        """Update ELO rating based on win rate."""
        expected_score = 0.5  # Assume opponents are at baseline
        actual_score = win_rate
        return current_elo + az_config.K_FACTOR * (actual_score - expected_score)
    
    def save_model(self, filepath: str):
        """Save model weights."""
        torch.save(self.network.state_dict(), filepath)
        print(f"Model saved to {filepath}")
    
    def save_checkpoint(self, filepath: str, iteration: int):
        """Save full checkpoint."""
        checkpoint = {
            'iteration': iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_win_rate': self.best_win_rate,
            'elo_rating': self.elo_rating,
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """Load checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.best_win_rate = checkpoint['best_win_rate']
        self.elo_rating = checkpoint['elo_rating']
        print(f"Checkpoint loaded from {filepath}")


def main():
    """Main training function."""
    # Set random seeds
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create trainer
    print("Creating AlphaZero Trainer...")
    trainer = AlphaZeroTrainer()
    
    # Train
    trainer.train(max_iterations=az_config.MAX_ITERATIONS)
    
    print("\n🎉 AlphaZero training completed!")


if __name__ == "__main__":
    main()

