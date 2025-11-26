"""
Optimized AlphaZero Training Script
Integrates all optimizations for maximum training throughput

Key Features:
1. Batched inference server for GPU efficiency
2. Parallel self-play with shared inference
3. FastBoard for efficient game state operations
4. Smaller, faster network architecture
5. Adaptive MCTS simulation count
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import time
import os
import sys
from datetime import datetime
from typing import Tuple, Optional
import argparse

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.alphazero.az_config_optimized import (
    az_config_optimized, FastDebugConfig, BalancedConfig, QualityConfig
)
from agents.alphazero.self_play_optimized import (
    ParallelSelfPlayEngine, SimpleSelfPlayEngine
)
from agents.alphazero.fast_board import FastBoard


class LightPolicyValueNetwork(nn.Module):
    """
    Lightweight Policy-Value Network optimized for ConnectX.
    
    Architecture:
    - 3-4 convolutional layers (instead of deep ResNet)
    - Small filter count (64 instead of 256)
    - Efficient for quick inference and training
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or az_config_optimized
        
        num_filters = self.config.NUM_FILTERS
        
        # Convolutional backbone (simpler than ResNet)
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        
        # Optional residual blocks
        self.use_residual = self.config.NUM_RES_BLOCKS > 0
        if self.use_residual:
            self.res_blocks = nn.ModuleList([
                self._make_res_block(num_filters) 
                for _ in range(self.config.NUM_RES_BLOCKS)
            ])
        
        # Policy head
        policy_filters = self.config.POLICY_FILTERS
        self.policy_conv = nn.Conv2d(num_filters, policy_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_fc = nn.Linear(policy_filters * 6 * 7, 7)
        
        # Value head
        value_filters = self.config.VALUE_FILTERS
        self.value_conv = nn.Conv2d(num_filters, value_filters, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_filters)
        self.value_fc1 = nn.Linear(value_filters * 6 * 7, self.config.VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(self.config.VALUE_HIDDEN, 1)
        
        self.dropout = nn.Dropout(self.config.DROPOUT)
    
    def _make_res_block(self, num_filters):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Backbone
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual blocks
        if self.use_residual:
            for res_block in self.res_blocks:
                residual = x
                x = res_block(x)
                x = F.relu(x + residual)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.dropout(policy)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value


class AlphaZeroTrainerOptimized:
    """
    Optimized AlphaZero trainer with all performance improvements.
    """
    
    def __init__(self, config=None, use_parallel: bool = True):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
            use_parallel: Whether to use parallel self-play
        """
        self.config = config or az_config_optimized
        self.device = self.config.DEVICE
        
        # Create network
        self.network = LightPolicyValueNetwork(self.config)
        self.network.to(self.device)
        
        # Print model info
        num_params = sum(p.numel() for p in self.network.parameters())
        print(f"Network parameters: {num_params:,}")
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.L2_REGULARIZATION
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.MAX_ITERATIONS * self.config.TRAINING_EPOCHS
        )
        
        # Mixed precision scaler
        self.scaler = GradScaler('cuda') if self.config.USE_AMP else None
        
        # Self-play engine
        if use_parallel:
            self.self_play = ParallelSelfPlayEngine(
                self.network,
                num_parallel_games=self.config.NUM_PARALLEL_GAMES,
                config=self.config,
                use_batched_inference=self.config.USE_BATCHED_INFERENCE
            )
        else:
            self.self_play = SimpleSelfPlayEngine(self.network, self.config)
        
        # Training stats
        self.iteration = 0
        self.total_games = 0
        self.best_win_rate = 0.0
        
        # Create directories
        self._create_directories()
        
        # Run name for logging
        self.run_name = f"alphazero_opt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def _create_directories(self):
        """Create necessary directories."""
        for dir_path in [self.config.MODEL_DIR, self.config.CHECKPOINT_DIR,
                        self.config.LOG_DIR, self.config.PLOT_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    def train_iteration(self) -> dict:
        """
        Run one training iteration.
        
        Returns:
            Dictionary with training statistics
        """
        stats = {}
        self.iteration += 1
        
        print(f"\n{'='*60}")
        print(f"Iteration {self.iteration}")
        print(f"{'='*60}")
        
        # Phase 1: Self-play
        print("\nPhase 1: Self-Play")
        self.network.eval()
        
        if hasattr(self.self_play, 'start'):
            self.self_play.start()
        
        selfplay_start = time.perf_counter()
        
        # Use batched generation for parallel engine
        if isinstance(self.self_play, ParallelSelfPlayEngine):
            num_examples = self.self_play.generate_games_batched(
                num_games=self.config.NUM_SELFPLAY_GAMES,
                batch_size=self.config.NUM_PARALLEL_GAMES
            )
        else:
            num_examples = self.self_play.generate_self_play_data(
                num_games=self.config.NUM_SELFPLAY_GAMES
            )
        
        selfplay_time = time.perf_counter() - selfplay_start
        
        if hasattr(self.self_play, 'stop'):
            self.self_play.stop()
        
        self.total_games += self.config.NUM_SELFPLAY_GAMES
        stats['selfplay_time'] = selfplay_time
        stats['selfplay_examples'] = num_examples
        stats['selfplay_games_per_sec'] = self.config.NUM_SELFPLAY_GAMES / selfplay_time
        
        print(f"  Generated {num_examples} examples in {selfplay_time:.1f}s "
              f"({stats['selfplay_games_per_sec']:.1f} games/s)")
        
        # Phase 2: Training
        print("\nPhase 2: Training")
        self.network.train()
        
        training_start = time.perf_counter()
        train_stats = self._train_network()
        training_time = time.perf_counter() - training_start
        
        stats.update(train_stats)
        stats['training_time'] = training_time
        
        print(f"  Trained for {self.config.TRAINING_EPOCHS} epochs in {training_time:.1f}s")
        print(f"  Policy loss: {train_stats['policy_loss']:.4f}, "
              f"Value loss: {train_stats['value_loss']:.4f}")
        
        # Phase 3: Evaluation (periodic)
        if self.iteration % self.config.EVAL_INTERVAL == 0:
            print("\nPhase 3: Evaluation")
            eval_stats = self._evaluate()
            stats.update(eval_stats)
        
        # Save checkpoint
        if self.iteration % self.config.SAVE_INTERVAL == 0:
            self._save_checkpoint()
        
        # Print summary
        total_time = selfplay_time + training_time
        print(f"\nIteration {self.iteration} complete in {total_time:.1f}s")
        print(f"  Total games played: {self.total_games}")
        print(f"  Buffer size: {len(self.self_play.buffer)}")
        
        return stats
    
    def _train_network(self) -> dict:
        """Train the network on collected data."""
        if len(self.self_play.buffer) < self.config.BATCH_SIZE:
            return {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0}
        
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for epoch in range(self.config.TRAINING_EPOCHS):
            # Sample batch
            states, policies, values = self.self_play.get_training_batch(
                self.config.BATCH_SIZE
            )
            
            # Convert to tensors
            states = torch.from_numpy(states).to(self.device)
            target_policies = torch.from_numpy(policies).to(self.device)
            target_values = torch.from_numpy(values).to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.USE_AMP and self.scaler:
                with autocast('cuda'):
                    policy_logits, values = self.network(states)
                    
                    # Losses
                    policy_loss = F.cross_entropy(policy_logits, target_policies)
                    value_loss = F.mse_loss(values, target_values)
                    total_loss = policy_loss + value_loss
                
                # Backward with scaling
                self.scaler.scale(total_loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy_logits, values = self.network(states)
                
                policy_loss = F.cross_entropy(policy_logits, target_policies)
                value_loss = F.mse_loss(values, target_values)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'total_loss': (total_policy_loss + total_value_loss) / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _evaluate(self) -> dict:
        """Evaluate against baseline opponents."""
        self.network.eval()
        stats = {}
        
        # Quick evaluation against random
        wins, losses, draws = self._play_against_random(
            num_games=self.config.EVAL_GAMES
        )
        
        win_rate = wins / self.config.EVAL_GAMES
        stats['vs_random_win_rate'] = win_rate
        stats['vs_random_wins'] = wins
        stats['vs_random_losses'] = losses
        stats['vs_random_draws'] = draws
        
        print(f"  vs Random: {wins}W / {losses}L / {draws}D ({win_rate*100:.1f}% win)")
        
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self._save_checkpoint("best")
            print(f"  New best model! (win rate: {win_rate*100:.1f}%)")
        
        return stats
    
    def _play_against_random(self, num_games: int) -> Tuple[int, int, int]:
        """Play games against random opponent."""
        from agents.alphazero.mcts_optimized import MCTSOptimized
        from agents.alphazero.batched_inference import SyncInferenceWrapper
        
        inference = SyncInferenceWrapper(self.network)
        mcts = MCTSOptimized(inference.inference, config=self.config)
        
        wins = losses = draws = 0
        
        for game_idx in range(num_games):
            # Alternate who goes first
            az_mark = 1 if game_idx % 2 == 0 else 2
            
            board = FastBoard()
            current_mark = 1
            
            while True:
                if current_mark == az_mark:
                    # AlphaZero move
                    action = mcts.get_best_action(board, current_mark)
                else:
                    # Random move
                    valid_moves = board.get_valid_moves()
                    action = np.random.choice(valid_moves)
                
                board.make_move_inplace(action, current_mark)
                
                is_terminal, winner = board.is_terminal()
                if is_terminal:
                    if winner == az_mark:
                        wins += 1
                    elif winner == 0:
                        draws += 1
                    else:
                        losses += 1
                    break
                
                current_mark = 3 - current_mark
        
        return wins, losses, draws
    
    def _save_checkpoint(self, name: str = None):
        """Save model checkpoint."""
        if name is None:
            name = f"iter{self.iteration}"
        
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f"{name}_{self.run_name}.pth"
        )
        
        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_games': self.total_games,
            'best_win_rate': self.best_win_rate,
            'config': str(self.config)
        }, checkpoint_path)
        
        print(f"  Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
        
        print(f"Loaded checkpoint from iteration {self.iteration}")
    
    def train(self, num_iterations: int = None):
        """
        Run full training loop.
        
        Args:
            num_iterations: Number of iterations (default from config)
        """
        if num_iterations is None:
            num_iterations = self.config.MAX_ITERATIONS
        
        print(f"\nStarting AlphaZero training")
        print(f"  Config: {self.config}")
        print(f"  Device: {self.device}")
        print(f"  Iterations: {num_iterations}")
        
        start_time = time.perf_counter()
        
        try:
            for _ in range(num_iterations):
                self.train_iteration()
                
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            self._save_checkpoint("interrupted")
        
        total_time = time.perf_counter() - start_time
        
        print(f"\n{'='*60}")
        print("Training Complete")
        print(f"  Total iterations: {self.iteration}")
        print(f"  Total games: {self.total_games}")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best win rate vs random: {self.best_win_rate*100:.1f}%")
        print(f"{'='*60}")
        
        # Save final model
        self._save_checkpoint("final")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train AlphaZero for ConnectX')
    parser.add_argument('--config', type=str, default='optimized',
                       choices=['debug', 'optimized', 'balanced', 'quality'],
                       help='Configuration preset')
    parser.add_argument('--iterations', type=int, default=None,
                       help='Number of training iterations')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint to resume from')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable parallel self-play')
    
    args = parser.parse_args()
    
    # Select config
    if args.config == 'debug':
        config = FastDebugConfig()
    elif args.config == 'balanced':
        config = BalancedConfig()
    elif args.config == 'quality':
        config = QualityConfig()
    else:
        config = az_config_optimized
    
    print(f"Using config: {args.config}")
    print(f"  Simulations: {config.NUM_SIMULATIONS}")
    print(f"  Network: {config.NUM_RES_BLOCKS} res blocks, {config.NUM_FILTERS} filters")
    print(f"  Self-play games per iteration: {config.NUM_SELFPLAY_GAMES}")
    print(f"  Parallel games: {config.NUM_PARALLEL_GAMES}")
    
    # Create trainer
    trainer = AlphaZeroTrainerOptimized(
        config=config,
        use_parallel=not args.no_parallel
    )
    
    # Load checkpoint if specified
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # Train
    trainer.train(num_iterations=args.iterations)


if __name__ == "__main__":
    main()

