"""
Self-Play Engine for AlphaZero
Generates training data through self-play games using MCTS
"""

import numpy as np
import random
from typing import List, Tuple, Dict
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.alphazero.az_config import az_config
from agents.alphazero.mcts import MCTS
from agents.base.utils import encode_state, get_valid_moves, make_move, is_terminal


class SelfPlayBuffer:
    """
    Replay buffer for self-play training data.
    
    Stores (state, policy, value) tuples from self-play games.
    """
    
    def __init__(self, capacity: int = None):
        """
        Initialize self-play buffer.
        
        Args:
            capacity: Maximum buffer size (default from config)
        """
        if capacity is None:
            capacity = az_config.REPLAY_BUFFER_SIZE
        
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, policy: np.ndarray, value: float):
        """
        Add a training example to buffer.
        
        Args:
            state: Board state encoding
            policy: MCTS policy distribution
            value: Game outcome from this state's perspective
        """
        self.buffer.append((state, policy, value))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of training examples.
        
        Args:
            batch_size: Number of examples to sample
        
        Returns:
            Tuple of (states, policies, values)
        """
        indices = np.random.choice(len(self.buffer), batch_size,
                                  replace=False if len(self.buffer) >= batch_size else True)
        
        batch = [self.buffer[i] for i in indices]
        states, policies, values = zip(*batch)
        
        return (
            np.array(states),
            np.array(policies),
            np.array(values, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        """Check if buffer has enough samples."""
        return len(self) >= min_size


class SelfPlayEngine:
    """
    Self-play engine for generating training data.
    
    Features:
    - Plays games against itself using MCTS
    - Stores (state, policy, outcome) for training
    - Supports data augmentation (horizontal flip)
    - Temperature-based exploration
    """
    
    def __init__(self, network, config=None):
        """
        Initialize self-play engine.
        
        Args:
            network: Policy-value neural network
            config: Configuration object (default: az_config)
        """
        self.network = network
        self.config = config or az_config
        self.mcts = MCTS(network, config)
        self.buffer = SelfPlayBuffer(self.config.REPLAY_BUFFER_SIZE)
    
    def play_game(self, temperature_threshold: int = None,
                 add_noise: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play one self-play game.
        
        Args:
            temperature_threshold: Move number to switch from exploration to exploitation
            add_noise: Whether to add Dirichlet noise at root
        
        Returns:
            List of (state, policy, value) tuples
        """
        if temperature_threshold is None:
            temperature_threshold = self.config.TEMP_THRESHOLD
        
        # Initialize game
        board = [0] * (az_config.ROWS * az_config.COLUMNS)
        current_mark = 1
        move_count = 0
        
        # Store game trajectory
        game_history = []  # List of (state, policy, mark)
        
        # Play until terminal
        while True:
            # Determine temperature
            if move_count < temperature_threshold:
                temperature = self.config.TEMPERATURE
            else:
                temperature = 0.0  # Greedy
            
            # Encode current state
            state = encode_state(board, current_mark)
            
            # Run MCTS to get policy
            policy, _ = self.mcts.search(
                board, current_mark,
                temperature=temperature,
                add_noise=add_noise
            )
            
            # Store state and policy
            game_history.append((state, policy, current_mark))
            
            # Sample action from policy
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                break
            
            # Sample action based on policy
            action = self._sample_action(policy, valid_moves)
            
            # Make move
            board = make_move(board, action, current_mark)
            move_count += 1
            
            # Check if game ended
            done, winner = is_terminal(board)
            if done:
                # Assign values based on outcome
                game_data = self._process_game_outcome(game_history, winner)
                return game_data
            
            # Switch player
            current_mark = 3 - current_mark
        
        # Should not reach here, but return empty if it does
        return []
    
    def _sample_action(self, policy: np.ndarray, valid_moves: List[int]) -> int:
        """
        Sample action from policy.
        
        Args:
            policy: Action probability distribution
            valid_moves: List of valid actions
        
        Returns:
            Selected action
        """
        # Ensure policy is valid
        policy = np.array(policy)
        policy = policy[valid_moves]
        
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = np.ones(len(valid_moves)) / len(valid_moves)
        
        # Sample action
        action_idx = np.random.choice(len(valid_moves), p=policy)
        return valid_moves[action_idx]
    
    def _process_game_outcome(self, game_history: List[Tuple],
                              winner: int) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Process game outcome and assign values.
        
        Args:
            game_history: List of (state, policy, mark) tuples
            winner: Winner of the game (0 for draw)
        
        Returns:
            List of (state, policy, value) tuples for training
        """
        game_data = []
        
        for state, policy, mark in game_history:
            # Assign value from this player's perspective
            if winner == 0:  # Draw
                value = 0.0
            elif winner == mark:  # Win
                value = 1.0
            else:  # Loss
                value = -1.0
            
            game_data.append((state, policy, value))
        
        return game_data
    
    def augment_data(self, state: np.ndarray, policy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Augment data with horizontal flip.
        
        Args:
            state: Board state
            policy: Policy distribution
        
        Returns:
            Tuple of (augmented_state, augmented_policy)
        """
        # Flip board horizontally
        flipped_state = np.flip(state, axis=2).copy()
        
        # Flip policy (reverse order)
        flipped_policy = np.flip(policy).copy()
        
        return flipped_state, flipped_policy
    
    def generate_self_play_data(self, num_games: int = None,
                               use_augmentation: bool = None) -> int:
        """
        Generate self-play data for training.
        
        Args:
            num_games: Number of games to play (default from config)
            use_augmentation: Whether to use data augmentation (default from config)
        
        Returns:
            Number of training examples generated
        """
        if num_games is None:
            num_games = self.config.NUM_SELFPLAY_GAMES
        if use_augmentation is None:
            use_augmentation = self.config.USE_AUGMENTATION
        
        examples_generated = 0
        
        print(f"Generating self-play data: {num_games} games...")
        
        for game_num in range(num_games):
            # Play one game
            game_data = self.play_game()
            
            # Add to buffer
            for state, policy, value in game_data:
                self.buffer.push(state, policy, value)
                examples_generated += 1
                
                # Data augmentation
                if use_augmentation:
                    aug_state, aug_policy = self.augment_data(state, policy)
                    self.buffer.push(aug_state, aug_policy, value)
                    examples_generated += 1
            
            # Progress reporting
            if (game_num + 1) % 10 == 0:
                print(f"  Generated {game_num + 1}/{num_games} games "
                      f"({examples_generated} examples)")
        
        print(f"Self-play complete: {examples_generated} examples generated")
        print(f"Buffer size: {len(self.buffer)}")
        
        return examples_generated
    
    def get_training_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get a batch of training data.
        
        Args:
            batch_size: Batch size
        
        Returns:
            Tuple of (states, policies, values)
        """
        return self.buffer.sample(batch_size)


class ParallelSelfPlay:
    """
    Parallel self-play using multiple processes (future extension).
    
    For now, this is a placeholder for sequential self-play.
    """
    
    def __init__(self, network, num_workers: int = 1):
        """
        Initialize parallel self-play.
        
        Args:
            network: Policy-value network
            num_workers: Number of parallel workers
        """
        self.engine = SelfPlayEngine(network)
        self.num_workers = num_workers
    
    def generate_games(self, num_games: int) -> SelfPlayBuffer:
        """
        Generate games using parallel workers.
        
        Args:
            num_games: Total number of games to generate
        
        Returns:
            Buffer containing all generated examples
        """
        # For now, just sequential generation
        # TODO: Implement multiprocessing for true parallelism
        self.engine.generate_self_play_data(num_games)
        return self.engine.buffer


if __name__ == "__main__":
    # Test self-play engine
    print("Testing Self-Play Engine...")
    print("=" * 60)
    
    import torch
    from agents.alphazero.az_model import DualHeadNetwork
    
    # Create a small network for testing
    network = DualHeadNetwork()
    network.to(az_config.DEVICE)
    network.eval()
    
    # Create self-play engine
    engine = SelfPlayEngine(network)
    
    # Play one game
    print("\nPlaying one self-play game...")
    game_data = engine.play_game()
    
    print(f"Game completed:")
    print(f"  Moves played: {len(game_data)}")
    print(f"  Final value: {game_data[-1][2] if game_data else 'N/A'}")
    
    # Add to buffer
    for state, policy, value in game_data:
        engine.buffer.push(state, policy, value)
    
    print(f"  Buffer size: {len(engine.buffer)}")
    
    # Test augmentation
    if game_data:
        state, policy, value = game_data[0]
        aug_state, aug_policy = engine.augment_data(state, policy)
        print(f"\nData augmentation test:")
        print(f"  Original policy: {policy}")
        print(f"  Flipped policy: {aug_policy}")
    
    # Generate multiple games
    print(f"\nGenerating self-play data (5 games)...")
    examples = engine.generate_self_play_data(num_games=5)
    print(f"  Total examples: {examples}")
    print(f"  Buffer size: {len(engine.buffer)}")
    
    # Sample a batch
    if len(engine.buffer) >= 32:
        states, policies, values = engine.get_training_batch(32)
        print(f"\nSampled training batch:")
        print(f"  States shape: {states.shape}")
        print(f"  Policies shape: {policies.shape}")
        print(f"  Values shape: {values.shape}")
        print(f"  Value range: [{values.min():.2f}, {values.max():.2f}]")
    
    print("\n✓ Self-play engine test passed!")

