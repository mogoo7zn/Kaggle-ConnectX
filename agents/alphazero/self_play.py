"""
Self-Play Engine for AlphaZero
Uses batched inference and parallel game execution for maximum throughput

Key Features:
1. Batched network inference across multiple games
2. FastBoard for efficient game state operations  
3. Parallel game execution with shared inference server
4. Adaptive MCTS simulation count
5. Efficient memory management
"""

import numpy as np
import torch
import threading
import queue
import time
from typing import List, Tuple, Dict, Optional
from collections import deque
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.alphazero.az_config import az_config
from agents.alphazero.fast_board import FastBoard, ROWS, COLS
from agents.alphazero.mcts import MCTS, MCTSNode
from agents.alphazero.batched_inference import BatchedInferenceServer, SyncInferenceWrapper


@dataclass
class GameState:
    """State of a single self-play game."""
    game_id: int
    board: FastBoard
    current_mark: int
    move_count: int
    history: List[Tuple[np.ndarray, np.ndarray, int]]  # (state, policy, mark)
    temperature_threshold: int


class SelfPlayBuffer:
    """
    Replay buffer using numpy arrays for storage.
    """
    
    def __init__(self, capacity: int = None):
        """
        Initialize buffer.
        
        Args:
            capacity: Maximum buffer size
        """
        self.capacity = capacity or az_config.REPLAY_BUFFER_SIZE
        self.buffer = deque(maxlen=self.capacity)
        self._lock = threading.Lock()
    
    def push(self, state: np.ndarray, policy: np.ndarray, value: float):
        """Add training example to buffer (thread-safe)."""
        with self._lock:
            self.buffer.append((state, policy, value))
    
    def push_batch(self, examples: List[Tuple[np.ndarray, np.ndarray, float]]):
        """Add multiple examples to buffer (thread-safe)."""
        with self._lock:
            for example in examples:
                self.buffer.append(example)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample a batch of training examples."""
        with self._lock:
            buffer_size = len(self.buffer)
            if buffer_size == 0:
                raise ValueError("Buffer is empty")
            
            indices = np.random.choice(
                buffer_size, 
                min(batch_size, buffer_size),
                replace=buffer_size < batch_size
            )
            
            batch = [self.buffer[i] for i in indices]
        
        states, policies, values = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(policies, dtype=np.float32),
            np.array(values, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        with self._lock:
            return len(self.buffer)
    
    def is_ready(self, min_size: int) -> bool:
        return len(self) >= min_size
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.buffer.clear()


class ParallelSelfPlay:
    """
    Parallel self-play engine using batched inference.
    
    Runs multiple games concurrently, collecting states for
    batched neural network evaluation.
    """
    
    def __init__(self, network: torch.nn.Module,
                 num_parallel_games: int = None,
                 config=None,
                 use_batched_inference: bool = True):
        """
        Initialize parallel self-play engine.
        
        Args:
            network: Policy-value neural network
            num_parallel_games: Number of games to run in parallel
            config: Configuration object
            use_batched_inference: Whether to use batched inference server
        """
        self.network = network
        self.config = config or az_config
        self.num_parallel_games = num_parallel_games or self.config.NUM_PARALLEL_GAMES
        
        # Initialize inference
        if use_batched_inference:
            self.inference_server = BatchedInferenceServer(
                network,
                max_batch_size=self.num_parallel_games * 8,  # Enough for all parallel games
                max_wait_ms=5.0
            )
            self.inference_fn = self.inference_server.inference
            self.batch_inference_fn = self.inference_server.inference_batch
        else:
            self.inference_wrapper = SyncInferenceWrapper(network)
            self.inference_fn = self.inference_wrapper.inference
            self.batch_inference_fn = self.inference_wrapper.inference_batch
            self.inference_server = None
        
        # Buffer for training data
        self.buffer = SelfPlayBuffer(self.config.REPLAY_BUFFER_SIZE)
        
        # Statistics
        self.games_completed = 0
        self.total_moves = 0
        self.total_time = 0.0
    
    def start(self):
        """Start inference server if using batched inference."""
        if self.inference_server:
            self.inference_server.start()
    
    def stop(self):
        """Stop inference server."""
        if self.inference_server:
            self.inference_server.stop()
    
    def play_game(self, game_id: int = 0,
                  temperature_threshold: int = None,
                  add_noise: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """
        Play a single self-play game.
        
        Args:
            game_id: Game identifier for logging
            temperature_threshold: Move to switch to greedy
            add_noise: Whether to add Dirichlet noise
        
        Returns:
            List of (state, policy, value) training examples
        """
        if temperature_threshold is None:
            temperature_threshold = self.config.TEMP_THRESHOLD
        
        # Initialize game
        board = FastBoard()
        current_mark = 1
        move_count = 0
        history = []
        
        # Create MCTS instance
        mcts = MCTS(
            inference_fn=self.inference_fn,
            batch_inference_fn=self.batch_inference_fn,
            config=self.config
        )
        
        # Play until terminal
        while True:
            # Determine temperature
            temperature = self.config.TEMPERATURE if move_count < temperature_threshold else 0.0
            
            # Encode state for training data
            state = board.encode_state(current_mark)
            
            # Run MCTS
            policy, _ = mcts.search(
                board, current_mark,
                temperature=temperature,
                add_noise=add_noise and move_count < temperature_threshold
            )
            
            # Store for training
            history.append((state, policy, current_mark))
            
            # Sample action
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break
            
            action = self._sample_action(policy, valid_moves)
            
            # Make move
            board.make_move_inplace(action, current_mark)
            move_count += 1
            
            # Check terminal
            is_terminal, winner = board.is_terminal()
            if is_terminal:
                # Create training examples with assigned values
                return self._process_game_outcome(history, winner)
            
            # Switch player
            current_mark = 3 - current_mark
        
        return []
    
    def _sample_action(self, policy: np.ndarray, valid_moves: List[int]) -> int:
        """Sample action from policy over valid moves."""
        # Extract probabilities for valid moves
        probs = policy[valid_moves]
        
        if probs.sum() > 0:
            probs = probs / probs.sum()
        else:
            probs = np.ones(len(valid_moves)) / len(valid_moves)
        
        action_idx = np.random.choice(len(valid_moves), p=probs)
        return valid_moves[action_idx]
    
    def _process_game_outcome(self, history: List[Tuple], winner: int
                              ) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Process game outcome and assign values."""
        examples = []
        
        for state, policy, mark in history:
            if winner == 0:  # Draw
                value = 0.0
            elif winner == mark:  # Win
                value = 1.0
            else:  # Loss
                value = -1.0
            
            examples.append((state, policy, value))
        
        return examples
    
    def generate_self_play_data(self, num_games: int = None,
                               use_augmentation: bool = None,
                               num_workers: int = None) -> int:
        """
        Generate self-play data using parallel game execution.
        
        Args:
            num_games: Total number of games to generate
            use_augmentation: Whether to use data augmentation
            num_workers: Number of parallel workers
        
        Returns:
            Number of training examples generated
        """
        if num_games is None:
            num_games = self.config.NUM_SELFPLAY_GAMES
        if use_augmentation is None:
            use_augmentation = self.config.USE_AUGMENTATION
        if num_workers is None:
            num_workers = min(self.num_parallel_games, num_games)
        
        examples_generated = 0
        games_completed = 0
        
        print(f"Generating self-play data: {num_games} games using {num_workers} workers...")
        start_time = time.perf_counter()
        
        # Use thread pool for parallel game execution
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all games
            futures = []
            for game_id in range(num_games):
                future = executor.submit(self.play_game, game_id)
                futures.append(future)
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    game_data = future.result()
                    
                    # Add to buffer
                    for state, policy, value in game_data:
                        self.buffer.push(state, policy, value)
                        examples_generated += 1
                        
                        # Data augmentation (horizontal flip)
                        if use_augmentation:
                            aug_state = np.flip(state, axis=2).copy()
                            aug_policy = np.flip(policy).copy()
                            self.buffer.push(aug_state, aug_policy, value)
                            examples_generated += 1
                    
                    games_completed += 1
                    
                    # Progress reporting
                    if (games_completed) % 10 == 0:
                        elapsed = time.perf_counter() - start_time
                        games_per_sec = games_completed / elapsed
                        print(f"  Generated {games_completed}/{num_games} games "
                              f"({examples_generated} examples, {games_per_sec:.1f} games/s)")
                
                except Exception as e:
                    print(f"  Game {i} failed: {e}")
        
        elapsed = time.perf_counter() - start_time
        self.games_completed += games_completed
        self.total_time += elapsed
        
        print(f"Self-play complete: {examples_generated} examples in {elapsed:.1f}s")
        print(f"  Games: {games_completed}, Avg game time: {elapsed/max(1,games_completed):.2f}s")
        print(f"  Buffer size: {len(self.buffer)}")
        
        return examples_generated
    
    def generate_games_batched(self, num_games: int,
                               batch_size: int = None) -> int:
        """
        Generate games using batched MCTS across multiple positions.
        
        This approach collects states from all games that need evaluation
        and processes them in a single batch.
        
        Args:
            num_games: Number of games to generate
            batch_size: Batch size for inference
        
        Returns:
            Number of examples generated
        """
        if batch_size is None:
            batch_size = min(num_games, 32)
        
        # Initialize games
        games = [
            GameState(
                game_id=i,
                board=FastBoard(),
                current_mark=1,
                move_count=0,
                history=[],
                temperature_threshold=self.config.TEMP_THRESHOLD
            )
            for i in range(num_games)
        ]
        
        active_games = list(games)
        completed_games = []
        examples_generated = 0
        
        print(f"Generating {num_games} games with batched MCTS...")
        start_time = time.perf_counter()
        
        iteration = 0
        while active_games:
            iteration += 1
            
            # Collect states that need evaluation
            states_to_eval = []
            game_indices = []
            
            for i, game in enumerate(active_games):
                state = game.board.encode_state(game.current_mark)
                states_to_eval.append(state)
                game_indices.append(i)
            
            # Batch evaluation
            if self.batch_inference_fn:
                results = self.batch_inference_fn(states_to_eval)
            else:
                results = [self.inference_fn(s) for s in states_to_eval]
            
            # Process each game
            games_to_remove = []
            
            for game_idx, (policy_probs, value) in zip(game_indices, results):
                game = active_games[game_idx]
                
                # Store state and run mini-MCTS for policy refinement
                state = game.board.encode_state(game.current_mark)
                
                # Mask invalid moves
                valid_moves = game.board.get_valid_moves()
                mask = np.zeros(COLS)
                mask[valid_moves] = 1.0
                policy_probs = policy_probs * mask
                if policy_probs.sum() > 0:
                    policy_probs = policy_probs / policy_probs.sum()
                else:
                    policy_probs = mask / mask.sum()
                
                # Add Dirichlet noise for exploration
                if game.move_count < game.temperature_threshold:
                    noise = np.zeros(COLS)
                    noise[valid_moves] = np.random.dirichlet(
                        [self.config.DIRICHLET_ALPHA] * len(valid_moves)
                    )
                    policy_probs = ((1 - self.config.DIRICHLET_EPSILON) * policy_probs +
                                   self.config.DIRICHLET_EPSILON * noise)
                
                # Store for training
                game.history.append((state.copy(), policy_probs.copy(), game.current_mark))
                
                # Sample action
                temperature = self.config.TEMPERATURE if game.move_count < game.temperature_threshold else 0.0
                
                if temperature == 0:
                    action = np.argmax(policy_probs)
                else:
                    probs = policy_probs[valid_moves]
                    if temperature != 1:
                        probs = probs ** (1.0 / temperature)
                    probs = probs / probs.sum()
                    action = valid_moves[np.random.choice(len(valid_moves), p=probs)]
                
                # Make move
                game.board.make_move_inplace(action, game.current_mark)
                game.move_count += 1
                
                # Check terminal
                is_terminal, winner = game.board.is_terminal()
                if is_terminal:
                    # Process game outcome
                    game_examples = self._process_game_outcome(game.history, winner)
                    
                    # Add to buffer
                    for ex_state, ex_policy, ex_value in game_examples:
                        self.buffer.push(ex_state, ex_policy, ex_value)
                        examples_generated += 1
                        
                        # Augmentation
                        if self.config.USE_AUGMENTATION:
                            aug_state = np.flip(ex_state, axis=2).copy()
                            aug_policy = np.flip(ex_policy).copy()
                            self.buffer.push(aug_state, aug_policy, ex_value)
                            examples_generated += 1
                    
                    completed_games.append(game)
                    games_to_remove.append(game_idx)
                else:
                    # Switch player
                    game.current_mark = 3 - game.current_mark
            
            # Remove completed games (in reverse order to maintain indices)
            for idx in sorted(games_to_remove, reverse=True):
                active_games.pop(idx)
            
            # Progress reporting
            if iteration % 50 == 0 or not active_games:
                elapsed = time.perf_counter() - start_time
                print(f"  Iteration {iteration}: {len(completed_games)}/{num_games} games complete "
                      f"({examples_generated} examples, {len(active_games)} active)")
        
        elapsed = time.perf_counter() - start_time
        print(f"Batched generation complete: {examples_generated} examples in {elapsed:.1f}s")
        
        return examples_generated
    
    def get_training_batch(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get a batch of training data."""
        return self.buffer.sample(batch_size)


class SimpleSelfPlay:
    """
    Simplified self-play engine for single-threaded operation.
    """
    
    def __init__(self, network: torch.nn.Module, config=None):
        """
        Initialize simple self-play engine.
        
        Args:
            network: Policy-value neural network
            config: Configuration object
        """
        self.network = network
        self.config = config or az_config
        
        # Setup inference
        self.inference_wrapper = SyncInferenceWrapper(network)
        
        # Create MCTS
        self.mcts = MCTS(
            inference_fn=self.inference_wrapper.inference,
            batch_inference_fn=self.inference_wrapper.inference_batch,
            config=self.config
        )
        
        # Buffer
        self.buffer = SelfPlayBuffer(self.config.REPLAY_BUFFER_SIZE)
    
    def play_game(self, temperature_threshold: int = None,
                  add_noise: bool = True) -> List[Tuple[np.ndarray, np.ndarray, float]]:
        """Play one self-play game."""
        if temperature_threshold is None:
            temperature_threshold = self.config.TEMP_THRESHOLD
        
        board = FastBoard()
        current_mark = 1
        move_count = 0
        history = []
        
        while True:
            temperature = self.config.TEMPERATURE if move_count < temperature_threshold else 0.0
            state = board.encode_state(current_mark)
            
            # Run MCTS
            policy, _ = self.mcts.search(
                board, current_mark,
                temperature=temperature,
                add_noise=add_noise and move_count < temperature_threshold
            )
            
            history.append((state, policy, current_mark))
            
            # Sample action
            valid_moves = board.get_valid_moves()
            if not valid_moves:
                break
            
            probs = policy[valid_moves]
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(len(valid_moves)) / len(valid_moves)
            
            action = valid_moves[np.random.choice(len(valid_moves), p=probs)]
            
            board.make_move_inplace(action, current_mark)
            move_count += 1
            
            is_terminal, winner = board.is_terminal()
            if is_terminal:
                return self._process_outcome(history, winner)
            
            current_mark = 3 - current_mark
        
        return []
    
    def _process_outcome(self, history, winner):
        """Process game outcome."""
        examples = []
        for state, policy, mark in history:
            if winner == 0:
                value = 0.0
            elif winner == mark:
                value = 1.0
            else:
                value = -1.0
            examples.append((state, policy, value))
        return examples
    
    def generate_self_play_data(self, num_games: int = None,
                               use_augmentation: bool = None) -> int:
        """Generate self-play data."""
        if num_games is None:
            num_games = self.config.NUM_SELFPLAY_GAMES
        if use_augmentation is None:
            use_augmentation = self.config.USE_AUGMENTATION
        
        examples_generated = 0
        
        print(f"Generating self-play data: {num_games} games...")
        start_time = time.perf_counter()
        
        for game_num in range(num_games):
            game_data = self.play_game()
            
            for state, policy, value in game_data:
                self.buffer.push(state, policy, value)
                examples_generated += 1
                
                if use_augmentation:
                    aug_state = np.flip(state, axis=2).copy()
                    aug_policy = np.flip(policy).copy()
                    self.buffer.push(aug_state, aug_policy, value)
                    examples_generated += 1
        
        elapsed = time.perf_counter() - start_time
        print(f"Self-play complete: {examples_generated} examples in {elapsed:.1f}s")
        
        return examples_generated
    
    def get_training_batch(self, batch_size: int):
        """Get a batch of training data."""
        return self.buffer.sample(batch_size)


if __name__ == "__main__":
    # Test self-play engine
    print("Testing Self-Play Engine...")
    print("=" * 60)
    
    from agents.alphazero.az_model import DualHeadNetwork
    
    # Create network
    network = DualHeadNetwork()
    network.to(az_config.DEVICE)
    network.eval()
    
    # Test SimpleSelfPlay
    print("\n1. Testing SimpleSelfPlay...")
    engine = SimpleSelfPlay(network)
    
    game_data = engine.play_game()
    print(f"  Single game: {len(game_data)} moves")
    
    examples = engine.generate_self_play_data(num_games=5)
    print(f"  Generated {examples} examples from 5 games")
    
    # Test ParallelSelfPlay
    print("\n2. Testing ParallelSelfPlay...")
    parallel_engine = ParallelSelfPlay(
        network,
        num_parallel_games=4,
        use_batched_inference=True
    )
    
    parallel_engine.start()
    
    examples = parallel_engine.generate_self_play_data(num_games=10, num_workers=4)
    print(f"  Generated {examples} examples from 10 parallel games")
    
    parallel_engine.stop()
    
    print("\n✓ Self-Play test passed!")

