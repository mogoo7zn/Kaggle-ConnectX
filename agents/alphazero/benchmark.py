"""
Benchmark Script for AlphaZero Optimizations
Compares performance between original and optimized implementations
"""

import time
import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.alphazero.az_model import DualHeadNetwork
from agents.alphazero.az_config import az_config, FastConfig
from agents.alphazero.batched_inference import SyncInferenceWrapper
from agents.alphazero.mcts import MCTS
from agents.alphazero.fast_board import FastBoard


def benchmark_mcts():
    """Benchmark current MCTS implementation."""
    network = DualHeadNetwork()
    network.to(az_config.DEVICE)
    network.eval()
    
    inference = SyncInferenceWrapper(network)
    mcts = MCTS(inference_fn=inference.inference, config=az_config)
    
    board = FastBoard()
    mark = 1
    
    # Warmup
    for _ in range(3):
        mcts.search(board, mark, num_simulations=10)
    
    # Benchmark
    num_searches = 10
    num_sims = 50
    
    start = time.perf_counter()
    for _ in range(num_searches):
        mcts.search(board, mark, num_simulations=num_sims)
    elapsed = time.perf_counter() - start
    
    return {
        'name': 'MCTS (current)',
        'total_time': elapsed,
        'searches': num_searches,
        'sims_per_search': num_sims,
        'total_sims': num_searches * num_sims,
        'sims_per_sec': num_searches * num_sims / elapsed
    }


def benchmark_board_operations():
    """Benchmark board state operations."""
    from agents.alphazero.fast_board import FastBoard
    from agents.base.utils import make_move, check_winner, is_terminal, get_valid_moves
    
    results = {}
    num_iterations = 10000
    
    # Original list-based operations
    board = [0] * 42
    start = time.perf_counter()
    for i in range(num_iterations):
        new_board = make_move(board, i % 7, 1)
        valid = get_valid_moves(new_board)
        done, winner = is_terminal(new_board)
    orig_time = time.perf_counter() - start
    results['original_board_ops'] = orig_time
    
    # FastBoard operations
    fast_board = FastBoard()
    start = time.perf_counter()
    for i in range(num_iterations):
        new_board = fast_board.make_move(i % 7, 1)
        valid = new_board.get_valid_moves()
        done, winner = new_board.is_terminal()
    fast_time = time.perf_counter() - start
    results['fast_board_ops'] = fast_time
    
    results['speedup'] = orig_time / fast_time
    
    return results


def benchmark_win_checking():
    """Benchmark win checking performance."""
    from agents.alphazero.fast_board import FastBoard
    from agents.base.utils import check_winner
    
    num_iterations = 100000
    
    # Original
    board = [0] * 42
    board[0] = board[7] = board[14] = 1  # 3 in a row vertically
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        check_winner(board, 1)
    orig_time = time.perf_counter() - start
    
    # FastBoard
    fast_board = FastBoard.from_list(board)
    
    start = time.perf_counter()
    for _ in range(num_iterations):
        fast_board.check_win(1)
    fast_time = time.perf_counter() - start
    
    return {
        'original_win_check': orig_time,
        'fast_win_check': fast_time,
        'speedup': orig_time / fast_time,
        'iterations': num_iterations
    }


def benchmark_self_play():
    """Benchmark simple self-play game generation."""
    from agents.alphazero.self_play import SimpleSelfPlay
    
    config = FastConfig()
    config.NUM_SIMULATIONS = 10  # speed up for benchmark
    
    network = DualHeadNetwork()
    network.to(az_config.DEVICE)
    network.eval()
    
    engine = SimpleSelfPlay(network, config)
    
    start = time.perf_counter()
    game_data = engine.play_game()
    elapsed = time.perf_counter() - start
    
    return {
        'game_time': elapsed,
        'moves': len(game_data)
    }


def main():
    """Run all benchmarks."""
    print("=" * 70)
    print("AlphaZero Optimization Benchmarks")
    print("=" * 70)
    
    # Board operations
    print("\n1. Board Operations Benchmark")
    print("-" * 40)
    board_results = benchmark_board_operations()
    print(f"  Original board ops: {board_results['original_board_ops']:.3f}s")
    print(f"  FastBoard ops:      {board_results['fast_board_ops']:.3f}s")
    print(f"  Speedup:            {board_results['speedup']:.1f}x")
    
    # Win checking
    print("\n2. Win Checking Benchmark")
    print("-" * 40)
    win_results = benchmark_win_checking()
    print(f"  Original win check: {win_results['original_win_check']:.3f}s ({win_results['iterations']:,} iterations)")
    print(f"  FastBoard win check: {win_results['fast_win_check']:.3f}s")
    print(f"  Speedup:             {win_results['speedup']:.1f}x")
    
    # MCTS benchmark
    print("\n3. MCTS Benchmark")
    print("-" * 40)
    
    mcts_stats = benchmark_mcts()
    print(f"  {mcts_stats['name']}:")
    print(f"    Time: {mcts_stats['total_time']:.3f}s for {mcts_stats['searches']} searches")
    print(f"    Speed: {mcts_stats['sims_per_sec']:.0f} sims/sec")
    
    # Self-play
    print("\n4. Self-Play Benchmark (1 game, simple engine)")
    print("-" * 40)
    selfplay_results = benchmark_self_play()
    print(f"  SimpleSelfPlay: {selfplay_results['game_time']:.3f}s ({selfplay_results['moves']} moves)")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Board Operations: {board_results['speedup']:.1f}x faster")
    print(f"  Win Checking:     {win_results['speedup']:.1f}x faster")
    print(f"  MCTS Sims/sec:    {mcts_stats['sims_per_sec']:.0f}")
    print(f"  Self-Play (1g):   {selfplay_results['game_time']:.3f}s")


if __name__ == "__main__":
    main()

