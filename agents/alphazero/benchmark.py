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
from agents.alphazero.az_config_strong import az_config_strong as az_config


def benchmark_original_mcts():
    """Benchmark original MCTS implementation."""
    from agents.alphazero.mcts_optimized import MCTSWrapper as MCTS
    
    network = DualHeadNetwork()
    network.to(az_config.DEVICE)
    network.eval()
    
    mcts = MCTS(network)
    
    # Initial board state
    board = [0] * 42
    mark = 1
    
    # Warmup
    for _ in range(3):
        mcts.search(board, mark, num_simulations=10)
    
    # Benchmark
    num_searches = 10
    num_sims = 50
    
    start = time.perf_counter()
    for _ in range(num_searches):
        policy, root = mcts.search(board, mark, num_simulations=num_sims)
    elapsed = time.perf_counter() - start
    
    return {
        'name': 'Original MCTS',
        'total_time': elapsed,
        'searches': num_searches,
        'sims_per_search': num_sims,
        'total_sims': num_searches * num_sims,
        'sims_per_sec': num_searches * num_sims / elapsed
    }


def benchmark_optimized_mcts():
    """Benchmark optimized MCTS implementation."""
    from agents.alphazero.mcts_optimized import MCTSOptimized
    from agents.alphazero.fast_board import FastBoard
    from agents.alphazero.batched_inference import SyncInferenceWrapper
    
    network = DualHeadNetwork()
    network.to(az_config.DEVICE)
    network.eval()
    
    inference = SyncInferenceWrapper(network)
    mcts = MCTSOptimized(inference_fn=inference.inference)
    
    # Initial board state
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
        policy, root = mcts.search(board, mark, num_simulations=num_sims)
    elapsed = time.perf_counter() - start
    
    return {
        'name': 'Optimized MCTS',
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
    """Benchmark self-play game generation."""
    from agents.alphazero.self_play_optimized import SelfPlayEngine
    from agents.alphazero.self_play_optimized import SimpleSelfPlayEngine
    from agents.alphazero.az_config_strong import az_config_strong as az_config
    from copy import deepcopy
    
    # Use a lightweight copy for quick benchmark
    config = deepcopy(az_config)
    config.NUM_SIMULATIONS = 10
    
    network = DualHeadNetwork()
    network.to(az_config.DEVICE)
    network.eval()
    
    results = {}
    
    # Original self-play (1 game)
    original_engine = SelfPlayEngine(network, az_config)
    
    start = time.perf_counter()
    game_data = original_engine.play_game()
    orig_time = time.perf_counter() - start
    results['original_game_time'] = orig_time
    results['original_moves'] = len(game_data)
    
    # Optimized self-play (1 game)
    optimized_engine = SimpleSelfPlayEngine(network, config)
    
    start = time.perf_counter()
    game_data = optimized_engine.play_game()
    opt_time = time.perf_counter() - start
    results['optimized_game_time'] = opt_time
    results['optimized_moves'] = len(game_data)
    
    results['speedup'] = orig_time / opt_time
    
    return results


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
    
    # MCTS comparison
    print("\n3. MCTS Benchmark")
    print("-" * 40)
    
    orig_mcts = benchmark_original_mcts()
    print(f"  {orig_mcts['name']}:")
    print(f"    Time: {orig_mcts['total_time']:.3f}s for {orig_mcts['searches']} searches")
    print(f"    Speed: {orig_mcts['sims_per_sec']:.0f} sims/sec")
    
    opt_mcts = benchmark_optimized_mcts()
    print(f"  {opt_mcts['name']}:")
    print(f"    Time: {opt_mcts['total_time']:.3f}s for {opt_mcts['searches']} searches")
    print(f"    Speed: {opt_mcts['sims_per_sec']:.0f} sims/sec")
    
    mcts_speedup = opt_mcts['sims_per_sec'] / orig_mcts['sims_per_sec']
    print(f"  Speedup: {mcts_speedup:.1f}x")
    
    # Self-play comparison
    print("\n4. Self-Play Benchmark (1 game)")
    print("-" * 40)
    selfplay_results = benchmark_self_play()
    print(f"  Original: {selfplay_results['original_game_time']:.3f}s ({selfplay_results['original_moves']} moves)")
    print(f"  Optimized: {selfplay_results['optimized_game_time']:.3f}s ({selfplay_results['optimized_moves']} moves)")
    print(f"  Speedup: {selfplay_results['speedup']:.1f}x")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Board Operations: {board_results['speedup']:.1f}x faster")
    print(f"  Win Checking:     {win_results['speedup']:.1f}x faster")
    print(f"  MCTS:             {mcts_speedup:.1f}x faster")
    print(f"  Self-Play:        {selfplay_results['speedup']:.1f}x faster")
    
    overall_estimate = (board_results['speedup'] + win_results['speedup'] + 
                       mcts_speedup + selfplay_results['speedup']) / 4
    print(f"\n  Estimated Overall Improvement: ~{overall_estimate:.1f}x")


if __name__ == "__main__":
    main()

