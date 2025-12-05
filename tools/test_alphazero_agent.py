"""
Test script for AlphaZero agent
Tests MCTS, tactics, and overall performance
"""

import sys
import os
import time
import numpy as np

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Import agent
from submission.main_alphazero_v2 import (
    AlphaZeroAgentV2, MCTS, PolicyValueNetwork,
    get_valid_moves, make_move, check_winner, is_terminal,
    find_winning_move, find_blocking_move, find_double_threat_move,
    is_losing_move, get_safe_moves, encode_state, config
)

import torch


def print_board(board):
    """Print board in human-readable format"""
    symbols = {0: '.', 1: 'X', 2: 'O'}
    print("\n  0 1 2 3 4 5 6")
    print("  -------------")
    for row in range(config.ROWS):
        line = f"{row} "
        for col in range(config.COLUMNS):
            idx = row * config.COLUMNS + col
            line += symbols[board[idx]] + " "
        print(line)
    print()


def test_utility_functions():
    """Test basic utility functions"""
    print("=" * 60)
    print("Testing Utility Functions")
    print("=" * 60)
    
    # Empty board
    board = [0] * 42
    assert get_valid_moves(board) == list(range(7)), "Empty board should have all moves valid"
    print("[OK] get_valid_moves on empty board")
    
    # Make some moves
    board = make_move(board, 3, 1)  # X in center
    board = make_move(board, 3, 2)  # O on top
    
    assert board[38] == 1, "Player 1 piece should be at bottom of column 3"
    assert board[31] == 2, "Player 2 piece should be above player 1"
    print("[OK] make_move")
    
    # Test horizontal win
    board = [0] * 42
    for i in range(4):
        board = make_move(board, i, 1)
    assert check_winner(board, 1), "Player 1 should have horizontal win"
    print("[OK] check_winner horizontal")
    
    # Test vertical win
    board = [0] * 42
    for _ in range(4):
        board = make_move(board, 0, 1)
    assert check_winner(board, 1), "Player 1 should have vertical win"
    print("[OK] check_winner vertical")
    
    # Test diagonal win
    board = [0] * 42
    # Build a diagonal: need supporting pieces
    for col in range(4):
        for _ in range(col):
            board = make_move(board, col, 2)
        board = make_move(board, col, 1)
    assert check_winner(board, 1), "Player 1 should have diagonal win"
    print("[OK] check_winner diagonal")
    
    print("\n[OK] All utility function tests passed!\n")


def test_tactical_functions():
    """Test tactical detection functions"""
    print("=" * 60)
    print("Testing Tactical Functions")
    print("=" * 60)
    
    # Test find_winning_move
    board = [0] * 42
    board = make_move(board, 0, 1)
    board = make_move(board, 1, 1)
    board = make_move(board, 2, 1)
    
    winning = find_winning_move(board, 1)
    assert winning == 3, f"Winning move should be column 3, got {winning}"
    print("[OK] find_winning_move")
    
    # Test find_blocking_move
    blocking = find_blocking_move(board, 2)  # From player 2's perspective
    assert blocking == 3, f"Blocking move should be column 3, got {blocking}"
    print("[OK] find_blocking_move")
    
    # Test is_losing_move
    board = [0] * 42
    # Setup: if we play col 3, opponent can win on col 3
    board = make_move(board, 0, 2)
    board = make_move(board, 1, 2)
    board = make_move(board, 2, 2)
    # Now if player 1 plays anywhere except col 3, player 2 wins
    
    losing = is_losing_move(board, 0, 1)  # Any move except blocking is losing
    # After player 1 plays col 0, player 2 can win on col 3
    assert losing, "Playing col 0 should give opponent winning move"
    print("[OK] is_losing_move")
    
    # Test get_safe_moves
    safe = get_safe_moves(board, 1)
    assert 3 in safe, "Column 3 should be a safe move (blocks opponent)"
    print("[OK] get_safe_moves")
    
    print("\n[OK] All tactical function tests passed!\n")


def test_state_encoding():
    """Test state encoding for neural network"""
    print("=" * 60)
    print("Testing State Encoding")
    print("=" * 60)
    
    board = [0] * 42
    board = make_move(board, 3, 1)
    board = make_move(board, 2, 2)
    
    state = encode_state(board, 1)
    
    assert state.shape == (3, 6, 7), f"State shape should be (3, 6, 7), got {state.shape}"
    print(f"[OK] State shape: {state.shape}")
    
    assert state[0].sum() == 1, "Player channel should have 1 piece"
    assert state[1].sum() == 1, "Opponent channel should have 1 piece"
    print("[OK] Piece channels correct")
    
    # Check valid moves channel
    valid_cols = np.where(state[2, 0, :] == 1)[0]
    assert len(valid_cols) == 7, "All columns should be valid"
    print("[OK] Valid moves channel correct")
    
    print("\n[OK] State encoding tests passed!\n")


def test_neural_network():
    """Test neural network forward pass"""
    print("=" * 60)
    print("Testing Neural Network")
    print("=" * 60)
    
    device = torch.device("cpu")
    network = PolicyValueNetwork()
    network = network.to(device)
    network.eval()
    
    # Count parameters
    params = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {params:,}")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 6, 7).to(device)
    
    with torch.no_grad():
        policy_logits, value = network(input_tensor)
    
    assert policy_logits.shape == (batch_size, 7), f"Policy shape wrong: {policy_logits.shape}"
    assert value.shape == (batch_size, 1), f"Value shape wrong: {value.shape}"
    print(f"[OK] Forward pass: policy {policy_logits.shape}, value {value.shape}")
    
    # Check value is in valid range
    assert (value >= -1).all() and (value <= 1).all(), "Value should be in [-1, 1]"
    print("[OK] Value in valid range [-1, 1]")
    
    print("\n[OK] Neural network tests passed!\n")


def test_mcts():
    """Test MCTS search"""
    print("=" * 60)
    print("Testing MCTS")
    print("=" * 60)
    
    device = torch.device("cpu")
    network = PolicyValueNetwork()
    network = network.to(device)
    network.eval()
    
    mcts = MCTS(network, device)
    
    # Test on empty board
    board = [0] * 42
    mark = 1
    
    print(f"Running MCTS with {config.NUM_SIMULATIONS_FAST} simulations...")
    start = time.perf_counter()
    policy, visit_counts = mcts.search(board, mark, num_simulations=100)
    elapsed = time.perf_counter() - start
    
    print(f"[OK] MCTS completed in {elapsed:.3f}s ({100/elapsed:.0f} sims/sec)")
    print(f"  Visit counts: {visit_counts}")
    print(f"  Best action: {np.argmax(policy)}")
    
    # Verify policy is valid
    assert policy.sum() > 0, "Policy should not be all zeros"
    assert abs(policy.sum() - 1.0) < 1e-5, "Policy should sum to 1"
    print("[OK] Policy is valid probability distribution")
    
    # Test get_best_action
    action = mcts.get_best_action(board, mark, num_simulations=50)
    assert 0 <= action < 7, f"Action should be in [0, 6], got {action}"
    print(f"[OK] get_best_action: {action}")
    
    print("\n[OK] MCTS tests passed!\n")


def test_agent():
    """Test full agent"""
    print("=" * 60)
    print("Testing AlphaZero Agent V2")
    print("=" * 60)
    
    agent = AlphaZeroAgentV2()
    
    # Test without model (uses heuristics)
    board = [0] * 42
    mark = 1
    
    print("Testing without model (heuristic fallback)...")
    action = agent.select_action(board, mark)
    assert action == 3, f"Opening move should be center (3), got {action}"
    print(f"[OK] Opening move: {action} (center)")
    
    # Test winning move detection
    board = [0] * 42
    board = make_move(board, 0, 1)
    board = make_move(board, 1, 1)
    board = make_move(board, 2, 1)
    
    action = agent.select_action(board, 1)
    assert action == 3, f"Should take winning move at col 3, got {action}"
    print("[OK] Takes winning move")
    
    # Test blocking
    action = agent.select_action(board, 2)  # As player 2
    assert action == 3, f"Should block at col 3, got {action}"
    print("[OK] Blocks opponent's win")
    
    # Test avoiding losing moves
    board = [0] * 42
    # Setup where playing some columns gives opponent win
    board = make_move(board, 0, 2)
    board = make_move(board, 1, 2)
    board = make_move(board, 2, 2)
    # Player 1 must block at col 3
    
    action = agent.select_action(board, 1)
    assert action == 3, f"Must block at col 3, got {action}"
    print("[OK] Blocks must-block threats")
    
    print("\n[OK] Agent tests passed!\n")


def play_self_game(agent, verbose=True):
    """Play a game with agent playing against itself"""
    board = [0] * 42
    current_mark = 1
    moves = []
    
    while True:
        action = agent.select_action(board, current_mark)
        moves.append((current_mark, action))
        board = make_move(board, action, current_mark)
        
        if verbose:
            print(f"Player {current_mark} plays column {action}")
        
        terminal, winner = is_terminal(board)
        if terminal:
            if verbose:
                print_board(board)
                if winner == 0:
                    print("Draw!")
                else:
                    print(f"Player {winner} wins!")
            return winner, moves
        
        current_mark = 3 - current_mark


def test_self_play():
    """Test self-play games"""
    print("=" * 60)
    print("Testing Self-Play")
    print("=" * 60)
    
    agent = AlphaZeroAgentV2()
    
    print("Playing 5 self-play games (heuristic mode)...")
    results = {0: 0, 1: 0, 2: 0}
    
    for i in range(5):
        winner, moves = play_self_game(agent, verbose=False)
        results[winner] += 1
        print(f"  Game {i+1}: Player {winner if winner else 'Draw'} wins in {len(moves)} moves")
    
    print(f"\nResults: P1={results[1]}, P2={results[2]}, Draw={results[0]}")
    print("\n[OK] Self-play tests completed!\n")


def benchmark_mcts():
    """Benchmark MCTS performance"""
    print("=" * 60)
    print("MCTS Performance Benchmark")
    print("=" * 60)
    
    device = torch.device("cpu")
    network = PolicyValueNetwork()
    network = network.to(device)
    network.eval()
    
    mcts = MCTS(network, device)
    board = [0] * 42
    
    for num_sims in [50, 100, 200, 400]:
        start = time.perf_counter()
        iterations = 5
        
        for _ in range(iterations):
            mcts.search(board, 1, num_simulations=num_sims)
        
        elapsed = time.perf_counter() - start
        avg_time = elapsed / iterations
        sims_per_sec = num_sims / avg_time
        
        print(f"  {num_sims:4d} sims: {avg_time:.3f}s avg ({sims_per_sec:.0f} sims/sec)")
    
    print("\n[OK] Benchmark completed!\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("AlphaZero Agent Test Suite")
    print("=" * 60 + "\n")
    
    try:
        test_utility_functions()
        test_tactical_functions()
        test_state_encoding()
        test_neural_network()
        test_mcts()
        test_agent()
        test_self_play()
        benchmark_mcts()
        
        print("\n" + "=" * 60)
        print("All Tests Passed!")
        print("=" * 60 + "\n")
        
    except AssertionError as e:
        print(f"\n[FAIL] Test failed: {e}")
        raise
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        raise


if __name__ == "__main__":
    main()

