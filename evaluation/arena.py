"""
Arena - Unified battle ground for agent evaluation
Supports fair head-to-head matches between any two agents
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import time
from typing import Callable, Dict, Tuple, List
from collections import defaultdict

from core.utils import get_valid_moves, make_move, is_terminal
from core.config import config


class Arena:
    """
    Arena for evaluating agents against each other.
    
    Features:
    - Fair alternating starts
    - Timeout handling
    - Detailed statistics
    - Move history tracking
    """
    
    def __init__(self, timeout: float = 5.0):
        """
        Initialize arena.
        
        Args:
            timeout: Maximum time per move in seconds
        """
        self.timeout = timeout
        self.rows = config.ROWS
        self.columns = config.COLUMNS
    
    def play_game(self, agent1_fn: Callable, agent2_fn: Callable,
                 agent1_starts: bool = True, verbose: bool = False) -> Dict:
        """
        Play one game between two agents.
        
        Args:
            agent1_fn: Agent 1 policy function (board, mark) -> action
            agent2_fn: Agent 2 policy function (board, mark) -> action
            agent1_starts: Whether agent 1 starts
            verbose: Whether to print game progress
        
        Returns:
            Dictionary with game result and statistics
        """
        board = [0] * (self.rows * self.columns)
        
        # Assign marks
        if agent1_starts:
            agent1_mark, agent2_mark = 1, 2
            agents = {1: agent1_fn, 2: agent2_fn}
        else:
            agent1_mark, agent2_mark = 2, 1
            agents = {2: agent1_fn, 1: agent2_fn}
        
        current_mark = 1
        move_count = 0
        move_history = []
        agent_times = {1: [], 2: []}
        
        if verbose:
            print(f"\nGame Start - Agent1: Mark {agent1_mark}, Agent2: Mark {agent2_mark}")
        
        # Play until terminal
        while True:
            valid_moves = get_valid_moves(board)
            if not valid_moves:
                # Board full, draw
                return self._create_result(0, move_count, move_history, agent_times,
                                          agent1_mark, "Board full")
            
            # Get agent's move
            agent_fn = agents[current_mark]
            
            try:
                start_time = time.time()
                action = agent_fn(board.copy(), current_mark)
                elapsed_time = time.time() - start_time
                
                agent_times[current_mark].append(elapsed_time)
                
                # Check timeout
                if elapsed_time > self.timeout:
                    winner = 3 - current_mark  # Opponent wins
                    return self._create_result(
                        winner, move_count, move_history, agent_times,
                        agent1_mark, f"Agent {current_mark} timeout"
                    )
                
                # Validate move
                if action not in valid_moves:
                    winner = 3 - current_mark  # Opponent wins
                    return self._create_result(
                        winner, move_count, move_history, agent_times,
                        agent1_mark, f"Agent {current_mark} invalid move: {action}"
                    )
                
            except Exception as e:
                winner = 3 - current_mark  # Opponent wins
                return self._create_result(
                    winner, move_count, move_history, agent_times,
                    agent1_mark, f"Agent {current_mark} error: {str(e)}"
                )
            
            # Make move
            board = make_move(board, action, current_mark)
            move_history.append((current_mark, action))
            move_count += 1
            
            if verbose:
                print(f"Move {move_count}: Agent {current_mark} -> Column {action}")
            
            # Check terminal
            done, winner = is_terminal(board)
            if done:
                return self._create_result(winner, move_count, move_history,
                                          agent_times, agent1_mark,
                                          "Game completed normally")
            
            # Switch player
            current_mark = 3 - current_mark
    
    def _create_result(self, winner: int, move_count: int,
                      move_history: List[Tuple[int, int]],
                      agent_times: Dict[int, List[float]],
                      agent1_mark: int, reason: str) -> Dict:
        """Create result dictionary."""
        # Determine outcome from agent1's perspective
        if winner == 0:
            outcome = 'draw'
        elif winner == agent1_mark:
            outcome = 'agent1_win'
        else:
            outcome = 'agent2_win'
        
        # Calculate statistics
        result = {
            'outcome': outcome,
            'winner': winner,
            'move_count': move_count,
            'move_history': move_history,
            'agent1_mark': agent1_mark,
            'agent2_mark': 3 - agent1_mark,
            'reason': reason,
            'agent1_avg_time': np.mean(agent_times[agent1_mark]) if agent_times[agent1_mark] else 0.0,
            'agent2_avg_time': np.mean(agent_times[3-agent1_mark]) if agent_times[3-agent1_mark] else 0.0,
            'agent1_max_time': max(agent_times[agent1_mark]) if agent_times[agent1_mark] else 0.0,
            'agent2_max_time': max(agent_times[3-agent1_mark]) if agent_times[3-agent1_mark] else 0.0,
        }
        
        return result
    
    def play_match(self, agent1_fn: Callable, agent2_fn: Callable,
                  num_games: int = 100, agent1_name: str = "Agent1",
                  agent2_name: str = "Agent2", verbose: bool = False) -> Dict:
        """
        Play a match (multiple games) between two agents.
        
        Args:
            agent1_fn: Agent 1 policy function
            agent2_fn: Agent 2 policy function
            num_games: Number of games to play
            agent1_name: Name of agent 1
            agent2_name: Name of agent 2
            verbose: Whether to print progress
        
        Returns:
            Match statistics dictionary
        """
        print(f"\n{'='*60}")
        print(f"Arena Match: {agent1_name} vs {agent2_name}")
        print(f"Games: {num_games}")
        print(f"{'='*60}\n")
        
        results = []
        agent1_wins = 0
        agent2_wins = 0
        draws = 0
        
        start_time = time.time()
        
        for game_num in range(num_games):
            # Alternate starting player
            agent1_starts = (game_num % 2 == 0)
            
            # Play game
            result = self.play_game(agent1_fn, agent2_fn, agent1_starts, verbose=False)
            results.append(result)
            
            # Count outcomes
            if result['outcome'] == 'agent1_win':
                agent1_wins += 1
            elif result['outcome'] == 'agent2_win':
                agent2_wins += 1
            else:
                draws += 1
            
            # Progress reporting
            if verbose and (game_num + 1) % 10 == 0:
                print(f"  Game {game_num + 1}/{num_games} - "
                      f"{agent1_name}: {agent1_wins}, "
                      f"{agent2_name}: {agent2_wins}, "
                      f"Draws: {draws}")
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        match_stats = {
            'agent1_name': agent1_name,
            'agent2_name': agent2_name,
            'num_games': num_games,
            'agent1_wins': agent1_wins,
            'agent2_wins': agent2_wins,
            'draws': draws,
            'agent1_win_rate': agent1_wins / num_games,
            'agent2_win_rate': agent2_wins / num_games,
            'draw_rate': draws / num_games,
            'avg_moves': np.mean([r['move_count'] for r in results]),
            'agent1_avg_time': np.mean([r['agent1_avg_time'] for r in results]),
            'agent2_avg_time': np.mean([r['agent2_avg_time'] for r in results]),
            'total_time': total_time,
            'games_per_second': num_games / total_time,
            'results': results
        }
        
        # Print summary
        print(f"\nMatch Results:")
        print(f"  {agent1_name}: {agent1_wins} wins ({match_stats['agent1_win_rate']:.1%})")
        print(f"  {agent2_name}: {agent2_wins} wins ({match_stats['agent2_win_rate']:.1%})")
        print(f"  Draws: {draws} ({match_stats['draw_rate']:.1%})")
        print(f"  Average moves: {match_stats['avg_moves']:.1f}")
        print(f"  {agent1_name} avg time: {match_stats['agent1_avg_time']*1000:.1f}ms")
        print(f"  {agent2_name} avg time: {match_stats['agent2_avg_time']*1000:.1f}ms")
        print(f"  Total time: {total_time:.1f}s ({match_stats['games_per_second']:.2f} games/s)")
        
        return match_stats


def create_agent_wrapper(agent_obj, agent_type: str) -> Callable:
    """
    Create a unified wrapper for different agent types.
    
    Args:
        agent_obj: Agent object (Rainbow, AlphaZero, or function)
        agent_type: Type of agent ('rainbow', 'alphazero', 'function')
    
    Returns:
        Callable policy function (board, mark) -> action
    """
    if agent_type == 'rainbow':
        def policy(board, mark):
            return agent_obj.select_action(board, mark, epsilon=0.0)
        return policy
    
    elif agent_type == 'alphazero':
        from alphazero.mcts import MCTS
        mcts = MCTS(agent_obj.network if hasattr(agent_obj, 'network') else agent_obj)
        
        def policy(board, mark):
            return mcts.get_best_action(board, mark)
        return policy
    
    elif agent_type == 'ppo':
        def policy(board, mark):
            return agent_obj.select_action(board, mark)
        return policy

    elif agent_type == 'function':
        return agent_obj
    
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")


if __name__ == "__main__":
    # Test arena
    print("Testing Arena...")
    print("=" * 60)
    
    import random
    
    # Create simple test agents
    def random_agent(board, mark):
        moves = get_valid_moves(board)
        return random.choice(moves) if moves else 0
    
    def center_agent(board, mark):
        moves = get_valid_moves(board)
        if not moves:
            return 0
        # Prefer center
        center = config.COLUMNS // 2
        return min(moves, key=lambda x: abs(x - center))
    
    # Create arena
    arena = Arena(timeout=1.0)
    
    # Play a match
    match_stats = arena.play_match(
        random_agent, center_agent,
        num_games=20,
        agent1_name="Random",
        agent2_name="Center",
        verbose=True
    )
    
    print("\n✓ Arena test passed!")

