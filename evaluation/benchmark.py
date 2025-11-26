"""
Benchmark Suite for Agent Evaluation
Standard opponents and comprehensive performance metrics
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import numpy as np
from typing import Dict, List, Callable
import json
from datetime import datetime

from evaluation.arena import Arena, create_agent_wrapper
from core.utils import get_valid_moves, get_negamax_move
from core.config import config


class StandardOpponents:
    """Collection of standard baseline opponents."""
    
    @staticmethod
    def random_policy(board: List[int], mark: int) -> int:
        """Random baseline."""
        moves = get_valid_moves(board)
        return random.choice(moves) if moves else 0
    
    @staticmethod
    def center_preference_policy(board: List[int], mark: int) -> int:
        """Prefers center columns."""
        moves = get_valid_moves(board)
        if not moves:
            return 0
        center = config.COLUMNS // 2
        return min(moves, key=lambda x: (abs(x - center), random.random()))
    
    @staticmethod
    def negamax_depth_4(board: List[int], mark: int) -> int:
        """Negamax with depth 4."""
        return get_negamax_move(board, mark, depth=4)
    
    @staticmethod
    def negamax_depth_6(board: List[int], mark: int) -> int:
        """Negamax with depth 6."""
        return get_negamax_move(board, mark, depth=6)
    
    @staticmethod
    def negamax_depth_8(board: List[int], mark: int) -> int:
        """Negamax with depth 8 (slow but strong)."""
        return get_negamax_move(board, mark, depth=8)


class Benchmark:
    """
    Comprehensive benchmark suite for agent evaluation.
    
    Provides:
    - Standardized opponent suite
    - Performance metrics (win rate, time, ELO)
    - JSON export for comparison
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        self.arena = Arena(timeout=10.0)
        
        # Standard opponent suite
        self.opponents = {
            'Random': StandardOpponents.random_policy,
            'Center': StandardOpponents.center_preference_policy,
            'Negamax-4': StandardOpponents.negamax_depth_4,
            'Negamax-6': StandardOpponents.negamax_depth_6,
            #'Negamax-8': StandardOpponents.negamax_depth_8,  # Too slow for routine testing
        }
        
        # Default test configuration
        self.default_games_per_opponent = 100
    
    def run_benchmark(self, agent_fn: Callable, agent_name: str = "TestAgent",
                     games_per_opponent: int = None,
                     opponents_to_test: List[str] = None) -> Dict:
        """
        Run comprehensive benchmark.
        
        Args:
            agent_fn: Agent policy function
            agent_name: Name of the agent
            games_per_opponent: Games to play vs each opponent
            opponents_to_test: List of opponent names (None = all)
        
        Returns:
            Dictionary of benchmark results
        """
        if games_per_opponent is None:
            games_per_opponent = self.default_games_per_opponent
        
        if opponents_to_test is None:
            opponents_to_test = list(self.opponents.keys())
        
        print(f"\n{'='*70}")
        print(f"Running Benchmark for: {agent_name}")
        print(f"Games per opponent: {games_per_opponent}")
        print(f"Opponents: {', '.join(opponents_to_test)}")
        print(f"{'='*70}\n")
        
        results = {
            'agent_name': agent_name,
            'timestamp': datetime.now().isoformat(),
            'games_per_opponent': games_per_opponent,
            'opponents': {}
        }
        
        total_wins = 0
        total_games = 0
        
        # Test against each opponent
        for opp_name in opponents_to_test:
            if opp_name not in self.opponents:
                print(f"Warning: Unknown opponent '{opp_name}', skipping...")
                continue
            
            opp_fn = self.opponents[opp_name]
            
            # Run match
            match_stats = self.arena.play_match(
                agent_fn, opp_fn,
                num_games=games_per_opponent,
                agent1_name=agent_name,
                agent2_name=opp_name,
                verbose=False
            )
            
            # Store results
            results['opponents'][opp_name] = {
                'wins': match_stats['agent1_wins'],
                'losses': match_stats['agent2_wins'],
                'draws': match_stats['draws'],
                'win_rate': match_stats['agent1_win_rate'],
                'avg_moves': match_stats['avg_moves'],
                'avg_time_ms': match_stats['agent1_avg_time'] * 1000,
                'total_time': match_stats['total_time'],
            }
            
            total_wins += match_stats['agent1_wins']
            total_games += games_per_opponent
        
        # Overall statistics
        results['overall'] = {
            'total_games': total_games,
            'total_wins': total_wins,
            'overall_win_rate': total_wins / total_games if total_games > 0 else 0.0,
            'estimated_elo': self._estimate_elo(results['opponents'])
        }
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _estimate_elo(self, opponent_results: Dict) -> float:
        """
        Estimate ELO rating based on performance against known opponents.
        
        Approximate baseline ELOs:
        - Random: 800
        - Center: 1000
        - Negamax-4: 1400
        - Negamax-6: 1600
        - Negamax-8: 1800
        """
        baseline_elos = {
            'Random': 800,
            'Center': 1000,
            'Negamax-4': 1400,
            'Negamax-6': 1600,
            'Negamax-8': 1800,
        }
        
        elo_estimates = []
        
        for opp_name, stats in opponent_results.items():
            if opp_name not in baseline_elos:
                continue
            
            opp_elo = baseline_elos[opp_name]
            win_rate = stats['win_rate']
            
            # Estimate ELO from win rate using logistic formula
            # win_rate ≈ 1 / (1 + 10^((opp_elo - agent_elo) / 400))
            if 0.01 < win_rate < 0.99:
                estimated_elo = opp_elo + 400 * np.log10((1 / win_rate) - 1)
                elo_estimates.append(estimated_elo)
        
        if elo_estimates:
            return float(np.mean(elo_estimates))
        return 1200.0  # Default baseline
    
    def _print_summary(self, results: Dict):
        """Print formatted benchmark summary."""
        print(f"\n{'='*70}")
        print(f"Benchmark Summary: {results['agent_name']}")
        print(f"{'='*70}\n")
        
        print(f"{'Opponent':<20} {'W-L-D':<15} {'Win Rate':<12} {'Avg Time':<12} {'Avg Moves'}")
        print(f"{'-'*70}")
        
        for opp_name, stats in results['opponents'].items():
            w, l, d = stats['wins'], stats['losses'], stats['draws']
            wr = stats['win_rate']
            at = stats['avg_time_ms']
            am = stats['avg_moves']
            
            print(f"{opp_name:<20} {w:>3}-{l:<3}-{d:<3}    {wr:>6.1%}      "
                  f"{at:>7.1f}ms    {am:>6.1f}")
        
        print(f"{'-'*70}")
        overall = results['overall']
        print(f"{'OVERALL':<20} {overall['total_wins']:>3}/{overall['total_games']:<7}  "
              f"{overall['overall_win_rate']:>6.1%}")
        print(f"{'Estimated ELO':<20} {overall['estimated_elo']:>6.0f}")
        print(f"{'='*70}\n")
    
    def save_results(self, results: Dict, filepath: str):
        """
        Save benchmark results to JSON.
        
        Args:
            results: Benchmark results dictionary
            filepath: Output file path
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {filepath}")
    
    def compare_agents(self, results_list: List[Dict]) -> Dict:
        """
        Compare multiple agents' benchmark results.
        
        Args:
            results_list: List of benchmark result dictionaries
        
        Returns:
            Comparison dictionary
        """
        comparison = {
            'timestamp': datetime.now().isoformat(),
            'num_agents': len(results_list),
            'agents': []
        }
        
        for result in results_list:
            agent_summary = {
                'name': result['agent_name'],
                'overall_win_rate': result['overall']['overall_win_rate'],
                'estimated_elo': result['overall']['estimated_elo'],
                'per_opponent_win_rates': {
                    opp: stats['win_rate']
                    for opp, stats in result['opponents'].items()
                }
            }
            comparison['agents'].append(agent_summary)
        
        # Sort by ELO
        comparison['agents'].sort(key=lambda x: x['estimated_elo'], reverse=True)
        
        return comparison


if __name__ == "__main__":
    # Test benchmark
    print("Testing Benchmark Suite...")
    print("=" * 70)
    
    # Create a simple test agent
    def test_agent(board, mark):
        moves = get_valid_moves(board)
        if not moves:
            return 0
        # Slightly better than random - prefer center
        center = config.COLUMNS // 2
        return min(moves, key=lambda x: abs(x - center))
    
    # Run benchmark
    benchmark = Benchmark()
    results = benchmark.run_benchmark(
        test_agent,
        agent_name="TestAgent",
        games_per_opponent=20,
        opponents_to_test=['Random', 'Center']
    )
    
    # Save results
    benchmark.save_results(results, 'experiments/test_benchmark.json')
    
    print("\n✓ Benchmark test passed!")

