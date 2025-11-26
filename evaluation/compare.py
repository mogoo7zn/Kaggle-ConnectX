"""
Agent Comparison and Visualization
Generate comparative reports, charts, and HTML dashboards
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from typing import List, Dict
from datetime import datetime


class AgentComparator:
    """
    Compare and visualize performance of multiple agents.
    
    Features:
    - Side-by-side win rate comparison
    - Radar charts for multi-dimensional comparison
    - Performance tables
    - HTML report generation
    """
    
    def __init__(self):
        """Initialize comparator."""
        self.results = []
    
    def load_results(self, filepaths: List[str]):
        """
        Load benchmark results from JSON files.
        
        Args:
            filepaths: List of paths to benchmark result files
        """
        self.results = []
        for filepath in filepaths:
            with open(filepath, 'r') as f:
                result = json.load(f)
                self.results.append(result)
        print(f"Loaded {len(self.results)} benchmark results")
    
    def add_result(self, result: Dict):
        """
        Add a benchmark result dictionary.
        
        Args:
            result: Benchmark result dictionary
        """
        self.results.append(result)
    
    def plot_win_rates(self, save_path: str = None, show: bool = False):
        """
        Plot win rates comparison bar chart.
        
        Args:
            save_path: Path to save figure
            show: Whether to display plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Get common opponents
        all_opponents = set()
        for result in self.results:
            all_opponents.update(result['opponents'].keys())
        opponents = sorted(list(all_opponents))
        
        # Extract data
        agent_names = [r['agent_name'] for r in self.results]
        win_rates = []
        
        for result in self.results:
            rates = []
            for opp in opponents:
                if opp in result['opponents']:
                    rates.append(result['opponents'][opp]['win_rate'] * 100)
                else:
                    rates.append(0.0)
            win_rates.append(rates)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(opponents))
        width = 0.8 / len(agent_names)
        
        for i, (name, rates) in enumerate(zip(agent_names, win_rates)):
            offset = (i - len(agent_names)/2 + 0.5) * width
            ax.bar(x + offset, rates, width, label=name, alpha=0.8)
        
        ax.set_xlabel('Opponent', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Agent Win Rates Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(opponents, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Win rates plot saved to: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_radar_chart(self, save_path: str = None, show: bool = False):
        """
        Plot radar chart for multi-dimensional comparison.
        
        Args:
            save_path: Path to save figure
            show: Whether to display plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        # Get common opponents
        all_opponents = set()
        for result in self.results:
            all_opponents.update(result['opponents'].keys())
        opponents = sorted(list(all_opponents))
        
        if len(opponents) < 3:
            print("Need at least 3 opponents for radar chart")
            return
        
        # Prepare data
        agent_names = [r['agent_name'] for r in self.results]
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2 * np.pi, len(opponents), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(agent_names)))
        
        for idx, result in enumerate(self.results):
            values = []
            for opp in opponents:
                if opp in result['opponents']:
                    values.append(result['opponents'][opp]['win_rate'] * 100)
                else:
                    values.append(0.0)
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=agent_names[idx],
                   color=colors[idx], alpha=0.7)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(opponents, size=10)
        ax.set_ylim(0, 100)
        ax.set_ylabel('Win Rate (%)', size=11)
        ax.set_title('Agent Performance Radar Chart', size=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Radar chart saved to: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def plot_elo_comparison(self, save_path: str = None, show: bool = False):
        """
        Plot ELO ratings comparison.
        
        Args:
            save_path: Path to save figure
            show: Whether to display plot
        """
        if not self.results:
            print("No results to plot")
            return
        
        agent_names = [r['agent_name'] for r in self.results]
        elo_ratings = [r['overall']['estimated_elo'] for r in self.results]
        
        # Sort by ELO
        sorted_data = sorted(zip(agent_names, elo_ratings), key=lambda x: x[1], reverse=True)
        agent_names, elo_ratings = zip(*sorted_data)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(agent_names)))
        bars = ax.barh(agent_names, elo_ratings, color=colors, alpha=0.8)
        
        ax.set_xlabel('Estimated ELO Rating', fontsize=12)
        ax.set_title('Agent ELO Ratings Comparison', fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for bar, elo in zip(bars, elo_ratings):
            width = bar.get_width()
            ax.text(width + 20, bar.get_y() + bar.get_height()/2,
                   f'{elo:.0f}', ha='left', va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ELO comparison saved to: {save_path}")
        
        if show:
            plt.show()
        
        plt.close()
    
    def generate_html_report(self, output_path: str):
        """
        Generate comprehensive HTML report.
        
        Args:
            output_path: Path to output HTML file
        """
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ConnectX Agent Comparison Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            background-color: white;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin: 20px 0;
        }}
        th {{
            background-color: #3498db;
            color: white;
            padding: 12px;
            text-align: left;
        }}
        td {{
            padding: 10px;
            border-bottom: 1px solid #ddd;
        }}
        tr:hover {{
            background-color: #f8f9fa;
        }}
        .metric {{
            display: inline-block;
            background-color: white;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-label {{
            font-size: 14px;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .winner {{
            background-color: #2ecc71;
            color: white;
            padding: 5px 10px;
            border-radius: 3px;
        }}
    </style>
</head>
<body>
    <h1>ConnectX Agent Comparison Report</h1>
    <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    
    <h2>Overall Rankings</h2>
    <table>
        <thead>
            <tr>
                <th>Rank</th>
                <th>Agent</th>
                <th>Estimated ELO</th>
                <th>Overall Win Rate</th>
                <th>Games Played</th>
            </tr>
        </thead>
        <tbody>
"""
        
        # Sort by ELO
        sorted_results = sorted(self.results,
                               key=lambda x: x['overall']['estimated_elo'],
                               reverse=True)
        
        for rank, result in enumerate(sorted_results, 1):
            agent_name = result['agent_name']
            elo = result['overall']['estimated_elo']
            win_rate = result['overall']['overall_win_rate'] * 100
            games = result['overall']['total_games']
            
            winner_badge = '<span class="winner">TOP</span>' if rank == 1 else ''
            
            html += f"""
            <tr>
                <td>{rank}</td>
                <td><strong>{agent_name}</strong> {winner_badge}</td>
                <td>{elo:.0f}</td>
                <td>{win_rate:.1f}%</td>
                <td>{games}</td>
            </tr>
"""
        
        html += """
        </tbody>
    </table>
    
    <h2>Detailed Performance by Opponent</h2>
"""
        
        for result in sorted_results:
            agent_name = result['agent_name']
            html += f"""
    <h3>{agent_name}</h3>
    <table>
        <thead>
            <tr>
                <th>Opponent</th>
                <th>W-L-D</th>
                <th>Win Rate</th>
                <th>Avg Moves</th>
                <th>Avg Time (ms)</th>
            </tr>
        </thead>
        <tbody>
"""
            
            for opp_name, stats in result['opponents'].items():
                w, l, d = stats['wins'], stats['losses'], stats['draws']
                wr = stats['win_rate'] * 100
                am = stats['avg_moves']
                at = stats['avg_time_ms']
                
                html += f"""
            <tr>
                <td>{opp_name}</td>
                <td>{w}-{l}-{d}</td>
                <td>{wr:.1f}%</td>
                <td>{am:.1f}</td>
                <td>{at:.1f}</td>
            </tr>
"""
            
            html += """
        </tbody>
    </table>
"""
        
        html += """
</body>
</html>
"""
        
        # Write to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"HTML report saved to: {output_path}")
    
    def generate_all_reports(self, output_dir: str):
        """
        Generate all comparison reports and visualizations.
        
        Args:
            output_dir: Directory to save all outputs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\nGenerating comparison reports in: {output_dir}")
        
        # Generate plots
        self.plot_win_rates(os.path.join(output_dir, 'win_rates_comparison.png'))
        self.plot_elo_comparison(os.path.join(output_dir, 'elo_comparison.png'))
        
        try:
            self.plot_radar_chart(os.path.join(output_dir, 'radar_comparison.png'))
        except:
            print("Skipped radar chart (requires >=3 opponents)")
        
        # Generate HTML report
        self.generate_html_report(os.path.join(output_dir, 'comparison_report.html'))
        
        print(f"\n✓ All reports generated successfully!")


if __name__ == "__main__":
    # Test comparator
    print("Testing Agent Comparator...")
    print("=" * 70)
    
    # Create dummy results for testing
    dummy_results = [
        {
            'agent_name': 'Agent A',
            'opponents': {
                'Random': {'wins': 90, 'losses': 8, 'draws': 2, 'win_rate': 0.9,
                          'avg_moves': 15.5, 'avg_time_ms': 10.2},
                'Center': {'wins': 75, 'losses': 20, 'draws': 5, 'win_rate': 0.75,
                          'avg_moves': 18.3, 'avg_time_ms': 12.5},
            },
            'overall': {'total_games': 200, 'total_wins': 165, 'overall_win_rate': 0.825,
                       'estimated_elo': 1450}
        },
        {
            'agent_name': 'Agent B',
            'opponents': {
                'Random': {'wins': 95, 'losses': 3, 'draws': 2, 'win_rate': 0.95,
                          'avg_moves': 14.2, 'avg_time_ms': 25.8},
                'Center': {'wins': 85, 'losses': 12, 'draws': 3, 'win_rate': 0.85,
                          'avg_moves': 17.1, 'avg_time_ms': 28.3},
            },
            'overall': {'total_games': 200, 'total_wins': 180, 'overall_win_rate': 0.9,
                       'estimated_elo': 1550}
        }
    ]
    
    # Create comparator
    comparator = AgentComparator()
    for result in dummy_results:
        comparator.add_result(result)
    
    # Generate reports
    comparator.generate_all_reports('experiments/comparison_test')
    
    print("\n✓ Comparator test passed!")

