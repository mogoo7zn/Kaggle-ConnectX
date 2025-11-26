"""
主实验脚本 - 运行双轨强化学习实验
Training both Rainbow DQN and AlphaZero agents
"""

import sys
import os
import argparse
import torch
import numpy as np
import random
from datetime import datetime

# Set random seeds for reproducibility
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_rainbow(quick_mode=False):
    """Train Rainbow DQN agent."""
    print("\n" + "="*80)
    print("TRAINING RAINBOW DQN")
    print("="*80 + "\n")
    
    from agents.rainbow.rainbow_config import rainbow_config
    from agents.rainbow.rainbow_agent import RainbowAgent
    from agents.rainbow.train_rainbow import RainbowTrainer
    
    # Adjust config for quick mode
    if quick_mode:
        rainbow_config.SELF_PLAY_EPISODES = 1000
        rainbow_config.OPPONENT_EPISODES = 500
        rainbow_config.EVAL_INTERVAL = 100
        rainbow_config.SAVE_INTERVAL = 500
    
    # Create agent
    agent = RainbowAgent(use_noisy=True, use_distributional=False)
    
    # Create trainer
    run_name = f"rainbow_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    trainer = RainbowTrainer(agent, run_name=run_name)
    
    # Self-play training
    print("\nPhase 1: Self-Play Training")
    trainer.train(
        num_episodes=rainbow_config.SELF_PLAY_EPISODES,
        mode='self_play',
        eval_interval=rainbow_config.EVAL_INTERVAL,
        save_interval=rainbow_config.SAVE_INTERVAL
    )
    
    return agent


def train_alphazero(quick_mode=False):
    """Train AlphaZero agent."""
    print("\n" + "="*80)
    print("TRAINING ALPHAZERO")
    print("="*80 + "\n")
    
    from agents.alphazero.az_config import az_config
    from agents.alphazero.train_alphazero import AlphaZeroTrainer
    
    # Adjust config for quick mode
    if quick_mode:
        az_config.MAX_ITERATIONS = 20
        az_config.NUM_SELFPLAY_GAMES = 50
        az_config.EVAL_INTERVAL = 5
        az_config.SAVE_INTERVAL = 10
    
    # Create trainer
    trainer = AlphaZeroTrainer()
    
    # Train
    trainer.train(max_iterations=az_config.MAX_ITERATIONS)
    
    return trainer.network


def run_benchmarks(rainbow_agent, alphazero_network, output_dir='experiments'):
    """Run comprehensive benchmarks on both agents."""
    print("\n" + "="*80)
    print("RUNNING BENCHMARKS")
    print("="*80 + "\n")
    
    from evaluation.benchmark import Benchmark
    from evaluation.arena import create_agent_wrapper
    
    benchmark = Benchmark()
    
    # Benchmark Rainbow
    print("\n[1/2] Benchmarking Rainbow DQN...")
    rainbow_fn = create_agent_wrapper(rainbow_agent, 'rainbow')
    rainbow_results = benchmark.run_benchmark(
        rainbow_fn,
        agent_name="Rainbow DQN",
        games_per_opponent=100
    )
    benchmark.save_results(rainbow_results,
                          os.path.join(output_dir, 'rainbow_benchmark.json'))
    
    # Benchmark AlphaZero
    print("\n[2/2] Benchmarking AlphaZero...")
    alphazero_fn = create_agent_wrapper(alphazero_network, 'alphazero')
    alphazero_results = benchmark.run_benchmark(
        alphazero_fn,
        agent_name="AlphaZero",
        games_per_opponent=100
    )
    benchmark.save_results(alphazero_results,
                          os.path.join(output_dir, 'alphazero_benchmark.json'))
    
    return rainbow_results, alphazero_results


def generate_comparison_report(rainbow_results, alphazero_results, output_dir='experiments'):
    """Generate comprehensive comparison report."""
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT")
    print("="*80 + "\n")
    
    from evaluation.compare import AgentComparator
    
    comparator = AgentComparator()
    comparator.add_result(rainbow_results)
    comparator.add_result(alphazero_results)
    
    # Generate all reports
    report_dir = os.path.join(output_dir, f'comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    comparator.generate_all_reports(report_dir)
    
    print(f"\nComparison report available at: {report_dir}/comparison_report.html")
    
    return report_dir


def main():
    """Main experimental pipeline."""
    parser = argparse.ArgumentParser(description='Run ConnectX dual-agent experiment')
    parser.add_argument('--quick', action='store_true',
                       help='Run in quick mode (reduced episodes for testing)')
    parser.add_argument('--skip-rainbow', action='store_true',
                       help='Skip Rainbow DQN training')
    parser.add_argument('--skip-alphazero', action='store_true',
                       help='Skip AlphaZero training')
    parser.add_argument('--skip-benchmark', action='store_true',
                       help='Skip benchmark evaluation')
    parser.add_argument('--output-dir', default='experiments',
                       help='Output directory for results')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set seeds
    set_seeds(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("CONNECTX DUAL-AGENT EXPERIMENT")
    print("="*80)
    print(f"Quick mode: {args.quick}")
    print(f"Output directory: {args.output_dir}")
    print(f"Random seed: {args.seed}")
    print("="*80 + "\n")
    
    rainbow_agent = None
    alphazero_network = None
    
    # Train Rainbow
    if not args.skip_rainbow:
        try:
            rainbow_agent = train_rainbow(quick_mode=args.quick)
            print("\n[OK] Rainbow DQN training completed!")
        except Exception as e:
            print(f"\n[ERROR] Rainbow DQN training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Train AlphaZero
    if not args.skip_alphazero:
        try:
            alphazero_network = train_alphazero(quick_mode=args.quick)
            print("\n[OK] AlphaZero training completed!")
        except Exception as e:
            print(f"\n[ERROR] AlphaZero training failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Run benchmarks
    if not args.skip_benchmark and rainbow_agent and alphazero_network:
        try:
            rainbow_results, alphazero_results = run_benchmarks(
                rainbow_agent, alphazero_network, args.output_dir
            )
            print("\n[OK] Benchmarks completed!")
            
            # Generate comparison report
            report_dir = generate_comparison_report(
                rainbow_results, alphazero_results, args.output_dir
            )
            print(f"\n[OK] Comparison report generated!")
            
        except Exception as e:
            print(f"\n[ERROR] Benchmark/comparison failed: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETED!")
    print("="*80)
    print(f"\nResults saved in: {args.output_dir}/")
    print("\nNext steps:")
    print("  1. Review comparison report (comparison_*/comparison_report.html)")
    print("  2. Check training logs in outputs/logs/")
    print("  3. Load best models from outputs/checkpoints/")
    print("  4. Prepare Kaggle submission using tools/prepare_submission.py")
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()

