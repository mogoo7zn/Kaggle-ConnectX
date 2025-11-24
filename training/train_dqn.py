"""
Training Pipeline for DQN ConnectX Agent
Supports self-play and opponent-based training with visualization
"""

import json
import numpy as np
import random
import time
import os
from typing import Callable, Dict, List, Tuple, Optional
from collections import deque

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'core'))

from config import config
from dqn_agent import DQNAgent, evaluate_agent
from utils import (
    encode_state, get_valid_moves, is_terminal, 
    make_move, calculate_reward, create_directories,
    get_negamax_move
)
from visualize import plot_training_metrics, plot_win_rates, create_training_summary


class TrainingMetrics:
    """Track and store training metrics."""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.win_rates = []
        self.loss_values = []
        self.epsilon_values = []
        self.eval_episodes = []
        
        # Running averages
        self.recent_rewards = deque(maxlen=100)
        self.recent_lengths = deque(maxlen=100)
    
    def add_episode(self, reward: float, length: int, epsilon: float):
        """Add episode statistics."""
        self.episode_rewards.append(reward)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)
        self.recent_rewards.append(reward)
        self.recent_lengths.append(length)
    
    def add_evaluation(self, episode: int, win_rate: float):
        """Add evaluation result."""
        self.eval_episodes.append(episode)
        self.win_rates.append(win_rate)
    
    def add_loss(self, loss: float):
        """Add training loss."""
        if loss is not None:
            self.loss_values.append(loss)
    
    def get_recent_avg_reward(self) -> float:
        """Get average reward over recent episodes."""
        return np.mean(self.recent_rewards) if self.recent_rewards else 0.0
    
    def get_recent_avg_length(self) -> float:
        """Get average episode length over recent episodes."""
        return np.mean(self.recent_lengths) if self.recent_lengths else 0.0
    
    def get_recent_avg_loss(self) -> float:
        """Get average loss over recent training steps."""
        return np.mean(self.loss_values[-100:]) if self.loss_values else 0.0


def random_bot_policy(board: List[int], mark: int) -> int:
    """Simple random policy that selects from valid moves."""
    valid_moves = get_valid_moves(board)
    return random.choice(valid_moves) if valid_moves else 0


def center_preference_policy(board: List[int], mark: int) -> int:
    """Heuristic bot that prioritizes center columns."""
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return 0

    center_col = config.COLUMNS // 2
    return min(valid_moves, key=lambda col: (abs(center_col - col), random.random()))


def negamax_bot_policy(board: List[int], mark: int) -> int:
    """Wrapper around the negamax heuristic for compatibility."""
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return 0
    return get_negamax_move(board, mark)


class OpponentSampler:
    """Manage opponent sampling from heuristics and saved checkpoints."""

    def __init__(self, checkpoint_dir: str, top_k: int = 3):
        self.checkpoint_dir = checkpoint_dir
        self.top_k = top_k
        self.snapshot_metadata: List[Dict] = []
        self.cached_agents: Dict[str, Callable[[List[int], int], int]] = {}
        self.latest_checkpoint: Optional[str] = None
        self.heuristic_policies = {
            'random': random_bot_policy,
            'center': center_preference_policy,
            'negamax': negamax_bot_policy,
        }

    def _load_checkpoint_policy(self, path: str) -> Callable[[List[int], int], int]:
        if path not in self.cached_agents:
            opponent_agent = DQNAgent(model_type='standard', use_double_dqn=True)
            opponent_agent.load_model(path)
            opponent_agent.epsilon = 0.0

            def policy(board: List[int], mark: int) -> int:
                return opponent_agent.select_action(board, mark, epsilon=0.0)

            self.cached_agents[path] = policy

        return self.cached_agents[path]

    def register_snapshot(self, path: str, win_rate: float, episode: int):
        """Register a new checkpoint for sampling and keep top-K by win rate."""
        self.latest_checkpoint = path

        metadata = {
            'path': path,
            'win_rate': win_rate,
            'episode': episode,
            'timestamp': time.time(),
        }

        # Merge or append metadata
        existing_paths = {m['path'] for m in self.snapshot_metadata}
        if path not in existing_paths:
            self.snapshot_metadata.append(metadata)
        else:
            self.snapshot_metadata = [m for m in self.snapshot_metadata if m['path'] != path]
            self.snapshot_metadata.append(metadata)

        # Keep top-K snapshots by validation win rate
        self.snapshot_metadata = sorted(
            self.snapshot_metadata, key=lambda m: m['win_rate'], reverse=True
        )[: self.top_k]

        # Drop cached opponents that fell out of the top-K
        valid_paths = {m['path'] for m in self.snapshot_metadata}
        for cached_path in list(self.cached_agents.keys()):
            if cached_path not in valid_paths and cached_path != self.latest_checkpoint:
                self.cached_agents.pop(cached_path, None)

    def set_latest_checkpoint(self, path: str):
        self.latest_checkpoint = path

    def sample_opponent(self, elo: float) -> Tuple[str, Callable[[List[int], int], int]]:
        """Sample an opponent policy based on current Elo and available snapshots."""
        pool: List[Tuple[str, Callable[[List[int], int], int]]] = []
        kinds: List[str] = []

        for name, policy in self.heuristic_policies.items():
            pool.append((name, policy))
            kinds.append('heuristic')

        if self.latest_checkpoint:
            pool.append(('latest', self._load_checkpoint_policy(self.latest_checkpoint)))
            kinds.append('latest')

        for meta in self.snapshot_metadata:
            pool.append((f"snapshot_{os.path.basename(meta['path'])}", self._load_checkpoint_policy(meta['path'])))
            kinds.append('snapshot')

        if not pool:
            return 'random', random_bot_policy

        def weight(kind: str) -> float:
            if elo < 1100:
                return {'heuristic': 3.0, 'latest': 1.0, 'snapshot': 0.5}.get(kind, 1.0)
            elif elo < 1250:
                return {'heuristic': 2.0, 'latest': 3.0, 'snapshot': 2.0}.get(kind, 1.5)
            else:
                return {'heuristic': 1.5, 'latest': 3.0, 'snapshot': 3.0}.get(kind, 2.0)

        weights = [weight(k) for k in kinds]
        opponent_name, opponent_policy = random.choices(pool, weights=weights, k=1)[0]
        return opponent_name, opponent_policy


def get_policy_from_name(name: str) -> Callable[[List[int], int], int]:
    """Return a board-level policy callable for a given bot name."""
    mapping = {
        'random': random_bot_policy,
        'center': center_preference_policy,
        'negamax': negamax_bot_policy,
    }
    return mapping.get(name, random_bot_policy)


def policy_to_env_bot(policy: Callable[[List[int], int], int]) -> Callable:
    """Wrap a board/mark policy into a Kaggle-environment compatible callable."""

    def bot_fn(observation, configuration):
        return int(policy(observation.board, observation.mark))

    return bot_fn


def evaluate_validation_suite(agent: DQNAgent, bot_names: List[str]) -> Tuple[Dict[str, dict], float]:
    """Evaluate the agent against a suite of validation bots."""
    results: Dict[str, dict] = {}

    for bot_name in bot_names:
        policy = get_policy_from_name(bot_name)
        eval_results = evaluate_agent(agent, policy_to_env_bot(policy), num_games=config.EVAL_GAMES)
        results[bot_name] = eval_results

    overall_win_rate = float(np.mean([r['win_rate'] for r in results.values()])) if results else 0.0
    return results, overall_win_rate


def update_elo_from_win_rate(current_elo: float, win_rate: float, k_factor: float = 64.0) -> float:
    """Update a synthetic Elo score based on recent validation win rate."""
    return max(800.0, current_elo + k_factor * (win_rate - 0.5))


def save_best_checkpoint(agent: DQNAgent, mode: str, episode: int, validation_results: Dict[str, dict], overall_win_rate: float) -> Tuple[str, str]:
    """Persist the best model and accompanying metadata for later conversion."""
    timestamp = int(time.time())
    base_name = f"best_{mode}_ep{episode}_{timestamp}"
    model_path = os.path.join(config.CHECKPOINT_DIR, f"{base_name}.pth")
    metadata_path = os.path.join(config.CHECKPOINT_DIR, f"{base_name}.json")

    agent.save_model(model_path)

    metadata = {
        'episode': episode,
        'mode': mode,
        'overall_validation_win_rate': overall_win_rate,
        'validation_results': validation_results,
        'timestamp': timestamp,
        'model_path': model_path,
        'artifacts': {
            'torchscript_target': model_path.replace('.pth', '_ts.pt'),
            'onnx_target': model_path.replace('.pth', '.onnx')
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Best checkpoint saved to {model_path} with metadata {metadata_path}")

    return model_path, metadata_path


def play_self_play_episode(agent: DQNAgent) -> Tuple[float, int]:
    """
    Play one episode of self-play (agent vs itself).
    
    Args:
        agent: DQN agent
    
    Returns:
        Tuple of (total_reward, episode_length)
    """
    # Initialize board
    board = [0] * (config.ROWS * config.COLUMNS)
    current_mark = 1
    episode_length = 0
    total_reward = 0.0
    
    # Store transitions for both players
    transitions_p1 = []
    transitions_p2 = []
    
    while True:
        # Encode current state
        state = encode_state(board, current_mark)
        
        # Select action
        action = agent.select_action(board, current_mark)
        
        # Make move
        next_board = make_move(board, action, current_mark)
        
        # Check if game ended
        done, winner = is_terminal(next_board)
        
        # Calculate reward
        reward = calculate_reward(board, action, current_mark, next_board, done, winner)
        
        # Encode next state
        next_state = encode_state(next_board, current_mark)
        
        # Store transition for current player
        if current_mark == 1:
            transitions_p1.append((state, action, reward, next_state, done))
        else:
            transitions_p2.append((state, action, reward, next_state, done))
        
        episode_length += 1
        
        if done:
            # Assign final rewards
            if winner == 1:
                p1_final_reward = config.REWARD_WIN
                p2_final_reward = config.REWARD_LOSS
            elif winner == 2:
                p1_final_reward = config.REWARD_LOSS
                p2_final_reward = config.REWARD_WIN
            else:  # Draw
                p1_final_reward = config.REWARD_DRAW
                p2_final_reward = config.REWARD_DRAW
            
            # Update final rewards in transitions
            if transitions_p1:
                last_state, last_action, _, last_next_state, last_done = transitions_p1[-1]
                transitions_p1[-1] = (last_state, last_action, p1_final_reward, last_next_state, last_done)
            
            if transitions_p2:
                last_state, last_action, _, last_next_state, last_done = transitions_p2[-1]
                transitions_p2[-1] = (last_state, last_action, p2_final_reward, last_next_state, last_done)
            
            # Store all transitions
            for transition in transitions_p1 + transitions_p2:
                agent.store_transition(*transition)
            
            # Return reward from player 1's perspective
            total_reward = p1_final_reward
            break
        
        # Switch player
        board = next_board
        current_mark = 3 - current_mark
    
    return total_reward, episode_length


def play_opponent_episode(agent: DQNAgent,
                          opponent_policy: Callable[[List[int], int], int],
                          opponent_name: str = "custom") -> Tuple[float, int]:
    """
    Play one episode against an opponent.
    
    Args:
        agent: DQN agent
        opponent_type: Type of opponent ('random' or 'negamax')
    
    Returns:
        Tuple of (total_reward, episode_length)
    """
    board = [0] * (config.ROWS * config.COLUMNS)
    episode_length = 0
    
    # Agent is player 1, opponent is player 2
    agent_mark = 1
    opponent_mark = 2
    current_mark = 1
    
    transitions = []
    
    while True:
        if current_mark == agent_mark:
            # Agent's turn
            state = encode_state(board, agent_mark)
            action = agent.select_action(board, agent_mark)
            next_board = make_move(board, action, agent_mark)
            
            done, winner = is_terminal(next_board)
            reward = calculate_reward(board, action, agent_mark, next_board, done, winner)
            next_state = encode_state(next_board, agent_mark)
            
            transitions.append((state, action, reward, next_state, done))
            
            episode_length += 1
            
            if done:
                # Store all transitions
                for transition in transitions:
                    agent.store_transition(*transition)
                
                if winner == agent_mark:
                    return config.REWARD_WIN, episode_length
                elif winner == 0:
                    return config.REWARD_DRAW, episode_length
                else:
                    return config.REWARD_LOSS, episode_length
            
            board = next_board
        else:
            # Opponent's turn
            valid_moves = get_valid_moves(board)

            if not valid_moves:
                break

            action = opponent_policy(board, opponent_mark)

            board = make_move(board, action, opponent_mark)
            
            done, winner = is_terminal(board)
            
            if done:
                # Update last transition with final outcome
                if transitions:
                    last_state, last_action, _, last_next_state, _ = transitions[-1]
                    
                    if winner == agent_mark:
                        final_reward = config.REWARD_WIN
                    elif winner == 0:
                        final_reward = config.REWARD_DRAW
                    else:
                        final_reward = config.REWARD_LOSS
                    
                    transitions[-1] = (last_state, last_action, final_reward, last_next_state, True)
                
                # Store all transitions
                for transition in transitions:
                    agent.store_transition(*transition)
                
                if winner == agent_mark:
                    return config.REWARD_WIN, episode_length
                elif winner == 0:
                    return config.REWARD_DRAW, episode_length
                else:
                    return config.REWARD_LOSS, episode_length
        
        current_mark = 3 - current_mark
    
    return 0.0, episode_length


def train_agent(agent: DQNAgent,
                mode: str = 'self_play',
                num_episodes: int = 5000,
                eval_interval: int = 100,
                save_interval: int = 500,
                opponent_type: str = 'random',
                opponent_sampler: Optional[OpponentSampler] = None,
                initial_elo: float = 1000.0) -> TrainingMetrics:
    """
    Train the DQN agent.
    
    Args:
        agent: DQN agent to train
        mode: Training mode ('self_play' or 'opponent')
        num_episodes: Number of episodes to train
        eval_interval: Evaluate every N episodes
        save_interval: Save checkpoint every N episodes
        opponent_type: Type of opponent for opponent mode
    
    Returns:
        TrainingMetrics object with training history
    """
    metrics = TrainingMetrics()
    best_win_rate = 0.0
    best_validation_win_rate = 0.0
    current_elo = initial_elo
    
    print(f"\n{'='*60}")
    print(f"Starting training: {mode} mode")
    print(f"Episodes: {num_episodes}")
    print(f"Device: {config.DEVICE}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    for episode in range(1, num_episodes + 1):
        episode_start = time.time()
        
        # Play episode
        opponent_label = opponent_type
        if mode == 'self_play':
            reward, length = play_self_play_episode(agent)
        else:  # opponent mode
            if opponent_sampler is not None:
                opponent_label, opponent_policy = opponent_sampler.sample_opponent(current_elo)
            else:
                opponent_policy = get_policy_from_name(opponent_type)

            reward, length = play_opponent_episode(agent, opponent_policy, opponent_label)

        # Train agent (multiple steps to leverage larger replay)
        loss = None
        for _ in range(config.TRAINING_STEPS_PER_EPISODE):
            step_loss = agent.train_step()
            if step_loss is not None:
                loss = step_loss

        # Update metrics
        metrics.add_episode(reward, length, agent.epsilon)
        metrics.add_loss(loss)
        
        # Progress reporting
        if episode % 10 == 0:
            episode_time = time.time() - episode_start
            avg_reward = metrics.get_recent_avg_reward()
            avg_length = metrics.get_recent_avg_length()
            avg_loss = metrics.get_recent_avg_loss()
            
            print(f"Episode {episode:5d}/{num_episodes} | "
                  f"Reward: {reward:6.2f} | "
                  f"Avg Reward: {avg_reward:6.2f} | "
                  f"Length: {length:3d} | "
                  f"Opp: {opponent_label:>8} | "
                  f"Epsilon: {agent.epsilon:.4f} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Buffer: {len(agent.memory):6d} | "
                  f"Time: {episode_time:.2f}s")
        
        # Evaluation
        if episode % eval_interval == 0:
            print(f"\n{'='*60}")
            print(f"Evaluation at episode {episode}")
            print(f"{'='*60}")

            # Evaluate against validation suite
            validation_results, overall_validation_win_rate = evaluate_validation_suite(agent, config.VALIDATION_BOTS)

            for bot_name, eval_results in validation_results.items():
                print(f"Win rate vs {bot_name:7s}: {eval_results['win_rate']:.2%} "
                      f"({eval_results['wins']}W-{eval_results['losses']}L-{eval_results['draws']}D)")

            print(f"Overall validation win rate: {overall_validation_win_rate:.2%}")

            metrics.add_evaluation(episode, overall_validation_win_rate)

            # Save best model by validation performance
            if overall_validation_win_rate > best_validation_win_rate:
                best_validation_win_rate = overall_validation_win_rate
                model_path, metadata_path = save_best_checkpoint(
                    agent, mode, episode, validation_results, overall_validation_win_rate
                )
                if opponent_sampler is not None:
                    opponent_sampler.register_snapshot(model_path, overall_validation_win_rate, episode)

            # Update synthetic Elo to schedule tougher opponents
            current_elo = update_elo_from_win_rate(current_elo, overall_validation_win_rate)

            # Track best historical win rate for logging continuity
            if overall_validation_win_rate > best_win_rate:
                best_win_rate = overall_validation_win_rate

            print(f"{'='*60}\n")
        
        # Save checkpoint
        if episode % save_interval == 0:
            checkpoint_path = os.path.join(config.MODEL_DIR,
                                          f'checkpoint_{mode}_ep{episode}.pth')
            checkpoint_metrics = {
                'episode': episode,
                'mode': mode,
                'avg_reward': metrics.get_recent_avg_reward(),
                'best_win_rate': best_win_rate
            }
            agent.save_checkpoint(checkpoint_path, episode, checkpoint_metrics)

            if opponent_sampler is not None:
                opponent_sampler.set_latest_checkpoint(checkpoint_path)
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"Best win rate: {best_win_rate:.2%}")
    print(f"Final epsilon: {agent.epsilon:.4f}")
    print(f"Total steps: {agent.steps_done}")
    print(f"{'='*60}\n")
    
    # Save final checkpoint
    final_path = os.path.join(config.MODEL_DIR, f'final_model_{mode}.pth')
    agent.save_model(final_path)
    print(f"Final model saved to {final_path}")
    
    # Generate visualization
    plot_path = os.path.join(config.PLOT_DIR, f'training_metrics_{mode}.png')
    plot_training_metrics(metrics, save_path=plot_path, show=False)
    
    return metrics


def main():
    """Main training function."""
    # Create directories
    create_directories()

    # Create agent
    agent = DQNAgent(model_type='standard', use_double_dqn=True)
    opponent_sampler = OpponentSampler(config.CHECKPOINT_DIR, top_k=config.TOP_K_SNAPSHOTS)
    
    # Phase 1: Self-play training
    print("\n" + "="*60)
    print("PHASE 1: SELF-PLAY TRAINING")
    print("="*60)
    
    self_play_metrics = train_agent(
        agent=agent,
        mode='self_play',
        num_episodes=config.SELF_PLAY_EPISODES,
        eval_interval=config.EVAL_INTERVAL,
        save_interval=config.SAVE_INTERVAL,
        opponent_sampler=opponent_sampler
    )
    
    # Phase 2: Fine-tuning against opponents
    print("\n" + "="*60)
    print("PHASE 2: FINE-TUNING AGAINST OPPONENTS")
    print("="*60)
    
    # Train against random opponent
    print("\nTraining against RANDOM opponent...")
    random_metrics = train_agent(
        agent=agent,
        mode='opponent',
        num_episodes=config.OPPONENT_EPISODES // 2,
        eval_interval=config.EVAL_INTERVAL,
        save_interval=config.SAVE_INTERVAL,
        opponent_type='random',
        opponent_sampler=opponent_sampler
    )
    
    # Train against negamax opponent
    print("\nTraining against NEGAMAX opponent...")
    negamax_metrics = train_agent(
        agent=agent,
        mode='opponent',
        num_episodes=config.OPPONENT_EPISODES // 2,
        eval_interval=config.EVAL_INTERVAL,
        save_interval=config.SAVE_INTERVAL,
        opponent_type='negamax',
        opponent_sampler=opponent_sampler
    )
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    print("\nEvaluating against RANDOM opponent...")
    random_eval = evaluate_agent(agent, policy_to_env_bot(random_bot_policy), num_games=100)
    print(f"Win rate: {random_eval['win_rate']:.2%} "
          f"({random_eval['wins']}W-{random_eval['losses']}L-{random_eval['draws']}D)")

    print("\nEvaluating against NEGAMAX opponent...")
    negamax_eval = evaluate_agent(agent, policy_to_env_bot(negamax_bot_policy), num_games=100)
    print(f"Win rate: {negamax_eval['win_rate']:.2%} "
          f"({negamax_eval['wins']}W-{negamax_eval['losses']}L-{negamax_eval['draws']}D)")
    
    # Generate comprehensive visualizations
    print("\nGenerating visualizations...")
    
    # Combined win rates plot
    metrics_dict = {
        'Self-Play': self_play_metrics,
        'vs Random': random_metrics,
        'vs Negamax': negamax_metrics
    }
    
    win_rates_path = os.path.join(config.PLOT_DIR, 'win_rates_comparison.png')
    plot_win_rates(metrics_dict, save_path=win_rates_path, show=False)
    
    # Training summary
    summary_path = os.path.join(config.LOG_DIR, 'training_summary.txt')
    create_training_summary(agent, metrics_dict, summary_path)
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Models saved in: {config.MODEL_DIR}")
    print(f"Plots saved in: {config.PLOT_DIR}")
    print(f"Logs saved in: {config.LOG_DIR}")
    print("="*60)
    
    return agent, self_play_metrics, random_metrics, negamax_metrics


if __name__ == "__main__":
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    import torch
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Run training
    agent, self_play_metrics, random_metrics, negamax_metrics = main()

