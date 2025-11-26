"""
Optimized AlphaZero Configuration
Tuned for ConnectX with faster training on Quadro 4000 8GB

Key Changes from Standard Config:
1. Smaller network (4-6 residual blocks instead of 10-20)
2. Reduced MCTS simulations (50-100 instead of 800)
3. Larger batch inference size for better GPU utilization
4. Adaptive simulation count based on game phase
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class AlphaZeroConfigOptimized:
    """
    Optimized AlphaZero configuration for ConnectX.
    
    Designed for:
    - Fast training iteration on medium GPUs (RTX 3060, Quadro 4000, etc.)
    - ConnectX game complexity (smaller than Go/Chess)
    - Maximizing self-play throughput
    """
    
    # ============== Environment Parameters ==============
    ROWS = 6
    COLUMNS = 7
    INAROW = 4
    
    # ============== MCTS Parameters (OPTIMIZED) ==============
    # Original AlphaZero uses 800 sims, but ConnectX is much simpler
    NUM_SIMULATIONS = 50  # Reduced from 800/100 - sufficient for ConnectX
    NUM_SIMULATIONS_EVAL = 100  # More sims during evaluation for accuracy
    
    # Adaptive simulation counts per game phase
    SIMS_EARLY_GAME = 50   # First 6 moves - need exploration
    SIMS_MID_GAME = 50     # Moves 6-20 - standard play
    SIMS_LATE_GAME = 30    # After move 20 - simpler positions
    
    C_PUCT = 1.5  # Exploration constant
    TEMPERATURE = 1.0
    TEMP_THRESHOLD = 15  # Reduced - fewer moves in ConnectX than Go
    
    # Dirichlet noise
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25
    
    # Virtual loss for parallel MCTS
    VIRTUAL_LOSS = 3
    
    # ============== Neural Network Architecture (OPTIMIZED) ==============
    # ConnectX doesn't need a huge network
    
    # Simplified input: just current state (no history needed for ConnectX)
    HISTORY_LENGTH = 1  # Reduced from 8 - ConnectX is fully observable
    INPUT_CHANNELS = 3  # player, opponent, valid_moves
    
    # Smaller ResNet (original AlphaGo Zero: 20-40 blocks)
    NUM_RES_BLOCKS = 4  # Reduced from 10 - sufficient for ConnectX
    NUM_FILTERS = 64    # Reduced from 128 - smaller game
    
    # Heads
    POLICY_FILTERS = 16  # Reduced from 32
    VALUE_FILTERS = 16   # Reduced from 32
    VALUE_HIDDEN = 64    # Reduced from 256
    
    # Regularization
    DROPOUT = 0.1
    L2_REGULARIZATION = 1e-4
    
    # ============== Training Parameters (OPTIMIZED) ==============
    LEARNING_RATE = 0.002  # Slightly lower for stability
    MOMENTUM = 0.9
    LR_SCHEDULE = "cosine"  # Changed from step - smoother
    LR_MILESTONES = [50000, 100000, 150000]
    LR_GAMMA = 0.1
    
    BATCH_SIZE = 256  # Reduced for faster iteration
    TRAINING_EPOCHS = 5  # Reduced from 10 - train more frequently
    
    # ============== Self-Play Parameters (OPTIMIZED) ==============
    NUM_SELFPLAY_GAMES = 50  # Reduced for faster iteration
    NUM_PARALLEL_GAMES = 16  # Increased parallelism
    
    # Replay buffer
    REPLAY_BUFFER_SIZE = 100000  # Reduced - games are shorter
    MIN_REPLAY_SIZE = 2000  # Reduced from 10000
    
    # ============== Batched Inference Parameters ==============
    MAX_BATCH_SIZE = 128  # Max states per GPU batch
    MAX_WAIT_MS = 5.0     # Max wait time before processing partial batch
    USE_BATCHED_INFERENCE = True
    
    # ============== Evaluation Parameters ==============
    EVAL_GAMES = 20  # Reduced from 40
    EVAL_WIN_RATE_THRESHOLD = 0.55
    
    ARENA_OPPONENTS = ["random", "negamax"]
    ARENA_GAMES_PER_OPPONENT = 50  # Reduced from 100
    
    # ============== Training Iterations ==============
    MAX_ITERATIONS = 500  # Reduced - should converge faster
    EVAL_INTERVAL = 5
    SAVE_INTERVAL = 10
    
    # ============== Data Augmentation ==============
    USE_AUGMENTATION = True
    
    # ============== Saving and Logging ==============
    MODEL_DIR = "alphazero/models"
    CHECKPOINT_DIR = "alphazero/checkpoints"
    LOG_DIR = "alphazero/logs"
    PLOT_DIR = "alphazero/plots"
    SELFPLAY_DIR = "alphazero/selfplay_data"
    
    # ============== Device ==============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============== Mixed Precision ==============
    USE_AMP = True
    
    # ============== ELO Tracking ==============
    INITIAL_ELO = 1500
    K_FACTOR = 32
    
    # ============== Performance Tuning ==============
    # Pin memory for faster data transfer
    PIN_MEMORY = True
    # Number of data loader workers
    NUM_WORKERS = 2
    # Compile model with torch.compile (PyTorch 2.0+)
    USE_TORCH_COMPILE = False
    
    def get_adaptive_simulations(self, move_count: int) -> int:
        """
        Get adaptive simulation count based on game phase.
        
        Args:
            move_count: Number of moves played so far
        
        Returns:
            Number of simulations to run
        """
        if move_count < 6:
            return self.SIMS_EARLY_GAME
        elif move_count < 20:
            return self.SIMS_MID_GAME
        else:
            return self.SIMS_LATE_GAME
    
    def __repr__(self):
        return (f"AlphaZeroConfigOptimized(sims={self.NUM_SIMULATIONS}, "
                f"res_blocks={self.NUM_RES_BLOCKS}, filters={self.NUM_FILTERS}, "
                f"parallel_games={self.NUM_PARALLEL_GAMES})")


# Preset configurations for different scenarios
class FastDebugConfig(AlphaZeroConfigOptimized):
    """Ultra-fast config for debugging."""
    NUM_SIMULATIONS = 10
    NUM_RES_BLOCKS = 2
    NUM_FILTERS = 32
    NUM_SELFPLAY_GAMES = 10
    BATCH_SIZE = 64
    MIN_REPLAY_SIZE = 200
    EVAL_GAMES = 5


class BalancedConfig(AlphaZeroConfigOptimized):
    """Balanced config for good quality with reasonable speed."""
    NUM_SIMULATIONS = 100
    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 128
    NUM_SELFPLAY_GAMES = 100
    BATCH_SIZE = 512


class QualityConfig(AlphaZeroConfigOptimized):
    """Higher quality config (slower but better)."""
    NUM_SIMULATIONS = 200
    NUM_RES_BLOCKS = 8
    NUM_FILTERS = 128
    NUM_SELFPLAY_GAMES = 200
    BATCH_SIZE = 512
    TRAINING_EPOCHS = 10


# Create default optimized config
az_config_optimized = AlphaZeroConfigOptimized()


def estimate_training_speed(config: AlphaZeroConfigOptimized) -> dict:
    """
    Estimate training speed characteristics.
    
    Args:
        config: Configuration to analyze
    
    Returns:
        Dictionary with speed estimates
    """
    # Rough estimates based on typical hardware
    
    # MCTS time per move (ms) - roughly linear in simulations
    mcts_time_per_sim_ms = 0.5  # Typical with good GPU batching
    avg_game_length = 21  # Average ConnectX game length
    
    mcts_time_per_game_ms = (
        config.NUM_SIMULATIONS * avg_game_length * mcts_time_per_sim_ms
    )
    
    # Self-play games per iteration
    total_selfplay_time_s = (
        config.NUM_SELFPLAY_GAMES * mcts_time_per_game_ms / 1000 / 
        max(1, config.NUM_PARALLEL_GAMES / 4)  # Parallelism factor
    )
    
    # Training time per iteration (rough estimate)
    training_time_s = config.TRAINING_EPOCHS * 5  # ~5s per epoch
    
    # Total iteration time
    iteration_time_s = total_selfplay_time_s + training_time_s
    
    return {
        "mcts_ms_per_game": mcts_time_per_game_ms,
        "selfplay_time_s": total_selfplay_time_s,
        "training_time_s": training_time_s,
        "iteration_time_s": iteration_time_s,
        "iterations_per_hour": 3600 / iteration_time_s,
        "games_per_hour": config.NUM_SELFPLAY_GAMES * 3600 / iteration_time_s
    }


if __name__ == "__main__":
    print("AlphaZero Configuration Comparison")
    print("=" * 70)
    
    configs = {
        "FastDebug": FastDebugConfig(),
        "Optimized (Default)": az_config_optimized,
        "Balanced": BalancedConfig(),
        "Quality": QualityConfig(),
    }
    
    for name, cfg in configs.items():
        print(f"\n{name}:")
        print(f"  Simulations: {cfg.NUM_SIMULATIONS}")
        print(f"  Network: {cfg.NUM_RES_BLOCKS} blocks, {cfg.NUM_FILTERS} filters")
        print(f"  Self-play games: {cfg.NUM_SELFPLAY_GAMES}")
        print(f"  Parallel games: {cfg.NUM_PARALLEL_GAMES}")
        print(f"  Batch size: {cfg.BATCH_SIZE}")
        
        est = estimate_training_speed(cfg)
        print(f"  Estimated iteration time: {est['iteration_time_s']:.0f}s")
        print(f"  Estimated games/hour: {est['games_per_hour']:.0f}")
    
    print("\n" + "=" * 70)
    print(f"Device: {az_config_optimized.DEVICE}")
    print(f"Using optimized config: {az_config_optimized}")

