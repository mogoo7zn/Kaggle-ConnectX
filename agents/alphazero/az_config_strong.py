"""
Strong AlphaZero Configuration
增强版训练配置，用于训练更强的模型

关键改进:
1. 更大的网络 (更多残差块和滤波器)
2. 更多的MCTS模拟次数
3. 更多的自对弈游戏
4. 更长的训练迭代
5. 优化的学习率调度
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class AlphaZeroConfigStrong:
    """
    强化版AlphaZero配置
    
    设计目标:
    - 训练出更强的Connect Four AI
    - 在有GPU的情况下，牺牲一些训练速度换取模型质量
    - 更充分的MCTS搜索
    """
    
    # ============== 环境参数 ==============
    ROWS = 6
    COLUMNS = 7
    INAROW = 4
    
    # ============== MCTS参数 (增强版) ==============
    # 关键: 更多的模拟次数 = 更强的策略
    NUM_SIMULATIONS = 400       # 训练时的模拟次数 (原100 -> 400)
    NUM_SIMULATIONS_EVAL = 800  # 评估时的模拟次数 (更多以确保准确)
    
    # 自适应模拟次数
    SIMS_EARLY_GAME = 400   # 开局 (前6步) - 关键决策
    SIMS_MID_GAME = 400     # 中盘 (6-20步) - 标准游戏
    SIMS_LATE_GAME = 200    # 终盘 (20步后) - 位置较简单
    
    C_PUCT = 1.5            # 探索常数
    TEMPERATURE = 1.0       # 动作选择温度
    TEMP_THRESHOLD = 15     # 15步后使用贪婪策略
    
    # Dirichlet噪声 (增加探索)
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.25  # 噪声权重
    
    # 虚拟损失 (并行MCTS)
    VIRTUAL_LOSS = 3
    
    # ============== 神经网络架构 (增强版) ==============
    HISTORY_LENGTH = 1      # ConnectX不需要历史
    INPUT_CHANNELS = 3      # 玩家、对手、有效动作
    
    # 更大的ResNet (原4块 -> 8块, 原64滤波器 -> 128)
    NUM_RES_BLOCKS = 8      # 残差块数量 (增强)
    NUM_FILTERS = 128       # 滤波器数量 (增强)
    
    # 头部网络
    POLICY_FILTERS = 32     # 策略头滤波器 (增强)
    VALUE_FILTERS = 32      # 价值头滤波器 (增强)
    VALUE_HIDDEN = 128      # 价值头隐藏层
    
    # 正则化
    DROPOUT = 0.1
    L2_REGULARIZATION = 1e-4
    
    # ============== 训练参数 (优化版) ==============
    LEARNING_RATE = 0.01    # 初始学习率 (SGD)
    LR_MIN = 0.0001         # 最小学习率
    MOMENTUM = 0.9
    LR_SCHEDULE = "cosine"  # 余弦退火
    
    BATCH_SIZE = 512        # 批大小
    TRAINING_EPOCHS = 10    # 每迭代训练轮数
    
    # ============== 自对弈参数 (增强版) ==============
    NUM_SELFPLAY_GAMES = 200    # 每迭代自对弈游戏数 (原50 -> 200)
    NUM_PARALLEL_GAMES = 16     # 并行游戏数
    
    # 经验回放缓冲区
    REPLAY_BUFFER_SIZE = 500000  # 缓冲区大小 (增大)
    MIN_REPLAY_SIZE = 10000      # 开始训练前的最小样本数
    
    # ============== 批量推理参数 ==============
    MAX_BATCH_SIZE = 256
    MAX_WAIT_MS = 5.0
    USE_BATCHED_INFERENCE = True
    
    # ============== 评估参数 ==============
    EVAL_GAMES = 50             # 评估游戏数
    EVAL_WIN_RATE_THRESHOLD = 0.55  # 替换模型的胜率阈值
    
    ARENA_OPPONENTS = ["random", "negamax"]
    ARENA_GAMES_PER_OPPONENT = 100
    
    # ============== 训练迭代 (增加) ==============
    MAX_ITERATIONS = 500        # 最大迭代次数 (增加)
    EVAL_INTERVAL = 5           # 评估间隔
    SAVE_INTERVAL = 10          # 保存间隔
    
    # ============== 数据增强 ==============
    USE_AUGMENTATION = True     # 水平翻转增强
    
    # ============== 保存路径 ==============
    MODEL_DIR = "alphazero/models"
    CHECKPOINT_DIR = "alphazero/checkpoints"
    LOG_DIR = "alphazero/logs"
    PLOT_DIR = "alphazero/plots"
    SELFPLAY_DIR = "alphazero/selfplay_data"
    
    # ============== 设备 ==============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ============== 混合精度 ==============
    USE_AMP = True
    
    # ============== ELO追踪 ==============
    INITIAL_ELO = 1500
    K_FACTOR = 32
    
    # ============== 性能调优 ==============
    PIN_MEMORY = True
    NUM_WORKERS = 4
    USE_TORCH_COMPILE = False
    
    def get_adaptive_simulations(self, move_count: int) -> int:
        """根据游戏阶段获取自适应模拟次数"""
        if move_count < 6:
            return self.SIMS_EARLY_GAME
        elif move_count < 20:
            return self.SIMS_MID_GAME
        else:
            return self.SIMS_LATE_GAME
    
    def __repr__(self):
        return (f"AlphaZeroConfigStrong(sims={self.NUM_SIMULATIONS}, "
                f"res_blocks={self.NUM_RES_BLOCKS}, filters={self.NUM_FILTERS})")


# 预设配置: 极限强度 (需要更多计算资源)
class UltraStrongConfig(AlphaZeroConfigStrong):
    """极限强度配置 (需要强力GPU)"""
    NUM_SIMULATIONS = 800
    NUM_SIMULATIONS_EVAL = 1600
    NUM_RES_BLOCKS = 12
    NUM_FILTERS = 256
    NUM_SELFPLAY_GAMES = 500
    BATCH_SIZE = 1024
    MAX_ITERATIONS = 1000


# 预设配置: 平衡版 (适中计算资源)
class BalancedStrongConfig(AlphaZeroConfigStrong):
    """平衡强度配置"""
    NUM_SIMULATIONS = 200
    NUM_SIMULATIONS_EVAL = 400
    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 128
    NUM_SELFPLAY_GAMES = 150
    BATCH_SIZE = 256


# 预设配置: 快速强化版 (快速训练但仍然增强)
class FastStrongConfig(AlphaZeroConfigStrong):
    """快速增强配置 (较快训练)"""
    NUM_SIMULATIONS = 150
    NUM_SIMULATIONS_EVAL = 300
    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 96
    NUM_SELFPLAY_GAMES = 100
    BATCH_SIZE = 256
    MAX_ITERATIONS = 300


# 创建默认强化配置实例
az_config_strong = AlphaZeroConfigStrong()


def print_config_comparison():
    """打印配置对比"""
    from agents.alphazero.az_config_optimized import az_config_optimized
    
    print("=" * 70)
    print("AlphaZero 配置对比")
    print("=" * 70)
    
    configs = {
        "原优化版": az_config_optimized,
        "强化版 (推荐)": az_config_strong,
        "平衡强化版": BalancedStrongConfig(),
        "快速强化版": FastStrongConfig(),
        "极限强度版": UltraStrongConfig(),
    }
    
    print(f"\n{'配置名称':<20} {'MCTS模拟':<12} {'残差块':<10} {'滤波器':<10} {'自对弈':<10}")
    print("-" * 70)
    
    for name, cfg in configs.items():
        print(f"{name:<20} {cfg.NUM_SIMULATIONS:<12} {cfg.NUM_RES_BLOCKS:<10} "
              f"{cfg.NUM_FILTERS:<10} {cfg.NUM_SELFPLAY_GAMES:<10}")
    
    print("\n" + "=" * 70)


def estimate_training_time(config, gpu_factor=1.0):
    """
    估算训练时间
    
    Args:
        config: 配置对象
        gpu_factor: GPU性能因子 (1.0 = RTX 3060级别)
    
    Returns:
        估算的训练时间 (分钟)
    """
    # 每次模拟大约 2-5ms (取决于GPU)
    ms_per_sim = 3.0 / gpu_factor
    avg_game_length = 21
    
    # 每局游戏的MCTS时间
    mcts_time_per_game = config.NUM_SIMULATIONS * avg_game_length * ms_per_sim / 1000
    
    # 自对弈时间 (考虑并行)
    parallel_factor = min(config.NUM_PARALLEL_GAMES, 8)  # 并行效率
    selfplay_time = config.NUM_SELFPLAY_GAMES * mcts_time_per_game / parallel_factor
    
    # 训练时间 (大约每epoch 5-10秒)
    training_time = config.TRAINING_EPOCHS * 7 / gpu_factor
    
    # 每迭代总时间
    iteration_time = selfplay_time + training_time
    
    # 总训练时间
    total_minutes = config.MAX_ITERATIONS * iteration_time / 60
    
    return {
        'per_iteration_seconds': iteration_time,
        'total_minutes': total_minutes,
        'total_hours': total_minutes / 60
    }


if __name__ == "__main__":
    print_config_comparison()
    
    print("\n训练时间估算 (RTX 3060级别GPU):")
    print("-" * 50)
    
    configs = {
        "强化版": az_config_strong,
        "平衡强化版": BalancedStrongConfig(),
        "快速强化版": FastStrongConfig(),
    }
    
    for name, cfg in configs.items():
        est = estimate_training_time(cfg)
        print(f"\n{name}:")
        print(f"  每迭代: ~{est['per_iteration_seconds']:.0f}秒")
        print(f"  总时间: ~{est['total_hours']:.1f}小时 ({est['total_minutes']:.0f}分钟)")
        print(f"  迭代数: {cfg.MAX_ITERATIONS}")

