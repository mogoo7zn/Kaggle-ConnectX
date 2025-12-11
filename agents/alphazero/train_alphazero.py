"""
AlphaZero训练脚本（统一版）
使用当前可用的配置与组件进行训练

使用方法:
    python train_alphazero.py --config strong      # 推荐 (平衡速度和质量)
    python train_alphazero.py --config strong+     # 强化+ (模型<100MB可提交)
    python train_alphazero.py --config balanced    # 平衡版 (较快)
    python train_alphazero.py --config fast        # 快速版 (最快)
    python train_alphazero.py --config ultra       # 极限版 (需要更强GPU)
    
    # 从检查点继续训练
    python train_alphazero.py --checkpoint path/to/checkpoint.pth
    
    # 指定迭代次数
    python train_alphazero.py --iterations 200
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
import numpy as np
import time
import os
import sys
from datetime import datetime
from typing import Tuple, Optional
import argparse
import logging
from logging.handlers import RotatingFileHandler

# 添加路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from agents.alphazero.az_config import (
    az_config,
    BalancedConfig,
    FastConfig,
    UltraConfig,
    StrongPlusConfig,
)
from agents.alphazero.self_play import (
    ParallelSelfPlay,
    SimpleSelfPlay,
)
from agents.alphazero.fast_board import FastBoard
from agents.alphazero.batched_inference import SyncInferenceWrapper
from agents.alphazero.mcts import MCTS


class StrongPolicyValueNetwork(nn.Module):
    """
    增强版策略-价值网络
    
    架构特点:
    - 更深的ResNet (8-12个残差块)
    - 更多滤波器 (128-256)
    - 更好的特征提取能力
    """
    
    def __init__(self, config=None):
        super().__init__()
        self.config = config or az_config
        
        num_filters = self.config.NUM_FILTERS
        num_res_blocks = self.config.NUM_RES_BLOCKS
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        
        # 残差块 (使用Sequential以匹配checkpoint格式)
        self.res_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(inplace=True),
                nn.Conv2d(num_filters, num_filters, 3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters)
            )
            for _ in range(num_res_blocks)
        ])
        
        # 策略头
        policy_filters = self.config.POLICY_FILTERS
        self.policy_conv = nn.Conv2d(num_filters, policy_filters, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(policy_filters)
        self.policy_fc = nn.Linear(policy_filters * 6 * 7, 7)
        
        # 价值头
        value_filters = self.config.VALUE_FILTERS
        self.value_conv = nn.Conv2d(num_filters, value_filters, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(value_filters)
        self.value_fc1 = nn.Linear(value_filters * 6 * 7, self.config.VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(self.config.VALUE_HIDDEN, 1)
        
        self.dropout = nn.Dropout(self.config.DROPOUT)
        
        # 初始化权重
        self._initialize_weights()
    
    def _initialize_weights(self):
        """He初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播"""
        # 初始卷积
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # 残差块
        for res_block in self.res_blocks:
            residual = x
            x = res_block(x)
            x = F.relu(x + residual)
        
        # 策略头
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.dropout(policy)
        policy_logits = self.policy_fc(policy)
        
        # 价值头
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value


class AlphaZeroStrongTrainer:
    """
    强化版AlphaZero训练器
    
    特点:
    - 更强的网络架构
    - 更充分的MCTS搜索
    - 更多的训练数据
    - 学习率热身和余弦退火
    """
    
    def __init__(self, config=None, use_parallel: bool = True):
        """
        初始化训练器
        
        Args:
            config: 配置对象
            use_parallel: 是否使用并行自对弈
        """
        self.config = config or az_config
        self.device = self.config.DEVICE
        
        # 创建网络
        self.network = StrongPolicyValueNetwork(self.config)
        self.network.to(self.device)
        
        # 打印模型信息
        num_params = sum(p.numel() for p in self.network.parameters())
        print(f"网络参数量: {num_params:,}")
        print(f"残差块数量: {self.config.NUM_RES_BLOCKS}")
        print(f"滤波器数量: {self.config.NUM_FILTERS}")
        print(f"MCTS模拟次数: {self.config.NUM_SIMULATIONS}")
        
        # 优化器 (SGD with momentum)
        self.optimizer = optim.SGD(
            self.network.parameters(),
            lr=self.config.LEARNING_RATE,
            momentum=self.config.MOMENTUM,
            weight_decay=self.config.L2_REGULARIZATION,
            nesterov=True  # Nesterov动量加速收敛
        )
        
        # 学习率调度器 (余弦退火 + 热身)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=50 * self.config.TRAINING_EPOCHS,  # 50个迭代一个周期
            T_mult=2,
            eta_min=self.config.LR_MIN if hasattr(self.config, 'LR_MIN') else 0.0001
        )
        
        # 混合精度
        self.scaler = GradScaler('cuda') if self.config.USE_AMP else None
        
        # 自对弈引擎
        if use_parallel:
            self.self_play = ParallelSelfPlay(
                self.network,
                num_parallel_games=self.config.NUM_PARALLEL_GAMES,
                config=self.config,
                use_batched_inference=self.config.USE_BATCHED_INFERENCE
            )
        else:
            self.self_play = SimpleSelfPlay(self.network, self.config)
        
        # 训练统计
        self.iteration = 0
        self.total_games = 0
        self.best_win_rate = 0.0
        self.training_history = []
        
        # 创建目录
        self._create_directories()
        
        # 运行名称
        self.run_name = f"alphazero_strong_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 设置详细日志记录
        self._setup_logging()
    
    def _create_directories(self):
        """创建必要的目录"""
        for dir_path in [self.config.MODEL_DIR, self.config.CHECKPOINT_DIR,
                        self.config.LOG_DIR, self.config.PLOT_DIR]:
            os.makedirs(dir_path, exist_ok=True)
    
    def _setup_logging(self):
        """设置详细的日志记录"""
        # 创建日志文件路径
        log_file = os.path.join(self.config.LOG_DIR, f"training_{self.run_name}.log")
        
        # 配置日志格式
        log_format = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # 创建logger
        self.logger = logging.getLogger(f'AlphaZero_{self.run_name}')
        self.logger.setLevel(logging.DEBUG)
        
        # 避免重复添加handler
        if not self.logger.handlers:
            # 文件handler (带轮转，最大10MB，保留5个备份)
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=10*1024*1024, 
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_format)
            
            # 控制台handler (只显示INFO及以上级别)
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(log_format)
            
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        # 记录训练开始信息
        self.logger.info("="*70)
        self.logger.info("AlphaZero强化训练开始")
        self.logger.info("="*70)
        self.logger.info(f"运行名称: {self.run_name}")
        self.logger.info(f"设备: {self.device}")
        self.logger.info(f"配置: {self.config}")
        self.logger.info(f"网络参数量: {sum(p.numel() for p in self.network.parameters()):,}")
        self.logger.info(f"残差块数量: {self.config.NUM_RES_BLOCKS}")
        self.logger.info(f"滤波器数量: {self.config.NUM_FILTERS}")
        self.logger.info(f"MCTS模拟次数: {self.config.NUM_SIMULATIONS}")
        self.logger.info(f"自对弈游戏/迭代: {self.config.NUM_SELFPLAY_GAMES}")
        self.logger.info(f"最大迭代数: {self.config.MAX_ITERATIONS}")
        self.logger.info(f"学习率: {self.config.LEARNING_RATE}")
        self.logger.info(f"批大小: {self.config.BATCH_SIZE}")
        self.logger.info(f"训练轮数/迭代: {self.config.TRAINING_EPOCHS}")
        self.logger.info(f"日志文件: {log_file}")
        self.logger.info("="*70)
    
    def train_iteration(self) -> dict:
        """
        运行一个训练迭代
        
        Returns:
            训练统计字典
        """
        stats = {}
        self.iteration += 1
        
        print(f"\n{'='*70}")
        print(f"迭代 {self.iteration} / {self.config.MAX_ITERATIONS}")
        print(f"{'='*70}")
        
        self.logger.info("="*70)
        self.logger.info(f"迭代 {self.iteration} / {self.config.MAX_ITERATIONS}")
        self.logger.info("="*70)
        
        # 阶段1: 自对弈
        print("\n[阶段1] 自对弈")
        self.logger.info("[阶段1] 开始自对弈")
        self.network.eval()
        
        if hasattr(self.self_play, 'start'):
            self.self_play.start()
        
        selfplay_start = time.perf_counter()
        
        # 使用批量生成
        if isinstance(self.self_play, ParallelSelfPlay):
            num_examples = self.self_play.generate_games_batched(
                num_games=self.config.NUM_SELFPLAY_GAMES,
                batch_size=self.config.NUM_PARALLEL_GAMES
            )
        else:
            num_examples = self.self_play.generate_self_play_data(
                num_games=self.config.NUM_SELFPLAY_GAMES
            )
        
        selfplay_time = time.perf_counter() - selfplay_start
        
        if hasattr(self.self_play, 'stop'):
            self.self_play.stop()
        
        self.total_games += self.config.NUM_SELFPLAY_GAMES
        stats['selfplay_time'] = selfplay_time
        stats['selfplay_examples'] = num_examples
        stats['games_per_sec'] = self.config.NUM_SELFPLAY_GAMES / selfplay_time
        
        print(f"  生成 {num_examples} 样本 ({selfplay_time:.1f}s, "
              f"{stats['games_per_sec']:.1f} 局/秒)")
        
        self.logger.info(f"[阶段1完成] 生成 {num_examples} 个训练样本")
        self.logger.info(f"  自对弈时间: {selfplay_time:.2f}秒")
        self.logger.info(f"  游戏速度: {stats['games_per_sec']:.2f} 局/秒")
        self.logger.info(f"  累计总游戏数: {self.total_games}")
        self.logger.info(f"  缓冲区大小: {len(self.self_play.buffer)}")
        
        # 阶段2: 训练
        print("\n[阶段2] 神经网络训练")
        self.logger.info("[阶段2] 开始神经网络训练")
        self.network.train()
        
        training_start = time.perf_counter()
        train_stats = self._train_network()
        training_time = time.perf_counter() - training_start
        
        stats.update(train_stats)
        stats['training_time'] = training_time
        
        print(f"  训练 {self.config.TRAINING_EPOCHS} epochs ({training_time:.1f}s)")
        print(f"  策略损失: {train_stats['policy_loss']:.4f}, "
              f"价值损失: {train_stats['value_loss']:.4f}")
        print(f"  学习率: {train_stats['learning_rate']:.6f}")
        
        self.logger.info(f"[阶段2完成] 训练 {self.config.TRAINING_EPOCHS} 个epochs")
        self.logger.info(f"  训练时间: {training_time:.2f}秒")
        self.logger.info(f"  策略损失: {train_stats['policy_loss']:.6f}")
        self.logger.info(f"  价值损失: {train_stats['value_loss']:.6f}")
        self.logger.info(f"  总损失: {train_stats['total_loss']:.6f}")
        self.logger.info(f"  当前学习率: {train_stats['learning_rate']:.8f}")
        
        # 阶段3: 评估 (定期)
        if self.iteration % self.config.EVAL_INTERVAL == 0:
            print("\n[阶段3] 评估")
            self.logger.info("[阶段3] 开始模型评估")
            eval_stats = self._evaluate()
            stats.update(eval_stats)
            
            # 定期保存训练曲线
            self._plot_training_curves()
        
        # 保存检查点
        if self.iteration % self.config.SAVE_INTERVAL == 0:
            self._save_checkpoint()
            self.logger.info(f"[检查点] 已保存迭代 {self.iteration} 的检查点")
        
        # 记录历史
        self.training_history.append(stats)
        
        # 打印总结
        total_time = selfplay_time + training_time
        print(f"\n迭代 {self.iteration} 完成 ({total_time:.1f}s)")
        print(f"  总游戏数: {self.total_games}")
        print(f"  缓冲区大小: {len(self.self_play.buffer)}")
        
        self.logger.info(f"[迭代完成] 迭代 {self.iteration} 完成")
        self.logger.info(f"  总耗时: {total_time:.2f}秒")
        self.logger.info(f"  累计总游戏数: {self.total_games}")
        self.logger.info(f"  当前缓冲区大小: {len(self.self_play.buffer)}")
        self.logger.info(f"  最佳胜率: {self.best_win_rate*100:.2f}%")
        
        return stats
    
    def _train_network(self) -> dict:
        """训练网络"""
        if len(self.self_play.buffer) < self.config.BATCH_SIZE:
            return {'policy_loss': 0, 'value_loss': 0, 'total_loss': 0, 'learning_rate': 0}
        
        total_policy_loss = 0
        total_value_loss = 0
        num_batches = 0
        
        for epoch in range(self.config.TRAINING_EPOCHS):
            # 采样批次
            states, policies, values = self.self_play.get_training_batch(
                self.config.BATCH_SIZE
            )
            
            # 转换为张量
            states = torch.from_numpy(states).to(self.device)
            target_policies = torch.from_numpy(policies).to(self.device)
            target_values = torch.from_numpy(values).to(self.device).unsqueeze(1)
            
            self.optimizer.zero_grad()
            
            # 前向传播 (混合精度)
            if self.config.USE_AMP and self.scaler:
                with autocast('cuda'):
                    policy_logits, values = self.network(states)
                    
                    # 分布交叉熵: target_policies 为概率分布
                    log_probs = F.log_softmax(policy_logits, dim=1)
                    policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
                    value_loss = F.mse_loss(values, target_values)
                    total_loss = policy_loss + value_loss
                
                # 反向传播 (缩放)
                self.scaler.scale(total_loss).backward()
                
                # 梯度裁剪 (防止梯度爆炸)
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                policy_logits, values = self.network(states)
                
                log_probs = F.log_softmax(policy_logits, dim=1)
                policy_loss = -(target_policies * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(values, target_values)
                total_loss = policy_loss + value_loss
                
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            num_batches += 1
        
        return {
            'policy_loss': total_policy_loss / num_batches,
            'value_loss': total_value_loss / num_batches,
            'total_loss': (total_policy_loss + total_value_loss) / num_batches,
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
    
    def _evaluate(self) -> dict:
        """评估模型"""
        self.network.eval()
        stats = {}
        
        # 对抗随机agent
        wins, losses, draws = self._play_against_random(
            num_games=self.config.EVAL_GAMES
        )
        
        win_rate = wins / self.config.EVAL_GAMES
        stats['vs_random_win_rate'] = win_rate
        stats['vs_random_wins'] = wins
        stats['vs_random_losses'] = losses
        stats['vs_random_draws'] = draws
        
        print(f"  vs Random: {wins}胜 / {losses}负 / {draws}平 ({win_rate*100:.1f}%)")
        
        self.logger.info(f"[评估结果] vs Random对手")
        self.logger.info(f"  胜: {wins}, 负: {losses}, 平: {draws}")
        self.logger.info(f"  胜率: {win_rate*100:.2f}%")
        self.logger.info(f"  之前最佳胜率: {self.best_win_rate*100:.2f}%")
        
        # 保存最佳模型
        if win_rate > self.best_win_rate:
            self.best_win_rate = win_rate
            self._save_checkpoint("best")
            print(f"  [新最佳模型!] 胜率: {win_rate*100:.1f}%")
            self.logger.info(f"[新最佳模型!] 胜率提升至: {win_rate*100:.2f}%")
        
        return stats
    
    def _play_against_random(self, num_games: int) -> Tuple[int, int, int]:
        """与随机对手对弈"""
        inference = SyncInferenceWrapper(self.network)
        mcts = MCTS(inference_fn=inference.inference, config=self.config)
        
        wins = losses = draws = 0
        
        for game_idx in range(num_games):
            az_mark = 1 if game_idx % 2 == 0 else 2
            
            board = FastBoard()
            current_mark = 1
            
            while True:
                if current_mark == az_mark:
                    action = mcts.get_best_action(board, current_mark)
                else:
                    valid_moves = board.get_valid_moves()
                    action = np.random.choice(valid_moves)
                
                board.make_move_inplace(action, current_mark)
                
                is_terminal, winner = board.is_terminal()
                if is_terminal:
                    if winner == az_mark:
                        wins += 1
                    elif winner == 0:
                        draws += 1
                    else:
                        losses += 1
                    break
                
                current_mark = 3 - current_mark
        
        return wins, losses, draws
    
    def _save_checkpoint(self, name: str = None):
        """保存检查点"""
        if name is None:
            name = f"iter{self.iteration}"
        
        checkpoint_path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f"{name}_{self.run_name}.pth"
        )
        
        # 保存完整检查点 (用于恢复训练)
        torch.save({
            'iteration': self.iteration,
            'model_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'total_games': self.total_games,
            'best_win_rate': self.best_win_rate,
            'training_history': self.training_history,
            'config': str(self.config)
        }, checkpoint_path)
        
        print(f"  保存检查点: {checkpoint_path}")
        self.logger.info(f"[检查点保存] {checkpoint_path}")
        
        # 同时也保存一个仅包含权重的版本 (用于提交，<100MB)
        if 'best' in name or 'final' in name:
            submission_path = os.path.join(
                self.config.CHECKPOINT_DIR,
                f"{name}_{self.run_name}_submission.pth"
            )
            torch.save(self.network.state_dict(), submission_path)
            print(f"  保存提交模型: {submission_path}")
            self.logger.info(f"[提交模型保存] {submission_path}")

        self.logger.debug(f"  迭代: {self.iteration}, 总游戏数: {self.total_games}, 最佳胜率: {self.best_win_rate*100:.2f}%")
    
    def _plot_training_curves(self):
        """绘制并保存训练曲线"""
        if not self.training_history:
            return
        
        try:
            import matplotlib
            matplotlib.use('Agg')  # 非交互式后端
            import matplotlib.pyplot as plt
            
            # 提取数据
            iterations = list(range(1, len(self.training_history) + 1))
            policy_losses = [h.get('policy_loss', 0) for h in self.training_history]
            value_losses = [h.get('value_loss', 0) for h in self.training_history]
            total_losses = [h.get('total_loss', 0) for h in self.training_history]
            learning_rates = [h.get('learning_rate', 0) for h in self.training_history]
            win_rates = [h.get('vs_random_win_rate', None) for h in self.training_history]
            selfplay_times = [h.get('selfplay_time', 0) for h in self.training_history]
            
            # 创建图形
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'AlphaZero Training Progress - {self.run_name}', fontsize=14)
            
            # 1. 策略损失
            ax1 = axes[0, 0]
            ax1.plot(iterations, policy_losses, 'b-', linewidth=1.5)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Policy Loss')
            ax1.set_title('Policy Loss')
            ax1.grid(True, alpha=0.3)
            
            # 2. 价值损失
            ax2 = axes[0, 1]
            ax2.plot(iterations, value_losses, 'r-', linewidth=1.5)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Value Loss')
            ax2.set_title('Value Loss')
            ax2.grid(True, alpha=0.3)
            
            # 3. 总损失
            ax3 = axes[0, 2]
            ax3.plot(iterations, total_losses, 'g-', linewidth=1.5)
            ax3.set_xlabel('Iteration')
            ax3.set_ylabel('Total Loss')
            ax3.set_title('Total Loss')
            ax3.grid(True, alpha=0.3)
            
            # 4. 学习率
            ax4 = axes[1, 0]
            ax4.plot(iterations, learning_rates, 'm-', linewidth=1.5)
            ax4.set_xlabel('Iteration')
            ax4.set_ylabel('Learning Rate')
            ax4.set_title('Learning Rate Schedule')
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
            
            # 5. 胜率 (只绘制有评估的点)
            ax5 = axes[1, 1]
            eval_iters = [i for i, w in zip(iterations, win_rates) if w is not None]
            eval_wins = [w for w in win_rates if w is not None]
            if eval_wins:
                ax5.plot(eval_iters, [w * 100 for w in eval_wins], 'c-o', linewidth=1.5, markersize=4)
                ax5.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50%')
                ax5.axhline(y=self.best_win_rate * 100, color='green', linestyle='--', alpha=0.5, label=f'Best: {self.best_win_rate*100:.1f}%')
                ax5.legend()
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('Win Rate vs Random (%)')
            ax5.set_title('Win Rate vs Random')
            ax5.set_ylim(0, 105)
            ax5.grid(True, alpha=0.3)
            
            # 6. 自对弈时间
            ax6 = axes[1, 2]
            ax6.plot(iterations, selfplay_times, 'y-', linewidth=1.5)
            ax6.set_xlabel('Iteration')
            ax6.set_ylabel('Time (seconds)')
            ax6.set_title('Self-Play Time per Iteration')
            ax6.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            os.makedirs(self.config.PLOT_DIR, exist_ok=True)
            plot_path = os.path.join(self.config.PLOT_DIR, f'training_curves_{self.run_name}.png')
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            
            print(f"  保存训练曲线: {plot_path}")
            
        except ImportError:
            print("  [警告] matplotlib未安装，跳过绘图")
        except Exception as e:
            print(f"  [警告] 绘图失败: {e}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.network.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.iteration = checkpoint['iteration']
        self.total_games = checkpoint['total_games']
        self.best_win_rate = checkpoint.get('best_win_rate', 0.0)
        self.training_history = checkpoint.get('training_history', [])
        
        print(f"加载检查点: 迭代 {self.iteration}")
    
    def train(self, num_iterations: int = None):
        """
        运行完整训练循环
        
        Args:
            num_iterations: 迭代次数 (默认使用配置)
        """
        if num_iterations is None:
            num_iterations = self.config.MAX_ITERATIONS
        
        print(f"\n{'='*70}")
        print("开始AlphaZero强化训练")
        print(f"{'='*70}")
        print(f"  配置: {self.config}")
        print(f"  设备: {self.device}")
        print(f"  迭代数: {num_iterations}")
        print(f"  MCTS模拟: {self.config.NUM_SIMULATIONS}")
        print(f"  自对弈游戏/迭代: {self.config.NUM_SELFPLAY_GAMES}")
        
        start_time = time.perf_counter()
        
        try:
            for _ in range(num_iterations):
                self.train_iteration()
                
        except KeyboardInterrupt:
            print("\n训练被用户中断")
            self.logger.warning("训练被用户中断 (KeyboardInterrupt)")
            self._save_checkpoint("interrupted")
        except Exception as e:
            self.logger.error(f"训练过程中发生错误: {str(e)}", exc_info=True)
            raise
        
        total_time = time.perf_counter() - start_time
        
        print(f"\n{'='*70}")
        print("训练完成")
        print(f"{'='*70}")
        print(f"  总迭代数: {self.iteration}")
        print(f"  总游戏数: {self.total_games}")
        print(f"  总时间: {total_time/60:.1f} 分钟")
        print(f"  最佳vs随机胜率: {self.best_win_rate*100:.1f}%")
        
        self.logger.info("="*70)
        self.logger.info("训练完成")
        self.logger.info("="*70)
        self.logger.info(f"总迭代数: {self.iteration}")
        self.logger.info(f"总游戏数: {self.total_games}")
        self.logger.info(f"总训练时间: {total_time/60:.2f} 分钟 ({total_time:.2f} 秒)")
        self.logger.info(f"平均每迭代时间: {total_time/self.iteration:.2f} 秒")
        self.logger.info(f"最佳vs随机胜率: {self.best_win_rate*100:.2f}%")
        self.logger.info("="*70)
        
        # 保存最终模型和绘图
        self._save_checkpoint("final")
        self._plot_training_curves()
        
        # 关闭日志handler
        for handler in self.logger.handlers:
            handler.close()
            self.logger.removeHandler(handler)


def main():
    """主入口"""
    parser = argparse.ArgumentParser(description='AlphaZero强化训练 for ConnectX')
    parser.add_argument('--config', type=str, default='strong',
                       choices=['strong', 'strong+', 'balanced', 'fast', 'ultra'],
                       help='配置预设 (strong=推荐, strong+=强化+可提交, balanced=平衡, fast=快速, ultra=极限)')
    parser.add_argument('--iterations', type=int, default=None,
                       help='训练迭代次数')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='从检查点继续训练')
    parser.add_argument('--no-parallel', action='store_true',
                       help='禁用并行自对弈')
    
    args = parser.parse_args()
    
    # 选择配置
    if args.config == 'balanced':
        config = BalancedConfig()
        print("使用配置: 平衡版")
    elif args.config == 'fast':
        config = FastConfig()
        print("使用配置: 快速版")
    elif args.config == 'ultra':
        config = UltraConfig()
        print("使用配置: 极限强度版 (注意: 模型可能更大)")
    elif args.config == 'strong+':
        config = StrongPlusConfig()
        print("使用配置: 强化+版 (推荐用于Kaggle提交, 模型<100MB)")
    else:
        config = az_config
        print("使用配置: 标准版 (推荐)")
    
    print(f"  MCTS模拟: {config.NUM_SIMULATIONS}")
    print(f"  网络: {config.NUM_RES_BLOCKS} 残差块, {config.NUM_FILTERS} 滤波器")
    print(f"  自对弈游戏/迭代: {config.NUM_SELFPLAY_GAMES}")
    print(f"  最大迭代: {config.MAX_ITERATIONS}")
    
    # 创建训练器
    trainer = AlphaZeroStrongTrainer(
        config=config,
        use_parallel=not args.no_parallel
    )
    
    # 加载检查点
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)
    
    # 开始训练
    trainer.train(num_iterations=args.iterations)


if __name__ == "__main__":
    main()

