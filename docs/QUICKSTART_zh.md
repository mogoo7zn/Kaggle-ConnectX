# 🚀 Quick Start Guide

快速上手 ConnectX 双智能体项目

## ⚡ 5分钟快速测试

```bash
# 1. 使用自动化脚本设置环境（推荐）
# Windows:
scripts\setup_env.bat

# Linux/Mac:
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh

# 或手动安装依赖
pip install -r requirements.txt

# 2. 激活虚拟环境（如果使用自动化脚本）
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 3. 快速测试（约5-10分钟）
python run_full_experiment.py --quick

# 4. 查看结果
ls experiments/comparison_*/comparison_report.html
```

## 📖 详细步骤

### 步骤1：环境准备

#### 方法A：自动化设置（推荐）

**Windows:**
```bash
# 运行自动化脚本
scripts\setup_env.bat
```

**Linux/Mac:**
```bash
# 添加执行权限并运行
chmod +x scripts/setup_env.sh
./scripts/setup_env.sh
```

脚本会自动：
- 检查 Python 版本（需要 3.8+）
- 创建虚拟环境 `venv/`
- 安装所有依赖（包括 PyTorch, NumPy, Matplotlib, Pygame, TensorBoard 等）

#### 方法B：手动设置

```bash
# 检查Python版本 (需要3.8+)
python --version

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows:
venv\Scripts\activate.bat
# Linux/Mac:
source venv/bin/activate

# 安装所有依赖
pip install -r requirements.txt

# 可选：CUDA支持（GPU加速）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 步骤2：选择训练方案

#### 方案A：快速测试模式（推荐新手）
```bash
# 训练时间：约10-20分钟
# 目的：验证代码正常工作
python run_full_experiment.py --quick
```

#### 方案B：仅训练Rainbow DQN
```bash
# 训练时间：数小时到1天
cd rainbow
python train_rainbow.py
```

#### 方案C：仅训练AlphaZero
```bash
# 训练时间：1-2天
cd alphazero
python train_alphazero.py
```

#### 方案D：完整训练（需要GPU）
```bash
# 训练时间：3-7天
python run_full_experiment.py
```

### 步骤3：监控训练进度

```bash
# 在新终端中启动TensorBoard
tensorboard --logdir rainbow/logs/runs --logdir alphazero/logs/runs

# 访问 http://localhost:6006
```

关注以下指标：
- **Loss**: 应该下降
- **Win Rate**: 应该上升
- **Q Values**: 应该趋于稳定
- **ELO Rating**: 应该增长

### 步骤4：评估模型

```bash
# 运行基准测试
python -m evaluation.benchmark

# 或使用Python脚本
python << EOF
from evaluation.benchmark import Benchmark
from agents.rainbow.rainbow_agent import RainbowAgent
from evaluation.arena import create_agent_wrapper

# 加载训练好的模型
agent = RainbowAgent()
agent.load_model('rainbow/checkpoints/best_rainbow_*.pth')

# 运行基准测试
benchmark = Benchmark()
results = benchmark.run_benchmark(
    create_agent_wrapper(agent, 'rainbow'),
    agent_name="My Rainbow Agent",
    games_per_opponent=50
)
EOF
```

### 步骤5：准备Kaggle提交

```bash
# Rainbow DQN提交
python tools/prepare_kaggle_submission.py \
    --agent rainbow \
    --model-path rainbow/checkpoints/best_rainbow_full_20251125_*.pth \
    --output submission/my_rainbow_agent.py

# AlphaZero提交  
python tools/prepare_kaggle_submission.py \
    --agent alphazero \
    --model-path alphazero/checkpoints/best_alphazero_20251125_*.pth \
    --output submission/my_alphazero_agent.py \
    --mcts-sims 100
```

### 步骤6：本地测试提交文件

```python
# 测试Rainbow agent
from submission.my_rainbow_agent import agent

# 模拟Kaggle observation
class Obs:
    def __init__(self):
        self.board = [0] * 42
        self.mark = 1

obs = Obs()
action = agent(obs, None)
print(f"Agent selected action: {action}")
```

## 🎮 交互式对弈（可选）

```python
from evaluation.arena import Arena
from agents.rainbow.rainbow_agent import RainbowAgent
from evaluation.benchmark import StandardOpponents

# 加载你的agent
my_agent = RainbowAgent()
my_agent.load_model('rainbow/checkpoints/best_rainbow.pth')

# 创建对战场
arena = Arena()

# 对战测试
results = arena.play_match(
    agent1_fn=lambda b,m: my_agent.select_action(b, m, epsilon=0),
    agent2_fn=StandardOpponents.negamax_depth_4,
    num_games=10,
    agent1_name="My Agent",
    agent2_name="Negamax-4",
    verbose=True
)
```

## 📊 查看结果

### 1. TensorBoard可视化
```bash
tensorboard --logdir experiments/
```

### 2. HTML报告
打开浏览器访问:
```
experiments/comparison_*/comparison_report.html
```

### 3. JSON数据
```python
import json

with open('experiments/rainbow_benchmark.json') as f:
    data = json.load(f)
    
print(f"Overall win rate: {data['overall']['overall_win_rate']:.1%}")
print(f"Estimated ELO: {data['overall']['estimated_elo']:.0f}")
```

## 🔧 常见问题

### Q: 训练很慢怎么办？
**A**: 几个解决方案：
```bash
# 1. 使用快速模式
python run_full_experiment.py --quick

# 2. 减少训练轮数
# 编辑 rainbow/rainbow_config.py
SELF_PLAY_EPISODES = 1000  # 默认8000

# 3. 减少MCTS模拟次数
# 编辑 alphazero/az_config.py
NUM_SIMULATIONS = 200  # 默认800
```

### Q: 内存不足？
**A**: 减小buffer大小：
```python
# rainbow/rainbow_config.py
REPLAY_BUFFER_SIZE = 100000  # 默认500000
BATCH_SIZE = 128  # 默认256

# alphazero/az_config.py
REPLAY_BUFFER_SIZE = 200000  # 默认500000
BATCH_SIZE = 256  # 默认512
```

### Q: 如何使用GPU？
**A**: PyTorch会自动检测GPU：
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
```

### Q: 如何加载预训练模型？
**A**: 
```python
from agents.rainbow.rainbow_agent import RainbowAgent

agent = RainbowAgent()
agent.load_model('path/to/model.pth')
# 或
agent.load_checkpoint('path/to/checkpoint.pth')
```

### Q: 训练中断了如何恢复？
**A**: 
```python
# Rainbow
from agents.rainbow.rainbow_agent import RainbowAgent
from rainbow.train_rainbow import RainbowTrainer

agent = RainbowAgent()
agent.load_checkpoint('rainbow/checkpoints/rainbow_ep5000.pth')

trainer = RainbowTrainer(agent)
trainer.train(num_episodes=3000)  # 继续训练

# AlphaZero
from alphazero.train_alphazero import AlphaZeroTrainer

trainer = AlphaZeroTrainer()
trainer.load_checkpoint('alphazero/checkpoints/alphazero_iter50.pth')
trainer.train(max_iterations=50)  # 继续训练
```

## 📝 下一步

完成快速开始后，你可以：

1. **调优超参数**
   - 修改 `rainbow/rainbow_config.py`
   - 修改 `alphazero/az_config.py`

2. **实验不同架构**
   - 尝试更深的网络
   - 调整ResBlock数量
   - 测试Distributional RL

3. **添加新对手**
   - 实现自定义策略
   - 添加到benchmark suite

4. **优化性能**
   - 使用模型量化
   - 实现批处理推理
   - 多GPU并行训练

5. **提交到Kaggle**
   - 准备submission文件
   - 本地测试
   - 上传并评估

## 🎯 推荐学习路径

### 初学者
1. 运行 `--quick` 模式理解流程
2. 阅读 `DUAL_AGENT_README.md`
3. 研究 `rainbow/rainbow_agent.py` 代码
4. 尝试修改简单参数重新训练

### 中级用户
1. 完整训练Rainbow DQN
2. 分析TensorBoard日志
3. 实现自定义评估指标
4. 优化超参数

### 高级用户
1. 完整训练两个agent
2. 实现分布式训练
3. 添加新的RL算法
4. 参与Kaggle竞赛

## 🆘 获取帮助

- 📖 完整文档: `DUAL_AGENT_README.md`
- 💡 实现细节: `IMPLEMENTATION_SUMMARY.md`
- 🐛 问题报告: GitHub Issues
- 💬 讨论: GitHub Discussions

---

如有问题随时查阅文档或提issue!

