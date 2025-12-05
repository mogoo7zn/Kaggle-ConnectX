# 📁 ConnectX 项目结构说明

**版本**: 2.0.0  
**状态**: ✅ 已重组  
**日期**: 2025-11-25

## 🎯 设计原则

1. **模块化**: 每个组件职责清晰，独立可测
2. **可扩展**: 易于添加新的agent实现
3. **标准化**: 遵循Python包管理最佳实践
4. **集中管理**: 输出和文档统一组织

## 📂 完整目录结构

```
connectX/
│
├── 📁 agents/                       # 所有智能体实现
│   ├── __init__.py                  # 包初始化
│   │
│   ├── 📁 base/                     # 共享基础组件
│   │   ├── __init__.py
│   │   ├── config.py                # 基础配置类
│   │   └── utils.py                 # 通用工具函数
│   │
│   ├── 📁 dqn/                      # 基础DQN实现
│   │   ├── __init__.py
│   │   ├── dqn_model.py             # DQN神经网络
│   │   ├── dqn_agent.py             # DQN智能体
│   │   ├── replay_buffer.py         # 经验回放
│   │   └── train_dqn.py             # 训练脚本
│   │
│   ├── 📁 rainbow/                  # Rainbow DQN实现
│   │   ├── __init__.py
│   │   ├── rainbow_config.py        # Rainbow配置
│   │   ├── rainbow_model.py         # Dueling + Noisy Nets
│   │   ├── rainbow_agent.py         # Rainbow智能体
│   │   ├── prioritized_buffer.py    # 优先经验回放
│   │   ├── train_rainbow.py         # 训练脚本
│   │   └── README.md                # Rainbow文档
│   │
│   └── 📁 alphazero/                # AlphaZero实现
│       ├── __init__.py
│       ├── az_config.py             # AlphaZero配置
│       ├── az_model.py              # Policy-Value网络
│       ├── mcts.py                  # MCTS实现
│       ├── self_play.py             # 自我对弈
│       ├── train_alphazero.py       # 训练脚本
│       └── README.md                # AlphaZero文档
│
├── 📁 evaluation/                   # 评估框架
│   ├── __init__.py
│   ├── arena.py                     # 对战竞技场
│   ├── benchmark.py                 # 基准测试
│   └── compare.py                   # 性能对比
│
├── 📁 tools/                        # 工具脚本
│   ├── __init__.py
│   ├── prepare_submission.py        # Kaggle提交准备
│   ├── visualize.py                 # 可视化工具
│   └── README.md
│
├── 📁 outputs/                      # 训练输出（统一管理）
│   ├── __init__.py
│   ├── 📁 checkpoints/             # 训练检查点
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   ├── 📁 logs/                    # 训练日志
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   ├── 📁 models/                  # 最终模型
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   └── 📁 plots/                   # 训练图表
│       ├── dqn/
│       ├── rainbow/
│       └── alphazero/
│
├── 📁 docs/                         # 文档
│   ├── README.md                    # 详细文档
│   ├── QUICKSTART.md                # 快速开始
│   ├── ARCHITECTURE.md              # 架构说明
│   └── REORGANIZATION.md            # 重组计划
│
├── 📁 tests/                        # 测试代码
│   ├── __init__.py
│   ├── test_dqn.py                  # DQN测试
│   ├── test_rainbow.py              # Rainbow测试
│   ├── test_alphazero.py            # AlphaZero测试
│   └── test_evaluation.py           # 评估测试
│
├── 📁 experiments/                  # 实验结果
│   ├── .gitkeep
│   └── README.md
│
├── 📁 submission/                   # Kaggle提交文件
│   ├── dqn_agent.py
│   ├── rainbow_agent.py
│   ├── alphazero_agent.py
│   └── README.md
│
├── 📄 run_experiment.py            # 主实验脚本
├── 📄 cleanup_old_files.py         # 清理脚本
├── 📄 requirements.txt             # 依赖
├── 📄 .gitignore                   # Git忽略文件
├── 📄 LICENSE                      # 许可证
├── 📄 README.md                    # 项目主README
├── 📄 REORGANIZATION_COMPLETE.md   # 重组完成说明
└── 📄 PROJECT_STRUCTURE.md         # 本文件
```

## 🔍 目录说明

### agents/ - 智能体实现

**作用**: 包含所有强化学习智能体的实现

**子目录**:
- `base/`: 共享的基础组件（配置、工具函数）
- `dqn/`: 基础DQN实现（baseline）
- `rainbow/`: Rainbow DQN（6大改进）
- `alphazero/`: AlphaZero（MCTS + 神经网络）

**特点**:
- 每个agent独立目录
- 共享组件在base/
- 易于添加新agent

### evaluation/ - 评估框架

**作用**: 统一的agent评估和对比工具

**组件**:
- `arena.py`: 公平的对战平台
- `benchmark.py`: 标准化性能测试
- `compare.py`: 多agent对比分析

**特点**:
- Agent无关的评估接口
- 标准化的性能指标
- 自动生成对比报告

### tools/ - 工具脚本

**作用**: 辅助开发和部署的工具

**包含**:
- Kaggle提交准备
- 训练可视化
- 诊断工具

### outputs/ - 训练输出

**作用**: 统一管理所有训练产生的文件

**结构**: 按agent类型和输出类型组织
- `checkpoints/`: 训练检查点
- `logs/`: TensorBoard日志
- `models/`: 最终训练模型
- `plots/`: 训练曲线图表

**优势**:
- 集中管理
- 易于清理
- 便于备份

### docs/ - 文档

**作用**: 集中管理所有项目文档

**包含**:
- 用户指南
- API文档
- 架构说明
- 开发文档

### tests/ - 测试

**作用**: 单元测试和集成测试

**组织**: 按模块组织测试文件

## 🚀 使用方法

### 训练Agent

```bash
# 使用模块方式
python -m agents.rainbow.train_rainbow
python -m agents.alphazero.train_alphazero

# 或直接运行
python agents/rainbow/train_rainbow.py
python agents/alphazero/train_alphazero.py
```

### 运行完整实验

```bash
# 快速测试
python run_experiment.py --quick

# 完整训练
python run_experiment.py
```

### 评估性能

```bash
# 基准测试
python -m evaluation.benchmark

# 生成对比报告
python -m evaluation.compare
```

### 准备提交

```bash
python tools/prepare_submission.py \
    --agent rainbow \
    --model-path outputs/models/rainbow/best.pth
```

## 📦 包导入示例

```python
# 导入基础组件
from agents.base.config import config
from agents.base.utils import encode_state, get_valid_moves

# 导入特定agent
from agents.rainbow.rainbow_agent import RainbowAgent
from agents.alphazero.mcts import MCTS

# 导入评估工具
from evaluation.arena import Arena
from evaluation.benchmark import Benchmark
```

## 🔄 添加新Agent

添加新agent的标准流程：

```bash
# 1. 创建目录
mkdir agents/new_agent

# 2. 创建必要文件
touch agents/new_agent/__init__.py
touch agents/new_agent/new_agent_config.py
touch agents/new_agent/new_agent_model.py
touch agents/new_agent/new_agent_agent.py
touch agents/new_agent/train_new_agent.py

# 3. 继承基础组件
# 在代码中: from agents.base import config, utils

# 4. 添加到评估
# 实现标准接口，可直接用evaluation框架评估
```

## 🛠️ 维护指南

### 清理输出

```bash
# 清理所有训练输出
rm -rf outputs/checkpoints/*
rm -rf outputs/logs/*
rm -rf outputs/plots/*

# 保留最新模型
# outputs/models/ 建议手动管理
```

### 备份重要文件

```bash
# 备份检查点
cp -r outputs/checkpoints/ backup/checkpoints_$(date +%Y%m%d)/

# 备份最佳模型
cp -r outputs/models/ backup/models_$(date +%Y%m%d)/
```

### 版本管理

```bash
# 仅跟踪源代码，忽略输出
git add agents/ evaluation/ tools/ docs/
git add run_experiment.py README.md requirements.txt

# outputs/ 应该在 .gitignore 中
```

## 📊 文件统计

- **Python文件**: ~35个
- **配置文件**: 3个
- **文档文件**: 8个
- **测试文件**: 4个 (待完善)
- **总代码行数**: ~7,500行

## ✅ 质量检查

### 代码规范

```bash
# 使用 black 格式化
black agents/ evaluation/ tools/

# 使用 flake8 检查
flake8 agents/ evaluation/ tools/

# 使用 mypy 类型检查
mypy agents/
```

### 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/test_rainbow.py

# 生成覆盖率报告
pytest --cov=agents tests/
```

## 🎯 最佳实践

1. **模块化开发**: 每个组件独立开发和测试
2. **文档先行**: 先写文档，再写代码
3. **测试驱动**: 关键功能都有测试覆盖
4. **版本控制**: 使用语义化版本号
5. **持续集成**: 自动化测试和部署

## 📚 相关文档

- [README.md](README.md) - 项目概览
- [docs/QUICKSTART.md](docs/QUICKSTART.md) - 快速开始
- [docs/README.md](docs/README.md) - 详细文档
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - 架构设计
- [REORGANIZATION_COMPLETE.md](REORGANIZATION_COMPLETE.md) - 重组说明

---

**清晰的结构 = 高效的开发**

*最后更新: 2025-11-25*

