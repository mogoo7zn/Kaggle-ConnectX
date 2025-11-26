# 项目重组计划

## 📋 当前问题分析

1. **目录重复**: `archive/` 和 `training/` 存储相同类型的内容
2. **文档重复**: `QUICK_START.md` 和 `QUICKSTART.md` 重复
3. **混合模式**: 旧DQN代码（`core/`, `training/`）和新代码（`rainbow/`, `alphazero/`）混在一起
4. **结构不清晰**: 文档、代码、数据混在根目录

## 🎯 新的清晰结构

```
connectX/
│
├── 📁 agents/                    # 所有智能体实现
│   ├── base/                     # 基础组件（共享）
│   │   ├── __init__.py
│   │   ├── config.py            # 通用配置
│   │   ├── utils.py             # 通用工具函数
│   │   └── environment.py       # 环境相关
│   │
│   ├── dqn/                      # 基础DQN（baseline）
│   │   ├── __init__.py
│   │   ├── dqn_config.py
│   │   ├── dqn_model.py
│   │   ├── dqn_agent.py
│   │   ├── replay_buffer.py
│   │   ├── train_dqn.py
│   │   └── README.md
│   │
│   ├── rainbow/                  # Rainbow DQN
│   │   ├── __init__.py
│   │   ├── rainbow_config.py
│   │   ├── rainbow_model.py
│   │   ├── rainbow_agent.py
│   │   ├── prioritized_buffer.py
│   │   ├── train_rainbow.py
│   │   └── README.md
│   │
│   └── alphazero/                # AlphaZero
│       ├── __init__.py
│       ├── az_config.py
│       ├── az_model.py
│       ├── az_agent.py
│       ├── mcts.py
│       ├── self_play.py
│       ├── train_alphazero.py
│       └── README.md
│
├── 📁 evaluation/                # 评估框架
│   ├── __init__.py
│   ├── arena.py
│   ├── benchmark.py
│   ├── compare.py
│   └── README.md
│
├── 📁 tools/                     # 工具脚本
│   ├── __init__.py
│   ├── prepare_submission.py    # Kaggle提交准备
│   ├── visualize.py             # 可视化工具
│   ├── diagnostics.py           # 诊断工具
│   └── README.md
│
├── 📁 experiments/               # 实验结果
│   ├── .gitkeep
│   └── README.md
│
├── 📁 outputs/                   # 训练输出（统一管理）
│   ├── checkpoints/             # 所有检查点
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   ├── logs/                    # 所有日志
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   ├── models/                  # 最终模型
│   │   ├── dqn/
│   │   ├── rainbow/
│   │   └── alphazero/
│   └── plots/                   # 训练图表
│       ├── dqn/
│       ├── rainbow/
│       └── alphazero/
│
├── 📁 submission/                # Kaggle提交文件
│   ├── dqn_agent.py
│   ├── rainbow_agent.py
│   ├── alphazero_agent.py
│   └── README.md
│
├── 📁 docs/                      # 所有文档
│   ├── README.md                # 主文档
│   ├── QUICKSTART.md            # 快速开始
│   ├── ARCHITECTURE.md          # 架构说明
│   ├── API.md                   # API文档
│   └── CONTRIBUTING.md          # 贡献指南
│
├── 📁 tests/                     # 测试代码
│   ├── __init__.py
│   ├── test_dqn.py
│   ├── test_rainbow.py
│   ├── test_alphazero.py
│   ├── test_evaluation.py
│   └── README.md
│
├── 📄 run_experiment.py         # 主实验脚本
├── 📄 requirements.txt          # 依赖
├── 📄 setup.py                  # 安装配置
├── 📄 .gitignore
├── 📄 LICENSE
└── 📄 README.md                 # 项目主README
```

## 🔄 迁移映射

### 文件迁移表

| 原路径 | 新路径 | 操作 |
|--------|--------|------|
| `core/config.py` | `agents/base/config.py` | 移动 |
| `core/utils.py` | `agents/base/utils.py` | 移动 |
| `core/dqn_*.py` | `agents/dqn/` | 移动 |
| `core/replay_buffer.py` | `agents/dqn/replay_buffer.py` | 移动 |
| `rainbow/*` | `agents/rainbow/` | 移动 |
| `alphazero/*` | `agents/alphazero/` | 移动 |
| `evaluation/*` | `evaluation/` | 保持 |
| `training/` | `outputs/` | 重组 |
| `archive/` | `outputs/` | 合并 |
| `QUICKSTART.md` | `docs/QUICKSTART.md` | 移动 |
| `QUICK_START.md` | 删除（重复） | 删除 |
| `DUAL_AGENT_README.md` | `docs/README.md` | 移动 |
| `IMPLEMENTATION_SUMMARY.md` | `docs/ARCHITECTURE.md` | 重命名+移动 |
| `tools/*` | `tools/` | 清理 |
| `submission/*` | `submission/` | 清理 |

### 目录操作

1. **创建新目录结构**
2. **移动文件到新位置**
3. **更新所有import路径**
4. **删除重复/过时文件**
5. **更新配置文件**
6. **测试所有功能**

## ✅ 实施步骤

### 步骤1: 创建新目录结构
```bash
mkdir -p agents/{base,dqn,rainbow,alphazero}
mkdir -p outputs/{checkpoints,logs,models,plots}/{dqn,rainbow,alphazero}
mkdir -p docs tests submission
```

### 步骤2: 移动文件
```bash
# Base组件
mv core/config.py agents/base/
mv core/utils.py agents/base/

# DQN
mv core/dqn_*.py agents/dqn/
mv core/replay_buffer.py agents/dqn/
mv training/train_dqn.py agents/dqn/

# Rainbow (已存在)
mv rainbow/* agents/rainbow/

# AlphaZero (已存在)
mv alphazero/* agents/alphazero/

# 文档
mv DUAL_AGENT_README.md docs/README.md
mv QUICKSTART.md docs/QUICKSTART.md
mv IMPLEMENTATION_SUMMARY.md docs/ARCHITECTURE.md
```

### 步骤3: 更新Import路径
所有代码中的import需要更新：
```python
# 旧
from agents.base.config import config
from agents.base.utils import encode_state

# 新
from agents.base.config import config
from agents.base.utils import encode_state
```

### 步骤4: 清理
```bash
# 删除空目录
rm -rf core/ training/ archive/

# 删除重复文档
rm QUICK_START.md

# 删除旧文件
rm diagnose.py
```

## 📝 配置更新

### setup.py (新建)
```python
from setuptools import setup, find_packages

setup(
    name="connectx-agents",
    version="2.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "tensorboard>=2.8.0",
    ],
)
```

### .gitignore (更新)
```
# 输出目录
outputs/
!outputs/.gitkeep

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# 环境
venv/
env/
.env

# IDE
.vscode/
.idea/
*.swp

# 实验结果
experiments/*
!experiments/.gitkeep
!experiments/README.md
```

## 🎯 优势

1. **清晰的模块边界**: 每个agent独立在自己的目录
2. **统一的输出管理**: 所有训练输出集中管理
3. **文档集中**: 所有文档在docs/目录
4. **易于扩展**: 添加新agent只需在agents/下创建新目录
5. **符合Python最佳实践**: 可以作为包安装

## ⚠️ 注意事项

1. 确保更新所有import路径
2. 更新README中的路径引用
3. 更新配置文件中的路径
4. 测试所有脚本确保功能正常
5. 保留outputs/目录的训练数据

