# ConnectX DQN Agent

深度 Q 网络 (DQN) 智能体，用于 Kaggle ConnectX 竞赛。

## 项目结构

```
connectX/
├── core/                   # 核心模块
│   ├── config.py          # 配置参数
│   ├── dqn_model.py       # DQN 神经网络模型
│   ├── dqn_agent.py       # DQN 智能体
│   ├── replay_buffer.py   # 经验回放缓冲区
│   └── utils.py           # 工具函数
│
├── training/              # 训练模块
│   ├── train_dqn.py       # 主训练脚本
│   ├── test_setup.py      # 训练环境测试
│   └── visualize.py       # 训练结果可视化
│
├── submission/            # Kaggle 提交文件
│   ├── main.py            # 嵌入模型的提交文件 (9.6 MB)
│   ├── main_backup.py     # 原始版本（需要 .pth 文件）
│   ├── best_model.pth     # 训练好的模型权重
│   └── prepare_submission.py  # 准备提交包
│
├── tools/                 # 工具脚本
│   ├── embed_model.py     # 模型转 Base64
│   ├── create_embedded_main.py  # 生成嵌入版本
│   ├── run_embed.bat      # Windows 一键脚本
│   └── run_embed.sh       # Linux/Mac 一键脚本
│
├── archive/               # 训练历史存档
│   ├── checkpoints/       # 历史检查点
│   ├── logs/              # 训练日志
│   └── plots/             # 训练图表
│
└── requirements.txt       # Python 依赖
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 训练模型

```bash
cd training
python train_dqn.py
```

### 3. 生成提交文件

两种方式：

**方式 A: 使用嵌入模型（推荐）**
```bash
cd tools
# Windows
run_embed.bat

# Linux/Mac
chmod +x run_embed.sh
./run_embed.sh
```
生成的 `submission/main.py` 包含完整模型，可直接提交。

~~**方式 B: 使用外部模型文件**~~
```bash
cd submission
python prepare_submission.py
```
生成的 `submission.zip` 包含 `main_backup.py` 和 `best_model.pth`。

### 4. 提交到 Kaggle

**方式 A（嵌入模型）：**
上传 `submission/main.py`

~~**方式 B（打包提交）：**~~
上传 `submission/submission.zip`

## 技术特点

### DQN 模型架构
- 3 层 CNN (64, 128, 128 通道)
- Batch Normalization
- Dropout (0.3)
- 全连接层 (256 → 128 → 7)

### 混合策略
1. 优先取胜
2. 阻止对手获胜
3. 阻止对手威胁 (3连)
4. DQN Q值决策
5. 中心优先回退

### 状态编码
- 3 通道输入 (6x7):
  - 玩家棋子位置
  - 对手棋子位置
  - 有效移动掩码

## 训练配置

详见 `core/config.py`:
- Episodes: 5000
- Batch size: 64
- Learning rate: 0.0001
- Gamma: 0.99
- Epsilon decay: 0.995

## 环境要求

- Python 3.7+
- PyTorch 1.10+
- NumPy
- Kaggle Environments

完整列表见 `requirements.txt`

## 模型嵌入说明

`main.py` 包含 Base64 编码的模型权重 (~9.6 MB)，无需外部文件。
如需重新生成，使用 `tools/` 下的脚本。

## 许可证

MIT License
