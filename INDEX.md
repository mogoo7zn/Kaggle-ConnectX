# ConnectX DQN Agent - 文档索引

欢迎！这是项目的文档导航中心。

---

## 📖 文档导航

### 🚀 开始使用

| 文档 | 说明 | 适合人群 |
|------|------|----------|
| **[QUICK_START.md](QUICK_START.md)** | 5分钟快速上手 | 新手、急用 |
| **[README.md](README.md)** | 项目完整说明 | 所有人 |
| **[WORKFLOW.md](WORKFLOW.md)** | 详细工作流程 | 进阶用户 |

### 📋 项目信息

| 文档 | 说明 |
|------|------|
| **[PROJECT_STATUS.md](PROJECT_STATUS.md)** | 当前项目状态 |
| **[requirements.txt](requirements.txt)** | Python 依赖 |
| **[.gitignore](.gitignore)** | Git 配置 |

### 📁 模块文档

| 位置 | 文档 | 说明 |
|------|------|------|
| `core/` | - | 核心模块（模型、Agent、配置）|
| `training/` | - | 训练脚本 |
| `submission/` | **[README.md](submission/README.md)** | 提交指南 |
| `tools/` | **[README.md](tools/README.md)** | 工具说明 |

---

## 🎯 按需求查找

### 我想...

#### 快速开始训练和提交
→ 看 [QUICK_START.md](QUICK_START.md)

#### 了解完整项目
→ 看 [README.md](README.md)

#### 学习详细工作流
→ 看 [WORKFLOW.md](WORKFLOW.md)

#### 了解项目状态
→ 看 [PROJECT_STATUS.md](PROJECT_STATUS.md)

#### 提交到 Kaggle
→ 看 [submission/README.md](submission/README.md)

#### 使用工具脚本
→ 看 [tools/README.md](tools/README.md)

---

## 📂 项目结构速览

```
connectX/
│
├── 📚 文档中心
│   ├── INDEX.md          ← 你在这里
│   ├── README.md         ← 项目说明
│   ├── QUICK_START.md    ← 快速开始
│   ├── WORKFLOW.md       ← 工作流程
│   └── PROJECT_STATUS.md ← 项目状态
│
├── 🧠 核心模块 (core/)
│   ├── config.py         ← 配置
│   ├── dqn_model.py      ← 模型
│   ├── dqn_agent.py      ← Agent
│   ├── replay_buffer.py  ← 回放缓冲
│   └── utils.py          ← 工具函数
│
├── 🎓 训练模块 (training/)
│   ├── train_dqn.py      ← 主训练脚本
│   ├── test_setup.py     ← 环境测试
│   └── visualize.py      ← 可视化
│
├── 📤 提交文件 (submission/)
│   ├── main.py           ← 嵌入模型版本 ⭐
│   ├── main_backup.py    ← 原始版本
│   ├── best_model.pth    ← 模型权重
│   └── README.md         ← 提交指南
│
├── 🔧 工具脚本 (tools/)
│   ├── embed_model.py    ← 模型编码
│   ├── create_embedded_main.py ← 生成嵌入版本
│   ├── run_embed.bat     ← Windows 脚本
│   ├── run_embed.sh      ← Linux/Mac 脚本
│   └── README.md         ← 工具说明
│
└── 📦 存档 (archive/)
    ├── checkpoints/      ← 历史检查点
    ├── logs/             ← 训练日志
    └── plots/            ← 训练图表
```

---

## ⚡ 常用命令速查

### 安装依赖
```bash
pip install -r requirements.txt
```

### 训练模型
```bash
cd training
python train_dqn.py
```

### 生成提交文件
```bash
cd tools
run_embed.bat      # Windows
./run_embed.sh     # Linux/Mac
```

### 提交到 Kaggle
```bash
kaggle competitions submit -c connectx -f submission/main.py -m "Version X"
```

---

## 🔗 外部链接

- [Kaggle ConnectX Competition](https://www.kaggle.com/c/connectx)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [DQN Paper](https://arxiv.org/abs/1312.5602)

---

## 🆘 需要帮助？

1. **新手入门** → [QUICK_START.md](QUICK_START.md)
2. **详细流程** → [WORKFLOW.md](WORKFLOW.md)
3. **提交问题** → [submission/README.md](submission/README.md)
4. **工具问题** → [tools/README.md](tools/README.md)

---

## 📊 项目关键信息

| 项目 | 信息 |
|------|------|
| **语言** | Python 3.7+ |
| **框架** | PyTorch |
| **竞赛** | Kaggle ConnectX |
| **模型** | DQN (Deep Q-Network) |
| **策略** | 混合（规则 + DQN） |
| **状态** | ✅ 生产就绪 |

---

**开始你的 ConnectX 之旅！🚀**

建议路径: 
1. [QUICK_START.md](QUICK_START.md) - 快速上手
2. [README.md](README.md) - 了解项目
3. [WORKFLOW.md](WORKFLOW.md) - 深入学习
4. 开始训练和提交！

