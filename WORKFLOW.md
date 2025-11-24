# ConnectX 项目工作流

## 📋 项目概览

这是一个用于 Kaggle ConnectX 竞赛的 DQN 智能体项目，包含完整的训练、测试和提交流程。

---

## 🚀 完整工作流

### 1. 环境设置

```bash
# 克隆/进入项目目录
cd connectX

# 安装依赖
pip install -r requirements.txt
```

### 2. 训练模型

```bash
cd training
python train_dqn.py
```

**训练参数配置：** 修改 `core/config.py`

**输出：**
- 模型保存到 `submission/best_model.pth`
- 检查点保存到 `archive/checkpoints/`
- 训练日志保存到 `archive/logs/`

**可视化训练结果：**
```bash
python visualize.py
```

### 3. 测试模型

```bash
# 测试训练环境
python test_setup.py

# 本地测试智能体
cd ..
python -c "from submission.main_backup import agent; print('Agent loaded successfully')"
```

### 4. 生成提交文件

#### 选项 A: 嵌入式模型（推荐）

```bash
cd tools

# Windows
run_embed.bat

# Linux/Mac
chmod +x run_embed.sh
./run_embed.sh
```

**生成：** `submission/main.py` (9.6 MB，包含完整模型)

#### 选项 B: 打包提交

```bash
cd submission
python prepare_submission.py
```

**生成：** `submission.zip` (包含 main.py + best_model.pth)

### 5. 提交到 Kaggle

#### 通过 Web 界面

1. 访问 [Kaggle ConnectX](https://www.kaggle.com/c/connectx)
2. 点击 "Submit Prediction"
3. 上传 `main.py` 或 `submission.zip`
4. 提交

#### 通过 API

```bash
# 提交嵌入式版本
kaggle competitions submit -c connectx -f submission/main.py -m "DQN v1.0 embedded"

# 提交打包版本
kaggle competitions submit -c connectx -f submission/submission.zip -m "DQN v1.0"
```

---

## 🔄 迭代优化工作流

### 修改模型架构

1. 编辑 `core/dqn_model.py`
2. 更新 `core/config.py` 中的超参数
3. 重新训练模型
4. 测试性能
5. 如果性能提升，重新生成提交文件

### 修改策略规则

1. 编辑 `submission/main_backup.py` 中的规则逻辑
2. 本地测试修改后的智能体
3. 如果效果好，重新嵌入模型：
   ```bash
   cd tools
   run_embed.bat  # 或 ./run_embed.sh
   ```

### 调整训练参数

主要参数在 `core/config.py`:

```python
# 训练轮数
EPISODES = 5000

# 学习率
LEARNING_RATE = 0.0001

# 探索率衰减
EPSILON_DECAY = 0.995

# 批大小
BATCH_SIZE = 64

# 折扣因子
GAMMA = 0.99
```

---

## 📁 关键文件说明

### 训练相关
- `training/train_dqn.py` - 主训练脚本
- `core/config.py` - 配置参数
- `core/dqn_model.py` - 模型架构
- `core/dqn_agent.py` - 训练智能体

### 提交相关
- `submission/main.py` - 嵌入模型版本（提交用）
- `submission/main_backup.py` - 原始版本（开发用）
- `submission/best_model.pth` - 模型权重

### 工具脚本
- `tools/embed_model.py` - 模型→Base64
- `tools/create_embedded_main.py` - 生成嵌入版本
- `tools/run_embed.bat/sh` - 一键生成脚本

---

## 🎯 最佳实践

### 训练
1. 从小规模测试开始（100-500 episodes）
2. 验证训练流程正常
3. 增加到完整训练轮数（5000+）
4. 定期保存检查点
5. 可视化训练曲线

### 测试
1. 本地测试基本功能
2. 使用 Kaggle 环境验证
3. 与基准智能体对战
4. 检查边界情况

### 提交
1. 使用嵌入式版本（更可靠）
2. 添加有意义的提交信息
3. 记录版本号和修改内容
4. 监控提交结果
5. 对比不同版本性能

---

## 🐛 常见问题

### 训练问题

**Q: 训练很慢？**
- 减少 EPISODES
- 增加 BATCH_SIZE
- 使用 GPU（修改 config.py 中的 DEVICE）

**Q: 模型不收敛？**
- 调整 LEARNING_RATE
- 修改 EPSILON_DECAY
- 检查奖励函数

### 提交问题

**Q: 文件太大？**
- 使用 main.py（已优化）
- 检查是否有重复文件

**Q: 模型加载失败？**
- 验证 PyTorch 版本兼容性
- 确保文件编码为 UTF-8
- 使用 `weights_only=True` 参数

**Q: 超时？**
- 优化推理代码
- 减少不必要的计算
- 使用规则优先策略

---

## 📊 性能监控

### 训练指标
- Episode Reward (趋势)
- Win Rate (目标 >70%)
- Loss (应逐渐下降)
- Epsilon (探索率衰减)

### 竞赛指标
- Public Leaderboard Score
- Private Leaderboard Score
- 对战胜率
- ELO Rating

---

## 🔧 高级功能

### 自定义对手
修改 `training/train_dqn.py` 中的对手智能体：

```python
from kaggle_environments import make
env = make("connectx", debug=True)
opponent = env.run([agent, "random"])[0]
```

### 多模型集成
训练多个模型，在提交时使用投票机制。

### 超参数搜索
使用 `optuna` 或网格搜索优化超参数。

---

## 📝 版本管理建议

```
v1.0 - 基础 DQN 模型
v1.1 - 添加威胁检测
v1.2 - 优化奖励函数
v2.0 - 改进网络架构
...
```

每次重要修改：
1. 更新版本号
2. 记录改动
3. 保存检查点到 archive/
4. 提交并记录成绩

---

## 🎓 学习资源

- [Kaggle ConnectX](https://www.kaggle.com/c/connectx)
- [DQN Paper](https://arxiv.org/abs/1312.5602)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Reinforcement Learning Book](http://incompleteideas.net/book/the-book.html)

---

**Happy Coding! 🚀**

