# Kaggle 提交文件

## 文件说明

### 提交文件
- **main.py** (9.6 MB) - 包含嵌入模型的完整智能体，可直接提交
- **main_backup.py** (14 KB) - 原始版本，需配合 `best_model.pth` 使用
- **best_model.pth** (6.5 MB) - 训练好的模型权重

### 工具脚本
- **prepare_submission.py** - 创建 submission.zip 打包文件

## 提交方式

### 方式 1: 直接提交 main.py（推荐）

```bash
# 上传 main.py 到 Kaggle
```

优点：
- 单文件，简单可靠
- 无需配置路径
- 包含完整模型

### 方式 2: 打包提交

```bash
python prepare_submission.py
# 上传生成的 submission.zip
```

包含：
- main_backup.py (重命名为 main.py)
- best_model.pth

## Kaggle API 提交

```bash
# 方式 1: 提交 main.py
kaggle competitions submit -c connectx -f main.py -m "DQN with embedded model"

# 方式 2: 提交 zip
kaggle competitions submit -c connectx -f submission.zip -m "DQN agent"
```

## 本地测试

```python
from main import agent

class Observation:
    board = [0] * 42
    mark = 1

class Configuration:
    columns = 7

action = agent(Observation(), Configuration())
print(f"Selected action: {action}")
```

## 文件对比

| 文件 | 大小 | 依赖 | 用途 |
|------|------|------|------|
| main.py | 9.6 MB | 无 | Kaggle 直接提交 |
| main_backup.py | 14 KB | best_model.pth | 本地开发/调试 |
| best_model.pth | 6.5 MB | - | 模型权重文件 |

## 重新生成 main.py

如果需要重新嵌入模型：

```bash
cd ../tools
# Windows
run_embed.bat

# Linux/Mac
./run_embed.sh
```

## 注意事项

1. `main.py` 包含完整模型，首次加载需要几秒钟
2. Kaggle 环境自动识别 `agent` 函数作为入口点
3. 确保文件编码为 UTF-8
4. 提交前可使用 Kaggle 本地环境测试

## 提交检查清单

- [ ] 文件可正常导入
- [ ] agent 函数存在
- [ ] 模型能成功加载
- [ ] 在有效列中返回动作
- [ ] 文件大小在限制内 (<10 MB)

