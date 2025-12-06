# Kaggle 提交指南

本目录包含准备将您的智能体提交到 Kaggle ConnectX 竞赛所需的工具和文件。

## ⚠️ 重要提示

Kaggle ConnectX 竞赛要求提交 **单个 python 文件**。由于我们的智能体（特别是 AlphaZero 和 Rainbow DQN）依赖于训练好的模型权重（`.pth` 文件），我们不能简单地提交代码。

**我们必须将模型权重直接嵌入到提交脚本中。**

## 🛠️ 准备工具

我们提供了一个脚本 `prepare_submission.py` 来自动处理此过程。它执行以下步骤：

1.  **读取** 您的训练模型文件 (`.pth`)。
2.  **压缩** 模型权重，使用 `gzip`。
3.  **编码** 压缩后的数据为 Base64 字符串。
4.  **嵌入** 此字符串到模板脚本中。
5.  **生成** 一个最终的 `main.py`，它在运行时解码并加载模型。

## 🚀 如何创建提交

### 1. 确定您的最佳模型

找到您想要提交的 `.pth` 文件。例如：`submission/alpha-zero-v1.pth`。

### 2. 运行准备脚本

从 `submission` 目录运行以下命令：

```bash
python prepare_submission.py --model <path_to_your_model> --output main.py
```

**示例:**

```bash
python prepare_submission.py --model alpha-zero-v1.pth --output main.py
```

### 3. 提交到 Kaggle

将生成的 `main.py` 文件上传到 Kaggle 竞赛页面。

## 📂 文件结构

- `prepare_submission.py`: 嵌入模型并生成提交文件的脚本。
- `main.py`: 生成的提交文件（请勿手动编辑）。
- `*.pth`: 您训练好的模型检查点。

## 🔍 验证

在提交之前，您可以通过运行或在 playground 中导入生成的 `main.py` 来在本地验证其是否工作。

```bash
# 测试是否无错误运行
python main.py
```
