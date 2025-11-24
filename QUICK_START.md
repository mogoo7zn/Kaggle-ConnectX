# 快速开始指南

## ⚡ 5分钟上手

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 训练模型
```bash
cd training
python train_dqn.py
```
等待训练完成（约30-60分钟，取决于配置）

### 3. 生成提交文件
```bash
cd ../tools
run_embed.bat  # Windows
# 或
./run_embed.sh  # Linux/Mac
```

### 4. 提交到 Kaggle
- 上传 `submission/main.py` 到 Kaggle ConnectX 竞赛
- 完成！

---

## 🎯 只想提交现有模型？

如果项目已包含训练好的 `best_model.pth`：

```bash
# 1. 生成提交文件
cd tools
run_embed.bat  # 或 ./run_embed.sh

# 2. 上传 submission/main.py
```

---

## 🔍 验证模型

```bash
python
>>> from submission.main import get_agent
>>> agent = get_agent()
>>> print(agent.model_loaded)  # 应显示 True
```

---

## 📚 更多信息

- 完整工作流：见 `WORKFLOW.md`
- 项目结构：见 `README.md`
- 工具说明：见 `tools/README.md`
- 提交指南：见 `submission/README.md`

---

## 🆘 遇到问题？

### 模型未加载
```bash
# 检查文件是否存在
ls submission/best_model.pth
```

### 工具脚本失败
```bash
# 手动执行
cd tools
python embed_model.py ../submission/best_model.pth model_weights_embedded.txt
python create_embedded_main.py
rm model_weights_embedded.txt
```

### 提交失败
- 检查文件大小 <10MB
- 确保 agent 函数存在
- 验证 Python 版本兼容性

---

**就这么简单！🎉**

