# 工具脚本

用于生成包含嵌入模型的 main.py 文件。

## 使用方法

### 1. 转换模型为 Base64 格式

```bash
cd tools
python embed_model.py ../submission/best_model.pth model_weights_embedded.txt
```

### 2. 生成包含嵌入模型的 main.py

```bash
python create_embedded_main.py --original ../submission/main_backup.py --model model_weights_embedded.txt --output ../submission/main.py
```

### 3. 一键执行（Windows）

```bash
run_embed.bat
```

### 4. 一键执行（Linux/Mac）

```bash
./run_embed.sh
```

## 文件说明

- `embed_model.py` - 将 .pth 模型转换为 Base64 字符串
- `create_embedded_main.py` - 创建包含嵌入模型的 main.py
- `run_embed.bat` - Windows 批处理脚本（一键执行）
- `run_embed.sh` - Linux/Mac shell 脚本（一键执行）

## 注意事项

1. 确保 `main_backup.py` 是原始的未嵌入版本
2. 生成的 `main.py` 文件约 9-10 MB
3. 过程需要几秒钟，请耐心等待
