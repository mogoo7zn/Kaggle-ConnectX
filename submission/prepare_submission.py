"""
准备Kaggle提交文件的脚本
清理不必要的文件，创建压缩包
"""

import os
import shutil
import zipfile
from pathlib import Path

def clean_submission_folder():
    """清理submission文件夹，移除不需要的文件"""
    submission_dir = Path(__file__).parent
    
    # 需要保留的文件
    keep_files = {
        'submission_agent.py',
        'main.py',
        'best_model.pth',
        '.gitignore',
        'SUBMIT.md',
        'README.md'  # 可选，但保留也无妨
    }
    
    # 需要删除的文件/文件夹
    to_remove = []
    
    for item in submission_dir.iterdir():
        if item.name.startswith('.'):
            continue  # 跳过隐藏文件（除了.gitignore）
        
        if item.is_file():
            if item.name not in keep_files:
                # 检查是否是测试文件
                if 'test' in item.name.lower() or item.suffix == '.pyc':
                    to_remove.append(item)
        elif item.is_dir():
            # 删除__pycache__等缓存目录
            if item.name in ['__pycache__', '.pytest_cache', '.mypy_cache']:
                to_remove.append(item)
    
    # 删除不需要的文件
    for item in to_remove:
        if item.is_file():
            print(f"删除文件: {item.name}")
            item.unlink()
        elif item.is_dir():
            print(f"删除目录: {item.name}")
            shutil.rmtree(item)
    
    print(f"\n清理完成！保留的文件：")
    for item in sorted(submission_dir.iterdir()):
        if item.is_file() and not item.name.startswith('.'):
            size = item.stat().st_size / (1024 * 1024)  # MB
            print(f"  - {item.name} ({size:.2f} MB)")


def create_submission_zip():
    """创建提交用的压缩包"""
    submission_dir = Path(__file__).parent
    
    # 需要包含的文件
    files_to_include = [
        'main.py',
        'best_model.pth'
    ]
    
    zip_path = submission_dir / 'submission.zip'
    
    # 检查文件是否存在
    missing_files = []
    for file in files_to_include:
        if not (submission_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"错误：以下文件不存在：{', '.join(missing_files)}")
        return False
    
    # 创建压缩包
    print(f"\n创建压缩包: {zip_path}")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file in files_to_include:
            file_path = submission_dir / file
            zipf.write(file_path, file)
            print(f"  添加: {file} ({file_path.stat().st_size / (1024*1024):.2f} MB)")
    
    zip_size = zip_path.stat().st_size / (1024 * 1024)
    print(f"\n压缩包创建成功！")
    print(f"文件大小: {zip_size:.2f} MB")
    print(f"位置: {zip_path.absolute()}")
    
    if zip_size > 100:
        print(f"[警告] 压缩包大小超过100MB限制！")
        return False
    
    return True


def verify_submission():
    """验证提交文件是否符合要求"""
    submission_dir = Path(__file__).parent
    
    print("\n验证提交文件...")
    
    # 检查必需文件
    required_files = {
        'best_model.pth': '模型权重文件'
    }
    
    all_ok = True
    for file, desc in required_files.items():
        file_path = submission_dir / file
        if file_path.exists():
            size = file_path.stat().st_size / (1024 * 1024)
            print(f"[OK] {file} ({desc}) - {size:.2f} MB")
        else:
            print(f"[ERROR] {file} ({desc}) - 缺失！")
            all_ok = False
    
    # 检查main.py（压缩包方式需要）
    main_file = submission_dir / 'main.py'
    if main_file.exists():
        content = main_file.read_text(encoding='utf-8')
        
        # 检查agent函数是否存在
        if 'def agent(observation, configuration):' in content:
            print("[OK] main.py存在且包含agent函数")
            
            # 检查agent函数是否是最后一个def
            lines = content.split('\n')
            last_def_line = -1
            agent_def_line = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('def '):
                    last_def_line = i
                    if 'def agent(observation, configuration):' in line:
                        agent_def_line = i
            
            if agent_def_line == last_def_line:
                print("[OK] agent函数是文件中最后一个def定义")
            else:
                print("[ERROR] agent函数不是文件中最后一个def定义！")
                all_ok = False
        else:
            print("[ERROR] main.py中agent函数不存在或格式不正确")
            all_ok = False
    else:
        print("[ERROR] main.py不存在")
        all_ok = False
    
    # 检查submission_agent.py（单文件提交备选）
    agent_file = submission_dir / 'submission_agent.py'
    if agent_file.exists():
        content = agent_file.read_text(encoding='utf-8')
        if 'def agent(observation, configuration):' in content:
            print("[OK] submission_agent.py存在（备选单文件提交方式）")
        else:
            print("[WARN] submission_agent.py缺少agent函数")
    
    # 检查文件大小
    total_size = 0
    for file in ['best_model.pth', 'main.py']:
        file_path = submission_dir / file
        if file_path.exists():
            total_size += file_path.stat().st_size
    
    total_size_mb = total_size / (1024 * 1024)
    print(f"\n总文件大小: {total_size_mb:.2f} MB")
    if total_size_mb > 100:
        print("[ERROR] 警告：总大小超过100MB限制！")
        all_ok = False
    else:
        print("[OK] 文件大小符合要求（<100MB）")
    
    return all_ok


if __name__ == '__main__':
    print("=" * 60)
    print("Kaggle ConnectX 提交文件准备工具")
    print("=" * 60)
    
    # 验证文件
    if not verify_submission():
        print("\n请先修复上述问题！")
        exit(1)
    
    # 清理文件
    print("\n" + "=" * 60)
    clean_submission_folder()
    
    # 创建压缩包
    print("\n" + "=" * 60)
    create_submission_zip()
    
    print("\n" + "=" * 60)
    print("准备完成！")
    print("\n提交方式：")
    print("【推荐】压缩包方式：上传 submission.zip")
    print("  - 包含: main.py + best_model.pth")
    print("  - main.py中agent函数是最后一个def定义")
    print("\n备选：单文件方式：直接上传 submission_agent.py")
    print("  - 注意：需要单独上传模型文件到Kaggle数据集")
    print("\n详细说明请查看 SUBMIT.md")

