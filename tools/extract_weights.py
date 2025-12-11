"""
从AlphaZero checkpoint中提取纯权重文件
用于将完整checkpoint转换为可提交的纯权重文件

使用方法:
    python extract_weights.py --input checkpoint.pth --output weights_only.pth
    python extract_weights.py --input checkpoint.pth  # 自动命名为 checkpoint_weights.pth
"""

import torch
import argparse
import os
import sys
import io
import gzip
import base64


def extract_weights(input_path: str, output_path: str = None) -> dict:
    """
    从checkpoint提取纯权重
    
    Args:
        input_path: 输入的checkpoint路径
        output_path: 输出的纯权重路径 (可选)
        
    Returns:
        包含大小信息的字典
    """
    print(f"读取: {input_path}")
    
    # 获取原始文件大小
    original_size = os.path.getsize(input_path)
    print(f"  原始大小: {original_size / 1024 / 1024:.2f} MB")
    
    # 加载checkpoint
    checkpoint = torch.load(input_path, map_location='cpu', weights_only=False)
    
    # 提取权重
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            print("  检测到完整checkpoint，提取 model_state_dict...")
            state_dict = checkpoint['model_state_dict']
            
            # 显示checkpoint包含的其他内容
            other_keys = [k for k in checkpoint.keys() if k != 'model_state_dict']
            if other_keys:
                print(f"  移除的字段: {other_keys}")
        else:
            print("  假设已经是纯权重...")
            state_dict = checkpoint
    else:
        print("  非字典格式，保持原样...")
        state_dict = checkpoint
    
    # 计算参数数量
    total_params = sum(v.numel() for v in state_dict.values())
    print(f"  参数总数: {total_params:,}")
    
    # 保存纯权重并测量大小
    if output_path is None:
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_weights{ext}"
    
    torch.save(state_dict, output_path)
    weights_size = os.path.getsize(output_path)
    
    print(f"\n保存: {output_path}")
    print(f"  纯权重大小: {weights_size / 1024 / 1024:.2f} MB")
    print(f"  节省空间: {(original_size - weights_size) / 1024 / 1024:.2f} MB "
          f"({(1 - weights_size/original_size) * 100:.1f}%)")
    
    # 估算提交文件大小
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    model_bytes = buffer.getvalue()
    
    compressed = gzip.compress(model_bytes, compresslevel=9)
    encoded = base64.b64encode(compressed)
    
    submission_estimate = len(encoded) + 20 * 1024  # 20KB代码开销
    
    print(f"\n提交文件估算:")
    print(f"  Gzip压缩后: {len(compressed) / 1024 / 1024:.2f} MB")
    print(f"  Base64编码后: {len(encoded) / 1024 / 1024:.2f} MB")
    print(f"  预估提交大小: {submission_estimate / 1024 / 1024:.2f} MB")
    
    can_submit = submission_estimate < 100 * 1024 * 1024
    print(f"  Kaggle可提交(<100MB): {'✓ 是' if can_submit else '✗ 否'}")
    
    return {
        'original_size': original_size,
        'weights_size': weights_size,
        'compressed_size': len(compressed),
        'encoded_size': len(encoded),
        'submission_estimate': submission_estimate,
        'can_submit': can_submit,
        'total_params': total_params,
        'output_path': output_path
    }


def main():
    parser = argparse.ArgumentParser(
        description='从AlphaZero checkpoint提取纯权重文件'
    )
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='输入的checkpoint文件路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出的纯权重文件路径 (可选)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在: {args.input}")
        sys.exit(1)
    
    extract_weights(args.input, args.output)


if __name__ == "__main__":
    main()
