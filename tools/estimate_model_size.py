"""
估算AlphaZero模型大小的脚本
"""

def estimate_model_size(num_res_blocks, num_filters, policy_filters=32, value_filters=32, value_hidden=128):
    """估算PyTorch模型文件大小 (MB)"""
    params = 0
    
    # 初始卷积层 (3层)
    params += 3 * num_filters * 3 * 3  # conv1
    params += num_filters * 2  # bn1 (weight + bias)
    params += num_filters * num_filters * 3 * 3  # conv2
    params += num_filters * 2  # bn2
    params += num_filters * num_filters * 3 * 3  # conv3
    params += num_filters * 2  # bn3
    
    # 残差块 (每块2个卷积 + 2个BN)
    for _ in range(num_res_blocks):
        params += num_filters * num_filters * 3 * 3 * 2  # 2个卷积
        params += num_filters * 2 * 2  # 2个BN (各有weight和bias)
    
    # 策略头
    params += num_filters * policy_filters * 1 * 1  # policy_conv
    params += policy_filters * 2  # policy_bn
    params += policy_filters * 6 * 7 * 7 + 7  # policy_fc (weights + bias)
    
    # 价值头
    params += num_filters * value_filters * 1 * 1  # value_conv
    params += value_filters * 2  # value_bn
    params += value_filters * 6 * 7 * value_hidden + value_hidden  # value_fc1
    params += value_hidden * 1 + 1  # value_fc2
    
    # 每个参数4字节(float32)，转换为MB
    size_mb = params * 4 / (1024 * 1024)
    return params, size_mb


if __name__ == "__main__":
    configs = [
        ('Fast', 6, 96, 32, 32, 128),
        ('Balanced', 6, 128, 32, 32, 128),
        ('Strong', 8, 128, 32, 32, 128),
        ('Strong+', 10, 160, 48, 48, 192),
        ('Ultra', 12, 256, 32, 32, 128),
    ]
    
    print('=' * 80)
    print('AlphaZero 模型大小估算对比')
    print('=' * 80)
    print(f"{'配置':<12} {'残差块':<8} {'滤波器':<8} {'参数量':<18} {'估算大小':<12} {'可提交(<100MB)'}")
    print('-' * 80)
    
    for name, res_blocks, filters, p_f, v_f, v_h in configs:
        params, size_mb = estimate_model_size(res_blocks, filters, p_f, v_f, v_h)
        submittable = '✓ 可以' if size_mb < 100 else '✗ 太大'
        print(f'{name:<12} {res_blocks:<8} {filters:<8} {params:>15,}   {size_mb:>8.2f} MB   {submittable}')
    
    print('=' * 80)
    print()
    print("说明:")
    print("  - Strong+配置 是专门为Kaggle提交设计的，模型<100MB")
    print("  - 通过更多MCTS模拟(600次)和更长训练(600迭代)弥补网络规模限制")
    print("  - Ultra配置模型太大(>100MB)，无法直接提交到Kaggle")
