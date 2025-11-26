"""
AlphaZero Policy-Value Neural Network
ResNet-style architecture with policy and value heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.alphazero.az_config import az_config


class ResidualBlock(nn.Module):
    """
    Residual block for deep network.
    
    Structure:
    x -> Conv -> BN -> ReLU -> Conv -> BN -> (+x) -> ReLU
    """
    
    def __init__(self, num_filters: int):
        """
        Initialize residual block.
        
        Args:
            num_filters: Number of convolutional filters
        """
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through residual block.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add residual connection
        out += residual
        out = F.relu(out)
        
        return out


class PolicyValueNetwork(nn.Module):
    """
    AlphaZero-style Policy-Value Network.
    
    Architecture:
    - Initial convolutional layer
    - Residual blocks (ResNet backbone)
    - Policy head (outputs action probabilities)
    - Value head (outputs position evaluation)
    """
    
    def __init__(self, input_channels: int = None, num_res_blocks: int = None,
                 num_filters: int = None):
        """
        Initialize Policy-Value Network.
        
        Args:
            input_channels: Number of input channels (default from config)
            num_res_blocks: Number of residual blocks (default from config)
            num_filters: Number of convolutional filters (default from config)
        """
        super(PolicyValueNetwork, self).__init__()
        
        # Use config defaults if not specified
        if input_channels is None:
            input_channels = az_config.INPUT_CHANNELS
        if num_res_blocks is None:
            num_res_blocks = az_config.NUM_RES_BLOCKS
        if num_filters is None:
            num_filters = az_config.NUM_FILTERS
        
        self.input_channels = input_channels
        self.num_res_blocks = num_res_blocks
        self.num_filters = num_filters
        
        # Initial convolution
        self.conv_input = nn.Conv2d(
            input_channels, num_filters,
            kernel_size=3, padding=1, bias=False
        )
        self.bn_input = nn.BatchNorm2d(num_filters)
        
        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])
        
        # Policy head
        self.policy_conv = nn.Conv2d(
            num_filters, az_config.POLICY_FILTERS,
            kernel_size=1, bias=False
        )
        self.policy_bn = nn.BatchNorm2d(az_config.POLICY_FILTERS)
        
        # Calculate policy fc input size
        policy_fc_size = az_config.POLICY_FILTERS * az_config.ROWS * az_config.COLUMNS
        self.policy_fc = nn.Linear(policy_fc_size, az_config.COLUMNS)
        
        # Value head
        self.value_conv = nn.Conv2d(
            num_filters, az_config.VALUE_FILTERS,
            kernel_size=1, bias=False
        )
        self.value_bn = nn.BatchNorm2d(az_config.VALUE_FILTERS)
        
        # Calculate value fc input size
        value_fc_size = az_config.VALUE_FILTERS * az_config.ROWS * az_config.COLUMNS
        self.value_fc1 = nn.Linear(value_fc_size, az_config.VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(az_config.VALUE_HIDDEN, 1)
        
        # Dropout
        self.dropout = nn.Dropout(az_config.DROPOUT)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through network.
        
        Args:
            x: Input tensor of shape (batch, channels, 6, 7)
        
        Returns:
            Tuple of (policy_logits, value)
            - policy_logits: shape (batch, 7)
            - value: shape (batch, 1) in range [-1, 1]
        """
        # Initial convolution
        x = self.conv_input(x)
        x = self.bn_input(x)
        x = F.relu(x)
        
        # Residual tower
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        policy = policy.view(policy.size(0), -1)  # Flatten
        policy = self.dropout(policy)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = self.dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]
        
        return policy_logits, value
    
    def predict(self, board_state: torch.Tensor) -> tuple:
        """
        Predict policy and value for a board state.
        
        Args:
            board_state: Board state tensor
        
        Returns:
            Tuple of (policy_probs, value)
        """
        self.eval()
        with torch.no_grad():
            policy_logits, value = self.forward(board_state)
            policy_probs = F.softmax(policy_logits, dim=1)
        return policy_probs, value


class DualHeadNetwork(nn.Module):
    """
    Simplified dual-head network (lighter than full AlphaZero).
    
    Good for faster training and testing.
    """
    
    def __init__(self):
        """Initialize simplified network."""
        super(DualHeadNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Policy head
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)
        
        # Value head
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """Forward pass."""
        # Shared layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.dropout(policy)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value


def create_alphazero_model(model_type: str = 'full') -> nn.Module:
    """
    Factory function to create AlphaZero model.
    
    Args:
        model_type: Type of model ('full' or 'light')
    
    Returns:
        Policy-Value network
    """
    if model_type == 'full':
        model = PolicyValueNetwork()
    elif model_type == 'light':
        model = DualHeadNetwork()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.to(az_config.DEVICE)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the AlphaZero model
    print("Testing AlphaZero Policy-Value Network...")
    print("=" * 60)
    
    # Test full model
    print("\nFull AlphaZero Model:")
    model_full = create_alphazero_model('full')
    print(f"  Parameters: {count_parameters(model_full):,}")
    
    # Test input
    batch_size = 4
    # Use 3 channels instead of full history for testing
    test_input = torch.randn(batch_size, 3, 6, 7).to(az_config.DEVICE)
    
    # Create a simpler model for testing
    print("\nTesting with simplified input (3 channels):")
    model_test = PolicyValueNetwork(input_channels=3, num_res_blocks=5, num_filters=128)
    model_test = model_test.to(az_config.DEVICE)
    print(f"  Parameters: {count_parameters(model_test):,}")
    
    # Forward pass
    model_test.train()
    policy_logits, value = model_test(test_input)
    
    print(f"\nForward Pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Policy logits shape: {policy_logits.shape}")
    print(f"  Value shape: {value.shape}")
    print(f"  Value range: [{value.min():.3f}, {value.max():.3f}]")
    
    # Test softmax on policy
    policy_probs = F.softmax(policy_logits, dim=1)
    print(f"  Policy probs sum: {policy_probs.sum(dim=1)}")
    
    # Test predict method
    model_test.eval()
    with torch.no_grad():
        pred_probs, pred_value = model_test.predict(test_input)
    print(f"\nPredict Method:")
    print(f"  Policy probs shape: {pred_probs.shape}")
    print(f"  Policy probs [0]: {pred_probs[0]}")
    print(f"  Value [0]: {pred_value[0].item():.3f}")
    
    # Test light model
    print("\nLight Model:")
    model_light = create_alphazero_model('light')
    print(f"  Parameters: {count_parameters(model_light):,}")
    
    policy_light, value_light = model_light(test_input)
    print(f"  Policy shape: {policy_light.shape}")
    print(f"  Value shape: {value_light.shape}")
    
    print("\n✓ AlphaZero model tests passed!")

