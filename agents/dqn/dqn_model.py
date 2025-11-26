"""
Deep Q-Network (DQN) Model
CNN-based architecture for processing ConnectX board states
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.base.config import config


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with CNN architecture.
    
    Architecture:
    - Input: (batch, 3, 6, 7) - 3 channels (player, opponent, valid moves)
    - Conv layers: Extract spatial features from board
    - Fully connected layers: Map features to Q-values
    - Output: (batch, 7) - Q-value for each column
    """
    
    def __init__(self):
        """Initialize DQN network architecture."""
        super(DQNNetwork, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=config.INPUT_CHANNELS,
            out_channels=config.CONV_CHANNELS[0],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.conv2 = nn.Conv2d(
            in_channels=config.CONV_CHANNELS[0],
            out_channels=config.CONV_CHANNELS[1],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        self.conv3 = nn.Conv2d(
            in_channels=config.CONV_CHANNELS[1],
            out_channels=config.CONV_CHANNELS[2],
            kernel_size=3,
            stride=1,
            padding=1
        )
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm2d(config.CONV_CHANNELS[0])
        self.bn2 = nn.BatchNorm2d(config.CONV_CHANNELS[1])
        self.bn3 = nn.BatchNorm2d(config.CONV_CHANNELS[2])
        
        # Calculate size after convolutions
        # Input: 6x7, after conv layers (with same padding): still 6x7
        conv_output_size = config.ROWS * config.COLUMNS * config.CONV_CHANNELS[2]
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, config.FC_HIDDEN)
        self.fc2 = nn.Linear(config.FC_HIDDEN, config.FC_HIDDEN // 2)
        self.fc3 = nn.Linear(config.FC_HIDDEN // 2, config.OUTPUT_SIZE)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.DROPOUT)
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 6, 7)
        
        Returns:
            Q-values tensor of shape (batch, 7)
        """
        # Convolutional layers with batch norm and ReLU
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def select_action(self, state, valid_moves=None):
        """
        Select action using the network (greedy policy).
        
        Args:
            state: State tensor of shape (1, 3, 6, 7) or (3, 6, 7)
            valid_moves: List of valid column indices (optional)
        
        Returns:
            Selected action (column index)
        """
        # Ensure state has batch dimension
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(state)
            
            # Mask invalid moves if provided
            if valid_moves is not None:
                mask = torch.full_like(q_values, float('-inf'))
                mask[0, valid_moves] = 0
                q_values = q_values + mask
            
            action = q_values.argmax(dim=1).item()
        
        return action


class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture (optional enhancement).
    Separates state value and action advantage estimation.
    """
    
    def __init__(self):
        """Initialize Dueling DQN architecture."""
        super(DuelingDQN, self).__init__()
        
        # Shared convolutional layers
        self.conv1 = nn.Conv2d(config.INPUT_CHANNELS, config.CONV_CHANNELS[0], 
                               kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(config.CONV_CHANNELS[0], config.CONV_CHANNELS[1], 
                               kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(config.CONV_CHANNELS[1], config.CONV_CHANNELS[2], 
                               kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm2d(config.CONV_CHANNELS[0])
        self.bn2 = nn.BatchNorm2d(config.CONV_CHANNELS[1])
        self.bn3 = nn.BatchNorm2d(config.CONV_CHANNELS[2])
        
        conv_output_size = config.ROWS * config.COLUMNS * config.CONV_CHANNELS[2]
        
        # Value stream
        self.value_fc1 = nn.Linear(conv_output_size, config.FC_HIDDEN)
        self.value_fc2 = nn.Linear(config.FC_HIDDEN, 1)
        
        # Advantage stream
        self.advantage_fc1 = nn.Linear(conv_output_size, config.FC_HIDDEN)
        self.advantage_fc2 = nn.Linear(config.FC_HIDDEN, config.OUTPUT_SIZE)
        
        self.dropout = nn.Dropout(config.DROPOUT)
    
    def forward(self, x):
        """
        Forward pass through Dueling DQN.
        
        Args:
            x: Input tensor of shape (batch, 3, 6, 7)
        
        Returns:
            Q-values tensor of shape (batch, 7)
        """
        # Shared convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        x = x.view(x.size(0), -1)
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.dropout(value)
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.dropout(advantage)
        advantage = self.advantage_fc2(advantage)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
    
    def select_action(self, state, valid_moves=None):
        """
        Select action using the network (greedy policy).
        
        Args:
            state: State tensor of shape (1, 3, 6, 7) or (3, 6, 7)
            valid_moves: List of valid column indices (optional)
        
        Returns:
            Selected action (column index)
        """
        if state.dim() == 3:
            state = state.unsqueeze(0)
        
        with torch.no_grad():
            q_values = self.forward(state)
            
            if valid_moves is not None:
                mask = torch.full_like(q_values, float('-inf'))
                mask[0, valid_moves] = 0
                q_values = q_values + mask
            
            action = q_values.argmax(dim=1).item()
        
        return action


def create_model(model_type='standard'):
    """
    Factory function to create a DQN model.
    
    Args:
        model_type: Type of model ('standard' or 'dueling')
    
    Returns:
        DQN model instance
    """
    if model_type == 'dueling':
        model = DuelingDQN()
    else:
        model = DQNNetwork()
    
    # Move to configured device
    model = model.to(config.DEVICE)
    
    return model


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print(f"Using device: {config.DEVICE}")
    
    model = create_model('standard')
    print(f"\nStandard DQN Model:")
    print(f"Parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 4
    dummy_input = torch.randn(batch_size, 3, config.ROWS, config.COLUMNS).to(config.DEVICE)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Test Dueling DQN
    dueling_model = create_model('dueling')
    print(f"\nDueling DQN Model:")
    print(f"Parameters: {count_parameters(dueling_model):,}")
    output_dueling = dueling_model(dummy_input)
    print(f"Output shape: {output_dueling.shape}")

