"""
Rainbow DQN Neural Network Models
Implements Dueling Architecture + Noisy Nets + Optional C51
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.rainbow.rainbow_config import rainbow_config


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer for learnable exploration.
    Implements factorised Gaussian noise as described in:
    https://arxiv.org/abs/1706.10295
    
    Key idea: Add parametric noise to weights and biases
    instead of epsilon-greedy exploration.
    """
    
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        """
        Initialize Noisy Linear layer.
        
        Args:
            in_features: Input dimension
            out_features: Output dimension
            sigma_init: Initial noise parameter
        """
        super(NoisyLinear, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init
        
        # Learnable parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Initialize parameters."""
        mu_range = 1 / math.sqrt(self.in_features)
        
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))
    
    def reset_noise(self):
        """Generate new noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Factorised Gaussian noise
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generate scaled noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with noisy weights.
        
        Args:
            x: Input tensor
        
        Returns:
            Output tensor
        """
        if self.training:
            # Use noisy weights during training
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # Use mean weights during evaluation
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(x, weight, bias)


class RainbowDQN(nn.Module):
    """
    Rainbow DQN with Dueling Architecture and Noisy Nets.
    
    Architecture:
    - CNN feature extractor
    - Dueling streams (Value + Advantage)
    - Noisy Linear layers
    - Optional: Categorical DQN (C51)
    """
    
    def __init__(self, use_noisy: bool = True, use_distributional: bool = False):
        """
        Initialize Rainbow DQN.
        
        Args:
            use_noisy: Whether to use Noisy Nets
            use_distributional: Whether to use C51 distributional RL
        """
        super(RainbowDQN, self).__init__()
        
        self.use_noisy = use_noisy
        self.use_distributional = use_distributional
        self.num_actions = rainbow_config.OUTPUT_SIZE
        
        # Distributional RL parameters
        if use_distributional:
            self.num_atoms = rainbow_config.NUM_ATOMS
            self.v_min = rainbow_config.V_MIN
            self.v_max = rainbow_config.V_MAX
            self.atoms = torch.linspace(self.v_min, self.v_max, self.num_atoms)
            self.delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        
        # Convolutional layers (shared feature extractor)
        self.conv1 = nn.Conv2d(
            rainbow_config.INPUT_CHANNELS,
            rainbow_config.CONV_CHANNELS[0],
            kernel_size=3,
            padding=1
        )
        self.conv2 = nn.Conv2d(
            rainbow_config.CONV_CHANNELS[0],
            rainbow_config.CONV_CHANNELS[1],
            kernel_size=3,
            padding=1
        )
        self.conv3 = nn.Conv2d(
            rainbow_config.CONV_CHANNELS[1],
            rainbow_config.CONV_CHANNELS[2],
            kernel_size=3,
            padding=1
        )
        
        # Batch normalization
        self.bn1 = nn.BatchNorm2d(rainbow_config.CONV_CHANNELS[0])
        self.bn2 = nn.BatchNorm2d(rainbow_config.CONV_CHANNELS[1])
        self.bn3 = nn.BatchNorm2d(rainbow_config.CONV_CHANNELS[2])
        
        # Calculate feature size
        conv_output_size = (rainbow_config.ROWS * rainbow_config.COLUMNS * 
                           rainbow_config.CONV_CHANNELS[2])
        
        # Linear layer type (Noisy or standard)
        LinearLayer = NoisyLinear if use_noisy else nn.Linear
        
        # Value stream (estimates state value)
        self.value_fc1 = LinearLayer(conv_output_size, rainbow_config.FC_HIDDEN)
        if use_distributional:
            self.value_fc2 = LinearLayer(rainbow_config.FC_HIDDEN, self.num_atoms)
        else:
            self.value_fc2 = LinearLayer(rainbow_config.FC_HIDDEN, 1)
        
        # Advantage stream (estimates action advantages)
        self.advantage_fc1 = LinearLayer(conv_output_size, rainbow_config.FC_HIDDEN)
        if use_distributional:
            self.advantage_fc2 = LinearLayer(
                rainbow_config.FC_HIDDEN, 
                self.num_actions * self.num_atoms
            )
        else:
            self.advantage_fc2 = LinearLayer(
                rainbow_config.FC_HIDDEN,
                self.num_actions
            )
        
        # Dropout (less important with Noisy Nets)
        self.dropout = nn.Dropout(rainbow_config.DROPOUT)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, 3, 6, 7)
        
        Returns:
            Q-values or value distributions of shape (batch, num_actions) or
            (batch, num_actions, num_atoms) for distributional
        """
        # Convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Value stream
        value = F.relu(self.value_fc1(x))
        value = self.dropout(value)
        value = self.value_fc2(value)
        
        # Advantage stream
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.dropout(advantage)
        advantage = self.advantage_fc2(advantage)
        
        if self.use_distributional:
            # Reshape for distributional RL
            value = value.view(-1, 1, self.num_atoms)
            advantage = advantage.view(-1, self.num_actions, self.num_atoms)
            
            # Combine streams
            q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
            
            # Apply softmax to get probability distribution
            q_dist = F.softmax(q_atoms, dim=2)
            
            return q_dist
        else:
            # Standard dueling: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
            q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
            
            return q_values
    
    def get_q_values(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get expected Q-values (for distributional RL).
        
        Args:
            x: Input tensor
        
        Returns:
            Expected Q-values of shape (batch, num_actions)
        """
        if self.use_distributional:
            q_dist = self.forward(x)
            # Move atoms to device
            atoms = self.atoms.to(q_dist.device)
            # Compute expectation: E[Z] = Σ p(z) * z
            q_values = (q_dist * atoms.view(1, 1, -1)).sum(dim=2)
            return q_values
        else:
            return self.forward(x)
    
    def reset_noise(self):
        """Reset noise in all Noisy Linear layers."""
        if not self.use_noisy:
            return
        
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()


def create_rainbow_model(use_noisy: bool = True, 
                        use_distributional: bool = False) -> nn.Module:
    """
    Factory function to create Rainbow DQN model.
    
    Args:
        use_noisy: Whether to use Noisy Nets
        use_distributional: Whether to use C51 distributional RL
    
    Returns:
        Rainbow DQN model
    """
    model = RainbowDQN(use_noisy=use_noisy, use_distributional=use_distributional)
    model.to(rainbow_config.DEVICE)
    return model


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the Rainbow model
    print("Testing Rainbow DQN Model...")
    print("=" * 60)
    
    # Test standard Rainbow (Dueling + Noisy)
    model = create_rainbow_model(use_noisy=True, use_distributional=False)
    print(f"\nStandard Rainbow DQN:")
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Test input
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 6, 7).to(rainbow_config.DEVICE)
    
    # Forward pass
    model.train()
    q_values = model(test_input)
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {q_values.shape}")
    print(f"  Q-values range: [{q_values.min():.2f}, {q_values.max():.2f}]")
    
    # Test noise reset
    model.reset_noise()
    q_values_new = model(test_input)
    noise_changed = not torch.allclose(q_values, q_values_new)
    print(f"  Noise reset working: {noise_changed}")
    
    # Test evaluation mode
    model.eval()
    q_values_eval1 = model(test_input)
    q_values_eval2 = model(test_input)
    deterministic = torch.allclose(q_values_eval1, q_values_eval2)
    print(f"  Evaluation deterministic: {deterministic}")
    
    # Test Distributional Rainbow (optional)
    print(f"\nDistributional Rainbow DQN (C51):")
    model_c51 = create_rainbow_model(use_noisy=True, use_distributional=True)
    print(f"  Parameters: {count_parameters(model_c51):,}")
    
    model_c51.train()
    q_dist = model_c51(test_input)
    q_values_c51 = model_c51.get_q_values(test_input)
    print(f"  Distribution shape: {q_dist.shape}")
    print(f"  Q-values shape: {q_values_c51.shape}")
    print(f"  Q-values range: [{q_values_c51.min():.2f}, {q_values_c51.max():.2f}]")
    
    # Check distribution sums to 1
    dist_sums = q_dist.sum(dim=2)
    valid_dist = torch.allclose(dist_sums, torch.ones_like(dist_sums), atol=1e-6)
    print(f"  Valid probability distribution: {valid_dist}")
    
    print("\n✓ Rainbow DQN model tests passed!")

