"""
Prepare Kaggle Submission from Trained Models
Creates submission-ready agent code with embedded model weights
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import argparse
import base64
import io
from datetime import datetime


def embed_model_weights(model_path: str) -> str:
    """
    Embed model weights as base64 string.
    
    Args:
        model_path: Path to model .pth file
    
    Returns:
        Base64 encoded string of model weights
    """
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Extract model_state_dict if it exists (for full checkpoints)
    # Otherwise use the checkpoint directly (for state_dict only files)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Save to bytes
    buffer = io.BytesIO()
    torch.save(state_dict, buffer)
    buffer.seek(0)
    
    # Encode to base64
    encoded = base64.b64encode(buffer.read()).decode('utf-8')
    
    return encoded


def create_rainbow_submission(model_path: str, output_path: str):
    """
    Create Kaggle submission file for Rainbow DQN.
    
    Args:
        model_path: Path to trained Rainbow model
        output_path: Output path for submission file
    """
    print(f"Creating Rainbow DQN submission from: {model_path}")
    
    # Embed model
    model_b64 = embed_model_weights(model_path)
    print(f"Model size: {len(model_b64) / 1024 / 1024:.2f} MB")
    
    submission_code = f'''
"""
ConnectX Rainbow DQN Agent - Kaggle Submission
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io

# Embedded model weights (base64)
MODEL_WEIGHTS_B64 = """{model_b64}"""

class NoisyLinear(nn.Module):
    """Noisy Linear layer for exploration."""
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer('bias_epsilon', torch.Tensor(out_features))
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        import math
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(0.5 / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(0.5 / math.sqrt(self.out_features))
    
    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

class RainbowDQN(nn.Module):
    """Rainbow DQN model."""
    def __init__(self):
        super().__init__()
        # Conv layers
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(256)
        
        # Dueling streams
        self.value_fc1 = NoisyLinear(256*6*7, 512)
        self.value_fc2 = NoisyLinear(512, 1)
        self.advantage_fc1 = NoisyLinear(256*6*7, 512)
        self.advantage_fc2 = NoisyLinear(512, 7)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        
        value = F.relu(self.value_fc1(x))
        value = self.dropout(value)
        value = self.value_fc2(value)
        
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.dropout(advantage)
        advantage = self.advantage_fc2(advantage)
        
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# Load model
model = RainbowDQN()
model.eval()

# Decode and load weights
model_bytes = base64.b64decode(MODEL_WEIGHTS_B64)
buffer = io.BytesIO(model_bytes)
state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
model.load_state_dict(state_dict, strict=False)

def encode_state(board, mark):
    """Encode board state."""
    state = np.zeros((3, 6, 7), dtype=np.float32)
    for i, cell in enumerate(board):
        row, col = divmod(i, 7)
        if cell == mark:
            state[0, row, col] = 1
        elif cell != 0:
            state[1, row, col] = 1
    # Valid moves
    for col in range(7):
        if board[col] == 0:
            state[2, 0, col] = 1
    return state

def get_valid_moves(board):
    """Get valid moves."""
    return [c for c in range(7) if board[c] == 0]

def agent(observation, configuration):
    """Main agent function for Kaggle."""
    board = observation.board
    mark = observation.mark
    
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return 0
    
    # Encode state
    state = encode_state(board, mark)
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    
    # Get Q-values
    with torch.no_grad():
        q_values = model(state_tensor)
        
        # Mask invalid moves
        mask = torch.full_like(q_values, float('-inf'))
        mask[0, valid_moves] = 0
        q_values = q_values + mask
        
        action = q_values.argmax(dim=1).item()
    
    return int(action)
'''
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(submission_code)
    
    print(f"Rainbow submission created: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def create_alphazero_submission(model_path: str, output_path: str, num_simulations: int = 100):
    """
    Create Kaggle submission file for AlphaZero.
    
    Args:
        model_path: Path to trained AlphaZero model
        output_path: Output path for submission file
        num_simulations: Number of MCTS simulations (reduce for faster inference)
    """
    print(f"Creating AlphaZero submission from: {model_path}")
    print(f"MCTS simulations: {num_simulations}")
    
    # Embed model
    model_b64 = embed_model_weights(model_path)
    print(f"Model size: {len(model_b64) / 1024 / 1024:.2f} MB")
    
    submission_code = f'''
"""
ConnectX AlphaZero Agent - Kaggle Submission
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
MCTS Simulations: {num_simulations}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io
import math
import random

# Embedded model weights
MODEL_WEIGHTS_B64 = """{model_b64}"""

class DualHeadNetwork(nn.Module):
    """Policy-Value network."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.policy_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 6 * 7, 7)
        
        self.value_conv = nn.Conv2d(128, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 6 * 7, 128)
        self.value_fc2 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy = self.dropout(policy)
        policy_logits = self.policy_fc(policy)
        
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = self.dropout(value)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value

# Load model
network = DualHeadNetwork()
network.eval()

model_bytes = base64.b64decode(MODEL_WEIGHTS_B64)
buffer = io.BytesIO(model_bytes)
state_dict = torch.load(buffer, map_location='cpu', weights_only=False)
network.load_state_dict(state_dict, strict=False)

# MCTS implementation
class MCTSNode:
    def __init__(self, state, mark, parent=None, action=None, prior_prob=0.0):
        self.state = state
        self.mark = mark
        self.parent = parent
        self.action = action
        self.children = {{}}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior_prob
    
    def select_child(self, c_puct=1.5):
        best_score = -float('inf')
        best_child = None
        sqrt_parent_n = math.sqrt(self.N)
        
        for child in self.children.values():
            if child.N == 0:
                q_value = 0.0
            else:
                q_value = -child.Q
            u_value = c_puct * child.P * sqrt_parent_n / (1 + child.N)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

def encode_state(board, mark):
    state = np.zeros((3, 6, 7), dtype=np.float32)
    for i, cell in enumerate(board):
        row, col = divmod(i, 7)
        if cell == mark:
            state[0, row, col] = 1
        elif cell != 0:
            state[1, row, col] = 1
    for col in range(7):
        if board[col] == 0:
            state[2, 0, col] = 1
    return state

def get_valid_moves(board):
    return [c for c in range(7) if board[c] == 0]

def make_move(board, col, mark):
    new_board = board[:]
    for row in range(5, -1, -1):
        idx = row * 7 + col
        if new_board[idx] == 0:
            new_board[idx] = mark
            return new_board
    return board

def is_terminal(board):
    # Check win
    for mark in [1, 2]:
        for row in range(6):
            for col in range(4):
                idx = row * 7 + col
                if all(board[idx + i] == mark for i in range(4)):
                    return True, mark
        for row in range(3):
            for col in range(7):
                idx = row * 7 + col
                if all(board[idx + 7*i] == mark for i in range(4)):
                    return True, mark
        for row in range(3):
            for col in range(4):
                idx = row * 7 + col
                if all(board[idx + 8*i] == mark for i in range(4)):
                    return True, mark
        for row in range(3):
            for col in range(3, 7):
                idx = row * 7 + col
                if all(board[idx + 6*i] == mark for i in range(4)):
                    return True, mark
    # Check draw
    if all(board[i] != 0 for i in range(7)):
        return True, 0
    return False, 0

def mcts_search(board, mark, num_sims={num_simulations}):
    root = MCTSNode(board, mark)
    
    # Evaluate root
    state = encode_state(board, mark)
    state_tensor = torch.from_numpy(state).unsqueeze(0)
    with torch.no_grad():
        policy_logits, _ = network(state_tensor)
        policy_probs = F.softmax(policy_logits, dim=1).numpy()[0]
    
    valid_moves = get_valid_moves(board)
    mask = np.zeros(7)
    mask[valid_moves] = 1.0
    policy_probs = policy_probs * mask
    if policy_probs.sum() > 0:
        policy_probs = policy_probs / policy_probs.sum()
    
    # Expand root
    for action in valid_moves:
        next_state = make_move(board, action, mark)
        next_mark = 3 - mark
        child = MCTSNode(next_state, next_mark, parent=root,
                        action=action, prior_prob=policy_probs[action])
        root.children[action] = child
    
    # Simulations
    for _ in range(num_sims):
        node = root
        path = [node]
        
        # Select
        while node.children and not is_terminal(node.state)[0]:
            node = node.select_child()
            path.append(node)
        
        # Evaluate
        if is_terminal(node.state)[0]:
            done, winner = is_terminal(node.state)
            if winner == 0:
                value = 0.0
            elif winner == node.mark:
                value = 1.0
            else:
                value = -1.0
        else:
            state = encode_state(node.state, node.mark)
            state_tensor = torch.from_numpy(state).unsqueeze(0)
            with torch.no_grad():
                policy_logits, pred_value = network(state_tensor)
                value = pred_value.item()
                policy_probs = F.softmax(policy_logits, dim=1).numpy()[0]
            
            valid_moves = get_valid_moves(node.state)
            mask = np.zeros(7)
            mask[valid_moves] = 1.0
            policy_probs = policy_probs * mask
            if policy_probs.sum() > 0:
                policy_probs = policy_probs / policy_probs.sum()
            
            for action in valid_moves:
                next_state = make_move(node.state, action, node.mark)
                next_mark = 3 - node.mark
                child = MCTSNode(next_state, next_mark, parent=node,
                                action=action, prior_prob=policy_probs[action])
                node.children[action] = child
        
        # Backprop
        for n in reversed(path):
            n.N += 1
            n.W += value
            n.Q = n.W / n.N
            value = -value
    
    # Select best action
    visit_counts = np.zeros(7)
    for action, child in root.children.items():
        visit_counts[action] = child.N
    return int(np.argmax(visit_counts))

def agent(observation, configuration):
    """Main agent function."""
    board = observation.board
    mark = observation.mark
    
    valid_moves = get_valid_moves(board)
    if not valid_moves:
        return 0
    
    try:
        action = mcts_search(board, mark)
        if action not in valid_moves:
            action = valid_moves[0]
    except:
        action = valid_moves[0]
    
    return int(action)
'''
    
    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(submission_code)
    
    print(f"AlphaZero submission created: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Prepare Kaggle submission')
    parser.add_argument('--agent', choices=['rainbow', 'alphazero'], required=True,
                       help='Agent type to prepare')
    parser.add_argument('--model-path', required=True,
                       help='Path to trained model weights')
    parser.add_argument('--output', default='submission/kaggle_agent.py',
                       help='Output path for submission file')
    parser.add_argument('--mcts-sims', type=int, default=100,
                       help='MCTS simulations for AlphaZero (default: 100)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("KAGGLE SUBMISSION PREPARATION")
    print("="*70)
    print(f"Agent: {args.agent}")
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output}")
    print("="*70 + "\n")
    
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    if args.agent == 'rainbow':
        create_rainbow_submission(args.model_path, args.output)
    elif args.agent == 'alphazero':
        create_alphazero_submission(args.model_path, args.output, args.mcts_sims)
    
    print("\n✓ Submission file created successfully!")
    print(f"\nNext steps:")
    print(f"  1. Test locally: python -c 'from {os.path.splitext(os.path.basename(args.output))[0]} import agent; print(agent)'")
    print(f"  2. Upload to Kaggle: {args.output}")
    print()


if __name__ == "__main__":
    main()

