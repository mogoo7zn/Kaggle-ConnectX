"""
Kaggle ConnectX Submission Agent V2
Enhanced AlphaZero with Full MCTS + Neural Network + Advanced Tactics

Features:
- Complete MCTS with PUCT selection (N(s,a) max for action)
- Neural network policy and value evaluation
- Advanced tactical detection (double threats, forks)
- Avoid giving opponent winning opportunities
- Optimized for Kaggle real-time play (~5s limit)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Optional, Dict


# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Configuration constants"""
    ROWS = 6
    COLUMNS = 7
    INAROW = 4
    
    # MCTS Parameters - Optimized for competition
    NUM_SIMULATIONS = 400      # Higher for stronger play
    NUM_SIMULATIONS_FAST = 100 # For time-critical situations
    C_PUCT = 1.5               # Exploration constant
    
    # Dirichlet noise (disabled for competition)
    DIRICHLET_ALPHA = 0.3
    DIRICHLET_EPSILON = 0.0
    
    # Network Architecture (matching trained model)
    INPUT_CHANNELS = 3
    NUM_RES_BLOCKS = 6
    NUM_FILTERS = 96
    POLICY_FILTERS = 32
    VALUE_FILTERS = 32
    VALUE_HIDDEN = 128


config = Config()


# ============================================================================
# Utility Functions
# ============================================================================

def get_valid_moves(board: List[int]) -> List[int]:
    """Get list of valid column indices"""
    return [col for col in range(config.COLUMNS) if board[col] == 0]


def make_move(board: List[int], col: int, mark: int) -> List[int]:
    """Make a move on the board (returns new board)"""
    board = board.copy()
    for row in range(config.ROWS - 1, -1, -1):
        idx = row * config.COLUMNS + col
        if board[idx] == 0:
            board[idx] = mark
            break
    return board


def check_winner(board: List[int], mark: int) -> bool:
    """Check if the given player has won"""
    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)
    
    # Check horizontal
    for row in range(config.ROWS):
        for col in range(config.COLUMNS - config.INAROW + 1):
            if all(board_2d[row, col + i] == mark for i in range(config.INAROW)):
                return True
    
    # Check vertical
    for row in range(config.ROWS - config.INAROW + 1):
        for col in range(config.COLUMNS):
            if all(board_2d[row + i, col] == mark for i in range(config.INAROW)):
                return True
    
    # Check diagonal (positive slope)
    for row in range(config.ROWS - config.INAROW + 1):
        for col in range(config.COLUMNS - config.INAROW + 1):
            if all(board_2d[row + i, col + i] == mark for i in range(config.INAROW)):
                return True
    
    # Check diagonal (negative slope)
    for row in range(config.INAROW - 1, config.ROWS):
        for col in range(config.COLUMNS - config.INAROW + 1):
            if all(board_2d[row - i, col + i] == mark for i in range(config.INAROW)):
                return True
    
    return False


def is_terminal(board: List[int]) -> Tuple[bool, int]:
    """Check if game is over."""
    if check_winner(board, 1):
        return True, 1
    if check_winner(board, 2):
        return True, 2
    if all(board[col] != 0 for col in range(config.COLUMNS)):
        return True, 0
    return False, -1


def find_winning_move(board: List[int], mark: int) -> Optional[int]:
    """Find immediate winning move"""
    valid_moves = get_valid_moves(board)
    for col in valid_moves:
        next_board = make_move(board, col, mark)
        if check_winner(next_board, mark):
            return col
    return None


def find_blocking_move(board: List[int], mark: int) -> Optional[int]:
    """Find move to block opponent's win"""
    opponent = 3 - mark
    return find_winning_move(board, opponent)


def find_double_threat_move(board: List[int], mark: int) -> Optional[int]:
    """Find a move that creates two winning threats simultaneously."""
    valid_moves = get_valid_moves(board)
    
    for col in valid_moves:
        next_board = make_move(board, col, mark)
        
        winning_threats = 0
        next_valid = get_valid_moves(next_board)
        
        for next_col in next_valid:
            test_board = make_move(next_board, next_col, mark)
            if check_winner(test_board, mark):
                winning_threats += 1
        
        if winning_threats >= 2:
            return col
    
    return None


def block_double_threat(board: List[int], mark: int) -> Optional[int]:
    """Block opponent's double threat if possible"""
    opponent = 3 - mark
    return find_double_threat_move(board, opponent)


def find_open_three_blocking_move(board: List[int], mark: int) -> Optional[int]:
    """
    Block opponent's open three (0 X X X 0) horizontally.
    Check both sides for safety.
    """
    opponent = 3 - mark
    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)
    
    for r in range(config.ROWS):
        for c in range(config.COLUMNS - 4):
            # Window of 5: [0, op, op, op, 0]
            window = board_2d[r, c:c+5]
            if np.all(window == [0, opponent, opponent, opponent, 0]):
                candidates = []
                # Check left side (c)
                if (r == config.ROWS - 1) or (board_2d[r+1, c] != 0):
                    candidates.append(c)
                # Check right side (c+4)
                if (r == config.ROWS - 1) or (board_2d[r+1, c+4] != 0):
                    candidates.append(c+4)
                
                for cand in candidates:
                    if not is_losing_move(board, cand, mark):
                        return cand
    return None


def find_open_two_blocking_move(board: List[int], mark: int) -> Optional[int]:
    """
    Block opponent's open two (0 X X 0) horizontally.
    Check both sides for safety.
    """
    opponent = 3 - mark
    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)
    
    for r in range(config.ROWS):
        for c in range(config.COLUMNS - 3):
            # Window of 4: [0, op, op, 0]
            window = board_2d[r, c:c+4]
            if np.all(window == [0, opponent, opponent, 0]):
                candidates = []
                # Check left side (c)
                if (r == config.ROWS - 1) or (board_2d[r+1, c] != 0):
                    candidates.append(c)
                # Check right side (c+3)
                if (r == config.ROWS - 1) or (board_2d[r+1, c+3] != 0):
                    candidates.append(c+3)
                
                for cand in candidates:
                    if not is_losing_move(board, cand, mark):
                        return cand
    return None


def is_losing_move(board: List[int], col: int, mark: int) -> bool:
    """Check if making this move gives opponent an immediate win."""
    next_board = make_move(board, col, mark)
    opponent = 3 - mark
    return find_winning_move(next_board, opponent) is not None


def get_safe_moves(board: List[int], mark: int) -> List[int]:
    """Get moves that don't immediately give opponent a win"""
    valid_moves = get_valid_moves(board)
    return [col for col in valid_moves if not is_losing_move(board, col, mark)]


def encode_state(board: List[int], mark: int) -> np.ndarray:
    """Encode board state as 3-channel tensor."""
    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)
    opponent = 3 - mark
    
    player_channel = (board_2d == mark).astype(np.float32)
    opponent_channel = (board_2d == opponent).astype(np.float32)
    
    valid_moves = np.zeros((config.ROWS, config.COLUMNS), dtype=np.float32)
    for col in range(config.COLUMNS):
        if board_2d[0, col] == 0:
            valid_moves[:, col] = 1.0
    
    return np.stack([player_channel, opponent_channel, valid_moves], axis=0)


# ============================================================================
# Neural Network Architecture (Matching Trained Model)
# ============================================================================

class PolicyValueNetwork(nn.Module):
    """
    AlphaZero-style Policy-Value Network.
    Architecture matches the trained checkpoint format.
    """
    
    def __init__(self, num_filters: int = 64, num_res_blocks: int = 4):
        super(PolicyValueNetwork, self).__init__()
        
        self.num_filters = num_filters
        self.num_res_blocks = num_res_blocks
        
        # Initial convolutional layers
        self.conv1 = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv3 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(num_filters)
        
        # Residual blocks (using Sequential to match checkpoint format)
        self.res_blocks = nn.ModuleList()
        for _ in range(num_res_blocks):
            block = nn.Sequential(
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters),
                nn.ReLU(),
                nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(num_filters)
            )
            self.res_blocks.append(block)
        
        # Policy head
        self.policy_conv = nn.Conv2d(num_filters, config.POLICY_FILTERS, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(config.POLICY_FILTERS)
        policy_fc_size = config.POLICY_FILTERS * config.ROWS * config.COLUMNS
        self.policy_fc = nn.Linear(policy_fc_size, config.COLUMNS)
        
        # Value head
        self.value_conv = nn.Conv2d(num_filters, config.VALUE_FILTERS, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(config.VALUE_FILTERS)
        value_fc_size = config.VALUE_FILTERS * config.ROWS * config.COLUMNS
        self.value_fc1 = nn.Linear(value_fc_size, config.VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(config.VALUE_HIDDEN, 1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial convolutions
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Residual blocks
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x = F.relu(x + residual)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value


# ============================================================================
# MCTS Implementation
# ============================================================================

class MCTSNode:
    """MCTS tree node with UCB statistics."""
    
    __slots__ = ['state', 'mark', 'parent', 'action', 'children',
                 'N', 'W', 'Q', 'P', '_valid_moves', '_is_terminal', '_terminal_winner']
    
    def __init__(self, state: List[int], mark: int, parent=None, 
                 action: int = None, prior_prob: float = 0.0):
        self.state = state
        self.mark = mark
        self.parent = parent
        self.action = action
        self.children: Dict[int, 'MCTSNode'] = {}
        
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior_prob
        
        self._valid_moves = None
        self._is_terminal = None
        self._terminal_winner = None
    
    @property
    def valid_moves(self) -> List[int]:
        if self._valid_moves is None:
            self._valid_moves = get_valid_moves(self.state)
        return self._valid_moves
    
    @property
    def is_terminal(self) -> bool:
        if self._is_terminal is None:
            self._is_terminal, self._terminal_winner = is_terminal(self.state)
        return self._is_terminal
    
    @property
    def terminal_winner(self) -> int:
        if self._is_terminal is None:
            self._is_terminal, self._terminal_winner = is_terminal(self.state)
        return self._terminal_winner
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def get_terminal_value(self, perspective_mark: int) -> float:
        if not self.is_terminal:
            return 0.0
        winner = self.terminal_winner
        if winner == 0:
            return 0.0
        elif winner == perspective_mark:
            return 1.0
        else:
            return -1.0
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """Select best child using PUCT formula."""
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_n = math.sqrt(max(1, self.N))
        
        for child in self.children.values():
            q_value = -child.Q if child.N > 0 else 0.0
            u_value = c_puct * child.P * sqrt_parent_n / (1 + child.N)
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs: np.ndarray):
        """Expand node by creating children for all valid moves."""
        for action in self.valid_moves:
            if action not in self.children:
                next_state = make_move(self.state, action, self.mark)
                next_mark = 3 - self.mark
                
                child = MCTSNode(
                    state=next_state,
                    mark=next_mark,
                    parent=self,
                    action=action,
                    prior_prob=policy_probs[action]
                )
                self.children[action] = child
    
    def backpropagate(self, value: float):
        """Backpropagate value up the tree."""
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """Monte Carlo Tree Search with neural network guidance."""
    
    def __init__(self, network: nn.Module, device: torch.device):
        self.network = network
        self.device = device
    
    def search(self, root_state: List[int], root_mark: int,
               num_simulations: int = None) -> Tuple[np.ndarray, Dict[int, int]]:
        """
        Perform MCTS search and return visit count distribution.
        """
        if num_simulations is None:
            num_simulations = config.NUM_SIMULATIONS
        
        # Create root node
        root = MCTSNode(root_state, root_mark)
        
        if root.is_terminal:
            policy = np.zeros(config.COLUMNS)
            return policy, {}
        
        # Evaluate root with neural network
        policy_probs, _ = self._evaluate(root_state, root_mark)
        policy_probs = self._mask_and_normalize(policy_probs, root.valid_moves)
        
        # Add Dirichlet noise for exploration
        if config.DIRICHLET_EPSILON > 0:
            noise = np.zeros(config.COLUMNS)
            noise[root.valid_moves] = np.random.dirichlet(
                [config.DIRICHLET_ALPHA] * len(root.valid_moves)
            )
            policy_probs = ((1 - config.DIRICHLET_EPSILON) * policy_probs +
                           config.DIRICHLET_EPSILON * noise)
        
        # Expand root
        root.expand(policy_probs)
        
        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)
        
        # Get visit counts
        visit_counts = {action: child.N for action, child in root.children.items()}
        
        # Compute policy (greedy - select action with max N)
        policy = np.zeros(config.COLUMNS)
        for action, count in visit_counts.items():
            policy[action] = count
        
        if policy.sum() > 0:
            best_action = np.argmax(policy)
            result_policy = np.zeros(config.COLUMNS)
            result_policy[best_action] = 1.0
            return result_policy, visit_counts
        
        return policy, visit_counts
    
    def _simulate(self, root: MCTSNode):
        """Run one simulation from root to leaf."""
        node = root
        
        while not node.is_leaf() and not node.is_terminal:
            node = node.select_child(config.C_PUCT)
        
        if node.is_terminal:
            value = node.get_terminal_value(node.mark)
            node.backpropagate(value)
            return
        
        policy_probs, value = self._evaluate(node.state, node.mark)
        policy_probs = self._mask_and_normalize(policy_probs, node.valid_moves)
        node.expand(policy_probs)
        
        node.backpropagate(value)
    
    def _evaluate(self, state: List[int], mark: int) -> Tuple[np.ndarray, float]:
        """Evaluate state using neural network."""
        encoded = encode_state(state, mark)
        state_tensor = torch.from_numpy(encoded).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            policy_probs = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()
        
        return policy_probs, value
    
    def _mask_and_normalize(self, policy: np.ndarray, valid_moves: List[int]) -> np.ndarray:
        """Mask invalid moves and renormalize."""
        mask = np.zeros(config.COLUMNS)
        mask[valid_moves] = 1.0
        policy = policy * mask
        
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = mask / mask.sum()
        
        return policy
    
    def get_best_action(self, state: List[int], mark: int,
                        num_simulations: int = None) -> int:
        """Get best action using MCTS (action with max N(s,a))."""
        policy, visit_counts = self.search(state, mark, num_simulations)
        
        if visit_counts:
            return max(visit_counts.keys(), key=lambda a: visit_counts[a])
        
        return int(np.argmax(policy))


# ============================================================================
# Agent Class
# ============================================================================

class AlphaZeroAgentV2:
    """Enhanced AlphaZero agent with advanced tactics."""
    
    def __init__(self):
        self.device = torch.device("cpu")
        self.network = None
        self.mcts = None
        self.model_loaded = False
        self.move_count = 0
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model weights."""
        try:
            self.network = PolicyValueNetwork()
            self.network = self.network.to(self.device)
            
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.network.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.network.load_state_dict(checkpoint['state_dict'])
                else:
                    self.network.load_state_dict(checkpoint)
            else:
                self.network.load_state_dict(checkpoint)
            
            self.network.eval()
            self.mcts = MCTS(self.network, self.device)
            self.model_loaded = True
            return True
        except Exception as e:
            return False
    
    def select_action(self, board: List[int], mark: int) -> int:
        """
        Select action using hierarchical strategy.
        
        Priority:
        1. Take winning move (instant win)
        2. Block opponent's win (must block)
        3. Create double threat (fork)
        4. Block opponent's double threat
        5. Avoid moves that give opponent instant win
        6. Use MCTS with N(s,a) max selection
        7. Fallback to center preference
        """
        valid_moves = get_valid_moves(board)
        
        if not valid_moves:
            return 0
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        # Count moves for game phase detection
        self.move_count = sum(1 for x in board if x != 0)
        
        # Priority 1: Take winning move
        winning_move = find_winning_move(board, mark)
        if winning_move is not None:
            return winning_move
        
        # Priority 2: Block opponent's win
        blocking_move = find_blocking_move(board, mark)
        if blocking_move is not None:
            return blocking_move
            
        # Priority 2.5: Block Open Three (Critical threat)
        open_three = find_open_three_blocking_move(board, mark)
        if open_three is not None:
            if not is_losing_move(board, open_three, mark):
                return open_three
        
        # Priority 3: Create double threat (fork)
        double_threat = find_double_threat_move(board, mark)
        if double_threat is not None:
            if not is_losing_move(board, double_threat, mark):
                return double_threat
        
        # Priority 4: Block opponent's double threat
        block_double = block_double_threat(board, mark)
        if block_double is not None:
            if not is_losing_move(board, block_double, mark):
                return block_double
                
        # Priority 4.5: Block Open Two (Preventative)
        open_two = find_open_two_blocking_move(board, mark)
        if open_two is not None:
            if not is_losing_move(board, open_two, mark):
                return open_two
        
        # Priority 5: Filter out losing moves
        safe_moves = get_safe_moves(board, mark)
        if not safe_moves:
            safe_moves = valid_moves
        
        # Priority 6: Use MCTS if model loaded
        if self.model_loaded and self.mcts is not None:
            try:
                # Adjust simulation count based on game phase
                if self.move_count < 4:
                    num_sims = config.NUM_SIMULATIONS
                elif self.move_count > 30:
                    num_sims = config.NUM_SIMULATIONS_FAST
                else:
                    num_sims = config.NUM_SIMULATIONS
                
                action = self.mcts.get_best_action(board, mark, num_sims)
                
                if action in safe_moves:
                    return action
                elif safe_moves:
                    return safe_moves[0]
            except Exception:
                pass
        
        # Priority 7: Heuristic fallback (center preference)
        center_priority = [3, 2, 4, 1, 5, 0, 6]
        
        for col in center_priority:
            if col in safe_moves:
                return col
        
        return safe_moves[0] if safe_moves else valid_moves[0]


# ============================================================================
# Global Agent Instance
# ============================================================================

_agent = None


def get_agent() -> AlphaZeroAgentV2:
    """Get or create global agent instance."""
    global _agent
    if _agent is None:
        _agent = AlphaZeroAgentV2()
        
        # Determine base path for resource loading
        import sys
        import os
        if getattr(sys, 'frozen', False):
            # Running in PyInstaller bundle
            base_path = sys._MEIPASS
        else:
            # Running in normal Python environment
            base_path = os.path.dirname(os.path.abspath(__file__))

        paths = [
            os.path.join(base_path, 'submission', 'alpha-zero-ultra-weights.pth'), # PyInstaller path
            os.path.join(base_path, 'alpha-zero-ultra-weights.pth'), # Local path
            '/kaggle_simulations/agent/alpha-zero-ultra-weights.pth',
            'alpha-zero-ultra-weights.pth',
            './alpha-zero-ultra-weights.pth',
            # Fallbacks
            os.path.join(base_path, 'submission', 'alpha-zero-v0.pth'),
            os.path.join(base_path, 'alpha-zero-v0.pth'),
        ]
        
        for path in paths:
            try:
                if _agent.load_model(path):
                    print(f"Successfully loaded model from {path}")
                    break
            except:
                continue
    
    return _agent


# ============================================================================
# Main Agent Function (Kaggle Entry Point)
# ============================================================================

def agent(observation, configuration):
    """
    Main agent function for Kaggle submission.
    Uses AlphaZero MCTS + Neural Network with advanced tactics.
    """
    try:
        my_agent = get_agent()
        board = list(observation.board)
        mark = observation.mark
        
        action = my_agent.select_action(board, mark)
        
        valid_moves = get_valid_moves(board)
        if action not in valid_moves:
            center = config.COLUMNS // 2
            action = center if center in valid_moves else valid_moves[0]
        
        return int(action)
    
    except Exception:
        valid_moves = [col for col in range(config.COLUMNS) if observation.board[col] == 0]
        if valid_moves:
            center = config.COLUMNS // 2
            return center if center in valid_moves else valid_moves[0]
        return config.COLUMNS // 2
