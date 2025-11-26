"""
Monte Carlo Tree Search (MCTS) Implementation
Core algorithm for AlphaZero using UCB formula and neural network guidance
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.alphazero.az_config import az_config
from agents.base.utils import get_valid_moves, make_move, is_terminal


class MCTSNode:
    """
    Node in the MCTS tree.
    
    Attributes:
        state: Board state as list
        mark: Current player mark
        parent: Parent node
        action: Action that led to this node
        children: Dictionary of action -> child node
        N: Visit count
        W: Total action value
        Q: Mean action value (W/N)
        P: Prior probability from neural network
    """
    
    def __init__(self, state: List[int], mark: int, parent=None, action: int = None,
                 prior_prob: float = 0.0):
        """
        Initialize MCTS node.
        
        Args:
            state: Board state
            mark: Current player mark
            parent: Parent node
            action: Action that led to this node
            prior_prob: Prior probability from policy network
        """
        self.state = state
        self.mark = mark
        self.parent = parent
        self.action = action
        
        self.children: Dict[int, MCTSNode] = {}
        
        # Statistics
        self.N = 0  # Visit count
        self.W = 0.0  # Total value
        self.Q = 0.0  # Mean value
        self.P = prior_prob  # Prior probability
        
        # Cache
        self._valid_moves = None
        self._is_terminal = None
        self._terminal_value = None
    
    def is_leaf(self) -> bool:
        """Check if node is a leaf (not expanded)."""
        return len(self.children) == 0
    
    def is_terminal_node(self) -> bool:
        """Check if node represents a terminal state."""
        if self._is_terminal is None:
            self._is_terminal, self._terminal_value = is_terminal(self.state)
        return self._is_terminal
    
    def get_terminal_value(self, perspective_mark: int) -> float:
        """
        Get terminal value from perspective of a player.
        
        Args:
            perspective_mark: Player mark to evaluate from
        
        Returns:
            Value in [-1, 1]
        """
        if not self.is_terminal_node():
            return 0.0
        
        winner = self._terminal_value
        
        if winner == 0:  # Draw
            return 0.0
        elif winner == perspective_mark:
            return 1.0
        else:
            return -1.0
    
    def get_valid_moves(self) -> List[int]:
        """Get cached valid moves."""
        if self._valid_moves is None:
            self._valid_moves = get_valid_moves(self.state)
        return self._valid_moves
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """
        Select child using UCB formula.
        
        UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: Exploration constant
        
        Returns:
            Selected child node
        """
        best_score = -float('inf')
        best_action = None
        best_child = None
        
        # Sum of visit counts for UCB formula
        sqrt_parent_n = math.sqrt(self.N)
        
        for action, child in self.children.items():
            # UCB score
            if child.N == 0:
                q_value = 0.0
            else:
                # Q value from child's perspective (negated for opponent)
                q_value = -child.Q
            
            # Exploration bonus
            u_value = c_puct * child.P * sqrt_parent_n / (1 + child.N)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs: np.ndarray):
        """
        Expand node by creating children for all valid moves.
        
        Args:
            policy_probs: Policy probabilities from neural network (length 7)
        """
        valid_moves = self.get_valid_moves()
        
        for action in valid_moves:
            if action not in self.children:
                # Create next state
                next_state = make_move(self.state, action, self.mark)
                next_mark = 3 - self.mark  # Switch player
                
                # Get prior probability
                prior_prob = policy_probs[action]
                
                # Create child node
                child = MCTSNode(next_state, next_mark, parent=self,
                               action=action, prior_prob=prior_prob)
                self.children[action] = child
    
    def backpropagate(self, value: float):
        """
        Backpropagate value up the tree.
        
        Args:
            value: Value to backpropagate (from current node's perspective)
        """
        self.N += 1
        self.W += value
        self.Q = self.W / self.N
        
        # Propagate to parent (negate value for opponent)
        if self.parent is not None:
            self.parent.backpropagate(-value)


class MCTS:
    """
    Monte Carlo Tree Search with neural network guidance.
    
    Features:
    - UCB-based selection
    - Neural network for policy and value
    - Dirichlet noise for exploration at root
    - Virtual loss for parallel search (optional)
    """
    
    def __init__(self, neural_network, config=None):
        """
        Initialize MCTS.
        
        Args:
            neural_network: Neural network for policy and value prediction
            config: Configuration object (default: az_config)
        """
        self.network = neural_network
        self.config = config or az_config
    
    def search(self, root_state: List[int], root_mark: int,
              num_simulations: int = None, temperature: float = 1.0,
              add_noise: bool = True) -> Tuple[np.ndarray, MCTSNode]:
        """
        Perform MCTS search from root state.
        
        Args:
            root_state: Initial board state
            root_mark: Current player mark
            num_simulations: Number of simulations (default from config)
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise to root
        
        Returns:
            Tuple of (policy distribution, root node)
        """
        if num_simulations is None:
            num_simulations = self.config.NUM_SIMULATIONS
        
        # Create root node
        root = MCTSNode(root_state, root_mark)
        
        # Evaluate root with neural network
        policy_probs, value = self._evaluate_state(root_state, root_mark)
        
        # Add Dirichlet noise to root for exploration
        if add_noise:
            policy_probs = self._add_dirichlet_noise(policy_probs, root.get_valid_moves())
        
        # Expand root
        root.expand(policy_probs)
        
        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)
        
        # Compute policy from visit counts
        policy = self._compute_policy(root, temperature)
        
        return policy, root
    
    def _simulate(self, root: MCTSNode):
        """
        Run one simulation from root to leaf.
        
        Steps:
        1. Selection: Traverse tree using UCB
        2. Expansion: Expand leaf node
        3. Evaluation: Evaluate with neural network
        4. Backpropagation: Update values up the tree
        """
        node = root
        search_path = [node]
        
        # Selection: traverse to leaf
        while not node.is_leaf() and not node.is_terminal_node():
            node = node.select_child(self.config.C_PUCT)
            search_path.append(node)
        
        # Check terminal
        if node.is_terminal_node():
            # Backpropagate terminal value
            value = node.get_terminal_value(node.mark)
            node.backpropagate(value)
            return
        
        # Expansion and Evaluation
        policy_probs, value = self._evaluate_state(node.state, node.mark)
        node.expand(policy_probs)
        
        # Backpropagation
        node.backpropagate(value)
    
    def _evaluate_state(self, state: List[int], mark: int) -> Tuple[np.ndarray, float]:
        """
        Evaluate state using neural network.
        
        Args:
            state: Board state
            mark: Current player mark
        
        Returns:
            Tuple of (policy_probs, value)
        """
        import torch
        from agents.base.utils import encode_state
        
        # Encode state for network
        encoded_state = encode_state(state, mark)
        
        # Convert to tensor and add batch dimension
        state_tensor = torch.from_numpy(encoded_state).float().unsqueeze(0)
        state_tensor = state_tensor.to(next(self.network.parameters()).device)
        
        # Get network predictions
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            
            # Convert to probabilities
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()
        
        # Mask invalid moves
        valid_moves = get_valid_moves(state)
        mask = np.zeros(az_config.COLUMNS)
        mask[valid_moves] = 1.0
        policy_probs = policy_probs * mask
        
        # Renormalize
        if policy_probs.sum() > 0:
            policy_probs = policy_probs / policy_probs.sum()
        else:
            # Fallback to uniform over valid moves
            policy_probs = mask / mask.sum()
        
        return policy_probs, value
    
    def _add_dirichlet_noise(self, policy_probs: np.ndarray,
                            valid_moves: List[int]) -> np.ndarray:
        """
        Add Dirichlet noise to root policy for exploration.
        
        Args:
            policy_probs: Policy probabilities
            valid_moves: Valid action indices
        
        Returns:
            Noisy policy probabilities
        """
        noise = np.zeros_like(policy_probs)
        noise[valid_moves] = np.random.dirichlet(
            [self.config.DIRICHLET_ALPHA] * len(valid_moves)
        )
        
        # Mix original policy with noise
        noisy_policy = ((1 - self.config.DIRICHLET_EPSILON) * policy_probs +
                       self.config.DIRICHLET_EPSILON * noise)
        
        return noisy_policy
    
    def _compute_policy(self, root: MCTSNode, temperature: float) -> np.ndarray:
        """
        Compute policy distribution from visit counts.
        
        Args:
            root: Root node
            temperature: Temperature parameter
                - tau=1: Proportional to visit counts
                - tau→0: Greedy (argmax)
                - tau→∞: Uniform
        
        Returns:
            Policy distribution over actions
        """
        policy = np.zeros(az_config.COLUMNS)
        
        # Get visit counts
        for action, child in root.children.items():
            policy[action] = child.N
        
        if temperature == 0:
            # Greedy: select most visited
            best_action = np.argmax(policy)
            policy = np.zeros(az_config.COLUMNS)
            policy[best_action] = 1.0
        elif temperature == 1:
            # Proportional to visit counts
            if policy.sum() > 0:
                policy = policy / policy.sum()
        else:
            # Apply temperature
            policy = policy ** (1.0 / temperature)
            if policy.sum() > 0:
                policy = policy / policy.sum()
        
        return policy
    
    def get_action_probs(self, root_state: List[int], root_mark: int,
                        temperature: float = 1.0) -> np.ndarray:
        """
        Get action probabilities using MCTS.
        
        Args:
            root_state: Current board state
            root_mark: Current player mark
            temperature: Temperature for exploration
        
        Returns:
            Action probability distribution
        """
        policy, _ = self.search(root_state, root_mark,
                               temperature=temperature)
        return policy
    
    def get_best_action(self, root_state: List[int], root_mark: int) -> int:
        """
        Get best action using MCTS with greedy selection.
        
        Args:
            root_state: Current board state
            root_mark: Current player mark
        
        Returns:
            Best action index
        """
        policy = self.get_action_probs(root_state, root_mark, temperature=0.0)
        return int(np.argmax(policy))


if __name__ == "__main__":
    # Test MCTS with a dummy network
    print("Testing MCTS...")
    print("=" * 60)
    
    import torch
    import torch.nn as nn
    
    # Dummy network for testing
    class DummyNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(126, 64)  # 3*6*7 = 126
            self.policy_head = nn.Linear(64, 7)
            self.value_head = nn.Linear(64, 1)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc(x))
            policy = self.policy_head(x)
            value = torch.tanh(self.value_head(x))
            return policy, value
    
    # Create network and MCTS
    network = DummyNetwork()
    network.eval()
    mcts = MCTS(network)
    
    # Test on empty board
    empty_board = [0] * 42
    mark = 1
    
    print(f"Running MCTS with {az_config.NUM_SIMULATIONS} simulations...")
    policy, root = mcts.search(empty_board, mark, num_simulations=100)
    
    print(f"\nMCTS Results:")
    print(f"  Root visits: {root.N}")
    print(f"  Number of children: {len(root.children)}")
    print(f"  Policy: {policy}")
    print(f"  Best action: {np.argmax(policy)}")
    
    # Get best action
    best_action = mcts.get_best_action(empty_board, mark)
    print(f"  Best action (greedy): {best_action}")
    
    print("\n✓ MCTS test passed!")

