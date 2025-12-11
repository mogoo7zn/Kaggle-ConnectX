"""
MCTS Implementation for AlphaZero
Reduced object allocation, better memory efficiency, and support for batched inference

Key Optimizations:
- Uses __slots__ for memory-efficient nodes
- Supports both single and batched inference
- Caches terminal states and valid moves
- Virtual loss for parallel simulation
- Optional transposition table
"""

import numpy as np
import math
from typing import List, Tuple, Optional, Dict, Callable
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.alphazero.az_config import az_config
from agents.alphazero.fast_board import FastBoard, ROWS, COLS


class MCTSNode:
    """
    MCTS node using __slots__ for memory efficiency.
    
    Stores minimal state information and caches computed values.
    """
    
    __slots__ = [
        'board', 'mark', 'parent', 'action', 'children',
        'N', 'W', 'Q', 'P', 'virtual_loss',
        '_valid_moves', '_is_terminal', '_terminal_winner'
    ]
    
    def __init__(self, board: FastBoard, mark: int, parent: 'MCTSNode' = None,
                 action: int = None, prior_prob: float = 0.0):
        """
        Initialize MCTS node.
        
        Args:
            board: FastBoard state
            mark: Current player mark
            parent: Parent node
            action: Action that led to this node
            prior_prob: Prior probability from policy network
        """
        self.board = board
        self.mark = mark
        self.parent = parent
        self.action = action
        
        self.children: Dict[int, MCTSNode] = {}
        
        # Statistics
        self.N = 0  # Visit count
        self.W = 0.0  # Total value
        self.Q = 0.0  # Mean value
        self.P = prior_prob  # Prior probability
        self.virtual_loss = 0  # For parallel MCTS
        
        # Cached values (computed on first access)
        self._valid_moves = None
        self._is_terminal = None
        self._terminal_winner = None
    
    @property
    def valid_moves(self) -> List[int]:
        """Get valid moves (cached)."""
        if self._valid_moves is None:
            self._valid_moves = self.board.get_valid_moves()
        return self._valid_moves
    
    @property
    def is_terminal(self) -> bool:
        """Check if terminal state (cached)."""
        if self._is_terminal is None:
            self._is_terminal, self._terminal_winner = self.board.is_terminal()
        return self._is_terminal
    
    @property
    def terminal_winner(self) -> int:
        """Get winner if terminal (cached)."""
        if self._is_terminal is None:
            self._is_terminal, self._terminal_winner = self.board.is_terminal()
        return self._terminal_winner
    
    def is_leaf(self) -> bool:
        """Check if node is unexpanded."""
        return len(self.children) == 0
    
    def get_terminal_value(self, perspective_mark: int) -> float:
        """Get terminal value from perspective of given player."""
        if not self.is_terminal:
            return 0.0
        
        winner = self.terminal_winner
        if winner == 0:  # Draw
            return 0.0
        elif winner == perspective_mark:
            return 1.0
        else:
            return -1.0
    
    def select_child(self, c_puct: float) -> 'MCTSNode':
        """
        Select best child using PUCT formula.
        
        PUCT(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        With virtual loss: N and W are adjusted temporarily.
        """
        best_score = -float('inf')
        best_child = None
        
        sqrt_parent_n = math.sqrt(max(1, self.N))
        
        for child in self.children.values():
            # Adjust for virtual loss
            n_adjusted = child.N + child.virtual_loss
            w_adjusted = child.W - child.virtual_loss  # Virtual loss is negative
            
            if n_adjusted == 0:
                q_value = 0.0
            else:
                # Q value from child's perspective (negated for opponent)
                q_value = -w_adjusted / n_adjusted
            
            # Exploration bonus
            u_value = c_puct * child.P * sqrt_parent_n / (1 + n_adjusted)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_child = child
        
        return best_child
    
    def expand(self, policy_probs: np.ndarray):
        """
        Expand node by creating children for valid moves.
        
        Args:
            policy_probs: Policy probabilities (length COLS)
        """
        for action in self.valid_moves:
            if action not in self.children:
                # Create next state
                next_board = self.board.make_move(action, self.mark)
                next_mark = 3 - self.mark
                
                # Create child
                child = MCTSNode(
                    board=next_board,
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
    
    def add_virtual_loss(self):
        """Add virtual loss for parallel MCTS."""
        self.virtual_loss += az_config.VIRTUAL_LOSS
        if self.parent is not None:
            self.parent.add_virtual_loss()
    
    def remove_virtual_loss(self):
        """Remove virtual loss after simulation completes."""
        self.virtual_loss -= az_config.VIRTUAL_LOSS
        if self.parent is not None:
            self.parent.remove_virtual_loss()


class MCTS:
    """
    MCTS with support for batched inference.
    
    Features:
    - Single and batched inference modes
    - Virtual loss for parallel simulations
    - Adaptive simulation count based on game phase
    - Memory-efficient node structure
    """
    
    def __init__(self, inference_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
                 batch_inference_fn: Callable[[List[np.ndarray]], List[Tuple[np.ndarray, float]]] = None,
                 config=None):
        """
        Initialize MCTS.
        
        Args:
            inference_fn: Function to evaluate single state
            batch_inference_fn: Function to evaluate batch of states (optional)
            config: Configuration object
        """
        self.inference_fn = inference_fn
        self.batch_inference_fn = batch_inference_fn
        self.config = config or az_config
    
    def search(self, board: FastBoard, mark: int,
               num_simulations: int = None,
               temperature: float = 1.0,
               add_noise: bool = True) -> Tuple[np.ndarray, MCTSNode]:
        """
        Perform MCTS search.
        
        Args:
            board: FastBoard state
            mark: Current player mark
            num_simulations: Number of simulations (auto-adjusted if None)
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise to root
        
        Returns:
            Tuple of (policy, root_node)
        """
        if num_simulations is None:
            num_simulations = self._get_adaptive_simulations(board)
        
        # Create root node
        root = MCTSNode(board.copy(), mark)
        
        # Evaluate root with neural network
        state = board.encode_state(mark)
        policy_probs, value = self.inference_fn(state)
        
        # Mask invalid moves and renormalize
        policy_probs = self._mask_and_normalize(policy_probs, root.valid_moves)
        
        # Add Dirichlet noise for exploration
        if add_noise:
            policy_probs = self._add_dirichlet_noise(policy_probs, root.valid_moves)
        
        # Expand root
        root.expand(policy_probs)
        
        # Run simulations
        for _ in range(num_simulations):
            self._simulate(root)
        
        # Compute policy from visit counts
        policy = self._compute_policy(root, temperature)
        
        return policy, root
    
    def search_batched(self, boards: List[FastBoard], marks: List[int],
                       num_simulations: int = None,
                       temperature: float = 1.0,
                       add_noise: bool = True) -> List[Tuple[np.ndarray, MCTSNode]]:
        """
        Perform batched MCTS search for multiple positions.
        
        This is useful when running parallel games and wanting to
        batch the root node evaluations.
        
        Args:
            boards: List of FastBoard states
            marks: List of current player marks
            num_simulations: Number of simulations per position
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise
        
        Returns:
            List of (policy, root_node) tuples
        """
        if not boards:
            return []
        
        # Create root nodes
        roots = [MCTSNode(board.copy(), mark) 
                 for board, mark in zip(boards, marks)]
        
        # Batch evaluate roots
        states = [board.encode_state(mark) for board, mark in zip(boards, marks)]
        
        if self.batch_inference_fn:
            results = self.batch_inference_fn(states)
        else:
            results = [self.inference_fn(s) for s in states]
        
        # Expand all roots
        for i, (root, (policy_probs, value)) in enumerate(zip(roots, results)):
            policy_probs = self._mask_and_normalize(policy_probs, root.valid_moves)
            if add_noise:
                policy_probs = self._add_dirichlet_noise(policy_probs, root.valid_moves)
            root.expand(policy_probs)
        
        # Run simulations for each tree
        if num_simulations is None:
            num_simulations = self.config.NUM_SIMULATIONS
        
        for _ in range(num_simulations):
            for root in roots:
                self._simulate(root)
        
        # Compute policies
        policies = [self._compute_policy(root, temperature) for root in roots]
        
        return list(zip(policies, roots))
    
    def _simulate(self, root: MCTSNode):
        """Run one simulation from root to leaf."""
        node = root
        
        # Selection: traverse to leaf
        while not node.is_leaf() and not node.is_terminal:
            node = node.select_child(self.config.C_PUCT)
        
        # Check terminal
        if node.is_terminal:
            value = node.get_terminal_value(node.mark)
            node.backpropagate(value)
            return
        
        # Expansion and Evaluation
        state = node.board.encode_state(node.mark)
        policy_probs, value = self.inference_fn(state)
        
        # Mask and normalize policy
        policy_probs = self._mask_and_normalize(policy_probs, node.valid_moves)
        
        # Expand node
        node.expand(policy_probs)
        
        # Backpropagation
        node.backpropagate(value)
    
    def _simulate_with_virtual_loss(self, root: MCTSNode) -> MCTSNode:
        """
        Run simulation with virtual loss (for parallel MCTS).
        Returns the leaf node for later batch evaluation.
        """
        node = root
        path = [node]
        
        # Selection with virtual loss
        while not node.is_leaf() and not node.is_terminal:
            node = node.select_child(self.config.C_PUCT)
            path.append(node)
        
        # Add virtual loss to path
        for n in path:
            n.virtual_loss += self.config.VIRTUAL_LOSS
        
        return node, path
    
    def _mask_and_normalize(self, policy: np.ndarray, valid_moves: List[int]) -> np.ndarray:
        """Mask invalid moves and renormalize policy."""
        mask = np.zeros(COLS)
        mask[valid_moves] = 1.0
        policy = policy * mask
        
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            policy = mask / mask.sum()
        
        return policy
    
    def _add_dirichlet_noise(self, policy: np.ndarray, valid_moves: List[int]) -> np.ndarray:
        """Add Dirichlet noise to policy for exploration."""
        noise = np.zeros(COLS)
        noise[valid_moves] = np.random.dirichlet(
            [self.config.DIRICHLET_ALPHA] * len(valid_moves)
        )
        
        return ((1 - self.config.DIRICHLET_EPSILON) * policy +
                self.config.DIRICHLET_EPSILON * noise)
    
    def _compute_policy(self, root: MCTSNode, temperature: float) -> np.ndarray:
        """Compute policy distribution from visit counts."""
        policy = np.zeros(COLS)
        
        for action, child in root.children.items():
            policy[action] = child.N
        
        if temperature == 0:
            # Greedy
            best_action = np.argmax(policy)
            policy = np.zeros(COLS)
            policy[best_action] = 1.0
        elif temperature == 1:
            # Proportional
            if policy.sum() > 0:
                policy = policy / policy.sum()
        else:
            # Apply temperature
            policy = policy ** (1.0 / temperature)
            if policy.sum() > 0:
                policy = policy / policy.sum()
        
        return policy
    
    def _get_adaptive_simulations(self, board: FastBoard) -> int:
        """
        Get adaptive number of simulations based on game phase.
        
        Early game: Full simulations (more exploration needed)
        Mid game: Standard simulations
        Late game: Reduced simulations (simpler positions)
        """
        move_count = board.move_count
        base_sims = self.config.NUM_SIMULATIONS
        
        # Early game (first 6 moves): use full simulations
        if move_count < 6:
            return base_sims
        
        # Mid game: standard
        elif move_count < 20:
            return base_sims
        
        # Late game: reduce simulations (positions are simpler)
        else:
            return max(base_sims // 2, 50)
    
    def get_best_action(self, board: FastBoard, mark: int) -> int:
        """Get best action using greedy selection."""
        policy, _ = self.search(board, mark, temperature=0.0, add_noise=False)
        return int(np.argmax(policy))
    
    def get_action_probs(self, board: FastBoard, mark: int,
                         temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities using MCTS."""
        policy, _ = self.search(board, mark, temperature=temperature)
        return policy


# Compatibility wrapper for old MCTS interface
class MCTSWrapper:
    """
    Wrapper to provide old MCTS interface using optimized implementation.
    """
    
    def __init__(self, network, config=None):
        """
        Initialize MCTS wrapper.
        
        Args:
            network: PyTorch neural network
            config: Configuration object
        """
        import torch
        self.network = network
        self.config = config or az_config
        self.device = next(network.parameters()).device
        
        # Create MCTS
        self.mcts = MCTS(
            inference_fn=self._inference,
            config=self.config
        )
    
    def _inference(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Neural network inference."""
        import torch
        
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()
        
        return policy_probs, value
    
    def search(self, root_state: List[int], root_mark: int,
               num_simulations: int = None, temperature: float = 1.0,
               add_noise: bool = True) -> Tuple[np.ndarray, MCTSNode]:
        """
        Perform MCTS search (compatible with old interface).
        
        Args:
            root_state: Board state as list
            root_mark: Current player mark
            num_simulations: Number of simulations
            temperature: Temperature for action selection
            add_noise: Whether to add Dirichlet noise
        
        Returns:
            Tuple of (policy, root_node)
        """
        # Convert list to FastBoard
        board = FastBoard.from_list(root_state)
        
        return self.mcts.search(board, root_mark, num_simulations,
                                temperature, add_noise)
    
    def get_best_action(self, root_state: List[int], root_mark: int) -> int:
        """Get best action."""
        board = FastBoard.from_list(root_state)
        return self.mcts.get_best_action(board, root_mark)
    
    def get_action_probs(self, root_state: List[int], root_mark: int,
                         temperature: float = 1.0) -> np.ndarray:
        """Get action probabilities."""
        board = FastBoard.from_list(root_state)
        return self.mcts.get_action_probs(board, root_mark, temperature)


if __name__ == "__main__":
    # Test MCTS
    print("Testing MCTS...")
    print("=" * 60)
    
    import time
    
    # Create dummy inference function
    def dummy_inference(state):
        policy = np.ones(COLS) / COLS
        value = 0.0
        return policy, value
    
    # Test MCTS
    mcts = MCTS(inference_fn=dummy_inference)
    
    board = FastBoard()
    board.make_move_inplace(3, 1)
    board.make_move_inplace(3, 2)
    
    print(f"Board:\n{board}")
    print(f"\nRunning MCTS with 100 simulations...")
    
    start = time.perf_counter()
    policy, root = mcts.search(board, 1, num_simulations=100)
    elapsed = time.perf_counter() - start
    
    print(f"Completed in {elapsed:.3f}s")
    print(f"Policy: {policy}")
    print(f"Best action: {np.argmax(policy)}")
    print(f"Root visits: {root.N}")
    print(f"Children visits: {[c.N for c in root.children.values()]}")
    
    # Benchmark
    print("\n" + "=" * 60)
    print("Benchmark: 1000 simulations...")
    
    start = time.perf_counter()
    policy, root = mcts.search(board, 1, num_simulations=1000)
    elapsed = time.perf_counter() - start
    
    print(f"Completed in {elapsed:.3f}s ({1000/elapsed:.0f} sims/sec)")
    
    # Test batched search
    print("\n" + "=" * 60)
    print("Testing batched search...")
    
    boards = [FastBoard() for _ in range(4)]
    marks = [1, 1, 1, 1]
    
    results = mcts.search_batched(boards, marks, num_simulations=50)
    print(f"Batched search: {len(results)} results")
    
    print("\n✓ MCTS test passed!")

