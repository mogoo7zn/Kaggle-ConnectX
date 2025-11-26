"""
Prioritized Experience Replay Buffer
Implements Sum Tree for efficient priority-based sampling
Based on: https://arxiv.org/abs/1511.05952
"""

import numpy as np
import random
from typing import Tuple, List


class SumTree:
    """
    Binary tree data structure for efficient priority-based sampling.
    
    Structure:
        - Leaf nodes store priorities
        - Internal nodes store sum of children priorities
        - Root node stores total sum
    
    This allows O(log n) sampling and O(log n) priority updates.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize Sum Tree.
        
        Args:
            capacity: Maximum number of leaf nodes (must be power of 2 for efficiency)
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)  # Internal + leaf nodes
        self.data = np.zeros(capacity, dtype=object)  # Store actual transitions
        self.data_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """
        Propagate priority change up the tree.
        
        Args:
            idx: Index of the changed node
            change: Amount of change in priority
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """
        Find sample index from priority sum.
        
        Args:
            idx: Current node index
            s: Target cumulative priority sum
        
        Returns:
            Leaf node index containing the target sum
        """
        left = 2 * idx + 1
        right = left + 1
        
        # If leaf node, return
        if left >= len(self.tree):
            return idx
        
        # Recurse based on cumulative sum
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """Return total priority (root node value)."""
        return self.tree[0]
    
    def add(self, priority: float, data: object):
        """
        Add new data with priority.
        
        Args:
            priority: Priority value (higher = more important)
            data: Transition data to store
        """
        # Calculate tree index for this data
        idx = self.data_pointer + self.capacity - 1
        
        # Store data and update tree
        self.data[self.data_pointer] = data
        self.update(idx, priority)
        
        # Move pointer (circular buffer)
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx: int, priority: float):
        """
        Update priority of a leaf node.
        
        Args:
            idx: Tree index (not data index)
            priority: New priority value
        """
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)
    
    def get(self, s: float) -> Tuple[int, float, object]:
        """
        Retrieve data based on priority sum.
        
        Args:
            s: Target cumulative priority sum
        
        Returns:
            Tuple of (tree_idx, priority, data)
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[data_idx])


class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer using Sum Tree.
    
    Features:
    - Priority-based sampling (important transitions sampled more frequently)
    - Importance sampling weights to correct bias
    - Efficient O(log n) operations
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta_start: float = 0.4,
                 beta_frames: int = 100000, epsilon: float = 1e-6):
        """
        Initialize Prioritized Replay Buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta_start: Initial importance sampling weight exponent
            beta_frames: Number of frames to anneal beta to 1.0
            epsilon: Small constant to ensure non-zero priority
        """
        self.tree = SumTree(capacity)
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon
        self.frame = 0
        self.max_priority = 1.0  # Track maximum priority seen
    
    def _get_priority(self, td_error: float) -> float:
        """
        Calculate priority from TD error.
        
        Args:
            td_error: Temporal difference error
        
        Returns:
            Priority value
        """
        return (abs(td_error) + self.epsilon) ** self.alpha
    
    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Add transition to buffer with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        transition = (state, action, reward, next_state, done)
        
        # New transitions get maximum priority for exploration
        self.tree.add(self.max_priority, transition)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """
        Sample batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, 
                     indices, weights)
        """
        batch = []
        indices = []
        priorities = []
        
        # Divide priority range into batch_size segments
        segment = self.tree.total() / batch_size
        
        # Update beta (anneal toward 1.0)
        self.beta = min(1.0, self.beta_start + self.frame * 
                       (1.0 - self.beta_start) / self.beta_frames)
        self.frame += 1
        
        # Sample one transition from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            s = random.uniform(a, b)
            
            idx, priority, data = self.tree.get(s)
            
            if data is not None:
                batch.append(data)
                indices.append(idx)
                priorities.append(priority)
        
        # Handle case where buffer not yet full
        if len(batch) < batch_size:
            # Pad with random samples
            for _ in range(batch_size - len(batch)):
                idx = random.randint(0, self.tree.n_entries - 1)
                tree_idx = idx + self.tree.capacity - 1
                priority = self.tree.tree[tree_idx]
                data = self.tree.data[idx]
                
                batch.append(data)
                indices.append(tree_idx)
                priorities.append(priority)
        
        # Calculate importance sampling weights
        sampling_probabilities = np.array(priorities) / self.tree.total()
        
        # Avoid division by zero
        sampling_probabilities = np.maximum(sampling_probabilities, 1e-10)
        
        weights = (self.tree.n_entries * sampling_probabilities) ** (-self.beta)
        weights /= weights.max()  # Normalize for stability
        
        # Unzip batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
            np.array(indices),
            np.array(weights, dtype=np.float32)
        )
    
    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.
        
        Args:
            indices: Tree indices of transitions to update
            td_errors: TD errors for each transition
        """
        for idx, td_error in zip(indices, td_errors):
            priority = self._get_priority(td_error)
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        """Return current buffer size."""
        return self.tree.n_entries
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples.
        
        Args:
            min_size: Minimum required samples
        
        Returns:
            True if buffer size >= min_size
        """
        return len(self) >= min_size


if __name__ == "__main__":
    # Test the prioritized replay buffer
    print("Testing Prioritized Replay Buffer...")
    
    buffer = PrioritizedReplayBuffer(capacity=100, alpha=0.6, beta_start=0.4)
    
    # Add some dummy transitions
    for i in range(50):
        state = np.random.rand(3, 6, 7)
        action = np.random.randint(0, 7)
        reward = np.random.randn()
        next_state = np.random.rand(3, 6, 7)
        done = False
        
        buffer.push(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    print(f"Total priority: {buffer.tree.total():.2f}")
    
    # Sample a batch
    batch = buffer.sample(batch_size=32)
    states, actions, rewards, next_states, dones, indices, weights = batch
    
    print(f"\nSampled batch:")
    print(f"  States shape: {states.shape}")
    print(f"  Actions shape: {actions.shape}")
    print(f"  Weights shape: {weights.shape}")
    print(f"  Weights range: [{weights.min():.3f}, {weights.max():.3f}]")
    print(f"  Beta: {buffer.beta:.3f}")
    
    # Update priorities with dummy TD errors
    td_errors = np.abs(np.random.randn(32))
    buffer.update_priorities(indices, td_errors)
    
    print(f"\nAfter priority update:")
    print(f"  Max priority: {buffer.max_priority:.3f}")
    
    print("\n✓ Prioritized Replay Buffer test passed!")

