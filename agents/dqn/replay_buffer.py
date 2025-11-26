"""
Experience Replay Buffer for DQN
Implements a circular buffer to store and sample transitions
"""

import numpy as np
import random
from typing import List, Tuple
from collections import deque


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling transitions.
    Uses a circular buffer (deque) for efficient memory management.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state (3, rows, cols)
            action: Action taken (column index)
            reward: Reward received
            next_state: Next state (3, rows, cols)
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray]:
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
            - states: (batch_size, 3, rows, cols)
            - actions: (batch_size,)
            - rewards: (batch_size,)
            - next_states: (batch_size, 3, rows, cols)
            - dones: (batch_size,)
        """
        # Sample random transitions
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack transitions
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return current size of buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all transitions from buffer."""
        self.buffer.clear()
    
    def is_ready(self, min_size: int) -> bool:
        """
        Check if buffer has enough samples for training.
        
        Args:
            min_size: Minimum number of samples required
        
        Returns:
            True if buffer has at least min_size samples
        """
        return len(self.buffer) >= min_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay Buffer (optional enhancement).
    Samples transitions based on their TD error priority.
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            alpha: Priority exponent (0 = uniform, 1 = fully prioritized)
            beta: Importance sampling exponent
        """
        super().__init__(capacity)
        self.alpha = alpha
        self.beta = beta
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add a transition with maximum priority.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        super().push(state, action, reward, next_state, done)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, 
                                                 np.ndarray, np.ndarray, np.ndarray, List[int]]:
        """
        Sample a batch of transitions based on priorities.
        
        Args:
            batch_size: Number of transitions to sample
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones, weights, indices)
        """
        # Calculate sampling probabilities
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices based on probabilities
        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        
        # Get transitions
        batch = [self.buffer[idx] for idx in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Calculate importance sampling weights
        weights = (len(self.buffer) * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize weights
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        weights = np.array(weights, dtype=np.float32)
        
        return states, actions, rewards, next_states, dones, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """
        Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priority values (typically TD errors)
        """
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def clear(self):
        """Clear all transitions and priorities."""
        super().clear()
        self.priorities.clear()
        self.max_priority = 1.0

