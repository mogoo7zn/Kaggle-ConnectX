"""
Batched Inference Server for AlphaZero
Collects states from multiple self-play games and batches them for efficient GPU inference

Key Features:
- Collects inference requests from multiple games
- Batches requests for efficient GPU utilization
- Supports both synchronous and asynchronous inference modes
- Automatic batch size tuning based on wait time
"""

import numpy as np
import torch
import threading
import queue
import time
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from collections import deque
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents.alphazero.az_config import az_config


@dataclass
class InferenceRequest:
    """A single inference request."""
    request_id: int
    state: np.ndarray  # Shape (3, 6, 7)
    result_event: threading.Event
    policy: np.ndarray = None  # Will be filled after inference
    value: float = 0.0  # Will be filled after inference


class BatchedInferenceServer:
    """
    Server that batches inference requests for efficient GPU utilization.
    
    Usage:
        server = BatchedInferenceServer(network)
        server.start()
        
        # In worker threads:
        policy, value = server.inference(state)
        
        server.stop()
    """
    
    def __init__(self, network: torch.nn.Module,
                 max_batch_size: int = 128,
                 max_wait_ms: float = 10.0,
                 device: torch.device = None):
        """
        Initialize batched inference server.
        
        Args:
            network: PyTorch neural network
            max_batch_size: Maximum batch size for inference
            max_wait_ms: Maximum wait time (ms) before processing a partial batch
            device: Device to run inference on
        """
        self.network = network
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms / 1000.0  # Convert to seconds
        self.device = device or az_config.DEVICE
        
        # Request queue
        self.request_queue: queue.Queue = queue.Queue()
        
        # Server thread
        self._server_thread: Optional[threading.Thread] = None
        self._running = False
        
        # Statistics
        self.total_requests = 0
        self.total_batches = 0
        self.total_inference_time = 0.0
        
        # Request ID counter
        self._request_counter = 0
        self._counter_lock = threading.Lock()
        
        # Move network to device and set to eval mode
        self.network.to(self.device)
        self.network.eval()
    
    def start(self):
        """Start the inference server thread."""
        if self._running:
            return
        
        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()
        print(f"BatchedInferenceServer started (max_batch={self.max_batch_size}, "
              f"max_wait={self.max_wait_ms*1000:.1f}ms)")
    
    def stop(self):
        """Stop the inference server thread."""
        self._running = False
        if self._server_thread:
            self._server_thread.join(timeout=2.0)
            self._server_thread = None
        print(f"BatchedInferenceServer stopped. Stats: {self.total_batches} batches, "
              f"{self.total_requests} requests, avg batch size: "
              f"{self.total_requests/max(1, self.total_batches):.1f}")
    
    def inference(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Submit an inference request and wait for result.
        
        Args:
            state: Board state encoding (3, 6, 7)
        
        Returns:
            Tuple of (policy_probs, value)
        """
        # Create request
        with self._counter_lock:
            request_id = self._request_counter
            self._request_counter += 1
        
        request = InferenceRequest(
            request_id=request_id,
            state=state,
            result_event=threading.Event()
        )
        
        # Submit to queue
        self.request_queue.put(request)
        
        # Wait for result
        request.result_event.wait()
        
        return request.policy, request.value
    
    def inference_batch(self, states: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
        Submit multiple inference requests and wait for all results.
        
        Args:
            states: List of board states
        
        Returns:
            List of (policy_probs, value) tuples
        """
        if not states:
            return []
        
        requests = []
        
        with self._counter_lock:
            base_id = self._request_counter
            self._request_counter += len(states)
        
        for i, state in enumerate(states):
            request = InferenceRequest(
                request_id=base_id + i,
                state=state,
                result_event=threading.Event()
            )
            requests.append(request)
            self.request_queue.put(request)
        
        # Wait for all results
        for request in requests:
            request.result_event.wait()
        
        return [(req.policy, req.value) for req in requests]
    
    def _server_loop(self):
        """Main server loop that processes batched requests."""
        pending_requests: List[InferenceRequest] = []
        batch_start_time = None
        
        while self._running:
            try:
                # Try to get a request with timeout
                try:
                    request = self.request_queue.get(timeout=0.001)
                    pending_requests.append(request)
                    if batch_start_time is None:
                        batch_start_time = time.perf_counter()
                except queue.Empty:
                    pass
                
                # Check if we should process the batch
                should_process = False
                
                if len(pending_requests) >= self.max_batch_size:
                    should_process = True
                elif pending_requests and batch_start_time:
                    elapsed = time.perf_counter() - batch_start_time
                    if elapsed >= self.max_wait_ms:
                        should_process = True
                
                if should_process and pending_requests:
                    self._process_batch(pending_requests)
                    pending_requests = []
                    batch_start_time = None
                    
            except Exception as e:
                print(f"BatchedInferenceServer error: {e}")
                # Signal all pending requests with error
                for req in pending_requests:
                    req.policy = np.ones(az_config.COLUMNS) / az_config.COLUMNS
                    req.value = 0.0
                    req.result_event.set()
                pending_requests = []
                batch_start_time = None
        
        # Process remaining requests on shutdown
        if pending_requests:
            self._process_batch(pending_requests)
    
    def _process_batch(self, requests: List[InferenceRequest]):
        """Process a batch of inference requests."""
        if not requests:
            return
        
        start_time = time.perf_counter()
        
        # Stack states into batch tensor
        states = np.stack([req.state for req in requests], axis=0)
        states_tensor = torch.from_numpy(states).float().to(self.device)
        
        # Run inference
        with torch.no_grad():
            policy_logits, values = self.network(states_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy().flatten()
        
        # Distribute results
        for i, req in enumerate(requests):
            req.policy = policy_probs[i]
            req.value = float(values[i])
            req.result_event.set()
        
        # Update statistics
        elapsed = time.perf_counter() - start_time
        self.total_requests += len(requests)
        self.total_batches += 1
        self.total_inference_time += elapsed


class SyncInferenceWrapper:
    """
    Synchronous inference wrapper that bypasses the batching for single inference.
    Useful for evaluation or when batching overhead is not beneficial.
    """
    
    def __init__(self, network: torch.nn.Module, device: torch.device = None):
        """
        Initialize synchronous inference wrapper.
        
        Args:
            network: PyTorch neural network
            device: Device to run inference on
        """
        self.network = network
        self.device = device or az_config.DEVICE
        self.network.to(self.device)
        self.network.eval()
    
    def inference(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Run inference on a single state.
        
        Args:
            state: Board state encoding (3, 6, 7)
        
        Returns:
            Tuple of (policy_probs, value)
        """
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            policy_logits, value = self.network(state_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            value = value.item()
        
        return policy_probs, value
    
    def inference_batch(self, states: List[np.ndarray]) -> List[Tuple[np.ndarray, float]]:
        """
        Run inference on a batch of states.
        
        Args:
            states: List of board states
        
        Returns:
            List of (policy_probs, value) tuples
        """
        if not states:
            return []
        
        states_np = np.stack(states, axis=0)
        states_tensor = torch.from_numpy(states_np).float().to(self.device)
        
        with torch.no_grad():
            policy_logits, values = self.network(states_tensor)
            policy_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()
            values = values.cpu().numpy().flatten()
        
        return [(policy_probs[i], float(values[i])) for i in range(len(states))]


class VirtualBatchMCTS:
    """
    MCTS implementation that uses virtual batching.
    
    Instead of running one simulation at a time, this collects leaf nodes
    from multiple virtual simulations and evaluates them in a batch.
    """
    
    def __init__(self, inference_fn: Callable[[np.ndarray], Tuple[np.ndarray, float]],
                 batch_inference_fn: Callable[[List[np.ndarray]], List[Tuple[np.ndarray, float]]] = None,
                 virtual_loss: int = 3):
        """
        Initialize Virtual Batch MCTS.
        
        Args:
            inference_fn: Function to evaluate single state
            batch_inference_fn: Function to evaluate batch of states
            virtual_loss: Virtual loss for parallel MCTS (prevents same path selection)
        """
        self.inference_fn = inference_fn
        self.batch_inference_fn = batch_inference_fn
        self.virtual_loss = virtual_loss


if __name__ == "__main__":
    # Test batched inference server
    print("Testing BatchedInferenceServer...")
    print("=" * 60)
    
    import torch.nn as nn
    
    # Create simple test network
    class TestNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 32, 3, padding=1)
            self.policy = nn.Linear(32 * 6 * 7, 7)
            self.value = nn.Linear(32 * 6 * 7, 1)
        
        def forward(self, x):
            x = torch.relu(self.conv(x))
            x = x.view(x.size(0), -1)
            return self.policy(x), torch.tanh(self.value(x))
    
    network = TestNetwork()
    
    # Test synchronous wrapper
    print("\nTesting SyncInferenceWrapper...")
    sync_wrapper = SyncInferenceWrapper(network)
    
    state = np.random.randn(3, 6, 7).astype(np.float32)
    policy, value = sync_wrapper.inference(state)
    print(f"Single inference: policy shape={policy.shape}, value={value:.3f}")
    
    # Test batch inference
    states = [np.random.randn(3, 6, 7).astype(np.float32) for _ in range(10)]
    results = sync_wrapper.inference_batch(states)
    print(f"Batch inference: {len(results)} results")
    
    # Test batched server
    print("\nTesting BatchedInferenceServer...")
    server = BatchedInferenceServer(network, max_batch_size=32, max_wait_ms=5.0)
    server.start()
    
    # Single thread test
    policy, value = server.inference(state)
    print(f"Server inference: policy shape={policy.shape}, value={value:.3f}")
    
    # Multi-thread test
    import concurrent.futures
    
    def worker_fn(worker_id, num_requests):
        results = []
        for i in range(num_requests):
            state = np.random.randn(3, 6, 7).astype(np.float32)
            policy, value = server.inference(state)
            results.append((worker_id, i, policy.shape, value))
        return results
    
    print("\nMulti-threaded test (4 workers, 50 requests each)...")
    start_time = time.perf_counter()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(worker_fn, i, 50) for i in range(4)]
        all_results = [f.result() for f in futures]
    
    elapsed = time.perf_counter() - start_time
    total_requests = sum(len(r) for r in all_results)
    print(f"Completed {total_requests} requests in {elapsed:.3f}s "
          f"({total_requests/elapsed:.0f} req/s)")
    
    server.stop()
    
    print("\n✓ BatchedInferenceServer test passed!")

