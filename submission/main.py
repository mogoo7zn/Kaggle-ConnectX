"""
Kaggle ConnectX Submission Agent
Hybrid DQN + Rule-based strategy for optimal play

Includes optional inference-time exploration controls via environment variables:
- INFER_EPS: probability of taking a random legal action (default: 0)
- INFER_TAU: temperature for softmax sampling over Q-values (default: None)
"""

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# Configuration
# ============================================================================

INFER_EPS = float(os.getenv("INFER_EPS", "0"))
INFER_TAU_ENV = os.getenv("INFER_TAU")
INFER_TAU = float(INFER_TAU_ENV) if INFER_TAU_ENV is not None else None


class Config:
    """Configuration constants"""

    ROWS = 6
    COLUMNS = 7
    INAROW = 4
    INPUT_CHANNELS = 3
    CONV_CHANNELS = [64, 128, 128]
    FC_HIDDEN = 256
    OUTPUT_SIZE = 7


config = Config()


# ============================================================================
# DQN Model Architecture
# ============================================================================


class DQNNetwork(nn.Module):
    """
    Deep Q-Network with CNN architecture
    Optimized for ConnectX board state evaluation
    """

    def __init__(self):
        super(DQNNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            config.INPUT_CHANNELS, config.CONV_CHANNELS[0], kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            config.CONV_CHANNELS[0], config.CONV_CHANNELS[1], kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            config.CONV_CHANNELS[1], config.CONV_CHANNELS[2], kernel_size=3, stride=1, padding=1
        )

        # Batch normalization
        self.bn1 = nn.BatchNorm2d(config.CONV_CHANNELS[0])
        self.bn2 = nn.BatchNorm2d(config.CONV_CHANNELS[1])
        self.bn3 = nn.BatchNorm2d(config.CONV_CHANNELS[2])

        # Fully connected layers
        conv_output_size = config.ROWS * config.COLUMNS * config.CONV_CHANNELS[2]
        self.fc1 = nn.Linear(conv_output_size, config.FC_HIDDEN)
        self.fc2 = nn.Linear(config.FC_HIDDEN, config.FC_HIDDEN // 2)
        self.fc3 = nn.Linear(config.FC_HIDDEN // 2, config.OUTPUT_SIZE)

        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# ============================================================================
# Utility Functions
# ============================================================================


def encode_state(board: List[int], mark: int) -> np.ndarray:
    """
    Encode board state as 3-channel tensor.

    Returns:
        3D numpy array of shape (3, rows, columns)
    """

    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)
    opponent_mark = 3 - mark

    # Create channels
    player_channel = (board_2d == mark).astype(np.float32)
    opponent_channel = (board_2d == opponent_mark).astype(np.float32)

    # Valid moves mask
    valid_moves = np.zeros((config.ROWS, config.COLUMNS), dtype=np.float32)
    for col in range(config.COLUMNS):
        if board_2d[0, col] == 0:
            valid_moves[:, col] = 1.0

    return np.stack([player_channel, opponent_channel, valid_moves], axis=0)


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
    opponent_mark = 3 - mark
    valid_moves = get_valid_moves(board)
    for col in valid_moves:
        next_board = make_move(board, col, opponent_mark)
        if check_winner(next_board, opponent_mark):
            return col
    return None


def detect_threats(board: List[int], mark: int) -> List[Tuple[int, int, str]]:
    """Detect 3-in-a-row threats with potential to make 4"""
    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)
    threats = []

    # Check horizontal threats
    for row in range(config.ROWS):
        for col in range(config.COLUMNS - config.INAROW + 1):
            window = board_2d[row, col : col + config.INAROW]
            if np.sum(window == mark) == 3 and np.sum(window == 0) == 1:
                threats.append((row, col, "horizontal"))

    # Check vertical threats
    for row in range(config.ROWS - config.INAROW + 1):
        for col in range(config.COLUMNS):
            window = board_2d[row : row + config.INAROW, col]
            if np.sum(window == mark) == 3 and np.sum(window == 0) == 1:
                threats.append((row, col, "vertical"))

    # Check diagonal threats (positive slope)
    for row in range(config.ROWS - config.INAROW + 1):
        for col in range(config.COLUMNS - config.INAROW + 1):
            window = [board_2d[row + i, col + i] for i in range(config.INAROW)]
            if window.count(mark) == 3 and window.count(0) == 1:
                threats.append((row, col, "diagonal_pos"))

    # Check diagonal threats (negative slope)
    for row in range(config.INAROW - 1, config.ROWS):
        for col in range(config.COLUMNS - config.INAROW + 1):
            window = [board_2d[row - i, col + i] for i in range(config.INAROW)]
            if window.count(mark) == 3 and window.count(0) == 1:
                threats.append((row, col, "diagonal_neg"))

    return threats


def find_threat_blocking_move(board: List[int], mark: int) -> Optional[int]:
    """Find move to block opponent's 3-in-a-row threats"""
    opponent_mark = 3 - mark
    valid_moves = get_valid_moves(board)
    threats = detect_threats(board, opponent_mark)

    board_2d = np.array(board).reshape(config.ROWS, config.COLUMNS)

    for threat_row, threat_col, direction in threats:
        if direction == "horizontal":
            for c in range(threat_col, min(threat_col + config.INAROW, config.COLUMNS)):
                if (
                    board_2d[threat_row, c] == 0
                    and (
                        threat_row == config.ROWS - 1
                        or board_2d[threat_row + 1, c] != 0
                    )
                    and c in valid_moves
                ):
                    return c

        elif direction == "vertical":
            for r in range(threat_row, min(threat_row + config.INAROW, config.ROWS)):
                if board_2d[r, threat_col] == 0 and threat_col in valid_moves:
                    return threat_col

        elif direction == "diagonal_pos":
            for i in range(config.INAROW):
                r, c = threat_row + i, threat_col + i
                if 0 <= r < config.ROWS and c < config.COLUMNS and board_2d[r, c] == 0:
                    if r == config.ROWS - 1 or board_2d[r + 1, c] != 0:
                        if c in valid_moves:
                            return c

        elif direction == "diagonal_neg":
            for i in range(config.INAROW):
                r, c = threat_row - i, threat_col + i
                if 0 <= r < config.ROWS and c < config.COLUMNS and board_2d[r, c] == 0:
                    if r == config.ROWS - 1 or board_2d[r + 1, c] != 0:
                        if c in valid_moves:
                            return c

    return None


# ============================================================================
# Agent Class
# ============================================================================


class HybridAgent:
    """
    Hybrid DQN + Rule-based agent for ConnectX.
    """

    def __init__(self):
        self.device = torch.device("cpu")  # Kaggle uses CPU
        self.model = DQNNetwork()
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model_loaded = False

    def load_model(self, model_path: str):
        """Load trained model weights"""
        try:
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
            return True
        except Exception:
            return False

    def _sample_action(self, q_values: np.ndarray, valid_moves: List[int]) -> Optional[int]:
        """Sample an action using epsilon and optional temperature."""
        if not valid_moves:
            return None

        # Epsilon-greedy exploration
        if INFER_EPS > 0 and random.random() < INFER_EPS:
            return random.choice(valid_moves)

        # Temperature sampling if provided
        if INFER_TAU is not None and INFER_TAU > 0:
            masked_q = np.full_like(q_values, float("-inf"))
            masked_q[valid_moves] = q_values[valid_moves]
            max_q = np.nanmax(masked_q)
            stable_q = masked_q - max_q
            exp_q = np.exp(stable_q / INFER_TAU)
            probs = exp_q[valid_moves] / np.sum(exp_q[valid_moves])
            return int(np.random.choice(valid_moves, p=probs))

        return int(np.argmax(q_values))

    def select_action(self, board: List[int], mark: int) -> int:
        """
        Select action using hybrid strategy.

        Priority:
        1. Take winning move
        2. Block opponent's win
        3. Block opponent's 3-in-a-row threat
        4. Use DQN Q-values with optional exploration
        5. Fallback to center preference
        """
        valid_moves = get_valid_moves(board)

        if not valid_moves:
            return 0

        if len(valid_moves) == 1:
            return valid_moves[0]

        # Rule 1: Take winning move
        winning_move = find_winning_move(board, mark)
        if winning_move is not None:
            return winning_move

        # Rule 2: Block opponent's winning move
        blocking_move = find_blocking_move(board, mark)
        if blocking_move is not None:
            return blocking_move

        # Rule 3: Block opponent's 3-in-a-row threat
        threat_block = find_threat_blocking_move(board, mark)
        if threat_block is not None:
            return threat_block

        # Rule 4: Use DQN if model is loaded
        if self.model_loaded:
            try:
                state = encode_state(board, mark)
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

                with torch.no_grad():
                    q_values = self.model(state_tensor).cpu().numpy()[0]

                # Mask invalid moves
                for col in range(config.COLUMNS):
                    if col not in valid_moves:
                        q_values[col] = float("-inf")

                action = self._sample_action(q_values, valid_moves)
                if action is not None and action in valid_moves:
                    return action
            except Exception:
                pass

        # Rule 5: Fallback to center preference
        center = config.COLUMNS // 2
        valid_moves_sorted = sorted(valid_moves, key=lambda x: abs(x - center))
        return valid_moves_sorted[0]


# ============================================================================
# Global Agent Instance
# ============================================================================


_agent = None


def get_agent():
    """Get or create global agent instance"""
    global _agent
    if _agent is None:
        _agent = HybridAgent()
        # Try to load model from common paths
        paths = [
            "/kaggle_simulations/agent/best_model.pth",
            "/kaggle/input/connectx-v1/best_model.pth",
            "best_model.pth",
            "./best_model.pth",
        ]
        for path in paths:
            if _agent.load_model(path):
                break
    return _agent


# ============================================================================
# Main Agent Function (Kaggle Entry Point)
# THIS MUST BE THE LAST 'def' IN THE FILE
# ============================================================================


def agent(observation, configuration):
    """
    Main agent function for Kaggle submission

    This is the entry point called by Kaggle environment
    """
    try:
        # Get agent instance
        my_agent = get_agent()

        # Extract board and mark
        board = observation.board
        mark = observation.mark

        # Select action
        action = my_agent.select_action(board, mark)

        # Validate action
        valid_moves = get_valid_moves(board)
        if action not in valid_moves:
            # Emergency fallback
            center = config.COLUMNS // 2
            action = center if center in valid_moves else valid_moves[0]

        return int(action)

    except Exception:
        # Emergency fallback - return center or first valid column
        valid_moves = [col for col in range(config.COLUMNS) if observation.board[col] == 0]
        if valid_moves:
            center = config.COLUMNS // 2
            return center if center in valid_moves else valid_moves[0]
        return config.COLUMNS // 2
