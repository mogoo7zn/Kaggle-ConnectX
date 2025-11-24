"""
Utility functions for ConnectX DQN Agent
Includes state encoding, board manipulation, and helper functions
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from config import config


def encode_state(board: List[int], mark: int) -> np.ndarray:
    """
    Encode the board state as a 3-channel image tensor.
    
    Args:
        board: Flattened board state (list of length rows*columns)
        mark: Current player's mark (1 or 2)
    
    Returns:
        3D numpy array of shape (3, rows, columns)
        - Channel 0: Current player's pieces (1s where player has pieces, 0s elsewhere)
        - Channel 1: Opponent's pieces (1s where opponent has pieces, 0s elsewhere)
        - Channel 2: Valid moves mask (1s for valid columns, 0s elsewhere)
    """
    rows, cols = config.ROWS, config.COLUMNS
    
    # Reshape board to 2D
    board_2d = np.array(board).reshape(rows, cols)
    
    # Determine opponent mark
    opponent_mark = 3 - mark  # if mark=1, opponent=2; if mark=2, opponent=1
    
    # Create three channels
    player_channel = (board_2d == mark).astype(np.float32)
    opponent_channel = (board_2d == opponent_mark).astype(np.float32)
    
    # Valid moves: check if top row of each column is empty
    valid_moves = np.zeros((rows, cols), dtype=np.float32)
    for col in range(cols):
        if board_2d[0, col] == 0:  # top row is empty
            valid_moves[:, col] = 1.0
    
    # Stack channels
    state = np.stack([player_channel, opponent_channel, valid_moves], axis=0)
    
    return state


def get_valid_moves(board: List[int]) -> List[int]:
    """
    Get list of valid column indices where a piece can be placed.
    
    Args:
        board: Flattened board state
    
    Returns:
        List of valid column indices
    """
    cols = config.COLUMNS
    valid = []
    for col in range(cols):
        if board[col] == 0:  # top position is empty
            valid.append(col)
    return valid


def is_valid_move(board: List[int], col: int) -> bool:
    """
    Check if placing a piece in the given column is valid.
    
    Args:
        board: Flattened board state
        col: Column index to check
    
    Returns:
        True if move is valid, False otherwise
    """
    return board[col] == 0


def make_move(board: List[int], col: int, mark: int) -> List[int]:
    """
    Make a move on the board (returns a new board, doesn't modify original).
    
    Args:
        board: Flattened board state
        col: Column to place piece in
        mark: Player's mark (1 or 2)
    
    Returns:
        New board state after move
    """
    rows, cols = config.ROWS, config.COLUMNS
    board = board.copy()
    
    # Find the lowest empty row in the column
    for row in range(rows - 1, -1, -1):
        idx = row * cols + col
        if board[idx] == 0:
            board[idx] = mark
            break
    
    return board


def check_winner(board: List[int], mark: int) -> bool:
    """
    Check if the given player has won the game.
    
    Args:
        board: Flattened board state
        mark: Player's mark to check for win
    
    Returns:
        True if player has won, False otherwise
    """
    rows, cols = config.ROWS, config.COLUMNS
    inarow = config.INAROW
    board_2d = np.array(board).reshape(rows, cols)
    
    # Check horizontal
    for row in range(rows):
        for col in range(cols - inarow + 1):
            if all(board_2d[row, col + i] == mark for i in range(inarow)):
                return True
    
    # Check vertical
    for row in range(rows - inarow + 1):
        for col in range(cols):
            if all(board_2d[row + i, col] == mark for i in range(inarow)):
                return True
    
    # Check diagonal (positive slope)
    for row in range(rows - inarow + 1):
        for col in range(cols - inarow + 1):
            if all(board_2d[row + i, col + i] == mark for i in range(inarow)):
                return True
    
    # Check diagonal (negative slope)
    for row in range(inarow - 1, rows):
        for col in range(cols - inarow + 1):
            if all(board_2d[row - i, col + i] == mark for i in range(inarow)):
                return True
    
    return False


def is_terminal(board: List[int]) -> Tuple[bool, Optional[int]]:
    """
    Check if the game has ended.
    
    Args:
        board: Flattened board state
    
    Returns:
        Tuple of (is_terminal, winner)
        - is_terminal: True if game has ended
        - winner: 1 or 2 if a player won, 0 for draw, None if game continues
    """
    # Check if either player won
    if check_winner(board, 1):
        return True, 1
    if check_winner(board, 2):
        return True, 2
    
    # Check if board is full (draw)
    if all(cell != 0 for cell in board):
        return True, 0
    
    return False, None


def board_to_string(board: List[int]) -> str:
    """
    Convert board to a human-readable string representation.
    
    Args:
        board: Flattened board state
    
    Returns:
        String representation of the board
    """
    rows, cols = config.ROWS, config.COLUMNS
    board_2d = np.array(board).reshape(rows, cols)
    
    symbols = {0: '.', 1: 'X', 2: 'O'}
    lines = []
    lines.append('  ' + ' '.join(str(i) for i in range(cols)))
    lines.append('  ' + '-' * (cols * 2 - 1))
    
    for row in range(rows):
        line = str(row) + ' ' + ' '.join(symbols[board_2d[row, col]] for col in range(cols))
        lines.append(line)
    
    return '\n'.join(lines)


def state_to_tensor(state: np.ndarray, device: torch.device = None) -> torch.Tensor:
    """
    Convert numpy state to PyTorch tensor.
    
    Args:
        state: Numpy array of shape (3, rows, cols)
        device: Device to put tensor on
    
    Returns:
        PyTorch tensor on specified device
    """
    if device is None:
        device = config.DEVICE
    
    tensor = torch.from_numpy(state).float()
    return tensor.to(device)


def calculate_reward(board: List[int], action: int, mark: int, 
                     next_board: List[int], done: bool, winner: Optional[int]) -> float:
    """
    Calculate reward for a state-action transition.
    
    Args:
        board: Current board state
        action: Action taken
        mark: Current player's mark
        next_board: Next board state
        done: Whether episode ended
        winner: Winner of the game (if done)
    
    Returns:
        Reward value
    """
    # Check if action was invalid
    if not is_valid_move(board, action):
        return config.REWARD_INVALID
    
    if done:
        if winner == mark:
            return config.REWARD_WIN
        elif winner == 0:
            return config.REWARD_DRAW
        else:
            return config.REWARD_LOSS
    
    return config.REWARD_STEP


def find_winning_move(board: List[int], mark: int) -> Optional[int]:
    """
    Find a move that immediately wins the game for the given player.
    
    Args:
        board: Current board state
        mark: Player's mark
    
    Returns:
        Column index that wins the game, or None if no winning move exists
    """
    valid_moves = get_valid_moves(board)
    
    for col in valid_moves:
        next_board = make_move(board, col, mark)
        if check_winner(next_board, mark):
            return col
    
    return None


def find_blocking_move(board: List[int], mark: int) -> Optional[int]:
    """
    Find a move that blocks the opponent from winning on their next turn.
    
    Args:
        board: Current board state
        mark: Current player's mark
    
    Returns:
        Column index to block opponent's win, or None if no blocking needed
    """
    opponent_mark = 3 - mark
    valid_moves = get_valid_moves(board)
    
    for col in valid_moves:
        next_board = make_move(board, col, opponent_mark)
        if check_winner(next_board, opponent_mark):
            return col
    
    return None


def count_consecutive(board_2d: np.ndarray, row: int, col: int, 
                     delta_row: int, delta_col: int, mark: int) -> int:
    """
    Count consecutive pieces in a given direction from a starting position.
    
    Args:
        board_2d: 2D board array
        row: Starting row
        col: Starting column
        delta_row: Row direction (-1, 0, 1)
        delta_col: Column direction (-1, 0, 1)
        mark: Player's mark to count
    
    Returns:
        Number of consecutive pieces in the direction
    """
    rows, cols = board_2d.shape
    count = 0
    
    while 0 <= row < rows and 0 <= col < cols:
        if board_2d[row, col] == mark:
            count += 1
            row += delta_row
            col += delta_col
        else:
            break
    
    return count


def detect_threats(board: List[int], mark: int) -> List[Tuple[int, int, str]]:
    """
    Detect all positions where the player has 3 in a row with potential to make 4.
    
    Args:
        board: Current board state
        mark: Player's mark to check for threats
    
    Returns:
        List of tuples (row, col, direction) indicating threat positions
        direction can be 'horizontal', 'vertical', 'diagonal_pos', 'diagonal_neg'
    """
    rows, cols = config.ROWS, config.COLUMNS
    inarow = config.INAROW
    board_2d = np.array(board).reshape(rows, cols)
    threats = []
    
    # Check horizontal threats (3 in a row with room to expand)
    for row in range(rows):
        for col in range(cols - inarow + 1):
            window = board_2d[row, col:col + inarow]
            if np.sum(window == mark) == 3 and np.sum(window == 0) == 1:
                threats.append((row, col, 'horizontal'))
    
    # Check vertical threats (3 in a column with room to expand)
    for row in range(rows - inarow + 1):
        for col in range(cols):
            window = board_2d[row:row + inarow, col]
            if np.sum(window == mark) == 3 and np.sum(window == 0) == 1:
                threats.append((row, col, 'vertical'))
    
    # Check diagonal threats (positive slope)
    for row in range(rows - inarow + 1):
        for col in range(cols - inarow + 1):
            window = [board_2d[row + i, col + i] for i in range(inarow)]
            if window.count(mark) == 3 and window.count(0) == 1:
                threats.append((row, col, 'diagonal_pos'))
    
    # Check diagonal threats (negative slope)
    for row in range(inarow - 1, rows):
        for col in range(cols - inarow + 1):
            window = [board_2d[row - i, col + i] for i in range(inarow)]
            if window.count(mark) == 3 and window.count(0) == 1:
                threats.append((row, col, 'diagonal_neg'))
    
    return threats


def find_threat_blocking_move(board: List[int], mark: int) -> Optional[int]:
    """
    Find a move that blocks opponent's 3-in-a-row threat.
    
    Args:
        board: Current board state
        mark: Current player's mark
    
    Returns:
        Column index to block threat, or None if no immediate threat
    """
    opponent_mark = 3 - mark
    rows, cols = config.ROWS, config.COLUMNS
    inarow = config.INAROW
    board_2d = np.array(board).reshape(rows, cols)
    valid_moves = get_valid_moves(board)
    
    # Detect opponent threats
    threats = detect_threats(board, opponent_mark)
    
    if not threats:
        return None
    
    # For each threat, find which column would block it
    for threat_row, threat_col, direction in threats:
        if direction == 'horizontal':
            # Check each position in the horizontal window
            for c in range(threat_col, min(threat_col + inarow, cols)):
                if board_2d[threat_row, c] == 0:
                    # Check if this column is valid (has support below or is bottom row)
                    if threat_row == rows - 1 or board_2d[threat_row + 1, c] != 0:
                        if c in valid_moves:
                            return c
        
        elif direction == 'vertical':
            # For vertical threats, block the top empty position
            for r in range(threat_row, min(threat_row + inarow, rows)):
                if board_2d[r, threat_col] == 0:
                    # The lowest empty position in this column
                    if threat_col in valid_moves:
                        return threat_col
                    break
        
        elif direction == 'diagonal_pos':
            # Check each position in diagonal
            for i in range(inarow):
                r, c = threat_row + i, threat_col + i
                if r < rows and c < cols and board_2d[r, c] == 0:
                    # Check if this position is playable
                    if r == rows - 1 or board_2d[r + 1, c] != 0:
                        if c in valid_moves:
                            return c
        
        elif direction == 'diagonal_neg':
            # Check each position in diagonal
            for i in range(inarow):
                r, c = threat_row - i, threat_col + i
                if 0 <= r < rows and c < cols and board_2d[r, c] == 0:
                    # Check if this position is playable
                    if r == rows - 1 or board_2d[r + 1, c] != 0:
                        if c in valid_moves:
                            return c
    
    return None


def get_negamax_move(board: List[int], mark: int, depth: int = 3) -> int:
    """
    Simple negamax agent for opponent play (heuristic-based).
    
    Args:
        board: Current board state
        mark: Player's mark
        depth: Search depth
    
    Returns:
        Best column to play
    """
    valid_moves = get_valid_moves(board)
    
    if not valid_moves:
        return 0
    
    # Check for immediate wins
    for col in valid_moves:
        next_board = make_move(board, col, mark)
        if check_winner(next_board, mark):
            return col
    
    # Check for blocking opponent wins
    opponent_mark = 3 - mark
    for col in valid_moves:
        next_board = make_move(board, col, opponent_mark)
        if check_winner(next_board, opponent_mark):
            return col
    
    # Simple heuristic: prefer center columns
    cols = config.COLUMNS
    center = cols // 2
    
    # Sort moves by distance from center
    valid_moves_sorted = sorted(valid_moves, key=lambda x: abs(x - center))
    
    return valid_moves_sorted[0]


def create_directories():
    """Create necessary directories for saving models, logs, and plots."""
    import os
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)
    os.makedirs(config.PLOT_DIR, exist_ok=True)

