"""
Fast Board Representation for ConnectX
Uses compact numpy arrays and bitboards for efficient game state operations

Optimizations:
- Bitboard representation for fast win checking
- Numpy array operations
- Minimal memory allocation
- Numba JIT for critical paths (optional)
"""

import numpy as np
from typing import List, Tuple, Optional

# Board dimensions
ROWS = 6
COLS = 7
INAROW = 4

# Precompute bit masks for win checking
# Using 64-bit integers for bitboard representation
# Board layout (bit positions):
#  35 36 37 38 39 40 41
#  28 29 30 31 32 33 34
#  21 22 23 24 25 26 27
#  14 15 16 17 18 19 20
#   7  8  9 10 11 12 13
#   0  1  2  3  4  5  6

def _create_win_masks():
    """Precompute all winning 4-in-a-row patterns as bitmasks."""
    masks = []
    
    # Horizontal wins
    for row in range(ROWS):
        for col in range(COLS - 3):
            base = row * COLS + col
            mask = (1 << base) | (1 << (base + 1)) | (1 << (base + 2)) | (1 << (base + 3))
            masks.append(mask)
    
    # Vertical wins
    for row in range(ROWS - 3):
        for col in range(COLS):
            base = row * COLS + col
            mask = (1 << base) | (1 << (base + COLS)) | (1 << (base + 2*COLS)) | (1 << (base + 3*COLS))
            masks.append(mask)
    
    # Diagonal (positive slope) wins
    for row in range(ROWS - 3):
        for col in range(COLS - 3):
            base = row * COLS + col
            mask = (1 << base) | (1 << (base + COLS + 1)) | (1 << (base + 2*COLS + 2)) | (1 << (base + 3*COLS + 3))
            masks.append(mask)
    
    # Diagonal (negative slope) wins
    for row in range(3, ROWS):
        for col in range(COLS - 3):
            base = row * COLS + col
            mask = (1 << base) | (1 << (base - COLS + 1)) | (1 << (base - 2*COLS + 2)) | (1 << (base - 3*COLS + 3))
            masks.append(mask)
    
    return masks

WIN_MASKS = _create_win_masks()


class FastBoard:
    """
    Fast board implementation using bitboards.
    
    Uses two 64-bit integers to represent positions for each player.
    This allows O(1) win checking through bitmask operations.
    """
    
    __slots__ = ['p1_bits', 'p2_bits', 'heights', 'move_count']
    
    def __init__(self):
        """Initialize empty board."""
        self.p1_bits = 0  # Bitboard for player 1
        self.p2_bits = 0  # Bitboard for player 2
        self.heights = np.zeros(COLS, dtype=np.int8)  # Height of each column
        self.move_count = 0
    
    def copy(self) -> 'FastBoard':
        """Create a copy of the board (minimal allocation)."""
        new_board = FastBoard.__new__(FastBoard)
        new_board.p1_bits = self.p1_bits
        new_board.p2_bits = self.p2_bits
        new_board.heights = self.heights.copy()
        new_board.move_count = self.move_count
        return new_board
    
    def get_valid_moves(self) -> List[int]:
        """Get list of valid columns."""
        return [col for col in range(COLS) if self.heights[col] < ROWS]
    
    def get_valid_moves_mask(self) -> np.ndarray:
        """Get valid moves as a boolean mask."""
        return self.heights < ROWS
    
    def is_valid_move(self, col: int) -> bool:
        """Check if a move is valid."""
        return self.heights[col] < ROWS
    
    def make_move(self, col: int, mark: int) -> 'FastBoard':
        """Make a move and return new board (immutable style)."""
        new_board = self.copy()
        new_board.make_move_inplace(col, mark)
        return new_board
    
    def make_move_inplace(self, col: int, mark: int):
        """Make a move in place (mutates board)."""
        row = int(self.heights[col])  # Convert from numpy int8 to Python int
        bit_pos = row * COLS + col
        
        if mark == 1:
            self.p1_bits |= (1 << bit_pos)
        else:
            self.p2_bits |= (1 << bit_pos)
        
        self.heights[col] += 1
        self.move_count += 1
    
    def undo_move(self, col: int, mark: int):
        """Undo a move (for MCTS backtracking)."""
        self.heights[col] -= 1
        row = int(self.heights[col])  # Convert from numpy int8 to Python int
        bit_pos = row * COLS + col
        
        if mark == 1:
            self.p1_bits &= ~(1 << bit_pos)
        else:
            self.p2_bits &= ~(1 << bit_pos)
        
        self.move_count -= 1
    
    def check_win(self, mark: int) -> bool:
        """Check if the given player has won using bitboard."""
        bits = int(self.p1_bits if mark == 1 else self.p2_bits)
        
        for mask in WIN_MASKS:
            if (bits & mask) == mask:
                return True
        return False
    
    def check_win_fast(self, mark: int, last_col: int) -> bool:
        """
        Fast win check after a move - only check patterns involving the last move.
        """
        bits = int(self.p1_bits if mark == 1 else self.p2_bits)
        last_row = int(self.heights[last_col]) - 1
        
        # Check horizontal
        count = 1
        for c in range(last_col - 1, -1, -1):
            if bits & (1 << (last_row * COLS + c)):
                count += 1
            else:
                break
        for c in range(last_col + 1, COLS):
            if bits & (1 << (last_row * COLS + c)):
                count += 1
            else:
                break
        if count >= 4:
            return True
        
        # Check vertical (only down since we just placed)
        count = 1
        for r in range(last_row - 1, -1, -1):
            if bits & (1 << (r * COLS + last_col)):
                count += 1
            else:
                break
        if count >= 4:
            return True
        
        # Check diagonal (positive slope)
        count = 1
        r, c = last_row - 1, last_col - 1
        while r >= 0 and c >= 0:
            if bits & (1 << (r * COLS + c)):
                count += 1
                r -= 1
                c -= 1
            else:
                break
        r, c = last_row + 1, last_col + 1
        while r < ROWS and c < COLS:
            if bits & (1 << (r * COLS + c)):
                count += 1
                r += 1
                c += 1
            else:
                break
        if count >= 4:
            return True
        
        # Check diagonal (negative slope)
        count = 1
        r, c = last_row + 1, last_col - 1
        while r < ROWS and c >= 0:
            if bits & (1 << (r * COLS + c)):
                count += 1
                r += 1
                c -= 1
            else:
                break
        r, c = last_row - 1, last_col + 1
        while r >= 0 and c < COLS:
            if bits & (1 << (r * COLS + c)):
                count += 1
                r -= 1
                c += 1
            else:
                break
        if count >= 4:
            return True
        
        return False
    
    def is_draw(self) -> bool:
        """Check if game is a draw (board full)."""
        return self.move_count >= ROWS * COLS
    
    def is_terminal(self) -> Tuple[bool, int]:
        """
        Check if game is over.
        
        Returns:
            (is_terminal, winner) where winner is 0 for draw, 1 or 2 for player win
        """
        if self.check_win(1):
            return True, 1
        if self.check_win(2):
            return True, 2
        if self.is_draw():
            return True, 0
        return False, -1
    
    def to_list(self) -> List[int]:
        """Convert to list format (for compatibility)."""
        p1_bits = int(self.p1_bits)
        p2_bits = int(self.p2_bits)
        board = [0] * (ROWS * COLS)
        for i in range(ROWS * COLS):
            bit_mask = 1 << i
            if p1_bits & bit_mask:
                board[i] = 1
            elif p2_bits & bit_mask:
                board[i] = 2
        return board
    
    def to_numpy(self) -> np.ndarray:
        """Convert to numpy array format."""
        p1_bits = int(self.p1_bits)
        p2_bits = int(self.p2_bits)
        board = np.zeros((ROWS, COLS), dtype=np.int8)
        for row in range(ROWS):
            for col in range(COLS):
                bit_pos = row * COLS + col
                bit_mask = 1 << bit_pos
                if p1_bits & bit_mask:
                    board[row, col] = 1
                elif p2_bits & bit_mask:
                    board[row, col] = 2
        return board
    
    @staticmethod
    def from_list(board_list: List[int]) -> 'FastBoard':
        """Create FastBoard from list format."""
        fb = FastBoard()
        for i, val in enumerate(board_list):
            if val != 0:
                row = i // COLS
                col = i % COLS
                if val == 1:
                    fb.p1_bits |= (1 << i)
                else:
                    fb.p2_bits |= (1 << i)
                # Update heights
                if fb.heights[col] <= row:
                    fb.heights[col] = row + 1
                fb.move_count += 1
        return fb
    
    def encode_state(self, mark: int) -> np.ndarray:
        """
        Encode board state as 3-channel tensor for neural network.
        
        Channels:
        - 0: Current player's pieces
        - 1: Opponent's pieces
        - 2: Valid moves mask
        
        Returns:
            numpy array of shape (3, ROWS, COLS)
        """
        # Ensure bits are Python ints for bitwise operations
        p1_bits = int(self.p1_bits)
        p2_bits = int(self.p2_bits)
        
        state = np.zeros((3, ROWS, COLS), dtype=np.float32)
        
        for row in range(ROWS):
            for col in range(COLS):
                bit_pos = row * COLS + col
                bit_mask = 1 << bit_pos
                if mark == 1:
                    if p1_bits & bit_mask:
                        state[0, row, col] = 1.0
                    elif p2_bits & bit_mask:
                        state[1, row, col] = 1.0
                else:
                    if p2_bits & bit_mask:
                        state[0, row, col] = 1.0
                    elif p1_bits & bit_mask:
                        state[1, row, col] = 1.0
        
        # Valid moves channel
        for col in range(COLS):
            if self.heights[col] < ROWS:
                state[2, :, col] = 1.0
        
        return state
    
    def get_hash(self) -> int:
        """Get unique hash for this board state (for transposition table)."""
        return hash((self.p1_bits, self.p2_bits))
    
    def __eq__(self, other: 'FastBoard') -> bool:
        """Check equality."""
        return self.p1_bits == other.p1_bits and self.p2_bits == other.p2_bits
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        symbols = {0: '.', 1: 'X', 2: 'O'}
        lines = ['  ' + ' '.join(str(i) for i in range(COLS))]
        lines.append('  ' + '-' * (COLS * 2 - 1))
        
        board = self.to_numpy()
        for row in range(ROWS - 1, -1, -1):  # Print from top to bottom
            line = str(row) + ' ' + ' '.join(symbols[board[row, col]] for col in range(COLS))
            lines.append(line)
        
        return '\n'.join(lines)


# Optional: Numba-accelerated functions
try:
    from numba import jit, njit
    
    @njit(cache=True)
    def fast_check_win_numba(bits: int) -> bool:
        """Numba-accelerated win checking."""
        # Horizontal
        for row in range(6):
            for col in range(4):
                base = row * 7 + col
                mask = (1 << base) | (1 << (base + 1)) | (1 << (base + 2)) | (1 << (base + 3))
                if (bits & mask) == mask:
                    return True
        
        # Vertical
        for row in range(3):
            for col in range(7):
                base = row * 7 + col
                mask = (1 << base) | (1 << (base + 7)) | (1 << (base + 14)) | (1 << (base + 21))
                if (bits & mask) == mask:
                    return True
        
        # Diagonal positive
        for row in range(3):
            for col in range(4):
                base = row * 7 + col
                mask = (1 << base) | (1 << (base + 8)) | (1 << (base + 16)) | (1 << (base + 24))
                if (bits & mask) == mask:
                    return True
        
        # Diagonal negative
        for row in range(3, 6):
            for col in range(4):
                base = row * 7 + col
                mask = (1 << base) | (1 << (base - 6)) | (1 << (base - 12)) | (1 << (base - 18))
                if (bits & mask) == mask:
                    return True
        
        return False
    
    HAS_NUMBA = True
    print("Numba acceleration available for fast_board")
    
except ImportError:
    HAS_NUMBA = False
    fast_check_win_numba = None


if __name__ == "__main__":
    # Test FastBoard
    print("Testing FastBoard...")
    print("=" * 60)
    
    import time
    
    # Basic test
    board = FastBoard()
    print(f"Empty board:\n{board}")
    print(f"Valid moves: {board.get_valid_moves()}")
    
    # Make some moves
    board.make_move_inplace(3, 1)
    board.make_move_inplace(3, 2)
    board.make_move_inplace(4, 1)
    board.make_move_inplace(4, 2)
    board.make_move_inplace(5, 1)
    board.make_move_inplace(5, 2)
    board.make_move_inplace(6, 1)  # Should win
    
    print(f"\nAfter moves:\n{board}")
    print(f"Player 1 wins: {board.check_win(1)}")
    print(f"Player 2 wins: {board.check_win(2)}")
    
    # Performance test
    print("\n" + "=" * 60)
    print("Performance test: 100,000 win checks")
    
    board = FastBoard()
    board.make_move_inplace(0, 1)
    board.make_move_inplace(1, 1)
    board.make_move_inplace(2, 1)
    
    start = time.perf_counter()
    for _ in range(100000):
        board.check_win(1)
    elapsed = time.perf_counter() - start
    print(f"Bitboard check_win: {elapsed:.3f}s ({100000/elapsed:.0f} checks/sec)")
    
    # Test encoding
    board = FastBoard()
    board.make_move_inplace(3, 1)
    board.make_move_inplace(2, 2)
    
    state = board.encode_state(1)
    print(f"\nEncoded state shape: {state.shape}")
    print(f"Player channel sum: {state[0].sum()}")
    print(f"Opponent channel sum: {state[1].sum()}")
    print(f"Valid moves channel sum: {state[2].sum()}")
    
    # Convert test
    board_list = board.to_list()
    board2 = FastBoard.from_list(board_list)
    print(f"\nConversion test passed: {board == board2}")
    
    print("\n✓ FastBoard test passed!")

