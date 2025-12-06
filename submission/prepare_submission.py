"""
准备Kaggle提交文件的脚本
将模型权重硬编码到main.py中，生成单文件提交

使用方法:
    python prepare_submission.py --model alpha-zero-v1.pth --output main.py
"""

import os
import sys
import base64
import gzip
import argparse
from pathlib import Path
import torch


def compress_and_encode_model(model_path: str) -> str:
    """压缩并编码模型为base64字符串"""
    print(f"读取模型: {model_path}")
    
    with open(model_path, 'rb') as f:
        model_bytes = f.read()
    
    original_size = len(model_bytes)
    print(f"  原始大小: {original_size / 1024 / 1024:.2f} MB")
    
    # 使用gzip压缩
    compressed = gzip.compress(model_bytes, compresslevel=9)
    compressed_size = len(compressed)
    print(f"  压缩后大小: {compressed_size / 1024 / 1024:.2f} MB")
    print(f"  压缩率: {compressed_size / original_size * 100:.1f}%")
    
    # Base64编码
    encoded = base64.b64encode(compressed).decode('ascii')
    encoded_size = len(encoded)
    print(f"  Base64编码后: {encoded_size / 1024 / 1024:.2f} MB")
    
    return encoded


def generate_submission_code(encoded_model: str) -> str:
    """生成带有硬编码模型的提交代码"""
    
    # 将编码后的模型分成多行存储，避免单行太长
    line_length = 76
    model_lines = [encoded_model[i:i+line_length] for i in range(0, len(encoded_model), line_length)]
    model_string = '"\n"'.join(model_lines)
    
    submission_code = '''"""
Kaggle ConnectX Submission - AlphaZero MCTS + Neural Network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import base64
import gzip
import io
from typing import List, Tuple, Optional, Dict


# ============================================================================
# Configuration
# ============================================================================

ROWS = 6
COLUMNS = 7
INAROW = 4
NUM_SIMULATIONS = 200
C_PUCT = 1.5
NUM_RES_BLOCKS = 6
NUM_FILTERS = 96
POLICY_FILTERS = 32
VALUE_FILTERS = 32
VALUE_HIDDEN = 128


# ============================================================================
# Embedded Model (Base64 + Gzip)
# ============================================================================

_MODEL_DATA = (
"''' + model_string + '''"
)


# ============================================================================
# Utility Functions
# ============================================================================

def get_valid_moves(board):
    return [c for c in range(COLUMNS) if board[c] == 0]


def make_move(board, col, mark):
    board = board.copy()
    for row in range(ROWS - 1, -1, -1):
        if board[row * COLUMNS + col] == 0:
            board[row * COLUMNS + col] = mark
            break
    return board


def check_winner(board, mark):
    b = np.array(board).reshape(ROWS, COLUMNS)
    for r in range(ROWS):
        for c in range(COLUMNS - 3):
            if all(b[r, c + i] == mark for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLUMNS):
            if all(b[r + i, c] == mark for i in range(4)):
                return True
    for r in range(ROWS - 3):
        for c in range(COLUMNS - 3):
            if all(b[r + i, c + i] == mark for i in range(4)):
                return True
    for r in range(3, ROWS):
        for c in range(COLUMNS - 3):
            if all(b[r - i, c + i] == mark for i in range(4)):
                return True
    return False


def is_terminal(board):
    if check_winner(board, 1):
        return True, 1
    if check_winner(board, 2):
        return True, 2
    if all(board[c] != 0 for c in range(COLUMNS)):
        return True, 0
    return False, -1


def find_winning_move(board, mark):
    for col in get_valid_moves(board):
        if check_winner(make_move(board, col, mark), mark):
            return col
    return None


def find_blocking_move(board, mark):
    return find_winning_move(board, 3 - mark)


def is_losing_move(board, col, mark):
    return find_winning_move(make_move(board, col, mark), 3 - mark) is not None


def get_safe_moves(board, mark):
    return [c for c in get_valid_moves(board) if not is_losing_move(board, c, mark)]


def encode_state(board, mark):
    b = np.array(board).reshape(ROWS, COLUMNS)
    p = (b == mark).astype(np.float32)
    o = (b == 3 - mark).astype(np.float32)
    v = np.zeros((ROWS, COLUMNS), dtype=np.float32)
    for c in range(COLUMNS):
        if b[0, c] == 0:
            v[:, c] = 1.0
    return np.stack([p, o, v], axis=0)


# ============================================================================
# Neural Network
# ============================================================================

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        nf = NUM_FILTERS
        self.conv1 = nn.Conv2d(3, nf, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(nf)
        self.conv2 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(nf)
        self.conv3 = nn.Conv2d(nf, nf, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(nf)
        
        self.res_blocks = nn.ModuleList()
        for _ in range(NUM_RES_BLOCKS):
            self.res_blocks.append(nn.Sequential(
                nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                nn.BatchNorm2d(nf),
                nn.ReLU(),
                nn.Conv2d(nf, nf, 3, padding=1, bias=False),
                nn.BatchNorm2d(nf)
            ))
        
        self.policy_conv = nn.Conv2d(nf, POLICY_FILTERS, 1)
        self.policy_bn = nn.BatchNorm2d(POLICY_FILTERS)
        self.policy_fc = nn.Linear(POLICY_FILTERS * ROWS * COLUMNS, COLUMNS)
        
        self.value_conv = nn.Conv2d(nf, VALUE_FILTERS, 1)
        self.value_bn = nn.BatchNorm2d(VALUE_FILTERS)
        self.value_fc1 = nn.Linear(VALUE_FILTERS * ROWS * COLUMNS, VALUE_HIDDEN)
        self.value_fc2 = nn.Linear(VALUE_HIDDEN, 1)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        for block in self.res_blocks:
            x = F.relu(block(x) + x)
        
        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = self.policy_fc(p.view(p.size(0), -1))
        
        v = F.relu(self.value_bn(self.value_conv(x)))
        v = F.relu(self.value_fc1(v.view(v.size(0), -1)))
        v = torch.tanh(self.value_fc2(v))
        
        return p, v


# ============================================================================
# MCTS
# ============================================================================

class Node:
    __slots__ = ['state', 'mark', 'parent', 'action', 'children', 'N', 'W', 'Q', 'P', '_vm', '_term']
    
    def __init__(self, state, mark, parent=None, action=None, prior=0.0):
        self.state = state
        self.mark = mark
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = prior
        self._vm = None
        self._term = None
    
    @property
    def valid_moves(self):
        if self._vm is None:
            self._vm = get_valid_moves(self.state)
        return self._vm
    
    @property
    def terminal(self):
        if self._term is None:
            self._term = is_terminal(self.state)
        return self._term
    
    def select(self, c):
        best_s, best_c = -1e9, None
        sqn = math.sqrt(max(1, self.N))
        for child in self.children.values():
            q = -child.Q if child.N > 0 else 0.0
            u = c * child.P * sqn / (1 + child.N)
            if q + u > best_s:
                best_s, best_c = q + u, child
        return best_c
    
    def expand(self, probs):
        for a in self.valid_moves:
            if a not in self.children:
                self.children[a] = Node(make_move(self.state, a, self.mark), 3 - self.mark, self, a, probs[a])
    
    def backup(self, v):
        self.N += 1
        self.W += v
        self.Q = self.W / self.N
        if self.parent:
            self.parent.backup(-v)


class MCTS:
    def __init__(self, net, dev):
        self.net = net
        self.dev = dev
    
    def search(self, state, mark, n_sims):
        root = Node(state, mark)
        if root.terminal[0]:
            return {}
        
        p, _ = self._eval(state, mark)
        p = self._mask(p, root.valid_moves)
        root.expand(p)
        
        for _ in range(n_sims):
            node = root
            while node.children and not node.terminal[0]:
                node = node.select(C_PUCT)
            
            if node.terminal[0]:
                w = node.terminal[1]
                v = 0.0 if w == 0 else (1.0 if w == node.mark else -1.0)
            else:
                p, v = self._eval(node.state, node.mark)
                p = self._mask(p, node.valid_moves)
                node.expand(p)
            
            node.backup(v)
        
        return {a: c.N for a, c in root.children.items()}
    
    def _eval(self, state, mark):
        s = torch.from_numpy(encode_state(state, mark)).float().unsqueeze(0).to(self.dev)
        with torch.no_grad():
            p, v = self.net(s)
            return F.softmax(p, dim=1).cpu().numpy()[0], v.item()
    
    def _mask(self, p, vm):
        m = np.zeros(COLUMNS)
        m[vm] = 1.0
        p = p * m
        return p / p.sum() if p.sum() > 0 else m / m.sum()


# ============================================================================
# Global State
# ============================================================================

_net = None
_mcts = None
_loaded = False


def _load_model():
    global _net, _mcts, _loaded
    if _loaded:
        return _net is not None
    _loaded = True
    
    try:
        data = base64.b64decode(_MODEL_DATA)
        data = gzip.decompress(data)
        buf = io.BytesIO(data)
        ckpt = torch.load(buf, map_location='cpu', weights_only=False)
        
        _net = Net()
        if 'model_state_dict' in ckpt:
            _net.load_state_dict(ckpt['model_state_dict'])
        else:
            _net.load_state_dict(ckpt)
        _net.eval()
        _mcts = MCTS(_net, torch.device('cpu'))
        return True
    except:
        return False


def find_open_three(board, mark):
    op = 3 - mark
    b = np.array(board).reshape(ROWS, COLUMNS)
    for r in range(ROWS):
        for c in range(COLUMNS - 4):
            if np.all(b[r, c:c+5] == [0, op, op, op, 0]):
                cands = []
                if (r == ROWS - 1) or (b[r+1, c] != 0): cands.append(c)
                if (r == ROWS - 1) or (b[r+1, c+4] != 0): cands.append(c+4)
                for cand in cands:
                    if not is_losing_move(board, cand, mark): return cand
    return None


def find_open_two(board, mark):
    op = 3 - mark
    b = np.array(board).reshape(ROWS, COLUMNS)
    for r in range(ROWS):
        for c in range(COLUMNS - 3):
            if np.all(b[r, c:c+4] == [0, op, op, 0]):
                cands = []
                if (r == ROWS - 1) or (b[r+1, c] != 0): cands.append(c)
                if (r == ROWS - 1) or (b[r+1, c+3] != 0): cands.append(c+3)
                for cand in cands:
                    if not is_losing_move(board, cand, mark): return cand
    return None


def _select_action(board, mark):
    valid = get_valid_moves(board)
    if not valid:
        return 0
    if len(valid) == 1:
        return valid[0]
    
    # Rule 1: Win
    w = find_winning_move(board, mark)
    if w is not None:
        return w
    
    # Rule 2: Block
    b = find_blocking_move(board, mark)
    if b is not None:
        return b

    # Rule 2.5: Block Open Three
    o3 = find_open_three(board, mark)
    if o3 is not None and not is_losing_move(board, o3, mark):
        return o3
        
    # Rule 2.6: Block Open Two
    o2 = find_open_two(board, mark)
    if o2 is not None and not is_losing_move(board, o2, mark):
        return o2
    
    # Rule 3: Safe moves
    safe = get_safe_moves(board, mark)
    if not safe:
        safe = valid
    
    # Rule 4: MCTS
    if _load_model() and _mcts:
        try:
            visits = _mcts.search(board, mark, NUM_SIMULATIONS)
            if visits:
                best = max(visits.keys(), key=lambda a: visits[a])
                if best in safe:
                    return best
        except:
            pass
    
    # Fallback: center preference
    for c in [3, 2, 4, 1, 5, 0, 6]:
        if c in safe:
            return c
    return safe[0] if safe else valid[0]


# ============================================================================
# Kaggle Entry Point (MUST BE LAST FUNCTION)
# ============================================================================

def agent(observation, configuration):
    try:
        return int(_select_action(list(observation.board), observation.mark))
    except:
        vm = [c for c in range(COLUMNS) if observation.board[c] == 0]
        return 3 if 3 in vm else (vm[0] if vm else 3)
'''
    
    # Ensure file ends with newline
    return submission_code + '\n'


def create_submission(model_path: str, output_path: str):
    """创建提交文件"""
    print("=" * 60)
    print("创建Kaggle提交文件")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在: {model_path}")
        return False
    
    # 压缩和编码模型
    print("\n[步骤1] 压缩和编码模型...")
    encoded_model = compress_and_encode_model(model_path)
    
    # 生成提交代码
    print("\n[步骤2] 生成提交代码...")
    submission_code = generate_submission_code(encoded_model)
    
    # 保存
    print(f"\n[步骤3] 保存到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(submission_code)
    
    output_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n生成文件大小: {output_size:.2f} MB")
    
    if output_size > 100:
        print("[警告] 文件大小超过100MB限制!")
        return False
    
    # 验证代码
    print("\n[步骤4] 验证代码...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("submission", output_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        if hasattr(module, 'agent'):
            print("[OK] agent函数存在")
        else:
            print("[错误] agent函数不存在!")
            return False
        
        # Test agent call
        class FakeObs:
            board = [0] * 42
            mark = 1
        
        action = module.agent(FakeObs(), None)
        print(f"[OK] agent返回动作: {action}")
        
    except Exception as e:
        print(f"[错误] 代码验证失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 60)
    print("提交文件创建成功!")
    print(f"文件: {output_path}")
    print(f"大小: {output_size:.2f} MB")
    print("=" * 60)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='准备Kaggle ConnectX提交文件')
    parser.add_argument('--model', type=str, default='alpha-zero-v2.pth',
                       help='模型文件路径')
    parser.add_argument('--output', type=str, default='main.py',
                       help='输出文件路径')
    
    args = parser.parse_args()
    
    submission_dir = Path(__file__).parent
    
    model_path = submission_dir / args.model
    output_path = submission_dir / args.output
    
    success = create_submission(str(model_path), str(output_path))
    
    if not success:
        sys.exit(1)


if __name__ == '__main__':
    main()
