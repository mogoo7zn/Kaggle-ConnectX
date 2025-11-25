#!/usr/bin/env python3
"""Evaluate checkpoints and export submission-ready weights."""

from __future__ import annotations

import argparse
import base64
import io
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

# Ensure local modules can be imported when running from tools/
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "core") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "core"))
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

from config import config  # type: ignore  # noqa: E402
from dqn_agent import DQNAgent, evaluate_agent  # type: ignore  # noqa: E402
from utils import get_negamax_move, get_valid_moves  # type: ignore  # noqa: E402

try:
    from embed_model import embed_model  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover - embed_model is optional
    embed_model = None


def policy_to_env_bot(policy_fn):
    """Convert a board-level policy into a Kaggle environment bot callable."""

    def bot_fn(observation, configuration):
        return int(policy_fn(observation.board, observation.mark))

    return bot_fn


def random_policy(board: List[int], mark: int) -> int:
    """Random agent that selects among valid moves."""
    valid_moves = get_valid_moves(board)
    return random.choice(valid_moves) if valid_moves else 0


def negamax_policy(board: List[int], mark: int) -> int:
    """Negamax heuristic wrapper (falls back to center preference)."""
    try:
        return get_negamax_move(board, mark)
    except Exception:
        # When negamax is unavailable, pick center-biased random move
        center = config.COLUMNS // 2
        valid_moves = get_valid_moves(board)
        if not valid_moves:
            return 0
        return min(valid_moves, key=lambda c: (abs(center - c), random.random()))


def load_agent_from_path(path: Path, device: torch.device) -> Tuple[DQNAgent, Dict]:
    """Load a DQNAgent from either a checkpoint or bare state_dict file."""
    config.DEVICE = device  # type: ignore[attr-defined]
    agent = DQNAgent(model_type="standard", use_double_dqn=True)
    agent.device = device
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    metadata: Dict = {}

    if isinstance(checkpoint, dict) and "policy_net_state_dict" in checkpoint:
        agent.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        agent.target_net.load_state_dict(checkpoint.get("target_net_state_dict", agent.policy_net.state_dict()))
        agent.optimizer.load_state_dict(checkpoint.get("optimizer_state_dict", {}))
        agent.steps_done = checkpoint.get("steps_done", 0)
        agent.epsilon = checkpoint.get("epsilon", 0.0)
        agent.losses = checkpoint.get("losses", [])
        metadata = checkpoint.get("metrics", {})
        metadata["episode"] = checkpoint.get("episode")
        metadata["checkpoint"] = True
    else:
        agent.policy_net.load_state_dict(checkpoint)
        agent.target_net.load_state_dict(agent.policy_net.state_dict())
        metadata["checkpoint"] = False

    agent.policy_net.eval()
    agent.target_net.eval()
    agent.epsilon = 0.0
    return agent, metadata


def evaluate_suite(agent: DQNAgent, games: int, snapshot_paths: Iterable[Path]) -> Dict[str, dict]:
    """Run evaluations against built-in bots and provided snapshots."""
    results: Dict[str, dict] = {}

    # Baseline heuristics
    results["random_bot"] = evaluate_agent(agent, policy_to_env_bot(random_policy), num_games=games)
    if get_negamax_move is not None:
        results["negamax_bot"] = evaluate_agent(agent, policy_to_env_bot(negamax_policy), num_games=games)

    # Past snapshots
    for snapshot_path in snapshot_paths:
        opponent, _ = load_agent_from_path(snapshot_path, device=config.DEVICE)

        def opponent_bot(obs, cfg, _opponent=opponent):
            return int(_opponent.select_action(obs.board, obs.mark, epsilon=0.0))

        key = f"snapshot::{snapshot_path.stem}"
        results[key] = evaluate_agent(agent, opponent_bot, num_games=games)

    return results


def discover_snapshots(snapshot_dir: Path, exclude: Optional[Path], limit: int) -> List[Path]:
    """Find recent snapshot files for comparison."""
    if not snapshot_dir.exists():
        return []

    exclude_resolved = exclude.resolve() if exclude else None
    candidates = [p for p in snapshot_dir.glob("*.pth") if not exclude_resolved or p.resolve() != exclude_resolved]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[:limit]


def export_weights(agent: DQNAgent, output_path: Path, embed_output: Optional[Path] = None):
    """Export CPU state_dict and optional Base64 payload for submission consumption."""
    cpu_state = {k: v.cpu() for k, v in agent.policy_net.state_dict().items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cpu_state, output_path)
    size_kb = output_path.stat().st_size / 1024
    print(f"Saved lightweight weights to {output_path} ({size_kb:.1f} KB)")

    if embed_output:
        embed_output.parent.mkdir(parents=True, exist_ok=True)
        if embed_model is None:
            # Fallback inline embedding
            buffer = io.BytesIO()
            torch.save(cpu_state, buffer)
            buffer.seek(0)
            encoded = base64.b64encode(buffer.read()).decode("utf-8")
            embed_output.write_text(encoded)
        else:
            embed_model(str(output_path), str(embed_output))
        print(f"Embedded Base64 weights written to {embed_output}")


def format_results(results: Dict[str, dict]) -> str:
    lines = ["\nEvaluation Results", "=================="]
    for name, stats in results.items():
        line = (
            f"- {name:15s}: "
            f"W {stats['wins']:3d} / D {stats['draws']:3d} / L {stats['losses']:3d} | "
            f"WinRate {stats['win_rate']:.1%}"
        )
        lines.append(line)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a checkpoint and export submission-ready weights.")
    parser.add_argument("--checkpoint", type=Path, default=REPO_ROOT / "submission" / "best_model.pth",
                        help="Path to checkpoint or state_dict file to evaluate")
    parser.add_argument("--snapshot-dir", type=Path, default=REPO_ROOT / "training" / "checkpoints",
                        help="Directory containing historical snapshot .pth files")
    parser.add_argument("--games", type=int, default=config.EVAL_GAMES,
                        help="Number of games to play per evaluation opponent")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="Device to load and evaluate the model on")
    parser.add_argument("--export-path", type=Path, default=REPO_ROOT / "submission" / "best_model.pth",
                        help="Destination to save stripped state_dict for submission")
    parser.add_argument("--embed-output", type=Path,
                        help="Optional path to also emit Base64-encoded weights")
    parser.add_argument("--max-snapshots", type=int, default=3,
                        help="Maximum number of historical snapshots to evaluate")

    args = parser.parse_args()

    # Configure device
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    config.DEVICE = device  # type: ignore[attr-defined]
    print(f"Using device: {device}")

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    agent, metadata = load_agent_from_path(args.checkpoint, device=device)
    print(f"Loaded agent from {args.checkpoint} (checkpoint={metadata.get('checkpoint', False)})")

    snapshots = discover_snapshots(args.snapshot_dir, exclude=args.checkpoint, limit=args.max_snapshots)
    if snapshots:
        print(f"Found {len(snapshots)} snapshot(s) for comparison: {[p.name for p in snapshots]}")
    else:
        print("No snapshot opponents found; skipping snapshot evaluation.")

    try:
        results = evaluate_suite(agent, games=args.games, snapshot_paths=snapshots)
    except ModuleNotFoundError as exc:
        if "kaggle_environments" in str(exc):
            print("kaggle_environments is required for evaluation. Install via `pip install kaggle-environments`.\n")
        raise

    print(format_results(results))

    export_weights(agent, args.export_path, embed_output=args.embed_output)
    print("\nDone. You can now embed the exported weights into submission/main.py")


if __name__ == "__main__":
    main()
