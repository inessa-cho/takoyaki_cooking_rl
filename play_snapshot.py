"""
Replay a saved Q-learning snapshot for Takoyaki Cooking RL.

Standard library only.

Usage:
    python play_snapshot.py
    python play_snapshot.py --checkpoint checkpoints/q_agent_ep100.pkl
    python play_snapshot.py --delay 0.1
    python play_snapshot.py --max-steps 80 --no-clear
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional

from env import TakoyakiEnv
from q_agent import QLearningAgent


DEFAULT_CHECKPOINT = os.path.join("checkpoints", "q_agent_ep100.pkl")


def clear_screen() -> None:
    # ANSI clear screen + cursor home. Works in most terminals.
    print("\033[2J\033[H", end="")


def replay(
    checkpoint: str,
    delay: float = 0.0,
    max_steps: Optional[int] = None,
    clear: bool = True,
    seed: int = 123,
) -> None:
    if not os.path.exists(checkpoint):
        raise FileNotFoundError(
            f"checkpoint not found: {checkpoint}\n"
            "먼저 python train.py 를 실행해서 체크포인트를 만들어줘."
        )

    env = TakoyakiEnv()
    agent = QLearningAgent.load(checkpoint, seed=seed)
    agent.epsilon = 0.0  # greedy replay

    obs = env.reset()
    done = False
    step_limit = max_steps if max_steps is not None else env.max_steps

    print(f"REPLAY checkpoint={checkpoint}")
    print(f"agent_stats={agent.stats()}")
    print("-" * 60)

    last_info = None
    total_reward = 0.0

    for frame in range(step_limit + 1):
        if clear:
            clear_screen()

        print(f"REPLAY checkpoint={checkpoint}")
        print(env.render_text())
        if last_info is not None:
            print(
                f"last_action={last_info['action']} "
                f"reward={last_info['reward']:+.3f} "
                f"served={last_info['served']} "
                f"burnt+={last_info['newly_burnt']}"
            )
        print(f"total_reward={total_reward:.3f}")
        print("legend: .=EMPTY b=BATTER o=OCTOPUS s=SHAPED C=COOKED F=FINISHED X=BURNT")

        if done or frame == step_limit:
            break

        valid = env.valid_actions()
        action = agent.choose_action(obs, valid, training=False)
        next_obs, reward, done, info = env.step(action)
        info["reward"] = reward
        last_info = info
        total_reward += reward
        obs = next_obs

        if delay > 0:
            time.sleep(delay)

    print("-" * 60)
    print("FINAL INFO")
    print(env.info())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--delay", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--no-clear", action="store_true", help="Do not clear terminal between frames")
    args = parser.parse_args()

    replay(
        checkpoint=args.checkpoint,
        delay=args.delay,
        max_steps=args.max_steps,
        clear=not args.no_clear,
        seed=args.seed,
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
