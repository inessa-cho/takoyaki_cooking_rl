"""
Train a minimal tabular Q-learning agent for Takoyaki Cooking RL.

Standard library only.

Usage:
    python train.py
    python train.py --episodes 10000
    python train.py --eval-episodes 20
"""

from __future__ import annotations

import argparse
import csv
import os
import random
from dataclasses import dataclass
from typing import Dict, List

from env import TakoyakiEnv
from q_agent import QLearningAgent


DEFAULT_CHECKPOINTS = [0, 100, 500, 1000, 5000, 10000, 50000]


@dataclass
class EpisodeResult:
    episode: int
    score: int
    burnt: int
    invalid_actions: int
    total_reward: float


def run_episode(env: TakoyakiEnv, agent: QLearningAgent, training: bool) -> EpisodeResult:
    obs = env.reset()
    done = False

    while not done:
        valid = env.valid_actions()
        action = agent.choose_action(obs, valid, training=training)
        next_obs, reward, done, _info = env.step(action)

        if training:
            next_valid = env.valid_actions()
            agent.update(obs, action, reward, next_obs, next_valid, done)

        obs = next_obs

    info = env.info()
    return EpisodeResult(
        episode=-1,
        score=info["score"],
        burnt=info["burnt"],
        invalid_actions=info["invalid_actions"],
        total_reward=info["total_reward"],
    )


def evaluate(agent: QLearningAgent, episodes: int, seed: int = 10_000) -> Dict[str, float]:
    scores: List[int] = []
    burnt: List[int] = []
    invalids: List[int] = []
    rewards: List[float] = []

    old_rng_state = agent.rng.getstate()
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for i in range(episodes):
        env = TakoyakiEnv()
        # Env is deterministic now, but keep this hook for future customer randomness.
        agent.rng.seed(seed + i)
        result = run_episode(env, agent, training=False)
        scores.append(result.score)
        burnt.append(result.burnt)
        invalids.append(result.invalid_actions)
        rewards.append(result.total_reward)

    agent.epsilon = old_epsilon
    agent.rng.setstate(old_rng_state)

    return {
        "avg_score": sum(scores) / len(scores),
        "avg_burnt": sum(burnt) / len(burnt),
        "avg_invalid": sum(invalids) / len(invalids),
        "avg_reward": sum(rewards) / len(rewards),
        "max_score": max(scores),
    }


def save_checkpoint(agent: QLearningAgent, episode: int, checkpoint_dir: str) -> str:
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"q_agent_ep{episode}.pkl")
    agent.save(path)
    return path


def write_leaderboard(rows: List[Dict[str, float]], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--leaderboard", default="leaderboard.csv")
    args = parser.parse_args()

    random.seed(args.seed)
    env = TakoyakiEnv()
    agent = QLearningAgent(
        n_actions=env.n_actions,
        alpha=0.15,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        seed=args.seed,
    )

    checkpoints = [c for c in DEFAULT_CHECKPOINTS if c <= args.episodes]
    if 0 not in checkpoints:
        checkpoints.insert(0, 0)
    if args.episodes not in checkpoints:
        checkpoints.append(args.episodes)
    checkpoints = sorted(set(checkpoints))

    leaderboard: List[Dict[str, float]] = []

    print("TAKOYAKI Q-LEARNING TRAINING")
    print(f"episodes={args.episodes} eval_episodes={args.eval_episodes} n_actions={env.n_actions}")
    print("-" * 72)

    # Save untrained snapshot.
    path = save_checkpoint(agent, 0, args.checkpoint_dir)
    metrics = evaluate(agent, args.eval_episodes, seed=args.seed)
    row = {
        "Episode": 0,
        "Score": round(metrics["avg_score"], 2),
        "Burnt": round(metrics["avg_burnt"], 2),
        "Invalid": round(metrics["avg_invalid"], 2),
        "Reward": round(metrics["avg_reward"], 3),
        "Epsilon": round(agent.epsilon, 4),
        "States": agent.stats()["states"],
        "Checkpoint": path,
    }
    leaderboard.append(row)
    print(row)

    for episode in range(1, args.episodes + 1):
        result = run_episode(env, agent, training=True)
        agent.decay_epsilon()

        if episode in checkpoints:
            path = save_checkpoint(agent, episode, args.checkpoint_dir)
            metrics = evaluate(agent, args.eval_episodes, seed=args.seed + episode)
            row = {
                "Episode": episode,
                "Score": round(metrics["avg_score"], 2),
                "Burnt": round(metrics["avg_burnt"], 2),
                "Invalid": round(metrics["avg_invalid"], 2),
                "Reward": round(metrics["avg_reward"], 3),
                "Epsilon": round(agent.epsilon, 4),
                "States": agent.stats()["states"],
                "Checkpoint": path,
            }
            leaderboard.append(row)
            print(row)

    leaderboard_path = args.leaderboard
    write_leaderboard(leaderboard, leaderboard_path)
    print("-" * 72)
    print(f"saved leaderboard: {leaderboard_path}")
    print("final agent stats:", agent.stats())


if __name__ == "__main__":
    main()
