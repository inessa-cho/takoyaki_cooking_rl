"""
Minimal tabular Q-learning agent for Takoyaki Cooking RL.

MVP goals:
- Standard library only
- Works with env.py action integers
- Uses only valid actions for exploration/exploitation
- Save/load Q-table with pickle for later checkpoints
"""

from __future__ import annotations

import pickle
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple


Observation = Tuple[Any, ...]
QTable = Dict[Observation, Dict[int, float]]


class QLearningAgent:
    def __init__(
        self,
        n_actions: int,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = random.Random(seed)
        self.q: QTable = defaultdict(dict)

    def choose_action(self, obs: Observation, valid_actions: List[int], training: bool = True) -> int:
        if not valid_actions:
            raise ValueError("valid_actions is empty")

        if training and self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)

        return self.best_action(obs, valid_actions)

    def best_action(self, obs: Observation, valid_actions: List[int]) -> int:
        """Return highest-Q valid action. Random tie-break to avoid fixed bias."""
        state_q = self.q.get(obs, {})
        best_value = max(state_q.get(a, 0.0) for a in valid_actions)
        best_actions = [a for a in valid_actions if state_q.get(a, 0.0) == best_value]
        return self.rng.choice(best_actions)

    def update(
        self,
        obs: Observation,
        action: int,
        reward: float,
        next_obs: Observation,
        next_valid_actions: List[int],
        done: bool,
    ) -> None:
        old_q = self.q[obs].get(action, 0.0)

        if done or not next_valid_actions:
            target = reward
        else:
            next_max_q = max(self.q[next_obs].get(a, 0.0) for a in next_valid_actions)
            target = reward + self.gamma * next_max_q

        self.q[obs][action] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        payload = {
            "n_actions": self.n_actions,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q": dict(self.q),
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, path: str, seed: int | None = None) -> "QLearningAgent":
        with open(path, "rb") as f:
            payload = pickle.load(f)

        agent = cls(
            n_actions=payload["n_actions"],
            alpha=payload["alpha"],
            gamma=payload["gamma"],
            epsilon=payload["epsilon"],
            epsilon_min=payload["epsilon_min"],
            epsilon_decay=payload["epsilon_decay"],
            seed=seed,
        )
        agent.q = defaultdict(dict, payload["q"])
        return agent

    def stats(self) -> Dict[str, Any]:
        n_states = len(self.q)
        n_state_actions = sum(len(actions) for actions in self.q.values())
        return {
            "states": n_states,
            "state_actions": n_state_actions,
            "epsilon": round(self.epsilon, 6),
        }


if __name__ == "__main__":
    from env import TakoyakiEnv

    env = TakoyakiEnv()
    agent = QLearningAgent(env.n_actions, seed=42)

    obs = env.reset()
    print("Q_AGENT SMOKE TEST")
    print("initial:", agent.stats())

    for step_idx in range(20):
        valid = env.valid_actions()
        action = agent.choose_action(obs, valid, training=True)
        next_obs, reward, done, info = env.step(action)
        agent.update(obs, action, reward, next_obs, env.valid_actions(), done)
        obs = next_obs
        print(f"step={step_idx + 1:02d} action={info['action']:<16} reward={reward:+.2f}")
        if done:
            break

    agent.decay_epsilon()
    print("final:", agent.stats())
    print(env.render_text())
