"""
Minimal Takoyaki Cooking RL environment.

Step 1 MVP goals:
- Standard library only
- 1 step = 1 second
- 4x4 independent cells
- Valid action generation
- Text rendering/debug info

No Gymnasium dependency yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


# Cell states
EMPTY = 0
BATTER = 1
OCTOPUS = 2
SHAPED = 3
COOKED = 4
FINISHED = 5
BURNT = 6

STATE_NAMES = {
    EMPTY: "EMPTY",
    BATTER: "BATTER",
    OCTOPUS: "OCTOPUS",
    SHAPED: "SHAPED",
    COOKED: "COOKED",
    FINISHED: "FINISHED",
    BURNT: "BURNT",
}

STATE_CHARS = {
    EMPTY: ".",
    BATTER: "b",
    OCTOPUS: "o",
    SHAPED: "s",
    COOKED: "C",
    FINISHED: "F",
    BURNT: "X",
}

# Action types
WAIT = 0
ACT_BATTER = 1
ACT_OCTOPUS = 2
ACT_FLIP = 3
ACT_MOVE_TO_TRAY = 4
ACT_SAUCE_TRAY = 5
ACT_SERVE_TRAY = 6
ACT_CLEAR = 7

ACTION_NAMES = {
    WAIT: "WAIT",
    ACT_BATTER: "BATTER",
    ACT_OCTOPUS: "OCTOPUS",
    ACT_FLIP: "FLIP",
    ACT_MOVE_TO_TRAY: "MOVE_TO_TRAY",
    ACT_SAUCE_TRAY: "SAUCE_TRAY",
    ACT_SERVE_TRAY: "SERVE_TRAY",
    ACT_CLEAR: "CLEAR",
}


@dataclass
class StepResult:
    observation: Tuple[Any, ...]
    reward: float
    done: bool
    info: Dict[str, Any]


class TakoyakiEnv:
    """Small deterministic environment for takoyaki cooking multitasking."""

    def __init__(self, max_steps: int = 200, rows: int = 4, cols: int = 4) -> None:
        self.max_steps = max_steps
        self.rows = rows
        self.cols = cols
        self.n_cells = rows * cols

        # Encoding design:
        # 0                          -> WAIT
        # 1..16                      -> BATTER(cell)
        # 17..32                     -> OCTOPUS(cell)
        # 33..48                     -> FLIP(cell)
        # 49..64                     -> MOVE_TO_TRAY(cell)
        # 65                         -> SAUCE_TRAY
        # 66                         -> SERVE_TRAY
        # 67..82                     -> CLEAR(cell)
        self.n_actions = 1 + self.n_cells * 4 + 2 + self.n_cells

        self.reset()

    def reset(self) -> Tuple[Any, ...]:
        self.t = 0
        self.cells: List[int] = [EMPTY for _ in range(self.n_cells)]
        self.timers: List[int] = [0 for _ in range(self.n_cells)]
        self.tray_count = 0
        self.tray_sauced = False
        self.score = 0
        self.burnt_count = 0
        self.invalid_actions = 0
        self.total_reward = 0.0
        return self.observe()

    def observe(self) -> Tuple[Any, ...]:
        """Return a hashable observation for tabular RL later."""
        return (
            self.t,
            tuple(self.cells),
            tuple(self.timers),
            self.tray_count,
            self.tray_sauced,
        )

    def encode_action(self, action_type: int, cell: int | None = None) -> int:
        if action_type == WAIT:
            return 0
        if action_type == ACT_BATTER:
            self._require_cell(cell)
            return 1 + int(cell)
        if action_type == ACT_OCTOPUS:
            self._require_cell(cell)
            return 1 + self.n_cells + int(cell)
        if action_type == ACT_FLIP:
            self._require_cell(cell)
            return 1 + self.n_cells * 2 + int(cell)
        if action_type == ACT_MOVE_TO_TRAY:
            self._require_cell(cell)
            return 1 + self.n_cells * 3 + int(cell)
        if action_type == ACT_SAUCE_TRAY:
            return 1 + self.n_cells * 4
        if action_type == ACT_SERVE_TRAY:
            return 1 + self.n_cells * 4 + 1
        if action_type == ACT_CLEAR:
            self._require_cell(cell)
            return 1 + self.n_cells * 4 + 2 + int(cell)
        raise ValueError(f"Unknown action_type: {action_type}")

    def decode_action(self, action: int) -> Tuple[int, int | None]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"Action out of range: {action}")
        if action == 0:
            return WAIT, None

        x = action - 1
        if x < self.n_cells:
            return ACT_BATTER, x
        x -= self.n_cells
        if x < self.n_cells:
            return ACT_OCTOPUS, x
        x -= self.n_cells
        if x < self.n_cells:
            return ACT_FLIP, x
        x -= self.n_cells
        if x < self.n_cells:
            return ACT_MOVE_TO_TRAY, x
        x -= self.n_cells
        if x == 0:
            return ACT_SAUCE_TRAY, None
        if x == 1:
            return ACT_SERVE_TRAY, None
        x -= 2
        return ACT_CLEAR, x

    def valid_actions(self) -> List[int]:
        actions = [self.encode_action(WAIT)]

        for cell, state in enumerate(self.cells):
            if state == EMPTY:
                actions.append(self.encode_action(ACT_BATTER, cell))
            elif state == BATTER:
                actions.append(self.encode_action(ACT_OCTOPUS, cell))
            elif state == OCTOPUS:
                actions.append(self.encode_action(ACT_FLIP, cell))
            elif state == COOKED:
                actions.append(self.encode_action(ACT_FLIP, cell))
            elif state == FINISHED and self.tray_count < 8:
                actions.append(self.encode_action(ACT_MOVE_TO_TRAY, cell))
            elif state == BURNT:
                actions.append(self.encode_action(ACT_CLEAR, cell))

        if self.tray_count == 8 and not self.tray_sauced:
            actions.append(self.encode_action(ACT_SAUCE_TRAY))
        if self.tray_count == 8 and self.tray_sauced:
            actions.append(self.encode_action(ACT_SERVE_TRAY))

        return actions

    def step(self, action: int) -> Tuple[Tuple[Any, ...], float, bool, Dict[str, Any]]:
        reward = -0.01  # time cost
        invalid = False
        served = False
        newly_burnt = 0

        if action not in self.valid_actions():
            reward -= 0.05
            invalid = True
            self.invalid_actions += 1
        else:
            action_type, cell = self.decode_action(action)
            shaping = self._apply_valid_action(action_type, cell)
            reward += shaping
            if action_type == ACT_SERVE_TRAY:
                served = True
                reward += 1.0

        # Time passes after the chosen action.
        newly_burnt = self._advance_time_and_cook()
        if newly_burnt:
            reward -= 0.2 * newly_burnt
            self.burnt_count += newly_burnt

        self.t += 1
        done = self.t >= self.max_steps
        self.total_reward += reward

        info = self.info()
        info.update({
            "invalid": invalid,
            "served": served,
            "newly_burnt": newly_burnt,
            "action": self.action_to_str(action),
        })
        return self.observe(), reward, done, info

    def render_text(self) -> str:
        lines = []
        lines.append(
            f"t={self.t}/{self.max_steps} score={self.score} "
            f"tray={self.tray_count}/8 sauced={self.tray_sauced} burnt={self.burnt_count}"
        )
        lines.append("cells: state(timer)")
        for r in range(self.rows):
            parts = []
            for c in range(self.cols):
                idx = r * self.cols + c
                parts.append(f"{STATE_CHARS[self.cells[idx]]}{self.timers[idx]:02d}")
            lines.append(" ".join(parts))
        return "\n".join(lines)

    def info(self) -> Dict[str, Any]:
        return {
            "time": self.t,
            "score": self.score,
            "burnt": self.burnt_count,
            "tray_count": self.tray_count,
            "tray_sauced": self.tray_sauced,
            "invalid_actions": self.invalid_actions,
            "total_reward": round(self.total_reward, 4),
            "n_actions": self.n_actions,
        }

    def action_to_str(self, action: int) -> str:
        action_type, cell = self.decode_action(action)
        name = ACTION_NAMES[action_type]
        if cell is None:
            return name
        return f"{name}({cell})"

    def _apply_valid_action(self, action_type: int, cell: int | None) -> float:
        """Apply already-validated action. Return optional shaping reward."""
        if action_type == WAIT:
            return 0.0

        if action_type == ACT_SAUCE_TRAY:
            self.tray_sauced = True
            return 0.02

        if action_type == ACT_SERVE_TRAY:
            self.score += 1
            self.tray_count = 0
            self.tray_sauced = False
            return 0.0

        assert cell is not None

        if action_type == ACT_BATTER:
            self.cells[cell] = BATTER
            self.timers[cell] = 0
            return 0.01

        if action_type == ACT_OCTOPUS:
            self.cells[cell] = OCTOPUS
            self.timers[cell] = 0
            return 0.01

        if action_type == ACT_FLIP:
            if self.cells[cell] == OCTOPUS:
                self.cells[cell] = SHAPED
                self.timers[cell] = 0
                return 0.02
            if self.cells[cell] == COOKED:
                self.cells[cell] = FINISHED
                self.timers[cell] = 0
                return 0.03

        if action_type == ACT_MOVE_TO_TRAY:
            self.cells[cell] = EMPTY
            self.timers[cell] = 0
            self.tray_count += 1
            return 0.05

        if action_type == ACT_CLEAR:
            self.cells[cell] = EMPTY
            self.timers[cell] = 0
            return 0.0

        raise RuntimeError(f"Unhandled valid action: {action_type}, {cell}")

    def _advance_time_and_cook(self) -> int:
        newly_burnt = 0
        for i, state in enumerate(self.cells):
            if state == EMPTY:
                self.timers[i] = 0
                continue

            self.timers[i] += 1

            # SHAPED -> COOKED after 6 seconds.
            if state == SHAPED and self.timers[i] >= 6:
                self.cells[i] = COOKED
                self.timers[i] = 0
                continue

            # COOKED / FINISHED burn after 5 seconds or more of neglect.
            # Since timers start at 0 when entering these states, timer >= 5 means
            # it has remained in that state for five full environment steps.
            if state in (COOKED, FINISHED) and self.timers[i] >= 5:
                self.cells[i] = BURNT
                self.timers[i] = 0
                newly_burnt += 1

        return newly_burnt

    def _require_cell(self, cell: int | None) -> None:
        if cell is None or not 0 <= int(cell) < self.n_cells:
            raise ValueError(f"Invalid cell index: {cell}")


if __name__ == "__main__":
    import random

    env = TakoyakiEnv()
    print("RESET OBSERVATION:")
    print(env.observe())
    print()
    print(env.render_text())
    print("-" * 40)

    random.seed(42)
    for step_idx in range(20):
        action = random.choice(env.valid_actions())
        obs, reward, done, info = env.step(action)
        print(
            f"step={step_idx + 1:02d} action={info['action']:<16} "
            f"reward={reward:+.2f} served={info['served']} burnt+={info['newly_burnt']}"
        )
        if done:
            break

    print("-" * 40)
    print(env.render_text())
    print("FINAL INFO:")
    print(env.info())

