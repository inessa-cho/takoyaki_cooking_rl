"""Small state/action abstraction over the existing tabular Q-learning agent."""
from env import (EMPTY, BATTER, OCTOPUS, SHAPED, COOKED, FINISHED, BURNT,
                 WAIT, ACT_BATTER, ACT_OCTOPUS, ACT_FLIP, ACT_MOVE_TO_TRAY,
                 ACT_SAUCE_TRAY, ACT_SERVE_TRAY, ACT_CLEAR)


def abstract_state(env):
    # Presence loses multiplicity; urgency loses exact timers. This is deliberately
    # an approximate, non-Markov representation, not an equivalent environment.
    presence = sum(1 << state for state in set(env.cells))
    cooked_urgent = any(s == COOKED and t >= 3 for s, t in zip(env.cells, env.timers))
    finished_urgent = any(s == FINISHED and t >= 3 for s, t in zip(env.cells, env.timers))
    return (min(4, env.t // 40), env.tray_count // 2, env.tray_sauced,
            presence, cooked_urgent, finished_urgent)


def action_candidates(env):
    """Nine action categories; oldest eligible cell, lowest index on ties.

    Cell selection is a fixed heuristic shared by exploration and exploitation.
    The agent learns which operation to perform, not which exact cell to use.
    """
    actions = {0: env.encode_action(WAIT)}
    for category, state, kind in (
        (1, EMPTY, ACT_BATTER), (2, BATTER, ACT_OCTOPUS),
        (3, OCTOPUS, ACT_FLIP), (4, COOKED, ACT_FLIP),
        (5, FINISHED, ACT_MOVE_TO_TRAY), (8, BURNT, ACT_CLEAR),
    ):
        cells = [i for i, s in enumerate(env.cells) if s == state]
        if cells and not (state == FINISHED and env.tray_count == 8):
            cell = max(cells, key=lambda i: (env.timers[i], -i))
            actions[category] = env.encode_action(kind, cell)
    if env.tray_count == 8:
        category, kind = (7, ACT_SERVE_TRAY) if env.tray_sauced else (6, ACT_SAUCE_TRAY)
        actions[category] = env.encode_action(kind)
    return actions
