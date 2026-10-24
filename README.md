# Takoyaki Cooking RL

A Python cooking simulation comparing rule-based agents and Q-learning on a timed 4-by-4 takoyaki grid.

A small Python reinforcement-learning experiment about cooking timing and multitasking
on a 4×4 takoyaki grid. Each episode lasts 200 one-second steps. No external packages,
GUI, business simulation or heavy RL framework are required.

## Run

Use Python 3.13.5 (the verified version) or the existing virtual environment:

```bash
.venv/bin/python rule_agent.py
.venv/bin/python random_agent.py
.venv/bin/python -m unittest -v
.venv/bin/python verify_results.py experiments/final_seed42
```

Train and plot a separate reproducible run (directory must not already exist):

```bash
.venv/bin/python final_experiment.py --output experiments/my_run --episodes 50000 --raw-episodes 1000 --seed 42 --eval-episodes 100
.venv/bin/python plot_results.py experiments/my_run
```

On a fresh machine, `python3` may replace `.venv/bin/python`; no dependency installation
is needed. Recorded experiment parameters and source hashes are in the configuration.

## Agent comparison

100 frozen evaluation episodes per row, seeds 10000–10099. Training seed: 42.
Score is cumulative reward; Served counts eight-item trays.

| Agent | Training episodes | Avg score (return) | Avg served (trays) | Avg burnt (items) | Avg invalid | Avg waste proxy |
|---|---:|---:|---:|---:|---:|---:|
| random_all | 0 | -10.8560 | 0.0000 | 6.6000 | 159.2900 | 161.8100 |
| random_valid | 0 | -7.3154 | 0.0000 | 37.9500 | 0.0000 | 12.6200 |
| random_abstract | 0 | -3.4308 | 0.6200 | 22.0900 | 0.0000 | 43.4800 |
| rule | 0 | 6.6300 | 4.0000 | 0.0000 | 0.0000 | 0.0000 |
| raw_q | 1000 | -7.3064 | 0.0000 | 37.9600 | 0.0000 | 12.3700 |
| abstract_q | 1000 | 6.5900 | 4.0000 | 0.0000 | 0.0000 | 1.0000 |
| abstract_q | 10000 | 6.5100 | 4.0000 | 0.0000 | 0.0000 | 3.0000 |
| abstract_q | 50000 | 6.5700 | 4.0000 | 0.0000 | 0.0000 | 3.0000 |

The Rule baseline exactly reproduces the historical 6.63 return and four trays without
burns. Raw-state Q-learning fails to serve a tray at 1,000 episodes. The compact learner
learns operation selection with an explicit oldest-cell heuristic, reaching four trays,
the maximum possible in 200 actions. This is one-seed evidence, not a general benchmark.
The waste proxy counts invalid actions and waits when another valid action exists;
it may count strategically useful waits.

## Report and artifacts

- [Final technical report](FINAL_REPORT.md)
- [Agent comparison CSV](experiments/final_seed42/comparison.csv)
- [Training curve CSV](experiments/final_seed42/learning_curve.csv)
- [Per-episode evaluations](experiments/final_seed42/evaluation_episodes.csv)
- [Run configuration](experiments/final_seed42/config.json)
- [Learning curve](experiments/final_seed42/learning_curve.svg)
- [Agent comparison plot](experiments/final_seed42/agent_comparison.svg)
- [Final verification record](FINAL_REVIEW.md)

Earlier checkpoints and the leaderboard are retained unchanged. `play_snapshot.py` supports
raw-state snapshots only; use the verification command above for compact snapshots.
The recorded closure results predate this GitHub publication.


## Implementation notes

The existing final report and measured experiment artifacts are retained. Rule, raw Q-learning and compact Q-learning have different results; the compact learner uses a hand-coded cell-selection heuristic. Historical verification statements remain dated evidence. No new full training run was performed during documentation review.

See [PRESERVATION.md](PRESERVATION.md) for the maintenance record, excluded files and verification scope.
