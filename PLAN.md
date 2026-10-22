# Closure plan

Use existing Python 3.13 and standard library; no dependencies or type checker setup.
Keep env.py unchanged. Correct baseline metric/import behavior and evaluation RNG
isolation. Retain raw-state Q-learning and run a fresh 1,000-episode reference;
limit it there because near-unique full observations cause memory growth.
Reuse QLearningAgent with a small state/action adapter for 50,000 episodes.
Use one training seed (42), 100 fixed evaluation seeds (10000–10099), identical
200-step episodes and frozen policies. Evaluate random-all, random-valid and
random-abstract baselines to expose action-mask and heuristic effects.
Use unittest for behavior, real runs for empirical evaluation, CSV/JSON/pickle
and standalone SVG for artifacts. Preserve originals before edits.
Document limitations, actual results and reproduce saved checkpoint evaluations.
No repository exists; do not initialize Git merely to create a closure commit.
