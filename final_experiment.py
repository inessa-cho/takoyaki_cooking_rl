"""Reproducible, standard-library-only closing experiment. Never overwrites a run."""
import argparse
import csv
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

from env import TakoyakiEnv, WAIT
from q_agent import QLearningAgent
from rule_agent import choose_action
from abstract_agent import abstract_state, action_candidates

METRICS = ['score', 'served', 'burnt', 'invalid', 'wait_actions', 'avoidable_wait_actions', 'wasted_actions']


def run_episode(kind, rng, agent=None, training=False):
    env = TakoyakiEnv()
    total = 0.0
    waits = avoidable = 0
    done = False
    while not done:
        valid = env.valid_actions()
        obs = abstract_state(env) if kind == 'abstract_q' else env.observe()
        candidates = action_candidates(env) if kind == 'abstract_q' else None
        if kind == 'rule':
            action = choose_action(env)
        elif kind == 'random_all':
            action = rng.randrange(env.n_actions)
        elif kind == 'random_valid':
            action = rng.choice(valid)
        elif kind == 'random_abstract':
            action = rng.choice(list(action_candidates(env).values()))
        else:
            choice = agent.choose_action(obs, list(candidates) if candidates else valid, training)
            action = candidates[choice] if candidates else choice
        if action == WAIT:
            waits += 1
            avoidable += len(valid) > 1
        _, reward, done, _ = env.step(action)
        total += reward
        if training:
            next_obs = abstract_state(env) if candidates else env.observe()
            next_actions = list(action_candidates(env)) if candidates else env.valid_actions()
            agent.update(obs, choice, reward, next_obs, next_actions, done)
    return dict(score=round(total, 8), served=env.score, burnt=env.burnt_count,
                invalid=env.invalid_actions, wait_actions=waits,
                avoidable_wait_actions=avoidable,
                wasted_actions=env.invalid_actions + avoidable)


def evaluate(kind, episodes=100, seed=10000, agent=None):
    # No updates, no Q-table insertion, and no disturbance of training RNG.
    previous = agent.rng.getstate() if agent else None
    rows = []
    try:
        for i in range(episodes):
            if agent:
                agent.rng.seed(seed + i)
            rows.append(dict(eval_episode=i + 1, eval_seed=seed + i,
                             **run_episode(kind, random.Random(seed + i), agent)))
    finally:
        if agent:
            agent.rng.setstate(previous)
    return rows


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summary(label, episode, rows, agent=None):
    return dict(agent=label, training_episodes=episode, evaluation_episodes=len(rows),
                **{'avg_' + m: mean(r[m] for r in rows) for m in METRICS},
                sd_score=stdev(r['score'] for r in rows) if len(rows) > 1 else 0,
                states=len(agent.q) if agent else 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='experiments/final_seed42')
    parser.add_argument('--episodes', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-episodes', type=int, default=100)
    parser.add_argument('--raw-episodes', type=int, default=1000)
    args = parser.parse_args()
    if min(args.episodes, args.eval_episodes, args.raw_episodes) < 1:
        parser.error('episode counts must be positive')
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    config = dict(vars(args), started_utc=datetime.now(timezone.utc).isoformat(),
                  python=sys.version, platform=platform.platform(),
                  eval_seed=10000, alpha=0.15, gamma=0.95, epsilon=1.0,
                  epsilon_min=0.05, epsilon_decay=0.995,
                  environment=dict(rows=4, cols=4, max_steps=200),
                  git_commit=None, dataset='No external data; deterministic simulator',
                  source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path('.').glob('*.py'))})
    (out/'config.json').write_text(json.dumps(config, indent=2))
    comparisons, episode_rows, curve = [], [], []
    def assess(kind, ep, agent=None):
        rows = evaluate(kind, args.eval_episodes, agent=agent)
        result = summary(kind, ep, rows, agent)
        comparisons.append(result)
        episode_rows.extend(dict(agent=kind, training_episodes=ep, **r) for r in rows)
        write_csv(out/'comparison.csv', comparisons)
        write_csv(out/'evaluation_episodes.csv', episode_rows)
        print(json.dumps(result), flush=True)
    for kind in ('random_all', 'random_valid', 'random_abstract', 'rule'):
        assess(kind, 0)
    for kind, limit in (('raw_q', args.raw_episodes), ('abstract_q', args.episodes)):
        agent = QLearningAgent(9 if kind == 'abstract_q' else 83, alpha=0.15,
                               gamma=0.95, seed=args.seed)
        assess(kind, 0, agent)
        checkpoints = {100, 500, 1000, 5000, 10000, 25000, 50000, limit}
        started = time.monotonic()
        for ep in range(1, limit + 1):
            metrics = run_episode(kind, random.Random(args.seed), agent, True)
            agent.decay_epsilon()
            curve.append(dict(agent=kind, episode=ep, epsilon=agent.epsilon, **metrics))
            if ep in checkpoints:
                assess(kind, ep, agent)
                write_csv(out/'learning_curve.csv', curve)
                if ep in (1000, 10000, 50000, limit):
                    agent.save(str(out/f'{kind}_ep{ep}.pkl'))
            if ep % 1000 == 0:
                print(f'{kind} ep={ep} seconds={time.monotonic()-started:.1f} states={len(agent.q)}', flush=True)
        config[kind + '_seconds'] = time.monotonic() - started
        del agent
    config['finished_utc'] = datetime.now(timezone.utc).isoformat()
    (out/'config.json').write_text(json.dumps(config, indent=2))
    write_csv(out/'learning_curve.csv', curve)


if __name__ == '__main__':
    main()
