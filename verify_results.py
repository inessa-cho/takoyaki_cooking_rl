"""Re-evaluate saved checkpoints and baselines against recorded CSV metrics."""
import argparse
import csv
import json
from pathlib import Path
from q_agent import QLearningAgent
from final_experiment import evaluate, METRICS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir')
    args = parser.parse_args()
    directory = Path(args.run_dir)
    config = json.loads((directory/'config.json').read_text())
    expected = list(csv.DictReader((directory/'evaluation_episodes.csv').open()))
    cases = [('random_all', 0), ('random_valid', 0), ('random_abstract', 0), ('rule', 0)]
    for path in sorted(directory.glob('*_ep*.pkl')):
        kind, episode = path.stem.rsplit('_ep', 1)
        cases.append((kind, int(episode)))
    for kind, episode in cases:
        agent = QLearningAgent.load(str(directory/f'{kind}_ep{episode}.pkl')) if episode else None
        observed = evaluate(kind, config['eval_episodes'], config['eval_seed'], agent)
        saved = [r for r in expected if r['agent'] == kind and int(r['training_episodes']) == episode]
        assert len(saved) == len(observed), (kind, episode)
        for a, b in zip(saved, observed):
            for key in METRICS + ['eval_episode', 'eval_seed']:
                assert abs(float(a[key])-b[key]) < 1e-8, (kind, episode, key, a[key], b[key])
        print(f'PASS {kind} episodes={episode}: {len(saved)} evaluation episodes reproduced')


if __name__ == '__main__':
    main()
