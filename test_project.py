"""Behavioral regression checks; run with python -m unittest -v."""
import random
import unittest
from env import *
from q_agent import QLearningAgent
from abstract_agent import abstract_state, action_candidates
from final_experiment import evaluate, run_episode
from train import evaluate as legacy_evaluate


class EnvironmentTests(unittest.TestCase):
    def test_action_encoding(self):
        env = TakoyakiEnv()
        self.assertEqual(env.n_actions, 83)
        for action in range(env.n_actions):
            self.assertEqual(env.encode_action(*env.decode_action(action)), action)

    def test_cooking_and_burning_boundaries(self):
        env = TakoyakiEnv()
        for kind in (ACT_BATTER, ACT_OCTOPUS, ACT_FLIP):
            env.step(env.encode_action(kind, 0))
        self.assertEqual((env.cells[0], env.timers[0]), (SHAPED, 1))
        for _ in range(4):
            env.step(WAIT)
        self.assertEqual(env.cells[0], SHAPED)
        env.step(WAIT)
        self.assertEqual((env.cells[0], env.timers[0]), (COOKED, 0))
        for _ in range(4):
            env.step(WAIT)
        self.assertEqual(env.cells[0], COOKED)
        _, reward, _, info = env.step(WAIT)
        self.assertEqual(env.cells[0], BURNT)
        self.assertEqual(info['burnt'], 1)
        self.assertAlmostEqual(reward, -0.21)
        env.step(env.encode_action(ACT_CLEAR, 0))
        self.assertEqual(env.cells[0], EMPTY)

    def test_tray_and_horizon(self):
        env = TakoyakiEnv()
        for i in range(8):
            env.cells[i] = FINISHED
            env.step(env.encode_action(ACT_MOVE_TO_TRAY, i))
        self.assertNotIn(env.encode_action(ACT_SERVE_TRAY), env.valid_actions())
        env.step(env.encode_action(ACT_SAUCE_TRAY))
        _, reward, _, _ = env.step(env.encode_action(ACT_SERVE_TRAY))
        self.assertAlmostEqual(reward, .99)
        self.assertEqual((env.score, env.tray_count, env.tray_sauced), (1, 0, False))
        while env.t < 199:
            self.assertFalse(env.step(WAIT)[2])
        self.assertTrue(env.step(WAIT)[2])

    def test_invalid_action(self):
        env = TakoyakiEnv()
        _, reward, _, info = env.step(env.encode_action(ACT_FLIP, 0))
        self.assertEqual(info['invalid_actions'], 1)
        self.assertAlmostEqual(reward, -.06)
        self.assertEqual(env.t, 1)

    def test_rule_regression(self):
        row = run_episode('rule', random.Random(0))
        self.assertAlmostEqual(row['score'], 6.63)
        self.assertEqual((row['served'], row['burnt'], row['invalid']), (4, 0, 0))

    def test_abstraction_actions_are_valid(self):
        rng = random.Random(99)
        env = TakoyakiEnv()
        for _ in range(200):
            candidates = action_candidates(env)
            self.assertTrue(set(candidates.values()).issubset(env.valid_actions()))
            self.assertEqual(len(abstract_state(env)), 6)
            env.step(rng.choice(list(candidates.values())))

    def test_evaluation_is_frozen_and_repeatable(self):
        agent = QLearningAgent(9, seed=42)
        run_episode('abstract_q', random.Random(42), agent, True)
        before = {s: dict(q) for s, q in agent.q.items()}
        rng = agent.rng.getstate()
        first = evaluate('abstract_q', 3, agent=agent)
        self.assertEqual(first, evaluate('abstract_q', 3, agent=agent))
        self.assertEqual(before, dict(agent.q))
        self.assertEqual(rng, agent.rng.getstate())

    def test_legacy_evaluation_seed_and_state(self):
        agent = QLearningAgent(83, seed=3)
        rng = agent.rng.getstate()
        a = legacy_evaluate(agent, 3)
        self.assertEqual(a, legacy_evaluate(agent, 3))
        self.assertEqual(rng, agent.rng.getstate())
        self.assertEqual(len(agent.q), 0)

    def test_terminal_q_update(self):
        agent = QLearningAgent(2, alpha=.5, seed=1)
        agent.q[('next',)][0] = 100
        agent.update(('state',), 1, 2., ('next',), [0], True)
        self.assertEqual(agent.q[('state',)][1], 1.)


if __name__ == '__main__':
    unittest.main()
