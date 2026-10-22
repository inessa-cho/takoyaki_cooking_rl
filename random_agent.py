import random
from env import TakoyakiEnv

EPISODES = 100

def main() -> None:
    random.seed(10000)
    scores = []
    served_counts = []
    invalid_counts = []

    for ep in range(EPISODES):
        env = TakoyakiEnv()
        obs = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = random.randint(0, env.n_actions - 1)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(total_reward)
        served_counts.append(info["score"])
        invalid_counts.append(info.get("invalid_actions", 0))

    print("=== Random Agent Result ===")
    print("episodes:", EPISODES)
    print("avg_score:", sum(scores) / EPISODES)
    print("avg_served:", sum(served_counts) / EPISODES)
    print("avg_invalid_actions:", sum(invalid_counts) / EPISODES)
    print("best_score:", max(scores))
    print("worst_score:", min(scores))


if __name__ == "__main__":
    main()
