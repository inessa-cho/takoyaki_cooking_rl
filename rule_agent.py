from env import (
    TakoyakiEnv,
    EMPTY, BATTER, OCTOPUS, SHAPED, COOKED, FINISHED, BURNT,
    WAIT, ACT_BATTER, ACT_OCTOPUS, ACT_FLIP,
    ACT_MOVE_TO_TRAY, ACT_SAUCE_TRAY, ACT_SERVE_TRAY, ACT_CLEAR,
)

EPISODES = 100


def choose_action(env: TakoyakiEnv) -> int:
    # 1. tray full이면 소스 -> 서빙
    if env.tray_count == 8 and not env.tray_sauced:
        return env.encode_action(ACT_SAUCE_TRAY)

    if env.tray_count == 8 and env.tray_sauced:
        return env.encode_action(ACT_SERVE_TRAY)

    # 2. 탄 것 치우기
    for i, state in enumerate(env.cells):
        if state == BURNT:
            return env.encode_action(ACT_CLEAR, i)

    # 3. 완성된 것 트레이로 옮기기
    for i, state in enumerate(env.cells):
        if state == FINISHED and env.tray_count < 8:
            return env.encode_action(ACT_MOVE_TO_TRAY, i)

    # 4. COOKED는 빨리 두 번째 뒤집기
    for i, state in enumerate(env.cells):
        if state == COOKED:
            return env.encode_action(ACT_FLIP, i)

    # 5. OCTOPUS는 바로 첫 뒤집기
    for i, state in enumerate(env.cells):
        if state == OCTOPUS:
            return env.encode_action(ACT_FLIP, i)

    # 6. BATTER에는 문어 넣기
    for i, state in enumerate(env.cells):
        if state == BATTER:
            return env.encode_action(ACT_OCTOPUS, i)

    # 7. 빈 칸에 반죽 붓기
    for i, state in enumerate(env.cells):
        if state == EMPTY:
            return env.encode_action(ACT_BATTER, i)

    # 8. 할 거 없으면 기다리기
    return env.encode_action(WAIT)


def main() -> None:
    scores = []
    served_counts = []
    invalid_counts = []
    burnt_counts = []

    for ep in range(EPISODES):
        env = TakoyakiEnv()
        obs = env.reset()

        total_reward = 0
        done = False

        while not done:
            action = choose_action(env)
            obs, reward, done, info = env.step(action)
            total_reward += reward

        scores.append(total_reward)
        served_counts.append(info["score"])
        invalid_counts.append(info["invalid_actions"])
        burnt_counts.append(info["burnt"])

    print("=== Rule Agent Result ===")
    print("episodes:", EPISODES)
    print("avg_score:", sum(scores) / EPISODES)
    print("avg_served:", sum(served_counts) / EPISODES)
    print("avg_invalid_actions:", sum(invalid_counts) / EPISODES)
    print("avg_burnt:", sum(burnt_counts) / EPISODES)
    print("best_score:", max(scores))
    print("worst_score:", min(scores))


if __name__ == "__main__":
    main()
