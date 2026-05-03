import argparse
import numpy as np

from stable_baselines3 import PPO

from godot_rl_vec_env import GodotVecEnv
from rl_bridge.utils.process_manager import ProcessManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot-exe", required=True)
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--env-path", default=None, help="Path to exported game .exe (optional)")
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--episodes", type=int, default=50)
    args = parser.parse_args()

    ports = [args.base_port]

    manager = None
    if args.env_path is None:
        manager = ProcessManager(args.godot_exe, args.project_path, base_port=args.base_port)
        manager.start(1, show_window=args.show_window)

    env = GodotVecEnv(ports=ports, env_path=args.env_path, show_window=args.show_window)
    model = PPO.load(args.model)

    rewards = []
    wins = 0
    for _ in range(args.episodes):
        obs = env.reset()
        done = False
        total = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += float(reward[0])
        rewards.append(total)
        if total > 0:
            wins += 1

    print(f"Avg reward: {np.mean(rewards):.2f}")
    print(f"Win rate: {wins / args.episodes:.2f}")

    env.close()
    if manager:
        manager.stop_all()


if __name__ == "__main__":
    main()
