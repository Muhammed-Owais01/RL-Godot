import argparse
import csv
import os
import time

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

    logs_root = os.path.join("logs", "eval_logs")
    os.makedirs(logs_root, exist_ok=True)
    run_idx = 1
    for name in os.listdir(logs_root):
        if not name.startswith("eval_log_") or not name.endswith(".csv"):
            continue
        suffix = name[len("eval_log_"):-len(".csv")]
        if suffix.isdigit():
            run_idx = max(run_idx, int(suffix) + 1)

    csv_path = os.path.join(logs_root, f"eval_log_{run_idx}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["episode", "return", "length", "elapsed_sec"])
    print(f"[eval] Logging to {os.path.abspath(csv_path)}", flush=True)

    ports = [args.base_port]

    manager = None
    if args.env_path is None:
        manager = ProcessManager(args.godot_exe, args.project_path, base_port=args.base_port)
        manager.start(1, show_window=args.show_window)

    env = GodotVecEnv(ports=ports, env_path=args.env_path, show_window=args.show_window)
    model = PPO.load(args.model)

    rewards = []
    wins = 0
    start_time = time.time()
    for ep_idx in range(1, args.episodes + 1):
        obs = env.reset()
        done = False
        total = 0.0
        length = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            total += float(reward[0])
            length += 1
        rewards.append(total)
        if total > 0:
            wins += 1

        elapsed = time.time() - start_time
        csv_writer.writerow([ep_idx, f"{total:.2f}", length, f"{elapsed:.1f}"])
        csv_file.flush()

    print(f"Avg reward: {np.mean(rewards):.2f}")
    print(f"Win rate: {wins / args.episodes:.2f}")

    csv_file.close()
    env.close()
    if manager:
        manager.stop_all()


if __name__ == "__main__":
    main()
