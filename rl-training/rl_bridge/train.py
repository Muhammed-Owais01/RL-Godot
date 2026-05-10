import argparse
import csv
import os
import time
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO

from godot_rl_vec_env import GodotVecEnv
from rl_bridge.utils.process_manager import ProcessManager


class ProgressCallback(BaseCallback):
    def __init__(self, n_envs: int, log_dir: str = "logs", max_episodes: int = 0, save_every_episodes: int = 25):
        super().__init__(verbose=0)
        self.n_envs = n_envs
        self.max_episodes = max_episodes
        self.save_every_episodes = save_every_episodes
        self.episode_count = 0
        self.ep_returns = [0.0] * n_envs
        self.ep_lengths = [0] * n_envs
        self.best_return = float("-inf")
        self.start_time = time.time()

        # CSV logging (unique per run, incremental)
        logs_root = os.path.join(log_dir, "training_logs")
        os.makedirs(logs_root, exist_ok=True)
        run_idx = self._get_next_run_index(logs_root)
        self.csv_path = os.path.join(logs_root, f"training_log_{run_idx}.csv")
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "episode", "return", "length", "best_return", "total_steps", "elapsed_sec"
        ])
        print(f"[train] Logging to {os.path.abspath(self.csv_path)}", flush=True)

    def _get_next_run_index(self, logs_root: str) -> int:
        max_idx = 0
        for name in os.listdir(logs_root):
            if not name.startswith("training_log_") or not name.endswith(".csv"):
                continue
            suffix = name[len("training_log_"):-len(".csv")]
            if suffix.isdigit():
                max_idx = max(max_idx, int(suffix))
        return max_idx + 1

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        if rewards is None or dones is None:
            return True

        for i in range(self.n_envs):
            self.ep_returns[i] += float(rewards[i])
            self.ep_lengths[i] += 1
            if dones[i]:
                self.episode_count += 1
                ep_return = self.ep_returns[i]
                ep_length = self.ep_lengths[i]

                if ep_return > self.best_return:
                    self.best_return = ep_return
                    self.model.save("checkpoints/best_model")

                elapsed = time.time() - self.start_time
                print(
                    "[train] ep=%d  return=%.2f  len=%d  best=%.2f  steps=%d  time=%.0fs"
                    % (self.episode_count, ep_return, ep_length, self.best_return, self.num_timesteps, elapsed),
                    flush=True,
                )

                # Write to CSV
                self.csv_writer.writerow([
                    self.episode_count, f"{ep_return:.2f}", ep_length,
                    f"{self.best_return:.2f}", self.num_timesteps, f"{elapsed:.1f}"
                ])
                self.csv_file.flush()

                # Periodic checkpoint
                if self.episode_count % self.save_every_episodes == 0:
                    path = f"checkpoints/checkpoint_ep{self.episode_count}"
                    self.model.save(path)
                    print(f"[train] Checkpoint saved: {path}", flush=True)

                self.ep_returns[i] = 0.0
                self.ep_lengths[i] = 0
                if self.max_episodes > 0 and self.episode_count >= self.max_episodes:
                    print("[train] Max episodes reached, stopping", flush=True)
                    return False

        return True

    def _on_training_end(self) -> None:
        self.csv_file.close()
        elapsed = time.time() - self.start_time
        print(f"[train] Training complete. {self.episode_count} episodes in {elapsed:.0f}s", flush=True)
        print(f"[train] Best return: {self.best_return:.2f}", flush=True)
        print(f"[train] Log saved to: {os.path.abspath(self.csv_path)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--godot-exe", required=True)
    parser.add_argument("--project-path", required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--base-port", type=int, default=11008)
    parser.add_argument("--env-path", default=None, help="Path to exported game .exe (optional)")
    parser.add_argument("--show-window", action="store_true")
    parser.add_argument("--speedup", type=int, default=None)
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--max-episodes", type=int, default=0)
    parser.add_argument("--resume", default=None, help="Path to saved model to resume training from")
    parser.add_argument("--device", default="auto", help="Device to use for training (cuda, cpu, auto)")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)

    ports = [args.base_port + i for i in range(args.num_envs)]

    manager = None
    if args.env_path is None:
        manager = ProcessManager(args.godot_exe, args.project_path, base_port=args.base_port)
        manager.start(args.num_envs, show_window=args.show_window)

    env = GodotVecEnv(
        ports=ports,
        env_path=args.env_path,
        show_window=args.show_window,
        speedup=args.speedup,
    )

    # Device selection
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"[train] Using device: {device}", flush=True)

    if args.resume:
        print(f"[train] Resuming from: {args.resume}", flush=True)
        model = PPO.load(args.resume, env=env, tensorboard_log="logs/", device=device)
    else:
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            clip_range=args.clip_range,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            verbose=1,
            tensorboard_log="logs/",
            device=device,
        )

    callback = ProgressCallback(env.num_envs, max_episodes=args.max_episodes)
    model.learn(total_timesteps=args.total_steps, callback=callback)
    model.save("checkpoints/final_model")
    print("[train] Final model saved to checkpoints/final_model", flush=True)

    env.close()
    if manager:
        manager.stop_all()


if __name__ == "__main__":
    main()
