import argparse
import csv
import os
from typing import List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt

    HAS_MPL = True
except Exception:
    HAS_MPL = False


def _read_training_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    episodes: List[int] = []
    returns: List[float] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            episodes.append(int(row["episode"]))
            returns.append(float(row["return"]))
    return np.asarray(episodes), np.asarray(returns)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) == 0:
        return values.copy()
    window = min(window, len(values))
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="valid")


def _latest_log_path(logs_dir: str, prefix: str) -> str:
    best_idx = -1
    best_path = ""
    for name in os.listdir(logs_dir):
        if not name.startswith(prefix) or not name.endswith(".csv"):
            continue
        suffix = name[len(prefix) : -len(".csv")]
        if not suffix.isdigit():
            continue
        idx = int(suffix)
        if idx > best_idx:
            best_idx = idx
            best_path = os.path.join(logs_dir, name)
    if best_path == "":
        raise FileNotFoundError(f"No CSVs found with prefix {prefix} in {logs_dir}")
    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-log", default=None, help="Path to training_log_*.csv")
    parser.add_argument("--window", type=int, default=25, help="Moving average window")
    parser.add_argument("--out-dir", default="logs/analysis", help="Output directory")
    args = parser.parse_args()

    logs_dir = os.path.join("logs", "training_logs")
    train_path = args.training_log or _latest_log_path(logs_dir, "training_log_")

    episodes, returns = _read_training_csv(train_path)
    if len(episodes) == 0:
        raise RuntimeError(f"No rows found in {train_path}")

    moving_avg = _moving_average(returns, args.window)
    best_so_far = np.maximum.accumulate(returns)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, "training_analysis.csv")
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["episode", "return", "moving_avg", "best_so_far"])
        for idx, ep in enumerate(episodes):
            ma_val = ""
            if idx >= args.window - 1:
                ma_val = f"{moving_avg[idx - (args.window - 1)]:.4f}"
            writer.writerow([ep, f"{returns[idx]:.4f}", ma_val, f"{best_so_far[idx]:.4f}"])

    if not HAS_MPL:
        print("matplotlib not installed; wrote analysis CSV only:", out_csv)
        print("Install with: pip install matplotlib")
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(episodes, returns, color="#9aa0a6", linewidth=1, alpha=0.6, label="return")
    ax.plot(episodes, best_so_far, color="#1a73e8", linewidth=2, label="best so far")

    if len(moving_avg) > 0:
        ma_x = episodes[args.window - 1 :]
        ax.plot(ma_x, moving_avg, color="#34a853", linewidth=2.5, label=f"moving avg ({args.window})")

    ax.set_title("Training Return Trend")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = os.path.join(args.out_dir, "training_return_trend.png")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print("Saved:", out_png)


if __name__ == "__main__":
    main()
