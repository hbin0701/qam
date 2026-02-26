#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


STEP_RE = re.compile(r"Step\s+(\d+)\s+Evaluation:\s+Success Rate\s+=\s+([0-9]*\.?[0-9]+)")
SEED_RE = re.compile(r"\[QAM,\s*Online,\s*V(\d+).+SEED\s+(\d+)\]", re.IGNORECASE)


def parse_log(path: Path, version: str) -> dict[int, dict[int, float]]:
    by_seed = defaultdict(dict)
    current_seed = None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            seed_match = SEED_RE.search(raw_line)
            if seed_match and seed_match.group(1) == version:
                current_seed = int(seed_match.group(2))
                continue
            step_match = STEP_RE.search(raw_line)
            if step_match and current_seed is not None:
                step = int(step_match.group(1))
                success = float(step_match.group(2))
                by_seed[current_seed][step] = success
    return by_seed


def compute_stats(by_seed: dict[int, dict[int, float]]) -> list[dict]:
    values_by_step = defaultdict(list)
    for _, step_map in by_seed.items():
        for step, success in step_map.items():
            values_by_step[step].append(success)

    rows = []
    for step in sorted(values_by_step):
        vals = values_by_step[step]
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        rows.append({"step": step, "mean_success": mean, "std_success": std, "count": len(vals)})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-dir",
        default="qam/logs/slurm",
        help="Directory with single_online_v*_0223*.out files.",
    )
    parser.add_argument(
        "--out-csv",
        default="qam/analysis/cube_task_single_0223_qam_online_v1_vs_v30_aggregated.csv",
    )
    parser.add_argument(
        "--out-png",
        default="qam/analysis/cube_task_single_0223_qam_online_v1_vs_v30_modern.png",
    )
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    files_v1 = sorted(log_dir.glob("single_online_v1_0223*.out"))
    files_v30 = sorted(log_dir.glob("single_online_v30_0223*.out"))
    if not files_v1 or not files_v30:
        raise RuntimeError("Could not find both v1 and v30 0223 logs.")

    by_seed_v1 = defaultdict(dict)
    by_seed_v30 = defaultdict(dict)
    for path in files_v1:
        for seed, step_map in parse_log(path, "1").items():
            by_seed_v1[seed].update(step_map)
    for path in files_v30:
        for seed, step_map in parse_log(path, "30").items():
            by_seed_v30[seed].update(step_map)

    stats_v1 = compute_stats(dict(by_seed_v1))
    stats_v30 = compute_stats(dict(by_seed_v30))

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["label", "step", "mean_success", "std_success", "count"]
        )
        writer.writeheader()
        for row in stats_v1:
            writer.writerow({"label": "Sparse", **row})
        for row in stats_v30:
            writer.writerow({"label": "Reward Shaped (Human)", **row})

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#fefefe")

    colors = {"Sparse": "#2563eb", "Reward Shaped (Human)": "#ef4444"}
    for label, stats in (
        ("Sparse", stats_v1),
        ("Reward Shaped (Human)", stats_v30),
    ):
        x = [row["step"] for row in stats]
        y = [row["mean_success"] for row in stats]
        s = [row["std_success"] for row in stats]
        ax.plot(x, y, label=label, color=colors[label], linewidth=2.8)
        ax.fill_between(
            x,
            [max(0.0, yy - ss) for yy, ss in zip(y, s)],
            [min(1.0, yy + ss) for yy, ss in zip(y, s)],
            color=colors[label],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_title("Cube_Task_Single [QAM, Online]", fontsize=14, pad=12)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved CSV: {out_csv}")
    print(f"Saved plot: {out_png}")
    print(f"Sparse seeds: {sorted(by_seed_v1.keys())}")
    print(f"Reward Shaped (Human) seeds: {sorted(by_seed_v30.keys())}")


if __name__ == "__main__":
    main()
