#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


PREFIX_RE = re.compile(r"^\s*\[?\s*datas?et\s*size\s*:", re.IGNORECASE)
SIZE_RE = re.compile(r"DS-(1k|10k|100k|250k|500k)", re.IGNORECASE)


def collect_rows(entity: str, project: str, metric: str, samples_per_run: int) -> list[dict]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}"))
    rows = []
    for run in runs:
        name = run.name or ""
        lowered = name.lower()

        size = None
        if PREFIX_RE.search(name):
            if " v1," not in lowered:
                continue
            m = SIZE_RE.search(name)
            if not m:
                continue
            size = m.group(1).lower()
        elif "[qam, online, v1 (sparse, dense), seed" in lowered:
            size = "1m"
        else:
            continue

        history = run.history(keys=["_step", metric], samples=samples_per_run, pandas=False)
        for item in history:
            step = item.get("_step")
            value = item.get(metric)
            if step is None or value is None:
                continue
            rows.append(
                {
                    "size": size,
                    "step": int(step),
                    "value": float(value),
                }
            )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["size"], row["step"])].append(row["value"])

    out = []
    for (size, step), vals in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out.append(
            {
                "size": size,
                "step": step,
                "mean_success": mean,
                "std_success": std,
                "count": len(vals),
            }
        )
    return out


def first_step_at_threshold(points: list[dict], threshold: float) -> int | None:
    for p in points:
        if p["mean_success"] >= threshold:
            return p["step"]
    return None


def plot_speed(agg: list[dict], out_png: Path) -> list[dict]:
    order = ["1m", "500k", "250k", "100k", "10k", "1k"]
    display = {"1m": "1M", "500k": "500K", "250k": "250K", "100k": "100K", "10k": "10K", "1k": "1K"}
    colors = {
        "1m": "#000000",
        "500k": "#7c3aed",
        "250k": "#2563eb",
        "100k": "#10b981",
        "10k": "#f59e0b",
        "1k": "#ef4444",
    }
    label_offset = {"1m": -0.02, "500k": -0.015, "250k": -0.01, "100k": -0.005, "10k": -0.012, "1k": -0.018}
    label_step = {"100k": 10000, "250k": 10000, "500k": 20000, "1m": 20000, "10k": 20000, "1k": 10000}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 7), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    by_size = defaultdict(list)
    for row in agg:
        by_size[row["size"]].append(row)

    speed_rows = []
    for size in order:
        points = sorted(by_size.get(size, []), key=lambda r: r["step"])
        if not points:
            continue
        points_50k = [p for p in points if p["step"] <= 50000]
        if not points_50k:
            continue
        if size == "1m":
            points_50k = [p for p in points_50k if p["step"] % 10000 == 0]
            if not points_50k:
                continue
        x = [p["step"] for p in points_50k]
        y = [p["mean_success"] for p in points_50k]
        s = [p["std_success"] for p in points_50k]
        if x[0] > 0:
            x = [0] + x
            y = [0.0] + y
            s = [0.0] + s

        line_style = "--" if size == "1m" else "-"
        line_width = 4.0 if size == "1m" else 2.6
        ax.plot(x, y, label=display[size], color=colors[size], linewidth=line_width, linestyle=line_style)
        ax.fill_between(
            x,
            [max(0.0, yy - ss) for yy, ss in zip(y, s)],
            [min(1.0, yy + ss) for yy, ss in zip(y, s)],
            color=colors[size],
            alpha=0.15,
            linewidth=0,
        )
        target_step = label_step[size]
        idx = min(range(len(x)), key=lambda i: abs(x[i] - target_step))
        text_x = min(49500, x[idx] + 900)
        text_y = min(1.02, max(0.02, y[idx] + label_offset[size]))
        ax.text(
            text_x,
            text_y,
            display[size],
            color=colors[size],
            fontsize=15,
            fontweight="bold" if size == "1m" else "normal",
            ha="left",
            va="center",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.7, "pad": 0.25},
        )

        speed_rows.append(
            {
                "size": display[size],
                "n_runs": points[0]["count"],
                "step_at_0.8": first_step_at_threshold(points, 0.8),
                "step_at_0.95": first_step_at_threshold(points, 0.95),
                "final_step": points[-1]["step"],
                "final_success": points[-1]["mean_success"],
            }
        )

    ax.set_title("Cube_Taks_Single [QAM, Online] | Sparse Only", fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("eval/success", fontsize=12)
    ax.set_xlim(0, 50000)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.22)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)
    return speed_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project", default="CUBE_TASK_SINGLE_0223")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument("--out-csv", default="qam/analysis/cube_task_single_0223_sparse_size_curves.csv")
    parser.add_argument("--out-speed-csv", default="qam/analysis/cube_task_single_0223_sparse_speed_summary.csv")
    parser.add_argument("--out-png", default="qam/analysis/cube_task_single_0223_sparse_size_speed_modern.png")
    args = parser.parse_args()

    rows = collect_rows(args.entity, args.project, args.metric, args.samples_per_run)
    if not rows:
        raise RuntimeError("No sparse (V1) runs found.")

    agg = aggregate(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["size", "step", "mean_success", "std_success", "count"])
        writer.writeheader()
        writer.writerows(agg)

    speed_rows = plot_speed(agg, Path(args.out_png))

    out_speed = Path(args.out_speed_csv)
    with out_speed.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["size", "n_runs", "step_at_0.8", "step_at_0.95", "final_step", "final_success"]
        )
        writer.writeheader()
        writer.writerows(speed_rows)

    print(f"Saved curve CSV: {out_csv}")
    print(f"Saved speed CSV: {out_speed}")
    print(f"Saved plot: {args.out_png}")
    print(f"Total raw points: {len(rows)}")
    print(f"Total aggregated points: {len(agg)}")


if __name__ == "__main__":
    main()
