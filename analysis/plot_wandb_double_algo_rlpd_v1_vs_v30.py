#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


RLPD_V1_RE = re.compile(r"\[RLPD,\s*Online,\s*V1\s*\(Sparse,\s*Dense\),\s*SEED\s+\d+\]", re.IGNORECASE)
RLPD_V30_RE = re.compile(
    r"\[RLPD,\s*Online,\s*V30\s*\(Sparse,\s*Dense\),\s*SEED\s+\d+\]", re.IGNORECASE
)


def label_from_run(run_name: str) -> str | None:
    if RLPD_V30_RE.search(run_name):
        return "Reward Shaped (Human)"
    if RLPD_V1_RE.search(run_name):
        return "Sparse"
    return None


def collect_rows(entity: str, project: str, metric: str, samples_per_run: int) -> list[dict]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}"))
    rows: list[dict] = []

    for run in runs:
        name = run.name or ""
        label = label_from_run(name)
        if label is None:
            continue

        history = run.history(keys=["_step", metric], samples=samples_per_run, pandas=False)
        for item in history:
            step = item.get("_step")
            value = item.get(metric)
            if step is None or value is None:
                continue
            rows.append(
                {
                    "label": label,
                    "run_id": run.id,
                    "run_name": name,
                    "step": int(step),
                    "value": float(value),
                }
            )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["label"], row["step"])].append(row["value"])

    out = []
    for (label, step), vals in sorted(by_key.items(), key=lambda x: (x[0][0], x[0][1])):
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out.append(
            {
                "label": label,
                "step": step,
                "mean_success": mean,
                "std_success": std,
                "count": len(vals),
            }
        )
    return out


def plot_modern(agg: list[dict], out_png: Path) -> None:
    colors = {
        "Reward Shaped (Human)": "#dc2626",
        "Sparse": "#2563eb",
    }
    order = ["Reward Shaped (Human)", "Sparse"]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    by_label = defaultdict(list)
    for row in agg:
        by_label[row["label"]].append(row)

    for label in order:
        pts = sorted(by_label.get(label, []), key=lambda r: r["step"])
        if not pts:
            continue
        x = [p["step"] for p in pts]
        y = [p["mean_success"] for p in pts]
        s = [p["std_success"] for p in pts]
        ax.plot(x, y, color=colors[label], linewidth=4.2, label=label)
        ax.fill_between(
            x,
            [max(0.0, yy - ss) for yy, ss in zip(y, s)],
            [min(1.0, yy + ss) for yy, ss in zip(y, s)],
            color=colors[label],
            alpha=0.18,
            linewidth=0,
        )

    ax.set_title("RLPD [Online]", fontsize=17, fontweight="bold", pad=12)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel("eval/success", fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.20)
    legend = ax.legend(loc="lower right", frameon=True, fontsize=11, title="Reward Type")
    legend.get_frame().set_facecolor("#ffffff")
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor("#d1d5db")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project", default="CUBE_TASK_DOUBLE_ALGO")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument(
        "--out-csv",
        default="qam/analysis/cube_task_double_algo_rlpd_v1_vs_v30_eval_success_mean_std.csv",
    )
    parser.add_argument(
        "--out-png",
        default="qam/analysis/cube_task_double_algo_rlpd_v1_vs_v30_modern.png",
    )
    args = parser.parse_args()

    rows = collect_rows(args.entity, args.project, args.metric, args.samples_per_run)
    if not rows:
        raise RuntimeError("No matching RLPD V1/V30 runs found in W&B.")

    agg = aggregate(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "step", "mean_success", "std_success", "count"])
        writer.writeheader()
        writer.writerows(agg)

    plot_modern(agg, Path(args.out_png))
    print(f"Saved CSV: {out_csv}")
    print(f"Saved plot: {args.out_png}")
    print(f"Raw points: {len(rows)}")
    print(f"Aggregated points: {len(agg)}")


if __name__ == "__main__":
    main()
