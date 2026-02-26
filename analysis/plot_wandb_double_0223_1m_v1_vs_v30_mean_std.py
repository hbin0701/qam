#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


V1_RE = re.compile(r"\[QAM,\s*Online,\s*V1\s*\(Sparse,\s*Dense\),\s*SEED\s+\d+\]", re.IGNORECASE)
V30_RE = re.compile(r"\[QAM,\s*Online,\s*V30\s*\(Sparse,\s*Dense\),\s*SEED\s+\d+\]", re.IGNORECASE)

DISPLAY = {
    "V1": "Sparse",
    "V30": "Reward Shaped (Human)",
}
COLORS = {
    "V1": "#2563eb",
    "V30": "#dc2626",
}


def label_from_name(name: str) -> str | None:
    if V1_RE.search(name):
        return "V1"
    if V30_RE.search(name):
        return "V30"
    return None


def collect_rows(entity: str, project: str, metric: str, samples_per_run: int) -> tuple[list[dict], list[dict]]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}"))
    rows: list[dict] = []
    matched_runs: list[dict] = []
    for run in runs:
        name = run.name or ""
        label = label_from_name(name)
        if label is None:
            continue
        matched_runs.append(
            {
                "run_id": run.id,
                "run_name": name,
                "label": label,
                "display_label": DISPLAY[label],
            }
        )
        history = run.history(keys=["_step", metric], samples=samples_per_run, pandas=False)
        for item in history:
            step = item.get("_step")
            value = item.get(metric)
            if step is None or value is None:
                continue
            rows.append(
                {
                    "label": label,
                    "display_label": DISPLAY[label],
                    "run_id": run.id,
                    "run_name": name,
                    "step": int(step),
                    "value": float(value),
                }
            )
    return rows, matched_runs


def aggregate(rows: list[dict]) -> list[dict]:
    by_key = defaultdict(list)
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
                "display_label": DISPLAY[label],
                "step": step,
                "mean_success": mean,
                "std_success": std,
                "count": len(vals),
            }
        )
    return out


def plot_rl_style(agg: list[dict], metric: str, out_png: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=180)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    by_label = defaultdict(list)
    for row in agg:
        by_label[row["label"]].append(row)

    for label in ("V1", "V30"):
        pts = sorted(by_label.get(label, []), key=lambda r: r["step"])
        if not pts:
            continue
        x = [p["step"] for p in pts]
        y = [p["mean_success"] for p in pts]
        s = [p["std_success"] for p in pts]
        ax.plot(x, y, color=COLORS[label], linewidth=3.0, label=DISPLAY[label])
        ax.fill_between(
            x,
            [max(0.0, yy - ss) for yy, ss in zip(y, s)],
            [min(1.0, yy + ss) for yy, ss in zip(y, s)],
            color=COLORS[label],
            alpha=0.20,
            linewidth=0,
        )

    ax.set_title("CUBE_TASK_DOUBLE_0223 (1M) | V1 vs V30", fontsize=15, fontweight="bold", pad=10)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", frameon=True, fontsize=11, title="Reward")
    fig.tight_layout()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project", default="CUBE_TASK_DOUBLE_0223")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument("--out-dir", default="qam/analysis/artifacts/CUBE_TASK_DOUBLE_0223_1m_v1_vs_v30")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, matched_runs = collect_rows(
        entity=args.entity,
        project=args.project,
        metric=args.metric,
        samples_per_run=args.samples_per_run,
    )
    if not rows:
        raise RuntimeError("No matching V1/V30 runs found.")

    agg = aggregate(rows)

    matched_csv = out_dir / "matched_runs.csv"
    with matched_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_id", "run_name", "label", "display_label"])
        writer.writeheader()
        writer.writerows(matched_runs)

    raw_csv = out_dir / "raw_points.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "display_label", "run_id", "run_name", "step", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_csv = out_dir / "aggregated_mean_std.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "display_label", "step", "mean_success", "std_success", "count"],
        )
        writer.writeheader()
        writer.writerows(agg)

    out_png = out_dir / "double_0223_1m_v1_vs_v30_mean_std.png"
    plot_rl_style(agg=agg, metric=args.metric, out_png=out_png)

    print(f"Saved matched runs: {matched_csv}")
    print(f"Saved raw points: {raw_csv}")
    print(f"Saved aggregate: {agg_csv}")
    print(f"Saved figure: {out_png}")
    print(f"Matched runs: {len(matched_runs)}")


if __name__ == "__main__":
    main()
