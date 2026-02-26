#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


ONLINE_V1_RE = re.compile(
    r"^\[Online\]\s*Double,\s*V1,\s*QAM,\s*DS-(10k|100k|250k|500k)\s*$",
    re.IGNORECASE,
)
ONE_M_V1_RE = re.compile(
    r"^\[QAM,\s*Online,\s*V1\s*\(Sparse,\s*Dense\),\s*SEED\s+\d+\]$",
    re.IGNORECASE,
)

SIZE_ORDER = ["10k", "100k", "250k", "500k", "1M"]
COLORS = {
    "10k": "#ef4444",   # red
    "100k": "#f59e0b",  # orange
    "250k": "#22c55e",  # green
    "500k": "#3b82f6",  # blue
}


def collect_sparse_rows(api: wandb.Api, entity: str, project_small: str, project_1m: str, metric: str, samples: int):
    rows = []
    runs_meta = []

    for run in api.runs(f"{entity}/{project_small}"):
        name = (run.name or "").strip()
        m = ONLINE_V1_RE.match(name)
        if not m:
            continue
        ds_size = m.group(1).lower()
        runs_meta.append(
            {
                "source_project": project_small,
                "run_id": run.id,
                "run_name": name,
                "ds_size": ds_size,
            }
        )
        history = run.history(keys=["_step", metric], samples=samples, pandas=False)
        for item in history:
            step = item.get("_step")
            val = item.get(metric)
            if step is None or val is None:
                continue
            rows.append(
                {
                    "source_project": project_small,
                    "run_id": run.id,
                    "run_name": name,
                    "ds_size": ds_size,
                    "step": int(step),
                    "value": float(val),
                }
            )

    for run in api.runs(f"{entity}/{project_1m}"):
        name = (run.name or "").strip()
        if not ONE_M_V1_RE.match(name):
            continue
        runs_meta.append(
            {
                "source_project": project_1m,
                "run_id": run.id,
                "run_name": name,
                "ds_size": "1M",
            }
        )
        history = run.history(keys=["_step", metric], samples=samples, pandas=False)
        for item in history:
            step = item.get("_step")
            val = item.get(metric)
            if step is None or val is None:
                continue
            rows.append(
                {
                    "source_project": project_1m,
                    "run_id": run.id,
                    "run_name": name,
                    "ds_size": "1M",
                    "step": int(step),
                    "value": float(val),
                }
            )

    return rows, runs_meta


def aggregate(rows):
    by_key = defaultdict(list)
    for row in rows:
        by_key[(row["ds_size"], row["step"])].append(row["value"])

    out = []
    for (ds_size, step), vals in sorted(by_key.items(), key=lambda x: (SIZE_ORDER.index(x[0][0]), x[0][1])):
        mean = sum(vals) / len(vals)
        std = 0.0
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        out.append(
            {
                "ds_size": ds_size,
                "step": step,
                "mean": mean,
                "std": std,
                "count": len(vals),
            }
        )
    return out


def plot_sparse_multisize(agg, metric: str, out_png: Path):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 7), dpi=200)
    fig.patch.set_facecolor("#f8fafc")
    ax.set_facecolor("#ffffff")

    by_size = defaultdict(list)
    for row in agg:
        by_size[row["ds_size"]].append(row)

    line_endpoints = []
    for ds_size in SIZE_ORDER:
        pts = sorted(by_size.get(ds_size, []), key=lambda r: r["step"])
        if not pts:
            continue
        x = [p["step"] for p in pts]
        y = [p["mean"] for p in pts]
        s = [p["std"] for p in pts]

        if ds_size == "1M":
            ax.plot(
                x,
                y,
                color="black",
                linestyle="--",
                linewidth=4.2,
                label="1M",
            )
            ax.fill_between(
                x,
                [max(0.0, yy - ss) for yy, ss in zip(y, s)],
                [min(1.0, yy + ss) for yy, ss in zip(y, s)],
                color="black",
                alpha=0.14,
                linewidth=0,
            )
            line_endpoints.append((ds_size, x[-1], y[-1], "black"))
        else:
            ax.plot(
                x,
                y,
                color=COLORS[ds_size],
                linewidth=2.8,
                label=f"{ds_size}",
            )
            line_endpoints.append((ds_size, x[-1], y[-1], COLORS[ds_size]))

    if line_endpoints:
        xmin, xmax = ax.get_xlim()
        xpad = max(1.0, 0.03 * (xmax - xmin))
        ax.set_xlim(xmin, xmax + 3.5 * xpad)
        for ds_size, x_end, y_end, color in line_endpoints:
            ax.text(
                x_end + xpad,
                y_end,
                f"{ds_size}",
                color=color,
                fontsize=10,
                va="center",
                ha="left",
                fontweight="bold" if ds_size == "1M" else "normal",
            )

    ax.set_title("Sparse Only | Double Task2 Across Dataset Sizes", fontsize=16, fontweight="bold", pad=12)
    ax.set_xlabel("Training Step", fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_ylim(0.0, 1.05)
    ax.grid(alpha=0.22)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=True,
        title="Dataset Size",
    )
    fig.tight_layout(rect=[0, 0.08, 1, 1])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project-small", default="0219-double-tmp1")
    parser.add_argument("--project-1m", default="CUBE_TASK_DOUBLE_0223")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument("--out-dir", default="qam/analysis/artifacts/double_sparse_multisize")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=120)
    rows, runs_meta = collect_sparse_rows(
        api=api,
        entity=args.entity,
        project_small=args.project_small,
        project_1m=args.project_1m,
        metric=args.metric,
        samples=args.samples_per_run,
    )
    if not rows:
        raise RuntimeError("No sparse runs found for multisize plot.")

    agg = aggregate(rows)

    runs_csv = out_dir / "matched_runs.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["source_project", "run_id", "run_name", "ds_size"])
        writer.writeheader()
        writer.writerows(runs_meta)

    raw_csv = out_dir / "raw_points.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["source_project", "run_id", "run_name", "ds_size", "step", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_csv = out_dir / "aggregated_mean_std.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["ds_size", "step", "mean", "std", "count"])
        writer.writeheader()
        writer.writerows(agg)

    out_png = out_dir / "double_sparse_multisize.png"
    plot_sparse_multisize(agg=agg, metric=args.metric, out_png=out_png)

    print(f"Saved matched runs: {runs_csv}")
    print(f"Saved raw points: {raw_csv}")
    print(f"Saved aggregate: {agg_csv}")
    print(f"Saved figure: {out_png}")
    print(f"Matched runs: {len(runs_meta)}")


if __name__ == "__main__":
    main()
