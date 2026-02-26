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
VERSION_RE = re.compile(r"\bV(1|30)\b", re.IGNORECASE)


def collect_rows(entity: str, project: str, metric: str, samples_per_run: int) -> list[dict]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}"))
    rows = []
    for run in runs:
        run_name = run.name or ""
        if not PREFIX_RE.search(run_name):
            # Include default (1M) online runs too.
            if "online" in run_name.lower() and "sparse, dense" in run_name.lower():
                version_match = VERSION_RE.search(run_name)
                if not version_match:
                    continue
                size = "1m"
                version = f"v{version_match.group(1)}"
            else:
                continue
        else:
            size_match = SIZE_RE.search(run_name)
            version_match = VERSION_RE.search(run_name)
            if not size_match or not version_match:
                continue
            size = size_match.group(1).lower()
            version = f"v{version_match.group(1)}"

        history = run.history(keys=["_step", metric], samples=samples_per_run, pandas=False)
        for item in history:
            step = item.get("_step")
            value = item.get(metric)
            if step is None or value is None:
                continue
            rows.append(
                {
                    "run_id": run.id,
                    "run_name": run_name,
                    "size": size,
                    "version": version,
                    "step": int(step),
                    "value": float(value),
                }
            )
    return rows


def aggregate(rows: list[dict]) -> list[dict]:
    bucket = defaultdict(list)
    for row in rows:
        bucket[(row["size"], row["version"], row["step"])].append(row["value"])

    out = []
    for (size, version, step), vals in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out.append(
            {
                "size": size,
                "version": version,
                "step": step,
                "mean_success": mean,
                "std_success": std,
                "count": len(vals),
            }
        )
    return out


def plot_modern(agg_rows: list[dict], out_png: Path) -> None:
    order = ["1m", "100k", "10k", "1k"]
    display_size = {"1m": "1M", "100k": "100K", "10k": "10K", "1k": "1K"}
    colors = {"v1": "#2563eb", "v30": "#dc2626"}
    labels = {"v1": "Sparse", "v30": "Reward Shaped (Human)"}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=180, sharey=True)
    fig.patch.set_facecolor("#f8fafc")

    title = "Cube_Taks_Single [QAM, Online]"
    fig.suptitle(title, fontsize=20, fontweight="bold", y=0.97)

    by_size_ver = defaultdict(list)
    for row in agg_rows:
        by_size_ver[(row["size"], row["version"])].append(row)

    for ax, panel in zip(axes.ravel(), order):
        ax.set_facecolor("#ffffff")

        ax.set_title(display_size[panel], fontsize=13, pad=8)
        for version in ("v1", "v30"):
            rows = sorted(by_size_ver.get((panel, version), []), key=lambda r: r["step"])
            if not rows:
                continue
            x = [r["step"] for r in rows]
            y = [r["mean_success"] for r in rows]
            s = [r["std_success"] for r in rows]
            ax.plot(x, y, color=colors[version], linewidth=2.6, alpha=0.95)
            ax.fill_between(
                x,
                [max(0.0, yy - ss) for yy, ss in zip(y, s)],
                [min(1.0, yy + ss) for yy, ss in zip(y, s)],
                color=colors[version],
                alpha=0.18,
                linewidth=0,
            )

        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.22)
        ax.tick_params(labelsize=10)
        ax.set_xlabel("Step", fontsize=11)
        ax.set_ylabel("eval/success", fontsize=11)

    legend_handles = []
    for version in ("v1", "v30"):
        (h,) = axes.ravel()[0].plot([], [], color=colors[version], linewidth=3, label=labels[version])
        legend_handles.append(h)
    fig.legend(
        handles=legend_handles,
        labels=[labels["v1"], labels["v30"]],
        loc="lower center",
        ncol=2,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
        fontsize=12,
    )

    fig.tight_layout(rect=[0, 0.08, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project", default="CUBE_TASK_SINGLE_0223")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument(
        "--out-csv",
        default="qam/analysis/cube_task_single_0223_dataset_size_v1_vs_v30_eval_success.csv",
    )
    parser.add_argument(
        "--out-png",
        default="qam/analysis/cube_task_single_0223_dataset_size_v1_vs_v30_eval_success_modern.png",
    )
    args = parser.parse_args()

    rows = collect_rows(args.entity, args.project, args.metric, args.samples_per_run)
    if not rows:
        raise RuntimeError("No matching dataset-size runs found in W&B.")

    agg_rows = aggregate(rows)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["size", "version", "step", "mean_success", "std_success", "count"]
        )
        writer.writeheader()
        writer.writerows(agg_rows)

    plot_modern(agg_rows, Path(args.out_png))
    print(f"Saved CSV: {out_csv}")
    print(f"Saved plot: {args.out_png}")
    print(f"Total raw points: {len(rows)}")
    print(f"Total aggregated points: {len(agg_rows)}")


if __name__ == "__main__":
    main()
