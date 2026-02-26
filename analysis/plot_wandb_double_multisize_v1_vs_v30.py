#!/usr/bin/env python3
import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


ONLINE_SIZE_RE = re.compile(
    r"^\[Online\]\s*Double,\s*(V1|V30),\s*QAM,\s*DS-(10k|100k|250k|500k)\s*$",
    re.IGNORECASE,
)
ONE_M_RE = re.compile(
    r"^\[QAM,\s*Online,\s*(V1|V30)\s*\(Sparse,\s*Dense\),\s*SEED\s+\d+\]$",
    re.IGNORECASE,
)

DISPLAY = {"V1": "Dense", "V30": "Reward Shaped (Human)"}
COLORS = {"V1": "#16a34a", "V30": "#dc2626"}
SIZE_ORDER = ["10k", "100k", "250k", "500k", "1M"]


def collect_0219_rows(api: wandb.Api, entity: str, project: str, metric: str, samples: int) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    runs_meta: list[dict] = []
    for run in api.runs(f"{entity}/{project}"):
        name = (run.name or "").strip()
        m = ONLINE_SIZE_RE.match(name)
        if not m:
            continue
        version = m.group(1).upper()
        ds_size = m.group(2).lower()
        runs_meta.append(
            {
                "source_project": project,
                "run_id": run.id,
                "run_name": name,
                "version": version,
                "display_label": DISPLAY[version],
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
                    "source_project": project,
                    "run_id": run.id,
                    "run_name": name,
                    "version": version,
                    "display_label": DISPLAY[version],
                    "ds_size": ds_size,
                    "step": int(step),
                    "value": float(val),
                }
            )
    return rows, runs_meta


def collect_1m_rows(api: wandb.Api, entity: str, project: str, metric: str, samples: int) -> tuple[list[dict], list[dict]]:
    rows: list[dict] = []
    runs_meta: list[dict] = []
    for run in api.runs(f"{entity}/{project}"):
        name = (run.name or "").strip()
        m = ONE_M_RE.match(name)
        if not m:
            continue
        version = m.group(1).upper()
        runs_meta.append(
            {
                "source_project": project,
                "run_id": run.id,
                "run_name": name,
                "version": version,
                "display_label": DISPLAY[version],
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
                    "source_project": project,
                    "run_id": run.id,
                    "run_name": name,
                    "version": version,
                    "display_label": DISPLAY[version],
                    "ds_size": "1M",
                    "step": int(step),
                    "value": float(val),
                }
            )
    return rows, runs_meta


def aggregate(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["ds_size"], row["version"], row["step"])].append(row["value"])

    out: list[dict] = []
    for (ds_size, version, step), vals in sorted(by_key.items(), key=lambda x: (SIZE_ORDER.index(x[0][0]), x[0][1], x[0][2])):
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = math.sqrt(var)
        else:
            std = 0.0
        out.append(
            {
                "ds_size": ds_size,
                "version": version,
                "display_label": DISPLAY[version],
                "step": step,
                "mean": mean,
                "std": std,
                "count": len(vals),
            }
        )
    return out


def plot_multisize(agg: list[dict], metric: str, out_png: Path) -> None:
    by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in agg:
        by_key[(row["ds_size"], row["version"])].append(row)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), dpi=200, sharex=False, sharey=True)
    axes = axes.flatten()

    for i, ds_size in enumerate(SIZE_ORDER):
        ax = axes[i]
        for version in ("V1", "V30"):
            pts = sorted(by_key.get((ds_size, version), []), key=lambda r: r["step"])
            if not pts:
                continue
            x = [p["step"] for p in pts]
            y = [p["mean"] for p in pts]
            s = [p["std"] for p in pts]
            ax.plot(x, y, color=COLORS[version], linewidth=2.8, label=DISPLAY[version])
            ax.fill_between(
                x,
                [max(0.0, yy - ss) for yy, ss in zip(y, s)],
                [min(1.0, yy + ss) for yy, ss in zip(y, s)],
                color=COLORS[version],
                alpha=0.18,
                linewidth=0,
            )

        ax.set_title(f"{ds_size}", fontsize=12, fontweight="bold")
        ax.set_ylim(0.0, 1.05)
        ax.grid(alpha=0.20)
        if i % 3 == 0:
            ax.set_ylabel(metric)
        if i >= 3:
            ax.set_xlabel("Step")

    axes[-1].axis("off")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, title="Reward")
    fig.suptitle("Double Task2 | V1 vs V30 Across Dataset Sizes", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project-small", default="0219-double-tmp1")
    parser.add_argument("--project-1m", default="CUBE_TASK_DOUBLE_0223")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument("--out-dir", default="qam/analysis/artifacts/double_multisize_v1_vs_v30")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=120)
    rows_small, runs_small = collect_0219_rows(
        api=api,
        entity=args.entity,
        project=args.project_small,
        metric=args.metric,
        samples=args.samples_per_run,
    )
    rows_1m, runs_1m = collect_1m_rows(
        api=api,
        entity=args.entity,
        project=args.project_1m,
        metric=args.metric,
        samples=args.samples_per_run,
    )

    rows = rows_small + rows_1m
    runs = runs_small + runs_1m
    if not rows:
        raise RuntimeError("No runs matched for multisize plotting.")

    agg = aggregate(rows)

    runs_csv = out_dir / "matched_runs.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_project", "run_id", "run_name", "version", "display_label", "ds_size"],
        )
        writer.writeheader()
        writer.writerows(runs)

    raw_csv = out_dir / "raw_points.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_project", "run_id", "run_name", "version", "display_label", "ds_size", "step", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_csv = out_dir / "aggregated_mean_std.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["ds_size", "version", "display_label", "step", "mean", "std", "count"],
        )
        writer.writeheader()
        writer.writerows(agg)

    out_png = out_dir / "double_multisize_v1_vs_v30_subfig.png"
    plot_multisize(agg=agg, metric=args.metric, out_png=out_png)

    print(f"Saved matched runs: {runs_csv}")
    print(f"Saved raw points: {raw_csv}")
    print(f"Saved aggregate: {agg_csv}")
    print(f"Saved figure: {out_png}")
    print(f"Matched runs total: {len(runs)}")
    print(f"Matched runs small sizes: {len(runs_small)}")
    print(f"Matched runs 1M: {len(runs_1m)}")


if __name__ == "__main__":
    main()
