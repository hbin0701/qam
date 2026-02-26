#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


ONLINE_RE = re.compile(
    r"^\[Online\]\s*Double,\s*(V1|V30),\s*QAM,\s*DS-(10k|100k|250k|500k)\s*$",
    re.IGNORECASE,
)

DISPLAY_LABEL = {
    "V1": "Sparse",
    "V30": "Reward Shaped (Human)",
}


def parse_online_run(name: str) -> tuple[str, str] | None:
    m = ONLINE_RE.match((name or "").strip())
    if not m:
        return None
    version = m.group(1).upper()
    ds_size = m.group(2).lower()
    return version, ds_size


def collect_rows(entity: str, project: str, metric: str, samples_per_run: int) -> tuple[list[dict], list[dict]]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs(f"{entity}/{project}"))
    rows: list[dict] = []
    matched_runs: list[dict] = []

    for run in runs:
        parsed = parse_online_run(run.name or "")
        if parsed is None:
            continue
        version, ds_size = parsed
        matched_runs.append(
            {
                "run_id": run.id,
                "run_name": run.name or "",
                "version": version,
                "reward_label": DISPLAY_LABEL.get(version, version),
                "ds_size": ds_size,
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
                    "run_id": run.id,
                    "run_name": run.name or "",
                    "version": version,
                    "reward_label": DISPLAY_LABEL.get(version, version),
                    "ds_size": ds_size,
                    "step": int(step),
                    "value": float(value),
                }
            )

    return rows, matched_runs


def aggregate_by_step(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    for row in rows:
        by_key[(row["ds_size"], row["version"], row["reward_label"], row["step"])].append(row["value"])

    out = []
    for (ds_size, version, reward_label, step), vals in sorted(
        by_key.items(), key=lambda x: (x[0][0], x[0][1], x[0][3])
    ):
        mean = sum(vals) / len(vals)
        if len(vals) > 1:
            var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            std = var ** 0.5
        else:
            std = 0.0
        out.append(
            {
                "ds_size": ds_size,
                "version": version,
                "reward_label": reward_label,
                "step": step,
                "mean": mean,
                "std": std,
                "count": len(vals),
            }
        )
    return out


def plot_2x2(agg: list[dict], metric: str, out_png: Path) -> None:
    order = ["10k", "100k", "250k", "500k"]
    colors = {"V1": "#2563eb", "V30": "#dc2626"}

    by_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in agg:
        by_pair[(row["ds_size"], row["version"])].append(row)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=180, sharex=True, sharey=True)
    axes = axes.flatten()

    for i, ds_size in enumerate(order):
        ax = axes[i]
        for version in ("V1", "V30"):
            pts = sorted(by_pair.get((ds_size, version), []), key=lambda r: r["step"])
            if not pts:
                continue
            x = [p["step"] for p in pts]
            y = [p["mean"] for p in pts]
            s = [p["std"] for p in pts]
            ax.plot(
                x,
                y,
                color=colors[version],
                linewidth=2.4,
                label=DISPLAY_LABEL.get(version, version),
            )
            ax.fill_between(
                x,
                [max(0.0, yy - ss) for yy, ss in zip(y, s)],
                [min(1.0, yy + ss) for yy, ss in zip(y, s)],
                color=colors[version],
                alpha=0.18,
                linewidth=0,
            )
        ax.set_title(f"Dataset size: {ds_size}", fontsize=11, fontweight="bold")
        ax.grid(alpha=0.22)
        ax.set_ylim(0.0, 1.05)
        if i % 2 == 0:
            ax.set_ylabel(metric)
        if i >= 2:
            ax.set_xlabel("Step")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    fig.suptitle("[Online] Double, V1 vs V30 (2x2 by dataset size)", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project", default="0219-double-tmp1")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--samples-per-run", type=int, default=5000)
    parser.add_argument("--out-dir", default="qam/analysis/artifacts/0219_online_v1_vs_v30")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, matched_runs = collect_rows(
        entity=args.entity,
        project=args.project,
        metric=args.metric,
        samples_per_run=args.samples_per_run,
    )
    if len(matched_runs) != 8:
        print(f"Warning: expected 8 runs, found {len(matched_runs)}")
    if not rows:
        raise RuntimeError("No matching rows found for [Online] V1/V30 runs.")

    agg = aggregate_by_step(rows)

    runs_csv = out_dir / "matched_runs.csv"
    with runs_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "run_name", "version", "reward_label", "ds_size"]
        )
        writer.writeheader()
        writer.writerows(matched_runs)

    raw_csv = out_dir / "raw_points.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["run_id", "run_name", "version", "reward_label", "ds_size", "step", "value"]
        )
        writer.writeheader()
        writer.writerows(rows)

    agg_csv = out_dir / "aggregated_mean_std.csv"
    with agg_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["ds_size", "version", "reward_label", "step", "mean", "std", "count"]
        )
        writer.writeheader()
        writer.writerows(agg)

    out_png = out_dir / "online_v1_vs_v30_2x2.png"
    plot_2x2(agg=agg, metric=args.metric, out_png=out_png)

    print(f"Saved matched runs: {runs_csv}")
    print(f"Saved raw points: {raw_csv}")
    print(f"Saved aggregate: {agg_csv}")
    print(f"Saved figure: {out_png}")
    print(f"Matched runs: {len(matched_runs)}")


if __name__ == "__main__":
    main()
