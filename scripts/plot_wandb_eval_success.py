#!/usr/bin/env python3
import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import wandb


PART_SUFFIX_RE = re.compile(
    r"\s*[\(\[\-_ ]*(?:part|pt)\s*[-_ ]*(\d+)\s*[\)\]]?\s*$",
    re.IGNORECASE,
)


def normalize_run_name(name: str) -> str:
    if not name:
        return "unnamed"
    normalized = PART_SUFFIX_RE.sub("", name).strip()
    normalized = re.sub(r"[\s\-_]+$", "", normalized).strip()
    return normalized or "unnamed"


def fetch_rows(
    api: wandb.Api,
    project_path: str,
    metric_key: str,
    max_runs: int | None = None,
    samples_per_run: int = 2000,
    name_contains: str | None = None,
) -> list[dict]:
    rows = []
    runs = list(api.runs(project_path))
    if max_runs is not None:
        runs = runs[:max_runs]
    total_runs = len(runs)
    for idx, run in enumerate(runs, start=1):
        run_name = run.name or "unnamed"
        if name_contains and name_contains.lower() not in run_name.lower():
            continue
        merged_name = normalize_run_name(run_name)
        if metric_key not in run.summary:
            continue
        print(f"[{idx}/{total_runs}] {run.id} | {run_name}", file=sys.stderr, flush=True)
        history = run.history(
            keys=["_step", metric_key],
            samples=samples_per_run,
            pandas=False,
        )
        for item in history:
            value = item.get(metric_key)
            if value is None:
                continue
            step = item.get("_step")
            if step is None:
                continue
            rows.append(
                {
                    "merged_name": merged_name,
                    "run_name": run_name,
                    "run_id": run.id,
                    "step": int(step),
                    metric_key: float(value),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and plot W&B eval/success data.")
    parser.add_argument("--entity", default="hbin0701")
    parser.add_argument("--project", default="qam-reproduce-final")
    parser.add_argument("--metric", default="eval/success")
    parser.add_argument("--out-dir", default="artifacts/wandb_eval_success")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--samples-per-run", type=int, default=2000)
    parser.add_argument("--name-contains", type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    api = wandb.Api(timeout=60)
    project_path = f"{args.entity}/{args.project}"
    df = fetch_rows(
        api,
        project_path,
        args.metric,
        max_runs=args.max_runs,
        samples_per_run=args.samples_per_run,
        name_contains=args.name_contains,
    )
    if not df:
        raise RuntimeError(f"No data found for metric '{args.metric}' in {project_path}")

    raw_csv = out_dir / "eval_success_raw.csv"
    with raw_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["merged_name", "run_name", "run_id", "step", args.metric]
        )
        writer.writeheader()
        writer.writerows(df)

    by_group_step = {}
    for row in sorted(df, key=lambda r: (r["merged_name"], r["step"])):
        by_group_step[(row["merged_name"], row["step"])] = row
    merged_df = sorted(by_group_step.values(), key=lambda r: (r["merged_name"], r["step"]))

    merged_csv = out_dir / "eval_success_merged.csv"
    with merged_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["merged_name", "run_name", "run_id", "step", args.metric]
        )
        writer.writeheader()
        writer.writerows(merged_df)

    plt.figure(figsize=(12, 7))
    grouped = defaultdict(list)
    for row in merged_df:
        grouped[row["merged_name"]].append(row)
    for name, group_rows in grouped.items():
        steps = [r["step"] for r in group_rows]
        values = [r[args.metric] for r in group_rows]
        plt.plot(steps, values, label=name, linewidth=1.8)

    plt.title(f"{project_path} - {args.metric}")
    plt.xlabel("Step")
    plt.ylabel(args.metric)
    plt.grid(alpha=0.25)
    plt.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8)
    plt.tight_layout()

    plot_path = out_dir / "eval_success_lineplot.png"
    plt.savefig(plot_path, dpi=180)
    plt.close()

    print(f"Saved raw data: {raw_csv}")
    print(f"Saved merged data: {merged_csv}")
    print(f"Saved plot: {plot_path}")
    print(f"Total points: {len(df)}")
    print(f"Merged run groups: {len(grouped)}")


if __name__ == "__main__":
    main()
