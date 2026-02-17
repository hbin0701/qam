#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

targets=(
  "scripts/slurm/single/online"
  # "scripts/slurm/double/online"
  # "scripts/slurm/triple/online"
)

submitted=0

for dir in "${targets[@]}"; do
  if [ ! -d "${dir}" ]; then
    echo "Skipping missing directory: ${dir}"
    continue
  fi

  while IFS= read -r -d '' slurm_file; do
    echo "Submitting ${slurm_file}"
    sbatch "${slurm_file}"
    submitted=$((submitted + 1))
  done < <(find "${dir}" -maxdepth 1 -type f -name "*.slurm" -print0 | sort -z)
done

echo "Submitted ${submitted} online jobs."
