#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_DIR}"

jobs=(
  "scripts/slurm/double/online/v1.slurm"
  "scripts/slurm/double/online/v7.slurm"
  "scripts/slurm/double/online/v8.slurm"
)

submitted=0

for slurm_file in "${jobs[@]}"; do
  if [ ! -f "${slurm_file}" ]; then
    echo "Skipping missing file: ${slurm_file}"
    continue
  fi

  echo "Submitting ${slurm_file}"
  sbatch "${slurm_file}"
  submitted=$((submitted + 1))
done

echo "Submitted ${submitted} online jobs."
