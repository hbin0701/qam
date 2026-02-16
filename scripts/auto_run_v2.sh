#!/bin/bash
# Submit one SLURM job per task (1 GPU each).
# Usage:
#   bash scripts/auto_run_v2.sh          # submit double + triple
#   bash scripts/auto_run_v2.sh double   # submit only double
#   bash scripts/auto_run_v2.sh triple   # submit only triple

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SLURM_DIR="${SCRIPT_DIR}/slurm"
TARGET="${1:-all}"

case "$TARGET" in
  all)
    mapfile -t JOB_FILES < <(find "${SLURM_DIR}/double" "${SLURM_DIR}/triple" -maxdepth 1 -type f -name '*.slurm' | sort)
    ;;
  double)
    mapfile -t JOB_FILES < <(find "${SLURM_DIR}/double" -maxdepth 1 -type f -name '*.slurm' | sort)
    ;;
  triple)
    mapfile -t JOB_FILES < <(find "${SLURM_DIR}/triple" -maxdepth 1 -type f -name '*.slurm' | sort)
    ;;
  *)
    echo "Usage: bash scripts/auto_run_v2.sh [all|double|triple]"
    exit 2
    ;;
esac

if [ "${#JOB_FILES[@]}" -eq 0 ]; then
  echo "No SLURM job files found under ${SLURM_DIR}."
  exit 1
fi

echo "Submitting ${#JOB_FILES[@]} jobs (target=${TARGET})..."

for job_file in "${JOB_FILES[@]}"; do
  submit_output="$(sbatch "$job_file")"
  job_id="$(echo "$submit_output" | awk '{print $4}')"
  echo "${job_file} -> ${job_id}"
done

echo "Submission complete."
