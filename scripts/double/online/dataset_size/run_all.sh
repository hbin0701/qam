#!/bin/bash
# Submit all Double Task2 Dense V30 dataset-size runs as separate SLURM jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

RUN_SCRIPTS=(
  "run_task2_dense_v30_10k.sh"
  "run_task2_dense_v30_50k.sh"
  "run_task2_dense_v30_100k.sh"
  "run_task2_dense_v30_500k.sh"
)

echo "Submitting ${#RUN_SCRIPTS[@]} SLURM jobs from ${SCRIPT_DIR}"

for script in "${RUN_SCRIPTS[@]}"; do
  script_path="${SCRIPT_DIR}/${script}"
  if [[ ! -f "${script_path}" ]]; then
    echo "missing script: ${script_path}" >&2
    exit 1
  fi

  submit_output="$(sbatch "${script_path}")"
  job_id="$(echo "${submit_output}" | awk '{print $NF}')"
  echo "submitted ${script} | job_id=${job_id}"
  sleep 0.2
done

echo "All V30 dataset-size jobs submitted."
