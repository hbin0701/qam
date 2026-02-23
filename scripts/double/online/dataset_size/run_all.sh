#!/bin/bash
# Launch all dataset-size variants (1k/10k/50k/100k) concurrently.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  "run_task2_dense_v1_1k.sh"
  "run_task2_dense_v1_10k.sh"
  "run_task2_dense_v1_50k.sh"
  "run_task2_dense_v1_100k.sh"
)

echo "Launching ${#SCRIPTS[@]} runs from ${SCRIPT_DIR}"

for script in "${SCRIPTS[@]}"; do
  log_file="${LOG_DIR}/${script%.sh}.log"
  bash "${SCRIPT_DIR}/${script}" > "${log_file}" 2>&1 &
  pid=$!
  echo "started ${script} | pid=${pid} | log=${log_file}"
done

echo "All runs launched."

