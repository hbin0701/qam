#!/bin/bash
# Launch all triple online dense variants concurrently.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"
mkdir -p "${LOG_DIR}"

SCRIPTS=(
  "run_task2_dense_v1.sh"
  "run_task2_dense_v2.sh"
  "run_task2_dense_v3.sh"
  "run_task2_dense_v4.sh"
  "run_task2_dense_v5.sh"
  "run_task2_dense_v6.sh"
  "run_task2_dense_v7.sh"
  "run_task2_dense_v30.sh"
)

echo "Launching ${#SCRIPTS[@]} runs from ${SCRIPT_DIR}"

for script in "${SCRIPTS[@]}"; do
  log_file="${LOG_DIR}/${script%.sh}.log"
  bash "${SCRIPT_DIR}/${script}" > "${log_file}" 2>&1 &
  pid=$!
  echo "started ${script} | pid=${pid} | log=${log_file}"
done

echo "All runs launched."
