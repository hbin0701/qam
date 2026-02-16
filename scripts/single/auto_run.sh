#!/bin/bash
set -euo pipefail

while true; do
  bash scripts/single/run_tasks_all.sh
  sleep 5
done
