#!/bin/bash
set -euo pipefail

echo "Launching all Single Task 2 variants..."

bash scripts/single/run_task2_sparse.sh &
sleep 1
bash scripts/single/run_task2_dense_v1.sh &
sleep 1
bash scripts/single/run_task2_dense_v2.sh &
sleep 1
bash scripts/single/run_task2_dense_v3.sh &
sleep 1
bash scripts/single/run_task2_dense_v4.sh &
sleep 1
bash scripts/single/run_task2_dense_v5.sh &
sleep 1
bash scripts/single/run_task2_dense_v6.sh &
sleep 1
bash scripts/single/run_task2_dense_v7.sh &

wait
