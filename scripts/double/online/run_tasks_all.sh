#!/bin/bash
# Master script to run all ONLINE-ONLY dense reward variants for Double-Play Task 2 in parallel.

# JAX Memory Management (inherited by child scripts)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

echo "Launching all ONLINE Task 2 dense variants (v1-v9)..."

bash scripts/double/online/run_task2_dense_v1.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v2.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v3.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v4.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v5.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v6.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v7.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v8.sh &
sleep 1
bash scripts/double/online/run_task2_dense_v9.sh &

echo "All online tasks launched. Waiting for completion..."
wait
echo "All online tasks finished."
