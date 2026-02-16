#!/bin/bash
# Master script to run all reward variants for Task 2 in parallel

# JAX Memory Management (inherited by child scripts)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.20

echo "Launching all Task 2 variants..."

# Launch Sparse on GPU 0
# bash scripts/double/run_task2_sparse.sh &
# sleep 1 # Offset to avoid simultaneous WandB initialization issues

# Launch Dense variants on GPU 1
# bash scripts/double/run_task2_dense_v1.sh &
# sleep 1
bash scripts/double/run_task2_dense_v2.sh &
sleep 1

bash scripts/double/run_task2_dense_v3.sh &
sleep 1
bash scripts/double/run_task2_dense_v4.sh &
sleep 1
bash scripts/double/run_task2_dense_v5.sh &
sleep 1
bash scripts/double/run_task2_dense_v6.sh &

echo "All tasks launched. Waiting for completion..."
wait
echo "All tasks finished."
