#!/bin/bash
set -euo pipefail

bash scripts/triple/run_task2_dense_v1.sh
bash scripts/triple/run_task2_dense_v2.sh
bash scripts/triple/run_task2_dense_v3.sh
bash scripts/triple/run_task2_dense_v4.sh
bash scripts/triple/run_task2_dense_v5.sh
bash scripts/triple/run_task2_dense_v6.sh
bash scripts/triple/run_task2_sparse.sh
