#!/usr/bin/env bash
set -euo pipefail

ROOT="/rlwrld3/home/hyeonbin/RL"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/qam/reward/results/${TS}"
mkdir -p "${OUT_DIR}"

PYTHONPATH="${ROOT}" "${ROOT}/qam/.venv/bin/python" -m qam.reward.pre.decompose \
  --model gemini-2.5-flash \
  --video "${ROOT}/qam/artifacts/double_demo.mp4" \
  --env-code-path "${ROOT}/qam/reward/envs/cube_double_env.py" \
  --pose-json-path "${ROOT}/qam/artifacts/double_demo.json" \
  --output "${OUT_DIR}/step1_result.json" \
  --write-raw-response "${OUT_DIR}/step1_raw.txt" \
  --write-prompt "${OUT_DIR}/step1_prompt.txt" \
  --max-frames 32 \
  --max-output-tokens 16384 \
  | tee "${OUT_DIR}/run.log"

echo "Saved under: ${OUT_DIR}"
