#!/usr/bin/env bash
set -euo pipefail

ROOT="/rlwrld3/home/hyeonbin/RL"
TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="${ROOT}/qam/reward/results/${TS}"
mkdir -p "${OUT_DIR}"

VIDEO="${1:-${ROOT}/qam/artifacts/double_demo.mp4}"
ENV_CODE="${2:-${ROOT}/qam/reward/envs/cube_double_env.py}"
POSE_JSON="${3:-${ROOT}/qam/artifacts/double_demo.json}"

echo "Writing results to: ${OUT_DIR}"
echo "Video: ${VIDEO}"
echo "Env code: ${ENV_CODE}"
echo "Pose JSON: ${POSE_JSON}"

PYTHONPATH="${ROOT}" "${ROOT}/qam/.venv/bin/python" -m qam.reward.pre.decompose \
  --model gemini-2.5-flash \
  --video "${VIDEO}" \
  --env-code-path "${ENV_CODE}" \
  --pose-json-path "${POSE_JSON}" \
  --output "${OUT_DIR}/step1_result.json" \
  --write-raw-response "${OUT_DIR}/step1_raw.txt" \
  --write-prompt "${OUT_DIR}/step1_prompt.txt" \
  --max-frames 32 \
  --max-output-tokens 16384 \
  | tee "${OUT_DIR}/run.log"

echo "Done. See:"
echo "  ${OUT_DIR}/step1_result.json"
echo "  ${OUT_DIR}/step1_raw.txt"
echo "  ${OUT_DIR}/step1_prompt.txt"
echo "  ${OUT_DIR}/run.log"
