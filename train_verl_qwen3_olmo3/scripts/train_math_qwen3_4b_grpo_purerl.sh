#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_PATH="${MODEL_PATH:-models/Qwen3-4B}"
export TRAIN_FILE="${TRAIN_FILE:-data/math/train.parquet}"
export VAL_FILE="${VAL_FILE:-data/math/val.parquet}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_math_grpo_purerl}"
export REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-${SCRIPT_DIR}/../recipe/creativity/reward_function_simplelr.py}"
export ADV_ESTIMATOR="grpo"

exec "${SCRIPT_DIR}/_run_grpo_local.sh"

