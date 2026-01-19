#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_PATH="${MODEL_PATH:-models/OLMo3-7B}" export TRAIN_FILE="${TRAIN_FILE:-data/math/train.parquet}" export VAL_FILE="${VAL_FILE:-data/math/val.parquet}" export EXPERIMENT_NAME="${EXPERIMENT_NAME:-olmo3_7b_math_grpo_weighted_mul}" export REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-${SCRIPT_DIR}/../recipe/creativity/reward_function_olmo3_math.py}" export ADV_ESTIMATOR="grpo_weighted_mul"

exec "${SCRIPT_DIR}/_run_grpo_local.sh"
