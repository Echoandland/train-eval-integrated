#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

export MODEL_PATH="${MODEL_PATH:-models/Qwen3-8B}" export TRAIN_FILE="${TRAIN_FILE:-data/medical/train.parquet}" export VAL_FILE="${VAL_FILE:-data/medical/val.parquet}" export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_8b_medical_grpo_purerl}" export REWARD_FUNCTION_PATH="${REWARD_FUNCTION_PATH:-${SCRIPT_DIR}/../recipe/medical/reward_function_medreasoning.py}" export ADV_ESTIMATOR="grpo"

exec "${SCRIPT_DIR}/_run_grpo_local.sh"
