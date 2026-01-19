#!/usr/bin/env bash
set -euo pipefail

# Local, non-cluster training entrypoint for the SimpleRL creativity setup (Qwen2.5).
# This script intentionally contains NO embedded secrets.
#
# You may need to set:
# - OPENAI_API_KEY (if SOLUTION_CLASSIFIER uses OpenAI)
# - DASHSCOPE_API_KEY (if SOLUTION_CLASSIFIER uses Qwen API)
#
# You can override paths via env vars (MODEL_PATH/TRAIN_FILE/VAL_FILE/EXPERIMENT_NAME).

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="${SCRIPT_DIR}"

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# Keep caches local to the repo by default.
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
export WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"
mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${WANDB_DIR}"

# Select solution-classifier implementation.
# Options depend on the code under train_simplerl_qwen25/verl/utils.
export SOLUTION_CLASSIFIER="${SOLUTION_CLASSIFIER:-o3}"

TRAIN_FILE="${TRAIN_FILE:-data/creativity/train.parquet}"
VAL_FILE="${VAL_FILE:-data/creativity/val.parquet}"

MODEL_PATH="${MODEL_PATH:-models/Qwen2.5-3B-Instruct}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen25_3b_creativity_grpo_weighted_mul}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"

BASE_CONFIG="\
    algorithm.adv_estimator=grpo_weighted_mul \
    data.train_batch_size=2 \
    data.val_batch_size=2 \
    data.max_prompt_length=2048 \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=2 \
    actor_rollout_ref.actor.ppo_mini_batch_size=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.logger=['console'] \
    trainer.project_name='zhiyuan-train' \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=1 \
    trainer.save_freq=1 \
    trainer.test_freq=1 \
    trainer.total_epochs=1"

cd "${REPO_ROOT}"

echo "[train] Starting SimpleRL creativity training"
echo "[train] model=${MODEL_PATH}"
echo "[train] train_file=${TRAIN_FILE}"
echo "[train] val_file=${VAL_FILE}"
echo "[train] experiment=${EXPERIMENT_NAME}"
echo "[train] solution_classifier=${SOLUTION_CLASSIFIER}"

COMMAND="python3 -m verl.trainer.main_ppo \
    ${BASE_CONFIG} \
    hydra.run.dir=. \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    actor_rollout_ref.model.path='\"${MODEL_PATH}\"' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    $@"

echo "[train] command=${COMMAND}"
eval "${COMMAND}"
