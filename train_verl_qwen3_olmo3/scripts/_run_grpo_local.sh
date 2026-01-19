#!/usr/bin/env bash
set -euo pipefail

# Shared local runner for GRPO-style RLVR training (no Slurm/PBS).
#
# Expected env vars (set by per-task wrapper scripts):
# - MODEL_PATH
# - TRAIN_FILE
# - VAL_FILE
# - EXPERIMENT_NAME
# - REWARD_FUNCTION_PATH
# - ADV_ESTIMATOR: grpo | grpo_weighted_mul
#
# Optional env vars:
# - N_GPUS_PER_NODE (default: 1)
# - TENSOR_MODEL_PARALLEL_SIZE (default: 1)
# - CKPT_ROOT (default: <repo>/outputs)
# - CACHE_ROOT (default: <repo>/.cache)

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

: "${MODEL_PATH:?MODEL_PATH is required}"
: "${TRAIN_FILE:?TRAIN_FILE is required}"
: "${VAL_FILE:?VAL_FILE is required}"
: "${EXPERIMENT_NAME:?EXPERIMENT_NAME is required}"
: "${REWARD_FUNCTION_PATH:?REWARD_FUNCTION_PATH is required}"
: "${ADV_ESTIMATOR:?ADV_ESTIMATOR is required}"

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
TENSOR_MODEL_PARALLEL_SIZE="${TENSOR_MODEL_PARALLEL_SIZE:-1}"

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/.cache}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/outputs}"

HF_HOME="${HF_HOME:-${CACHE_ROOT}/hf}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-${CACHE_ROOT}/xdg}"
WANDB_DIR="${WANDB_DIR:-${CACHE_ROOT}/wandb}"

mkdir -p "${HF_HOME}" "${XDG_CACHE_HOME}" "${WANDB_DIR}" "${CKPT_ROOT}"

export HF_HOME XDG_CACHE_HOME WANDB_DIR
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME}"

# Make runs reproducible-ish by default; override if you want.
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

RUN_DIR="${CKPT_ROOT}/${EXPERIMENT_NAME}"
mkdir -p "${RUN_DIR}"

cd "${REPO_ROOT}"

BASE_CONFIG="\
  algorithm.adv_estimator=${ADV_ESTIMATOR} \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.001 \
  data.train_batch_size=2 \
  +data.gen_batch_size=2 \
  data.val_batch_size=2 \
  data.max_prompt_length=2048 \
  data.max_response_length=2048 \
  actor_rollout_ref.actor.optim.lr=5e-7 \
  actor_rollout_ref.model.path='\"${MODEL_PATH}\"' \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE} \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.80 \
  actor_rollout_ref.rollout.n=2 \
  actor_rollout_ref.actor.ppo_mini_batch_size=1 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  trainer.project_name='zhiyuan-train' \
  trainer.experiment_name='${EXPERIMENT_NAME}' \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
  trainer.nnodes=1 \
  trainer.save_freq=1 \
  trainer.test_freq=1 \
  trainer.total_epochs=1 \
  trainer.default_local_dir=${RUN_DIR} \
  actor_rollout_ref.actor.checkpoint.save_contents=[hf_model] \
  custom_reward_function.path='${REWARD_FUNCTION_PATH}' \
  custom_reward_function.name=compute_score \
"

COMMAND="python -m verl.trainer.main_ppo \
  ${BASE_CONFIG} \
  hydra.run.dir=. \
  data.train_files=${TRAIN_FILE} \
  data.val_files=${VAL_FILE}"

echo "[train] repo_root=${REPO_ROOT}"
echo "[train] model=${MODEL_PATH}"
echo "[train] train_file=${TRAIN_FILE}"
echo "[train] val_file=${VAL_FILE}"
echo "[train] experiment=${EXPERIMENT_NAME}"
echo "[train] reward_fn=${REWARD_FUNCTION_PATH}"
echo "[train] adv_estimator=${ADV_ESTIMATOR}"
echo "[train] n_gpus_per_node=${N_GPUS_PER_NODE}"
echo "[train] tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE}"
echo "[train] run_dir=${RUN_DIR}"
echo "[train] command=${COMMAND}"

eval "${COMMAND}"

