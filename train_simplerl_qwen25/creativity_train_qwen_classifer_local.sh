#!/usr/bin/env bash
set -euo pipefail
set -x

############################################
# ===== API 配置（Stage2/3 & Stage1鉴权）=====
############################################
# OpenAI (用于 Stage 2 & 3 的 o3 / 其他 Responses API)
export OPENAI_API_KEY="REDACTED_OPENAI_API_KEY"
export OPENAI_GPT_MODEL="o3-mini"

# DashScope 变量名被分类器代码复用为 Bearer，
# 本脚本里用它来给 vLLM 做 --api-key 校验（本地也可用同一个 key）
export DASHSCOPE_API_KEY="REDACTED_SECRET"

# 默认仍指向 DashScope（会在 vLLM 就绪后被覆盖为本地）
export QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

############################################
# ===== vLLM 本地服务：启动 & 健康检查 =====
############################################
# 推荐与你训练用的 CUDA_VISIBLE_DEVICES 不重叠
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8002}"      #########################################################################################################
VLLM_DEVICES="${VLLM_DEVICES:-0,1}"                 #########################################################################################################
VLLM_API_KEY="${VLLM_API_KEY:-${DASHSCOPE_API_KEY}}" 
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-72B-Instruct}"
VLLM_SERVED_NAME="${VLLM_SERVED_NAME:-qwen2.5-72b-instruct}"
VLLM_TP="${VLLM_TP:-2}" #########################################################################################################
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAXLEN="${VLLM_MAXLEN:-32768}"
VLLM_MEMUTIL="${VLLM_MEMUTIL:-0.60}" #########################################################################################################
STOP_VLLM_ON_EXIT="${STOP_VLLM_ON_EXIT:-1}"         # 1=训练结束自动停 vLLM；0=保留后台
VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"

# 健康检查工具
_vllm_ready() {
  curl -fsS -m 5 -H "Authorization: Bearer ${VLLM_API_KEY}" "${VLLM_BASE_URL}/models" >/dev/null
}
_wait_ready() {
  local deadline=$((SECONDS + 180))
  local last=""
  while (( SECONDS < deadline )); do
    if _vllm_ready; then return 0; fi
    sleep 1.5
  done
  return 1
}

# 与训练 GPU 是否冲突（仅提示）
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a _train <<< "${CUDA_VISIBLE_DEVICES}"
  IFS=',' read -r -a _vllm  <<< "${VLLM_DEVICES}"
  for a in "${_train[@]}"; do
    for b in "${_vllm[@]}"; do
      if [[ "$a" == "$b" ]]; then
        echo "WARNING: 训练 CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} 与 vLLM 使用的 VLLM_DEVICES=${VLLM_DEVICES} 存在重叠（GPU ${a}）。建议改成不重叠的两组卡位。"
      fi
    done
  done
fi

# 启动或复用 vLLM
VLLM_PID=""
if _vllm_ready; then
  echo "[vLLM] Reusing existing service at ${VLLM_BASE_URL}"
else
  echo "[vLLM] Launching on ${VLLM_BASE_URL} with GPUs ${VLLM_DEVICES} ..."
  CUDA_VISIBLE_DEVICES="${VLLM_DEVICES}" nohup \
    python -m vllm.entrypoints.openai.api_server \
      --model "${VLLM_MODEL}" \
      --host "${VLLM_HOST}" --port "${VLLM_PORT}" \
      --tensor-parallel-size "${VLLM_TP}" \
      --dtype "${VLLM_DTYPE}" \
      --served-model-name "${VLLM_SERVED_NAME}" \
      --max-model-len "${VLLM_MAXLEN}" \
      --gpu-memory-utilization "${VLLM_MEMUTIL}" \
      --api-key "${VLLM_API_KEY}" \
      > vllm_server.log 2>&1 &

  VLLM_PID=$!
  echo "[vLLM] PID=${VLLM_PID} (logs: vllm_server.log)"
  echo "[vLLM] Waiting for readiness ..."
  if ! _wait_ready; then
    echo "[vLLM] ERROR: vLLM failed to become ready within timeout."
    [[ -n "${VLLM_PID}" ]] && kill "${VLLM_PID}" >/dev/null 2>&1 || true
    exit 1
  fi
fi

# 覆盖分类器使用的端点与鉴权：Stage1 将直连本地 vLLM
export QWEN_BASE_URL="${VLLM_BASE_URL}"
export DASHSCOPE_API_KEY="${VLLM_API_KEY}"

# 训练结束自动清理我们启动的 vLLM
if [[ "${STOP_VLLM_ON_EXIT}" == "1" && -n "${VLLM_PID}" ]]; then
  trap 'echo "[vLLM] Stopping PID=${VLLM_PID}"; kill ${VLLM_PID} >/dev/null 2>&1 || true; sleep 1; kill -9 ${VLLM_PID} >/dev/null 2>&1 || true' EXIT
fi

echo "[vLLM] Ready: ${QWEN_BASE_URL}  (served-name=${VLLM_SERVED_NAME}, tp=${VLLM_TP}, dtype=${VLLM_DTYPE}, max-len=${VLLM_MAXLEN})"

############################################
# ===== 训练参数（保留你原有配置） =====
############################################
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"  #########################################################################################################

BASE_CONFIG="\
    algorithm.adv_estimator=grpo_weighted_mul \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=1024 \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.grad_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.20 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='Creativity' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=80 \
    trainer.total_epochs=1"

# 数据集与初始模型
TRAIN_FILE="${TRAIN_FILE:-huggingface/simplelr_qwen_level3to5/train.parquet}"
VAL_FILE="${VAL_FILE:-huggingface/simplelr_qwen_level3to5/test.parquet}"
MODEL_PATH="${MODEL_PATH:-hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-simplerl_ckpt_creativity_training_grpo_weighted_mul_api_01Sep_testing}"

# 友好提示：密钥是否设置
if [[ -z "${DASHSCOPE_API_KEY}" || "${DASHSCOPE_API_KEY}" == "REPLACE_WITH_LOCAL_KEY" ]]; then
  echo "WARNING: DASHSCOPE_API_KEY 未设置有效值；Stage1 若需要 Bearer 校验将失败（本地 vLLM 若未启用 --api-key 则可忽略）。"
fi
if [[ -z "${OPENAI_API_KEY}" || "${OPENAI_API_KEY}" == "REPLACE_WITH_OPENAI_KEY" ]]; then
  echo "WARNING: OPENAI_API_KEY 未设置；Stage2/3 将回退到 fallback 逻辑或报错。"
fi

echo "Starting API-based training for dataset: ${TRAIN_FILE}"
echo "Using local vLLM(Qwen2.5-72B-Instruct) for Stage1 + OpenAI ${OPENAI_GPT_MODEL} for Stage2/3"

# 训练命令（保持你的原样）
COMMAND="python3 -m verl.trainer.main_ppo \
    ${BASE_CONFIG} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    actor_rollout_ref.model.path='\"${MODEL_PATH}\"' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    $@"

echo "Executing command: ${COMMAND}"
eval ${COMMAND}

# （可选）更新模型路径到最新检查点
MODEL_PATH="ckpt/${EXPERIMENT_NAME}/actor"

echo "API-based training finished!"
