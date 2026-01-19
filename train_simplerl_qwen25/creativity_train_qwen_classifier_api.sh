set -x

# ===== API配置 - 新的solution_classifier_qwen_api.py需要的环境变量 =====
# OpenAI API key (for stage 2/3)
export CUDA_VISIBLE_DEVICES=1,2
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"

# DashScope API key (for stage 1)
export DASHSCOPE_API_KEY="${DASHSCOPE_API_KEY:-}"

# 可选：指定使用的OpenAI模型 (默认: o3-mini)
export OPENAI_GPT_MODEL="o3-mini"

# 可选：自定义Qwen API端点 (默认: DashScope国际版)
export QWEN_BASE_URL="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

export VLLM_ATTENTION_BACKEND=XFORMERS
export SOLUTION_CLASSIFIER=${SOLUTION_CLASSIFIER:-qwen_api}
BASE_CONFIG="\
    algorithm.adv_estimator=grpo_weighted_mul \
    data.train_batch_size=4 \
    data.val_batch_size=8 \
    data.max_prompt_length=4096 \
    data.max_response_length=2048 \
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
    +actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
    +actor_rollout_ref.model.override_config.torch_dtype=bfloat16 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=160 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.ref.log_prob_micro_batch_size=160 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    +actor_rollout_ref.ref.fsdp_config.model_dtype=bfloat16 \
    algorithm.kl_ctrl.kl_coef=0.0001 \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.project_name='Creativity' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=80 \
    trainer.total_epochs=1"

# 训练集 & 测试集路径
TRAIN_FILE="huggingface/simplelr_qwen_level3to5/train.parquet"
# TRAIN_FILE="huggingface/textbookreasoning_physics/train.parquet"
VAL_FILE="huggingface/simplelr_qwen_level3to5/test.parquet"

# 初始模型路径
# MODEL_PATH="hkust-nlp/Qwen-2.5-7B-SimpleRL-Zoo"
MODEL_PATH="Qwen/Qwen2.5-3B-Instruct"
EXPERIMENT_NAME="simplerl_7b_creativity_training_grpo_weighted_mul_api_13Sep_testing"

# 验证API Keys是否已设置
if [ -z "$DASHSCOPE_API_KEY" ] || [ "$DASHSCOPE_API_KEY" = "your_dashscope_api_key_here" ]; then
    echo "WARNING: DASHSCOPE_API_KEY未设置，solution classification将使用fallback模式"
fi

if [ -z "$OPENAI_API_KEY" ]; then
    echo "WARNING: OPENAI_API_KEY未设置，solution classification将使用fallback模式"
fi

echo "Starting API-based training for dataset: ${TRAIN_FILE}"
echo "Using solution classifier: ${SOLUTION_CLASSIFIER} (options: o3, qwen_api, qwen_api_physics, qwen_api_medical, qwen_local)"
echo "OpenAI model for GPT stages: ${OPENAI_GPT_MODEL}"

# 构建训练命令
COMMAND="python3 -m verl.trainer.main_ppo \
    ${BASE_CONFIG} \
    hydra.run.dir=. \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    actor_rollout_ref.model.path='\"${MODEL_PATH}\"' \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    $@"

echo "Executing command: ${COMMAND}"
eval ${COMMAND}

# 更新模型路径到最新检查点
MODEL_PATH="ckpt/${EXPERIMENT_NAME}/actor" 

echo "API-based training finished!"