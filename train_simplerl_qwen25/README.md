Our code is implemented based on [Verl](https://github.com/volcengine/verl). We provide basic environment setup for training as follows, which only support custom environment setup and [FSDP training](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html). 

```bash
conda create -n verl python==3.10
conda activate verl
pip3 install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip3 install flash-attn --no-build-isolation
pip3 install -e . 
```

## Training with Creativity Script

We provide a training script `creativity_train.sh` for creativity training using GRPO weighted advantage methods. This script implements advanced reinforcement learning techniques for mathematical reasoning tasks.

### Basic Usage

```bash
# Activate the environment
conda activate verl

# Run the training script
bash creativity_train.sh
```

### Script Parameters Explanation

The `creativity_train.sh` script contains several key parameters that control the training process:

#### Algorithm Parameters
- `algorithm.adv_estimator=grpo_weighted_mul`: Uses the multiplication-based GRPO weighted advantage method
  - `grpo_weighted_add`: Addition-based weighting (combined_advantages = grpo_advantages + weights)
  - `grpo_weighted_mul`: Multiplication-based weighting (combined_advantages = grpo_advantages * weights)

#### Data Parameters
- `data.train_batch_size=4`: Number of training samples per batch
- `data.val_batch_size=8`: Number of validation samples per batch
- `data.max_prompt_length=4096`: Maximum length of input prompts
- `data.max_response_length=1024`: Maximum length of generated responses

#### Model Parameters
- `actor_rollout_ref.model.path`: Path to the base model (e.g., "Qwen/Qwen2.5-3B-Instruct")
- `actor_rollout_ref.model.use_remove_padding=True`: Enables padding removal for efficiency
- `actor_rollout_ref.model.enable_gradient_checkpointing=True`: Enables gradient checkpointing to save memory

#### Training Parameters
- `actor_rollout_ref.actor.optim.lr=5e-7`: Learning rate for the actor model
- `actor_rollout_ref.actor.ppo_mini_batch_size=64`: Mini-batch size for PPO updates
- `actor_rollout_ref.actor.ppo_micro_batch_size=8`: Micro-batch size per GPU
- `actor_rollout_ref.actor.use_kl_loss=True`: Enables KL divergence loss
- `actor_rollout_ref.actor.kl_loss_coef=0.001`: Coefficient for KL loss
- `actor_rollout_ref.actor.kl_loss_type=low_var_kl`: Type of KL loss calculation

#### Rollout Parameters
- `actor_rollout_ref.rollout.n=8`: Number of responses generated per prompt (important for GRPO weighted methods)
- `actor_rollout_ref.rollout.temperature=1.0`: Sampling temperature for text generation
- `actor_rollout_ref.rollout.top_p=0.95`: Top-p sampling parameter
- `actor_rollout_ref.rollout.gpu_memory_utilization=0.8`: GPU memory utilization for VLLM

#### Distributed Training Parameters
- `trainer.n_gpus_per_node=2`: Number of GPUs per node
- `trainer.nnodes=1`: Number of nodes
- `trainer.project_name='Creativity'`: Project name for logging
- `trainer.save_freq=10`: Frequency of model checkpointing
- `trainer.test_freq=80`: Frequency of validation testing

#### FSDP Configuration
- `actor_rollout_ref.actor.fsdp_config.param_offload=False`: Disables parameter offloading
- `actor_rollout_ref.actor.fsdp_config.grad_offload=False`: Disables gradient offloading
- `actor_rollout_ref.actor.fsdp_config.optimizer_offload=False`: Disables optimizer offloading

### Customization

You can modify the script to use different configurations:

1. **Change the advantage estimator**:
   ```bash
   # For addition-based weighting
   algorithm.adv_estimator=grpo_weighted_add
   
   # For multiplication-based weighting (default)
   algorithm.adv_estimator=grpo_weighted_mul
   ```

2. **Adjust batch sizes**:
   ```bash
   data.train_batch_size=8  # Increase for more parallel training
   actor_rollout_ref.actor.ppo_mini_batch_size=128  # Increase for larger updates
   ```

3. **Modify the number of responses per prompt**:
   ```bash
   actor_rollout_ref.rollout.n=4  # Reduce for faster training
   actor_rollout_ref.rollout.n=16  # Increase for better diversity
   ```

4. **Change the model**:
   ```bash
   MODEL_PATH="microsoft/DialoGPT-medium"  # Use a different base model
   ```

### Environment Variables

The script sets several important environment variables:

- `OPENAI_API_KEY`: Required for GPT-4 solution classification (when enabled)
- `VLLM_ATTENTION_BACKEND=XFORMERS`: Uses XFormers for efficient attention computation
- `CUDA_VISIBLE_DEVICES=0,1`: Specifies which GPUs to use

### Output and Logging

- Training logs are saved to `wandb` with project name "Creativity"
- Model checkpoints are saved to `ckpt/${EXPERIMENT_NAME}/actor`
- The experiment name follows the pattern: `qwen2.5_3b_instruct_creativity_training_grpo_weighted_mul_testing`

### Troubleshooting

1. **Memory Issues**: Reduce `gpu_memory_utilization` or `ppo_micro_batch_size`
2. **Slow Training**: Increase `rollout.n` or reduce `max_prompt_length`
3. **API Errors**: Ensure `OPENAI_API_KEY` is set correctly for solution classification