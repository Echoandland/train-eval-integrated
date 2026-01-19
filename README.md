Unified Training and Evaluation Framework

A unified framework for training and evaluating LLMs (Qwen2.5, Qwen3, OLMo3) on creativity-focused reinforcement learning tasks across Math and Physics (with optional medical evaluation utilities).

## Overview

This repository integrates three components:

| Directory | Description |
|-----------|-------------|
| `train_verl_qwen3_olmo3/` | VERL/RLVR-based training for Qwen3 and OLMo3 models |
| `train_simplerl_qwen25/` | SimpleRL-based training for Qwen2.5 models |
| `eval_passn/` | Pass@k evaluation with optional LLM-as-judge |

## Installation

Our code is based on [verl](https://github.com/volcengine/verl). We recommend using the Docker image for consistent environment:

### Option 1: Docker (Recommended)

```bash
docker pull hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0
```

To run locally (single node), mount this repo into the container and enable GPUs:

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -it --rm \
  -v "$PWD":/workspace/train-eval-integrated \
  -w /workspace/train-eval-integrated \
  hiyouga/verl:ngc-th2.7.1-cu12.6-vllm0.10.0 bash
```

### Option 2: Conda Environment

```bash
# Create conda environment
conda create -n verl_rlvr python=3.10 -y
conda activate verl_rlvr

# Install PyTorch (adjust CUDA version as needed)
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install base requirements
pip install -r requirements.txt

# vLLM is required for VERL/RLVR training scripts by default (rollout backend = vLLM)
pip install vllm>=0.6.3

# Optional: Install Flash Attention
pip install flash-attn --no-build-isolation
```

### Option 3: Conda from environment.yml

```bash
conda env create -f environment.yml
conda activate verl_rlvr
```

## Configure API Keys

Copy the environment template and fill in your API keys:

```bash
cp env.example .env
# Edit .env with your API keys:
# - DASHSCOPE_API_KEY (for Qwen API classifier)
# - OPENAI_API_KEY (for GPT-based classification)
# - HF_TOKEN (for HuggingFace model access)
```

Then source the environment:

```bash
source .env
```

## Data Preparation

See:
- `train_verl_qwen3_olmo3/data/README.md` for VERL/RLVR (Qwen3/OLMo3) dataset format and expected file locations.
- `train_simplerl_qwen25/data/README.md` for SimpleRL (Qwen2.5) dataset format.

At minimum, the default Qwen3 scripts expect:
- Math: `train_verl_qwen3_olmo3/data/math/train.parquet` and `train_verl_qwen3_olmo3/data/math/val.parquet`
- Physics: `train_verl_qwen3_olmo3/data/physics/train.parquet` and `train_verl_qwen3_olmo3/data/physics/val.parquet`

## Training

### Qwen3 Training (VERL/RLVR)

```bash
cd train_verl_qwen3_olmo3/scripts

# Math domain - Pure RL (8B model)
bash train_math_qwen3_8b_grpo_purerl.sh

# Math domain - Weighted creativity (8B model)
bash train_math_qwen3_8b_grpo_weighted_mul.sh

# Physics domain - Pure RL (8B model)
bash train_physics_qwen3_8b_grpo_purerl.sh

# Physics domain - Weighted creativity (8B model)
bash train_physics_qwen3_8b_grpo_weighted_mul.sh
```

### OLMo3 Training (VERL/RLVR)

```bash
cd train_verl_qwen3_olmo3/scripts

# Math domain (7B model)
bash train_math_olmo3_7b_grpo_purerl.sh
bash train_math_olmo3_7b_grpo_weighted_mul.sh

# Physics domain
bash train_physics_olmo3_7b_grpo_purerl.sh
bash train_physics_olmo3_7b_grpo_weighted_mul.sh
```

### Qwen2.5 Training (SimpleRL)

```bash
cd train_simplerl_qwen25

# Math domain with creativity classifier
bash creativity_train.sh

# With local classifier
bash creativity_train_qwen_classifer_local.sh

# With API classifier (Physics)
bash creativity_train_qwen_classifier_api_megascience.sh

# With API classifier (Medical)
bash creativity_train_qwen_classifier_api_medreasoning.sh
```

## Evaluation

### Pass@k Evaluation

```bash
cd eval_passn

# Basic pass@k evaluation
python run_pass_at_k_v2.py \
    --model_path /path/to/model \
    --dataset olympiadbench \
    --n_samples 16 \
    --k_values 1,4,8,16

# With LLM-as-judge for answer verification
python run_pass_at_k_v2.py \
    --model_path /path/to/model \
    --dataset olympiadbench \
    --use_llm_judge \
    --judge_model qwen2.5-72b-instruct

# OlympiadBench specific evaluation
python run_pass_at_k_olympiadbench.py \
    --model_path /path/to/model \
    --n_samples 16
```

## Project Structure

```
zhiyuan-train-eval-integrated/
├── README.md                    # This file
├── requirements.txt             # Python dependencies (verl_rlvr compatible)
├── environment.yml              # Conda environment file
├── env.example                  # Template for environment variables
├── SECURITY.md                  # Security guidelines
│
├── train_verl_qwen3_olmo3/     # Qwen3/OLMo3 training (VERL)
│   ├── scripts/                 # Training entry scripts
│   │   ├── _run_grpo_local.sh   # Shared local runner
│   │   ├── train_math_qwen3_*.sh
│   │   ├── train_physics_qwen3_*.sh
│   │   └── train_*_olmo3_*.sh
│   ├── recipe/                  # Domain-specific reward functions
│   │   ├── creativity/          # Math creativity
│   │   ├── physics/             # Physics domain
│   │   └── (more domains may be added later)
│   ├── verl/                    # VERL library
│   └── data/                    # Training data
│
├── train_simplerl_qwen25/      # Qwen2.5 training (SimpleRL)
│   ├── creativity_train*.sh     # Training scripts
│   ├── verl/                    # Modified VERL for SimpleRL
│   └── data/                    # Training data
│
└── eval_passn/                  # Evaluation framework
    ├── run_pass_at_k_v2.py      # Main evaluation script
    ├── run_pass_at_k_olympiadbench.py
    ├── OlympiadBench/           # OlympiadBench evaluator
    └── data/                    # Evaluation datasets
```



