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

# Math domain - Pure RL (4B model)
bash train_math_qwen3_4b_grpo_purerl.sh

# Math domain - Weighted creativity (4B model)
bash train_math_qwen3_4b_grpo_weighted_mul.sh

# Math domain - 8B model
bash train_math_qwen3_8b_grpo_purerl.sh

# Physics domain
bash train_physics_qwen3_4b_grpo_purerl.sh
```

### Quickstart: Train Qwen3 4B end-to-end (single node)

This is a minimal, local “smoke test” to make sure the environment, model loading, dataset reading, vLLM rollout,
and reward computation are all wired correctly.

#### 0) Environment

- Docker: use the container command above; then run the following steps inside the container.
- Conda: follow Option 2/3 above. For VERL/RLVR scripts, make sure `vllm` is installed.

#### 1) Model path (Qwen3 4B)

The training scripts read `MODEL_PATH` and default to `models/Qwen3-4B`.
You can set it to either:
- a **local path** containing `config.json` / tokenizer files / weights, or
- a **HuggingFace repo id** (will download into `HF_HOME` cache).

Examples:

```bash
# Local directory under this repo (recommended for offline runs)
export MODEL_PATH="models/Qwen3-4B"

# OR: HuggingFace repo id (requires HF_TOKEN for gated/private models)
# export MODEL_PATH="Qwen/Qwen3-4B"
```

If you need to authenticate:

```bash
export HF_TOKEN="..."
huggingface-cli login --token "$HF_TOKEN"
```

#### 2) Minimal math dataset (Parquet)

The default math scripts expect `TRAIN_FILE=data/math/train.parquet` and `VAL_FILE=data/math/val.parquet`
under `train_verl_qwen3_olmo3/`.

Generate a tiny dataset:

```bash
cd train_verl_qwen3_olmo3
python - <<'PY'
import os
import pandas as pd

os.makedirs("data/math", exist_ok=True)
rows = [
  {
    "data_source": "simplelr/gsm8k",
    "prompt": [{"role":"user","content":"1+1=? Please output final answer in \\\\boxed{...}."}],
    "reward_model": {"style":"rule", "ground_truth":"2"},
  },
  {
    "data_source": "simplelr/gsm8k",
    "prompt": [{"role":"user","content":"Compute 10-3. Output final answer in \\\\boxed{...}."}],
    "reward_model": {"style":"rule", "ground_truth":"7"},
  },
]
df = pd.DataFrame(rows)
df.to_parquet("data/math/train.parquet", index=False)
df.to_parquet("data/math/val.parquet", index=False)
print("Wrote:", "data/math/train.parquet", "data/math/val.parquet")
PY
```

#### 3) Run training (Qwen3 4B, math, pure RL)

```bash
cd train_verl_qwen3_olmo3/scripts

# Optional: control output/cache roots
export CKPT_ROOT="$PWD/../outputs"
export CACHE_ROOT="$PWD/../.cache"

# Optional: multi-GPU (single node)
export N_GPUS_PER_NODE=1
export TENSOR_MODEL_PARALLEL_SIZE=1

bash train_math_qwen3_4b_grpo_purerl.sh
```

Outputs go to `train_verl_qwen3_olmo3/outputs/<EXPERIMENT_NAME>/` by default, where `EXPERIMENT_NAME` defaults to
`qwen3_4b_math_grpo_purerl`.

If you want to change data/model without editing scripts:

```bash
export MODEL_PATH="/abs/path/or/hf_repo_id"
export TRAIN_FILE="data/math/train.parquet"
export VAL_FILE="data/math/val.parquet"
export EXPERIMENT_NAME="my_qwen3_4b_run"
bash train_math_qwen3_4b_grpo_purerl.sh
```

### OLMo3 Training (VERL/RLVR)

```bash
cd train_verl_qwen3_olmo3/scripts

# Math domain (7B model)
bash train_math_olmo3_7b_grpo_purerl.sh
bash train_math_olmo3_7b_grpo_weighted_mul.sh

# Physics domain
bash train_physics_olmo3_7b_grpo_purerl.sh
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



