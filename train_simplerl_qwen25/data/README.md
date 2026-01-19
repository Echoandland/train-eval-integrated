# Training Data for Qwen2.5 (SimpleRL)

Place your training and validation data files in this directory.

## Expected Format

Data should be in **Parquet** or **JSONL** format with the following fields:

```json
{
  "prompt": "The problem statement or question",
  "solution": "Optional: reference solution for reward computation",
  "answer": "Optional: expected final answer"
}
```

## Directory Structure

```
data/
├── creativity/
│   ├── train.parquet
│   └── val.parquet
├── math/
│   ├── train.parquet
│   └── val.parquet
└── medical/
    ├── train.parquet
    └── val.parquet
```

## Usage

The main training script `creativity_train.sh` expects:

```bash
export TRAIN_FILE="data/creativity/train.parquet"
export VAL_FILE="data/creativity/val.parquet"
export MODEL_PATH="/path/to/Qwen2.5-3B-Instruct"
bash creativity_train.sh
```
