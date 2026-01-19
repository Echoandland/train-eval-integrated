# Training Data for Qwen3/OLMo3 (VERL)

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

Organize data by domain:

```
data/
├── math/
│   ├── train.parquet
│   └── val.parquet
├── physics/
│   ├── train.parquet
│   └── val.parquet
```

## Data Sources

- **Math**: OlympiadBench, MATH, GSM8K
- **Physics**: MegaScience-Physics

## Usage

Training scripts expect data paths via environment variables:

```bash
export TRAIN_FILE="data/math/train.parquet"
export VAL_FILE="data/math/val.parquet"
bash scripts/train_math_qwen3_4b_grpo_purerl.sh
```
