# Training Data for Qwen3/OLMo3 (VERL)

Place your training and validation data files in this directory.

## Expected Format

Data should be in **Parquet** or **JSONL** format. For the default GRPO scripts in `scripts/`,
each row/item should include (at least) the following fields:

```json
{
  "data_source": "simplelr/gsm8k",
  "prompt": [{"role": "user", "content": "Question text..."}],
  "reward_model": {
    "style": "rule",
    "ground_truth": "72"
  }
}
```

Notes:
- `prompt` is stored in **chat format** (a list of `{role, content}` dicts). This matches
  `verl/utils/dataset/README.md` and the internal dataset loader.
- For the math creativity reward function used by Qwen3 scripts (`recipe/creativity/reward_function_simplelr.py`),
  `data_source` must contain the substring `"simplelr"` (otherwise it will raise `NotImplementedError`).
- `reward_model.ground_truth` should be a **string** containing the expected final answer. The scorer will normalize
  both prediction and ground truth into `\\boxed{...}` form before equivalence checking.

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
cd train_verl_qwen3_olmo3
export TRAIN_FILE="data/math/train.parquet"
export VAL_FILE="data/math/val.parquet"
cd scripts
bash train_math_qwen3_4b_grpo_purerl.sh
```
