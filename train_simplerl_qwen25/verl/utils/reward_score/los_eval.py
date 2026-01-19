# Copyright 2025
# A reward wrapper that reuses the correctness logic from lm-open-science-evaluation

import os
import sys
import importlib.util
from typing import Callable, Dict


def _load_eval_script_module():
    """Load lm-open-science-evaluation/eval/eval_script.py as a module."""
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    eval_script_path = os.path.join(base_dir, "lm-open-science-evaluation", "eval", "eval_script.py")
    if not os.path.exists(eval_script_path):
        raise FileNotFoundError(f"eval_script.py not found at {eval_script_path}")

    spec = importlib.util.spec_from_file_location("los_eval_script", eval_script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {eval_script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["los_eval_script"] = module
    spec.loader.exec_module(module)
    return module


def _get_eval_fn() -> Callable[[Dict, str, float], bool]:
    """Select an eval function from lm-open-science-evaluation by name.

    Controlled via env var LOS_EVAL_FN. Defaults to 'eval_math'.
    """
    eval_fn_name = os.environ.get("LOS_EVAL_FN", "eval_math")
    # Import lazily to avoid import cost when unused
    los_eval = _load_eval_script_module()
    if not hasattr(los_eval, eval_fn_name):
        raise ValueError(f"LOS_EVAL_FN={eval_fn_name} not found in eval_script.py")
    return getattr(los_eval, eval_fn_name)


def _extract_prediction(solution_str: str) -> str:
    """Extract the final answer from full prompt+response text.

    Reuse the robust extractor used in existing math verifier.
    """
    try:
        from .hf_math_verify import extract_solution as extract_solution_hf
        pred, _ = extract_solution_hf(solution_str)
        return pred if pred is not None else ""
    except Exception:
        # Fallback: return raw string (eval side may still normalize)
        return solution_str


def compute_score(solution_str: str, ground_truth: str, prec: float = 1e-3) -> dict:
    """Compute reward using Open-Science evaluation correctness logic.

    Returns a dict compatible with RewardManager expectations:
      {"score": float, "correctness": bool}

    Env overrides:
      - LOS_EVAL_FN: which eval_... function to use (default: eval_math)
      - LOS_CORRECT_SCORE: score when correct (default: 1.0)
      - LOS_INCORRECT_SCORE: score when incorrect (default: 0.0)
    """
    eval_fn = _get_eval_fn()

    # Build the expected item format for eval functions
    prediction = _extract_prediction(solution_str)

    item = {
        "prediction": [prediction],  # many eval_* accept list and handle de-dup
        "answer": ground_truth,
    }

    try:
        correct = bool(eval_fn(item, pred_key="prediction", prec=prec))
    except TypeError:
        # Some eval_* take only (item); retry without extra args
        correct = bool(eval_fn(item))

    correct_score = float(os.environ.get("LOS_CORRECT_SCORE", 1.0))
    incorrect_score = float(os.environ.get("LOS_INCORRECT_SCORE", 0.0))
    score = correct_score if correct else incorrect_score

    return {"score": float(score), "correctness": bool(correct)}


