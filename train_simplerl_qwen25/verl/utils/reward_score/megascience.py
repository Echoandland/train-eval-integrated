# Copyright 2025
# Thin adapter to reuse lm-open-science-evaluation's extraction and
# correctness logic with a gsm8k-like compute_score signature.

import os
import sys
import importlib.util
import inspect
from typing import Any, Callable, Dict, Optional
import re


def _repo_root_dir() -> str:
    # this file: .../verl/utils/reward_score/megascience.py
    # repo root: go up three levels
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))


def _los_repo_dir() -> str:
    return os.path.join(_repo_root_dir(), "lm-open-science-evaluation")


def _ensure_sys_path_for_los() -> None:
    """Ensure the lm-open-science-evaluation dir is on sys.path so that
    its intra-repo imports like `from utils import ...` resolve correctly.
    """
    los_dir = _los_repo_dir()
    if los_dir not in sys.path:
        sys.path.insert(0, los_dir)


def _load_module_from(path: str, module_name: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Module not found at {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_EXTRACT_DEFAULT = "extract_gsm_few_shot_cot_answer"  # official example extractor (robust for text/expr/numeric)
_EVAL_DEFAULT = "eval_last_single_answer"  # eval/eval_script.py
_EXTRACT_FALLBACKS_DEFAULT = [
    "extract_gsm_few_shot_cot_answer",
    "extract_boxed_answers",
    "extract_answer",
]
_EVAL_FALLBACKS_DEFAULT = [
    "eval_last_single_answer",
    "eval_math",
    "eval_ocwcourses",
]


def _load_extract_fn() -> Callable[..., Any]:
    """Load answer extractor from lm-open-science-evaluation.

    Env overrides:
      - GAIR_EXTRACT_FN (preferred)
      - LOS_EXTRACT_FN (alias)
    Defaults to `extract_answer` (string-based, robust for math-like tasks).
    """
    _ensure_sys_path_for_los()
    mod_path = os.path.join(_los_repo_dir(), "data_processing", "answer_extraction.py")
    mod = _load_module_from(mod_path, "los_answer_extraction")
    fn_name = os.environ.get("GAIR_EXTRACT_FN", os.environ.get("LOS_EXTRACT_FN", _EXTRACT_DEFAULT))
    if not hasattr(mod, fn_name):
        raise ValueError(f"Extractor {fn_name} not found in answer_extraction.py")
    return getattr(mod, fn_name)


def _load_extractor_by_name(name: str) -> Callable[..., Any]:
    _ensure_sys_path_for_los()
    mod_path = os.path.join(_los_repo_dir(), "data_processing", "answer_extraction.py")
    mod = _load_module_from(mod_path, "los_answer_extraction")
    if not hasattr(mod, name):
        raise ValueError(f"Extractor {name} not found in answer_extraction.py")
    return getattr(mod, name)


def _load_eval_fn() -> Callable[..., Any]:
    """Load evaluation function from lm-open-science-evaluation.

    Env overrides:
      - GAIR_EVAL_FN (preferred)
      - LOS_EVAL_FN (alias)
    Defaults to `eval_last_single_answer`.
    """
    _ensure_sys_path_for_los()
    mod_path = os.path.join(_los_repo_dir(), "eval", "eval_script.py")
    mod = _load_module_from(mod_path, "los_eval_script")
    fn_name = os.environ.get("GAIR_EVAL_FN", os.environ.get("LOS_EVAL_FN", _EVAL_DEFAULT))
    if not hasattr(mod, fn_name):
        raise ValueError(f"Eval function {fn_name} not found in eval_script.py")
    return getattr(mod, fn_name)


def _load_eval_fn_by_name(name: str) -> Callable[..., Any]:
    """Load a specific evaluation function by name from eval_script.py."""
    _ensure_sys_path_for_los()
    mod_path = os.path.join(_los_repo_dir(), "eval", "eval_script.py")
    mod = _load_module_from(mod_path, "los_eval_script")
    if not hasattr(mod, name):
        raise ValueError(f"Eval function {name} not found in eval_script.py")
    return getattr(mod, name)


def _call_extractor(extract_fn: Callable[..., Any], solution_str: str) -> Optional[Any]:
    """Call extractor with either (item) or (pred_str) depending on its signature."""
    try:
        sig = inspect.signature(extract_fn)
        params = list(sig.parameters.values())
        if params and (params[0].name == "item" or params[0].annotation in (dict, Dict)):
            # Build a minimal item expected by many extractors
            item = {"model_output": solution_str, "messages": [{"role": "user", "content": ""}, {"role": "assistant", "content": ""}]}
            return extract_fn(item)
        else:
            # Assume string-based extractor, e.g., extract_answer(pred_str)
            try:
                return extract_fn(solution_str)
            except TypeError:
                # Some string extractors accept (pred_str, exhaust=...)
                return extract_fn(solution_str, exhaust=False)
    except Exception:
        return None


def _extract_last_boxed(solution_str: str) -> Optional[str]:
    """Simple fallback: extract the last \\boxed{...} content from raw text."""
    try:
        matches = re.findall(r"\\boxed\{([^}]*)\}", solution_str, flags=re.DOTALL)
        if matches:
            return matches[-1].strip()
    except Exception:
        pass
    return None


def _normalize_pred_str(s: str) -> str:
    """Light normalization for LaTeX-ish answers.
    - strip whitespace
    - remove surrounding single $ ... $
    - collapse internal whitespace
    """
    if s is None:
        return ""
    s = str(s).strip()
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    s = re.sub(r"\s+", " ", s)
    return s


def _infer_unit(s: str) -> Optional[str]:
    """Heuristic: try to infer a unit token from the ground truth string.
    Examples: '3 m/s', '5 N', '10 J', '2 kg', '9.8 m/s^2'.
    Returns just the unit part (e.g., 'm/s', 'N'), or None if not found.
    """
    try:
        txt = str(s)
        # Common pattern: number [space]? unit (allow / and ^ in unit)
        m = re.search(r"[+-]?\d+(?:[.,]\d+)?\s*([A-Za-zμΩ°]+(?:[/*·][A-Za-zμΩ°]+)*(?:\^[+-]?\d+)?)", txt)
        if m:
            unit = m.group(1).strip()
            # Filter out cases where unit is likely a variable name (single letter without typical unit chars)
            if len(unit) == 1 and unit.lower() not in {"m","s","n","j","w","v","a","k","g"}:
                return None
            return unit
    except Exception:
        pass
    return None


def compute_score(solution_str: str, ground_truth: str, method: str = 'strict', format_score: float = 0.0, score: float = 1.0) -> dict:
    """Compute reward using lm-open-science-evaluation's logic, with gsm8k-compatible signature.

    Args:
        solution_str: the raw model output text
        ground_truth: the reference answer string
        method: kept for signature compatibility (unused)
        format_score: score when an answer is extracted but judged incorrect
        score: score when judged correct
    Returns:
        dict with keys:
          - 'score': float
          - 'correctness': float (1.0 for correct, else 0.0)
    """
    debug = os.environ.get("LOS_DEBUG", os.environ.get("GAIR_DEBUG", "0")) in {"1", "true", "True"}
    extract_fn = _load_extract_fn()
    eval_fn = _load_eval_fn()

    # parse configurable fallbacks from env
    def _parse_list_env(key: str, default_list: list[str]) -> list[str]:
        raw = os.environ.get(key, "").strip()
        if not raw:
            return default_list
        return [x.strip() for x in raw.split(',') if x.strip()]

    extract_chain = _parse_list_env("GAIR_EXTRACT_FALLBACKS", _EXTRACT_FALLBACKS_DEFAULT)
    eval_chain = _parse_list_env("GAIR_EVAL_FALLBACKS", _EVAL_FALLBACKS_DEFAULT)

    # 1) Extract a normalized prediction using official extractor
    # Try extractor chain in order until we get a valid prediction
    tried_extractors = []
    pred = None
    for name in extract_chain:
        tried_extractors.append(name)
        if name == "extract_boxed_answers":
            # official function returns list; if missing in repo use regex fallback
            try:
                fn = _load_extractor_by_name(name)
                out = _call_extractor(fn, solution_str)
                if isinstance(out, list) and out:
                    pred = out[-1]
                elif isinstance(out, str):
                    pred = out
                else:
                    pred = None
            except Exception:
                pred = _extract_last_boxed(solution_str)
        elif name == getattr(extract_fn, "__name__", ""):
            pred = _call_extractor(extract_fn, solution_str)
        else:
            try:
                fn = _load_extractor_by_name(name)
                pred = _call_extractor(fn, solution_str)
            except Exception:
                pred = None
        if pred not in (None, "", "[invalid]", "placeholder"):
            break

    # Normalize to a string for default evals like eval_last_single_answer
    if pred is None:
        return {"score": 0.0, "correctness": 0.0}
    if isinstance(pred, list):
        # Use the last element as many extractors do, or join if needed
        pred_str = pred[-1] if pred else ""
    else:
        pred_str = str(pred)

    if not pred_str or pred_str in {"[invalid]", "placeholder"}:
        return {"score": 0.0, "correctness": 0.0}

    pred_str = _normalize_pred_str(pred_str)
    gt_str = _normalize_pred_str(ground_truth)

    # 2) Build item for eval function
    # Try evaluator chain in order until one says correct
    def _build_item_for(name: str, unit: Optional[str] = None) -> dict:
        if name in ("eval_last_single_answer", "eval_ocwcourses"):
            return {"prediction": pred_str, "answer": gt_str}
        if name == "eval_scibench":
            # Ensure 'unit' key exists to avoid KeyError inside scibench
            return {"prediction": [pred_str], "answer": gt_str, "unit": unit or ""}
        return {"prediction": [pred_str], "answer": gt_str}

    # Precision for numeric closeness; can be overridden via env
    try:
        prec = float(os.environ.get("LOS_PREC", "1e-3"))
    except Exception:
        prec = 1e-3

    # 3) Evaluate correctness using official eval function
    def _run_eval(fn: Callable[..., Any], item: dict) -> bool:
        try:
            return bool(fn(item, pred_key="prediction", prec=prec))
        except TypeError:
            try:
                return bool(fn(item, pred_key="prediction"))
            except TypeError:
                try:
                    return bool(fn(item))
                except Exception:
                    return False

    # First attempt: configured/default evaluator
    tried_evals = []
    correct = False
    for name in eval_chain:
        tried_evals.append(name)
        try:
            fn = eval_fn if name == getattr(eval_fn, "__name__", "") else _load_eval_fn_by_name(name)
        except Exception:
            continue
        # Skip scibench unless we either infer a unit or the user explicitly enabled it
        item = _build_item_for(name)
        ok = _run_eval(fn, item)
        if debug:
            print(f"[LOS] eval={name}, pred='{pred_str}', gt='{gt_str}', correct={ok}")
        if ok:
            correct = True
            break

    score_value = float(score if correct else format_score)
    return {"score": score_value, "correctness": float(1.0 if correct else 0.0)}


