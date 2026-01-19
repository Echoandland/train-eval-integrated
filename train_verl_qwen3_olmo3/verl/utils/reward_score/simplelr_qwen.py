import re
import random
import logging
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from functools import wraps
from typing import Optional, Callable, Any, Dict

from .qwen_math_eval_toolkit.parser import extract_answer, extract_last_boxed
from .qwen_math_eval_toolkit.grader import math_equal


class GlobalProcessPool:
    """A small wrapper around ProcessPoolExecutor with automatic recovery."""

    _instance = None

    def __init__(self, max_workers: int = 16):
        self.max_workers = max_workers
        self.executor: Optional[ProcessPoolExecutor] = None
        self.logger = logging.getLogger(__name__)
        self._initialize_executor()

    def _initialize_executor(self) -> None:
        if self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None
        self.executor = ProcessPoolExecutor(max_workers=self.max_workers)
        self.logger.warning(f"Initialized ProcessPoolExecutor with {self.max_workers} workers")

    @classmethod
    def get_instance(cls, max_workers: int = 16) -> "GlobalProcessPool":
        if cls._instance is None:
            cls._instance = cls(max_workers=max_workers)
        return cls._instance

    def submit(self, fn: Callable, *args, **kwargs):
        try:
            if self.executor is None:
                self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)
        except Exception as e:
            self.logger.warning(f"Process pool broken, recreating: {str(e)}")
            self._initialize_executor()
            return self.executor.submit(fn, *args, **kwargs)


global_executor = GlobalProcessPool.get_instance(max_workers=16)


def hf_math_equal_subprocess(gold: str, target: str, timeout_seconds: int = 10) -> bool:
    """
    Run math_equal in a subprocess with timeout protection.
    """
    try:
        future = global_executor.submit(math_equal, prediction=target, reference=gold)
        result = future.result(timeout=timeout_seconds)
        return bool(result)
    except TimeoutError:
        print(f"Timeout occurred for gold {gold} and target {target}.")
        return False
    except Exception as e:
        print(f"Gold: {gold} Target: {target} Error: {str(e)}")
        return False


def compute_score(solution_str: str, ground_truth: str, method: str = "strict") -> Dict[str, Any]:
    """
    Scoring function for simplelr-style math datasets.

    Mirrors the behavior of the original simpleRL `hf_math_verify.compute_score`:
      1. Extract the final answer from the model output (using Qwen-style parser)
      2. Normalize answers into \\boxed{...} form
      3. Use a robust math equivalence checker (symbolic + numeric) for correctness
      4. Map correctness -> reward score (1.0 if correct, 0/penalty otherwise)
    """
    # 1. Extract predicted answer (string) and whether there was an explicit boxed answer
    extract_answer_str, is_boxed_matched = extract_solution(solution_str=solution_str)

    # 2. Ensure both prediction and ground truth are wrapped in \boxed{...}
    if "\\boxed" not in extract_answer_str:
        boxed_answer = f"\\boxed{{{extract_answer_str}}}"
    else:
        boxed_answer = extract_answer_str

    if "\\boxed" not in ground_truth:
        boxed_ground_truth = f"\\boxed{{{ground_truth}}}"
    else:
        boxed_ground_truth = ground_truth

    # 3. Compare using math_equal via subprocess (for robustness)
    correct = hf_math_equal_subprocess(gold=boxed_ground_truth, target=boxed_answer)

    # 4. Reward mapping: mimic original 'mix' behavior
    if correct:
        box_match = 1.0
    else:
        # If desired, a negative format penalty can be applied for badly formatted answers.
        # For now we keep it as 0 to match the common "correct => 1, else 0" scheme.
        box_match = 0.0

    # Occasionally print debug info
    if random.random() < 0.05:
        print(f"\n[Model Response]\n{solution_str}")
        print(f"\n[Ground Truth]\n{ground_truth}")
        print(f"\n[Extracted Answer]\n{extract_answer_str}")
        print(f"\n[Reward Score]\n{box_match}")

    # `acc` is added here so that components like DAPO's filter_groups logic
    # can use `metric='acc'` on simplelr-style datasets, matching the math_dapo API.
    return {
        "score": float(box_match),
        "correctness": bool(correct),
        "acc": bool(correct),
    }


def extract_solution(solution_str: str):
    """
    Approximate the original simpleRL `extract_solution` behavior:
      - Trim to assistant segment
      - Drop trailing special tokens
      - Use Qwen math parser to extract the final answer
      - Detect whether there was an explicit \\boxed answer
    """
    model_output = re.sub(
        r"^.*?<\|im_start\|>assistant", "<|im_start|>assistant", solution_str, flags=re.DOTALL, count=1
    )
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"]
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()

    predict_answer = extract_answer(model_output, data_name="math")
    extract_boxed_answer = extract_last_boxed(model_output)

    if extract_boxed_answer is not None:
        return predict_answer, True
    else:
        return predict_answer, False



