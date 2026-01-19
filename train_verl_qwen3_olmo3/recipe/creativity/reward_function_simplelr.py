from typing import Any, Dict

from verl.utils.reward_score import simplelr_qwen


"""
Custom reward function for `simplelr_qwen`-style datasets, implemented fully
inside RLVR (no dependency on the simpleRL repository).

This simply delegates to `verl.utils.reward_score.simplelr_qwen.compute_score`,
which mirrors the behavior of the original simpleRL hf_math_verify-based scorer.
"""


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    if "simplelr" not in str(data_source):
        raise NotImplementedError(f"Reward function is not implemented for data_source={data_source!r}")

    return simplelr_qwen.compute_score(solution_str=solution_str, ground_truth=ground_truth)
