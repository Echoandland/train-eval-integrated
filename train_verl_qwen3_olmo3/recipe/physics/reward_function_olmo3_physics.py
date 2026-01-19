"""
Custom reward function for Olmo3 physics training.

This is essentially the same as reward_function_megascience.py but kept separate
in case Olmo3-specific preprocessing is needed in the future.
"""

from typing import Any, Dict

from verl.utils.reward_score import megascience


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute reward for physics problems using megascience evaluation."""
    return megascience.compute_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        **kwargs,
    )

