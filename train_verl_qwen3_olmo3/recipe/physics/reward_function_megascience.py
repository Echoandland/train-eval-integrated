"""
Custom reward function for physics datasets (MegaScience / lm-open-science-evaluation style).

This delegates to `verl.utils.reward_score.megascience.compute_score`,
which uses lm-open-science-evaluation's extraction and evaluation logic.
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

