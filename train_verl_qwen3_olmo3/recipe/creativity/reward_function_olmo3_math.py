from typing import Any, Dict

from verl.utils.reward_score import simplelr_qwen


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Olmo-3 specific reward wrapper for simplelr-style math datasets.

    This keeps the core simplelr_qwen scoring logic (which mirrors hf_math_verify),
    but is separated so that Olmo-3 can evolve independently from Qwen-specific
    assumptions if needed.
    """
    if "simplelr" not in str(data_source):
        raise NotImplementedError(f"Reward function is not implemented for data_source={data_source!r}")

    return simplelr_qwen.compute_score(solution_str=solution_str, ground_truth=ground_truth)


