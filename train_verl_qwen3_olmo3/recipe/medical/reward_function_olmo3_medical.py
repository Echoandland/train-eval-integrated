"""
Custom reward function for Olmo3 MedCaseReasoning training.

This delegates to `verl.utils.reward_score.medcase_reasoning.compute_score`,
which uses LLM-as-judge to evaluate diagnostic accuracy.
The judge compares the predicted diagnosis against the ground truth,
handling synonyms, abbreviations, and cross-language equivalence.
"""

from typing import Any, Dict

from verl.utils.reward_score import medcase_reasoning


def compute_score(
    data_source: Any,
    solution_str: str,
    ground_truth: str,
    extra_info: Dict[str, Any] | None = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Compute reward for medical diagnosis problems using LLM-as-judge.
    
    The LLM judge evaluates whether the predicted diagnosis is semantically
    equivalent to the ground truth, handling:
    - Synonyms and abbreviations (MI = myocardial infarction = heart attack)
    - Parent-subtype relationships (STEMI/NSTEMI = myocardial infarction)
    - Cross-language equivalence (the Chinese term for myocardial infarction)
    """
    return medcase_reasoning.compute_score(
        response_text=solution_str,
        ground_truth_label=ground_truth,
        **kwargs,
    )
