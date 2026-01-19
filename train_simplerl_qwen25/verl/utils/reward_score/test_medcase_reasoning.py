#!/usr/bin/env python3

import os
import json
from typing import List, Tuple

import medcase_reasoning as mcr


def print_case(title: str, response_text: str, gold: str) -> None:
    print(f"\n=== {title} ===")
    print("Response snippet:")
    print("-" * 40)
    snippet = (response_text or "").splitlines()
    for line in snippet[:8]:
        print(line)
    if len(snippet) > 8:
        print("...")
    print("-" * 40)
    extracted = None
    try:
        extracted = mcr._extract_predicted_dx(response_text)
    except Exception as e:
        extracted = f"ERROR extracting: {type(e).__name__}: {e}"
    print(f"Extracted answer: {extracted}")

    try:
        det = mcr.compute_score_deterministic(response_text, gold)
    except Exception as e:
        det = f"ERROR deterministic: {type(e).__name__}: {e}"
    print(f"Deterministic score: {det}")

    # Judge test (optional)
    judge_score = None
    try:
        if os.getenv("OPENAI_API_KEY") and mcr.openai is not None:
            judge_score = mcr.compute_score_judge(
                response_text,
                gold,
                api_key=os.getenv("OPENAI_API_KEY"),
                # keep model as defined in module (currently o3-mini)
                max_tokens=16,
            )
        else:
            judge_score = "SKIP (no OPENAI_API_KEY or openai SDK)"
    except Exception as e:
        judge_score = f"ERROR judge: {type(e).__name__}: {e}"
    print(f"Judge score: {judge_score}")


def main() -> None:
    cases: List[Tuple[str, str, str]] = []

    # 1) With <answer> tags inside assistant block (should extract Phototoxic reaction)
    resp1 = (
        "<|im_start|>assistant\n"
        "<think>reasoning...</think>\n"
        "<answer>\nPhototoxic reaction\n</answer>\n"
        "<|im_end|>\n"
    )
    cases.append(("with_answer_tags", resp1, "Phototoxic reaction"))

    # 2) With explicit Final diagnosis line
    resp2 = (
        "<|im_start|>assistant\n"
        "Differential: cellulitis vs DVT; labs normal.\n"
        "Final diagnosis: Acute appendicitis.\n"
        "<|im_end|>\n"
    )
    cases.append(("final_diagnosis_line", resp2, "Acute appendicitis"))

    # 3) No explicit pattern; last non-empty line is the dx
    resp3 = (
        "<|im_start|>assistant\n"
        "Reasoning paragraphs...\n"
        "Acute myocarditis\n"
        "<|im_end|>\n"
    )
    cases.append(("last_line_fallback", resp3, "Acute myocarditis"))

    # 4) Negative case (should yield 0.0 deterministic)
    resp4 = (
        "<|im_start|>assistant\n"
        "<answer>Cellulitis</answer>\n"
        "<|im_end|>\n"
    )
    cases.append(("negative_case", resp4, "Phototoxic reaction"))

    for title, resp, gold in cases:
        print_case(title, resp, gold)


if __name__ == "__main__":
    main()
