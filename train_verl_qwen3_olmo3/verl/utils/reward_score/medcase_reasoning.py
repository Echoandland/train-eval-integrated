"""
Diagnostic accuracy reward aligned with Stanford MedCaseReasoning's
"Evaluating Diagnostic Accuracy" (LLM-as-judge) setup.

Reference project and metric description:
- MedCaseReasoning (evaluate.py reports Diagnostic accuracy with LLM-as-judge)
  https://github.com/kevinwu23/Stanford-MedCaseReasoning/tree/main#

This module provides two scorers:
- compute_score_judge(...): LLM-as-judge binary decision (1.0/0.0). This is the
  default and is intended to be consistent in spirit with MedCaseReasoning.
- compute_score_deterministic(...): Non-LLM fallback using normalized string
  matching plus optional alias mapping.

Top-level compute_score delegates to the judge-based scorer by default.
"""

from __future__ import annotations

import os
import re
import unicodedata
from typing import Any, Dict, Optional

try:
    import openai  # type: ignore
except Exception:
    openai = None  # allow importing the module without OpenAI installed


FINAL_PATTERNS = [
    r"final\s*diagnosis\s*[:\-]\s*(.+)",
    r"diagnosis\s*[:\-]\s*(.+)",
    r"dx\s*[:\-]\s*(.+)",
]


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def _normalize_label(text: str) -> str:
    if text is None:
        return ""
    text = _strip_diacritics(text)
    text = text.lower()
    # Remove punctuation, keep alphanumerics and whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


# —— only take"the last assistant block"；if missing ending <|im_end|>，take until end of text ——
_ASSISTANT_START_RE = re.compile(r"<\|im_start\|>\s*assistant\b", re.I)
_IM_END_RE = re.compile(r"<\|im_end\|>", re.I)


def _assistant_chunk(text: str) -> str:
    """Return the last assistant block content; fallback to original text if none.
    If the last assistant block lacks a closing tag, take from its start to end of text.
    """
    t = text or ""
    last_start = None
    for m in _ASSISTANT_START_RE.finditer(t):
        last_start = m
    if last_start is None:
        return t.strip()
    start = last_start.end()
    m_end = _IM_END_RE.search(t, pos=start)
    if m_end:
        return t[start:m_end.start()].strip()
    return t[start:].strip()


def _extract_predicted_dx(response_text: str) -> str:
    """Extract final diagnosis within the single assistant block:
    1) <answer>...</answer>
    2) Final diagnosis / Diagnosis / Dx
    3) last line of this block（if too long, truncate to first sentence）
    """
    t = _assistant_chunk(response_text)

    # 1) <answer>...</answer>
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", t, flags=re.S | re.I)
    if m:
        return m.group(1).strip()

    # 2) Final diagnosis / Diagnosis / Dx
    m = re.search(r"(?:final\s*)?diagnosis\s*[:：-]\s*(.+)", t, flags=re.I)
    if not m:
        m = re.search(r"\bdx\s*[:：-]\s*(.+)", t, flags=re.I)
    if m:
        cand = m.group(1).strip()
        cand = cand.splitlines()[0].strip()
        cand = cand.split("。")[0].split(".")[0].strip()
        return cand

    #3)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last) > 256:
            last = last.split("。")[0].split(".")[0].strip()
        return last
    return t


def _predicted_or_full(response_text: str) -> str:
    pred = _extract_predicted_dx(response_text).strip()
    #normalize （/）
    if _normalize_label(pred):
        return pred
    #assistant ；original
    return _assistant_chunk(response_text) or (response_text or "")


def _build_judge_messages(response_text: str, gold_label: str) -> list:
    """Strict y/n judge prompt for medical-diagnosis equivalence (drop-in)."""
    predicted = _predicted_or_full(response_text)

    system = (
        "You are a strict grader for medical diagnosis equivalence. "
        "Return EXACTLY one lowercase character: 'y' or 'n'. "
        "No spaces, no punctuation, no explanations. If uncertain, answer 'n'."
    )

    user = f"""TASK
Judge whether the predicted final diagnosis names the SAME disease entity as the true diagnosis.

SCORING POLICY — answer 'y' (SAME) if ANY apply:
• Synonyms / abbreviations: MI≡myocardial infarction≡heart attack; UTI≡urinary tract infection;
  CAP/HAP≡pneumonia; AF/A-fib≡atrial fibrillation; CKD≡chronic kidney disease; AKI≡acute kidney injury;
  COPD; CAD; TB≡tuberculosis.
• Ignore non-essential modifiers: acute/chronic/recurrent; mild/moderate/severe; laterality
  (left/right/bilateral); acquisition site (community/hospital); suspected/probable/likely/possible.
• Parent–subtype if same entity: STEMI/NSTEMI ≡ myocardial infarction; pyelonephritis ≡ urinary tract infection;
  viral URI ≡ upper respiratory infection.
• Cross-language equivalence allowed: 心肌梗死↔myocardial infarction; 肺炎↔pneumonia; 结核↔tuberculosis; 尿路感染↔urinary tract infection.

Answer 'n' (NOT the same) if ANY apply:
• Different diseases or systems (pneumonia ≠ bronchitis; sepsis ≠ pneumonia; angina ≠ myocardial infarction).
• Vague category vs. specific distinct disease ("infection"/"fever/发热"/"pain/疼痛"/"cancer/肿瘤") when the true label is a
  different specific disease.
• Different concrete etiologies that change the disease entity (bacterial pneumonia ≠ tuberculosis).

Only judge the final/main diagnosis string below.
Predicted diagnosis: {predicted}
True diagnosis: {gold_label}

Output exactly one character: y or n.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _yn_to_score(text: str) -> float:
    t = (text or "").strip().lower()
    if t.startswith("y"):
        return 1.0
    if t.startswith("n"):
        return 0.0
    #：output "yes/correct/true"
    if t.startswith(("yes", "correct", "true")):
        return 1.0
    return 0.0


def compute_score_judge(
    response_text: str,
    ground_truth_label: str,
    *,
    api_key: Optional[str] = None,
    model: str = "gpt-4o-mini",
    max_tokens: int = 16,
    temperature: float = 0.0,
    base_url: Optional[str] = None,
) -> float:
    """LLM-as-judge diagnostic accuracy; returns 1.0 (CORRECT) or 0.0 (INCORRECT).

    Uses OpenAI Chat Completions API for compatibility with various models.
    Default model is gpt-4o-mini for cost efficiency.
    """
    #API key（prefer，）
    api_key = api_key or os.getenv("OPENAI_API_KEY")

    if openai is None:
        print("[medcase_reasoning] OpenAI SDK not installed; cannot run LLM judge.")
        return 0.0

    if not api_key:
        print("[medcase_reasoning] OPENAI_API_KEY not set; cannot run LLM judge.")
        return 0.0

    #messages
    messages = _build_judge_messages(response_text or "", ground_truth_label or "")

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        #Chat Completions API
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        #extract
        text = ""
        if resp.choices:
            text = resp.choices[0].message.content or ""

        #y/n
        return _yn_to_score(text)

    except Exception as e:
        print(f"[medcase_reasoning] LLM judge call failed: {type(e).__name__}: {e}")
        return 0.0


def compute_score_deterministic(
    response_text: str,
    ground_truth_label: str,
) -> float:
    """Non-LLM deterministic matching; returns 1.0 if labels match after normalization.
    
    This is a fallback when LLM judge is not available.
    """
    try:
        pred_raw = _extract_predicted_dx(response_text)
        gold_raw = ground_truth_label or ""

        pred = _normalize_label(pred_raw)
        gold = _normalize_label(gold_raw)

        return 1.0 if pred and gold and pred == gold else 0.0
    except Exception:
        return 0.0


def compute_score(
    response_text: str,
    ground_truth_label: str,
    **kwargs: Any,
) -> Dict[str, float]:
    """Default to LLM-as-judge to align with MedCaseReasoning's diagnostic accuracy.

    Returns a dict with 'score' and 'correctness' keys for compatibility with training loop.
    
    kwargs are passed to compute_score_judge, e.g., model, api_key, base_url.
    """
    score = compute_score_judge(response_text, ground_truth_label, **kwargs)
    return {"score": score, "correctness": score}

