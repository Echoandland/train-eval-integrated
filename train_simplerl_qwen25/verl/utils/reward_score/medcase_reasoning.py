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

import re
import unicodedata
from typing import Dict, Optional

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


# def _extract_predicted_label(response_text: str) -> str:
#     """Attempt to extract the final diagnosis from a free-form response.

#     Heuristics:
#     - Look for lines containing "Final diagnosis:", "Diagnosis:", or "Dx:" and
#       take the trailing text.
#     - Otherwise, fallback to the last non-empty line.
#     - If nothing is found, return the original text.
#     """
#     if not response_text:
#         return ""

#     text = response_text.strip()
#     # Try explicit patterns
#     for pat in FINAL_PATTERNS:
#         m = re.search(pat, text, flags=re.IGNORECASE)
#         if m:
#             candidate = m.group(1).strip()
#             # Stop at common line/sentence boundaries
#             candidate = candidate.split("\n")[0].strip()
#             candidate = candidate.split(".")[0].strip()
#             return candidate

#     # Fallback: use last non-empty line as the "prediction"
#     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#     if lines:
#         # Take last line; if it is very long, take first sentence
#         last = lines[-1]
#         if len(last) > 256 and "." in last:
#             last = last.split(".")[0]
#         return last

#     return text


# def _apply_alias(label: str, aliases: Optional[Dict[str, str]]) -> str:
#     if not aliases:
#         return label
#     # Match on normalized keys
#     normalized_label = _normalize_label(label)
#     for k, v in aliases.items():
#         if _normalize_label(k) == normalized_label:
#             return v
#     return label


# def compute_score_deterministic(response_text: str,
#                                 ground_truth_label: str,
#                                 aliases: Optional[Dict[str, str]] = None) -> float:
#     """Non-LLM deterministic matching; returns 1.0 if labels match after normalization."""
#     try:
#         pred_raw = _extract_predicted_label(response_text)
#         gold_raw = ground_truth_label or ""

#         # Apply alias mapping before normalization (canonicalization step)
#         pred_canon = _apply_alias(pred_raw, aliases)
#         gold_canon = _apply_alias(gold_raw, aliases)

#         pred = _normalize_label(pred_canon)
#         gold = _normalize_label(gold_canon)

#         return 1.0 if pred and gold and pred == gold else 0.0
#     except Exception:
#         return 0.0


import re

# —— only take“the last assistant block”；if missing ending <|im_end|>，take until end of text ——
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


# def _build_judge_messages(response_text: str, gold_label: str) -> list:
#""" y/n judge ；，response"""
#     predicted = _predicted_or_full(response_text)
#     user = (
#         "Is our predicted diagnosis correct (y/n)?\n"
#         f"Predicted diagnosis: {predicted}, True diagnosis: {gold_label}\n"
#         "Answer only with[y/n]."
#     )
#     return [{"role": "user", "content": user}]
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
• Vague category vs. specific distinct disease (“infection”/“fever/发热”/“pain/疼痛”/“cancer/肿瘤”) when the true label is a
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
    if t.startswith("y"): return 1.0
    if t.startswith("n"): return 0.0
    #：output "yes/correct/true"
    if t.startswith(("yes", "correct", "true")): return 1.0
    return 0.0



def compute_score_judge(response_text: str,
                        ground_truth_label: str,
                        *,
                        api_key: Optional[str] = None,
                        model: str = "o3",
                        max_tokens: int = 128,
                        temperature: float = 0.0,   # kept for compat
                        base_url: Optional[str] = None) -> float:
    """LLM-as-judge diagnostic accuracy; returns 1.0 (CORRECT) or 0.0 (INCORRECT).
    
    FIXED TO USE ONLY o3 MODEL.
    Ignores the 'model' parameter and always uses o3 with Responses API.
    """
    #1) messages（）
    raw_messages = _build_judge_messages(response_text or "", ground_truth_label or "")

    #2) Responses API content
    def to_structured(messages):
        out = []
        for m in messages:
            txt = m.get("content", "")
            out.append({
                "role": m.get("role", "user"),
                "content": [{"type": "input_text", "text": txt}],
            })
        return out

    structured_input = to_structured(raw_messages)

    if openai is None:
        print("[medcase_reasoning] OpenAI SDK not installed; cannot run LLM judge.")
        return 0.0

    try:
        client = openai.OpenAI(api_key=api_key, base_url=base_url)

        #o3 Responses API
        resp = client.responses.create(
            model="o3",
            input=structured_input,
            max_output_tokens=max_tokens,
        )

        #--- extract ---
        text = getattr(resp, "output_text", None) or ""
        if not text:
            parts = getattr(resp, "output", None) or []
            chunks = []
            for p in parts:
                msg = getattr(p, "message", None)
                if msg and getattr(msg, "content", None):
                    for c in msg.content:
                        t = getattr(c, "text", None)
                        if t:
                            chunks.append(t)
            text = " ".join(chunks).strip()

        #y/n
        return _yn_to_score(text)

    except Exception as e:
        print(f"[medcase_reasoning] LLM judge call failed: {type(e).__name__}: {e}")
        return 0.0

def compute_score(response_text: str,
                  ground_truth_label: str,
                  **judge_kwargs) -> float:
    """Default to LLM-as-judge to align with MedCaseReasoning's diagnostic accuracy.

    judge_kwargs are passed to compute_score_judge, e.g., model, api_key, base_url.
    """
    return compute_score_judge(response_text, ground_truth_label, **judge_kwargs)
