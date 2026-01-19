# or save as medreason_eval.py then import it in the trainer
from __future__ import annotations
import re
import unicodedata
import string
from typing import Optional

#only takethe last assistant block（ ChatML / OpenAI format）
_ASSISTANT_BLOCK_RE = re.compile(
    r"<\|im_start\|>\s*assistant\s*(.*?)<\|im_end\|>", re.S | re.I
)

def _assistant_chunk(text: str) -> str:
    text = text or ""
    blocks = list(_ASSISTANT_BLOCK_RE.finditer(text))
    if blocks:
        return blocks[-1].group(1).strip()
    return text.strip()

# —— normalize/clean —— #
_CJK_PUNC = "，。、；：！？【】（）《》“”‘’—·\u3000"
_PUNC_TABLE = str.maketrans({c: " " for c in (string.punctuation + _CJK_PUNC)})

def _clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    #ASCII CJK，->
    s = s.translate(_PUNC_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

#—— will gold —— #
#：'A'..'F' / 'a'..'f' / '0'..'5'（0->A）/ '1'..'6'（1->A）
#"Answer: C"、" D" （）
def _normalize_gold(gold: str) -> Optional[str]:
    g = (gold or "").strip()
    # (translated comment)
    m = re.search(r"\b([A-Fa-f])\b", g)
    if m:
        return m.group(1).upper()
    #：prefer 0->A ； 0..5 1->A
    m = re.search(r"\b([0-9])\b", g)
    if m:
        n = int(m.group(1))
        if 0 <= n <= 5:
            return chr(ord('A') + n)
        if 1 <= n <= 6:
            return chr(ord('A') + (n - 1))
    return None

#—— “” —— #
#MedReason ：→→。
#，“/LLM ”。
_ANSWER_PATTERNS = [
    #XML-ish /
    r"<final_answer>\s*([A-F])\s*</final_answer>",
    r"<answer>\s*([A-F])\s*</answer>",
    r"<dx>\s*([A-F])\s*</dx>",
    # (translated comment)
    r"(?:final\s*)?answer\s*[:：\-]\s*([A-F])\b",
    r"the\s+answer\s+is\s*[:：\-]?\s*([A-F])\b",
    r"(?:choose|select)\s*[:：\-]?\s*([A-F])\b",
    r"option\s*[:：\-]?\s*([A-F])\b",
    r"\b(?:therefore|so|thus)[, ]+\s*([A-F])\b",
    # (translated comment)
    r"(?:final|last|correct)?\s*答案\s*[:：\-]?\s*([A-F])\b",
    r"(?:选择|option)\s*[:：\-]?\s*([A-F])\b",
]

#/（ "，C" "：D" ）
#100 A-F
def _tail_pick(text: str) -> Optional[str]:
    tail = text[-200:]  # only check tail to reduce false hits
    #prefer“answer///”
    key = re.findall(r"(?:answer|答案|option|选择)[:：\-]?\s*([A-F])\b", tail, flags=re.I)
    if key:
        return key[-1].upper()
    #： A-F（）
    m = re.findall(r"\b([A-F])\b", tail)
    if m:
        return m[-1].upper()
    return None

def _extract_predicted_choice(response_text: str) -> Optional[str]:
    t = _assistant_chunk(response_text)
    if not t:
        return None

    #1) /（prefer）
    for pat in _ANSWER_PATTERNS:
        m = re.search(pat, t, flags=re.I | re.S)
        if m:
            return m.group(1).upper()

    #2) ：“”
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        #prefer“final/answer///”（）
        for ln in reversed(lines[-6:]):  # look at last few lines
            if re.search(r"(final|answer|答案|option|选择)", ln, flags=re.I):
                m = re.search(r"\b([A-F])\b", ln)
                if m:
                    return m.group(1).upper()

    #3)
    pick = _tail_pick(t)
    if pick:
        return pick

    #4) ：clean
    ct = _clean(t)
    m = re.findall(r"\b([A-F])\b", ct)
    if m:
        return m[-1].upper()

    return None

#—— API： —— #
def compute_score(response_text: str,
                  ground_truth_label: str,
                  **kwargs) -> dict:
    """
    MedReason equivalent reward for evaluation：whether multiple choice option letters match。
    return：dictcontains 'score' 和 'correctness' key，value为 1.0（correct）或 0.0（error/无法extract）。

    parameter
    - response_text: model's complete output（may contain ChatML wrapper etc）
    - ground_truth_label: ground truth option；supports：
        'A'..'F' / 'a'..'f' / '0'..'5'（0->A） / '1'..'6'（1->A）
        以及诸如 "Answer: C" 这class会被进一步parse出字母的string
    """
    gold = _normalize_gold(ground_truth_label)
    if not gold:
        #gold 0.0（：）
        return {'score': 0.0, 'correctness': 0.0}

    pred = _extract_predicted_choice(response_text)
    if not pred:
        return {'score': 0.0, 'correctness': 0.0}

    score_value = 1.0 if pred == gold else 0.0
    return {'score': score_value, 'correctness': score_value}
