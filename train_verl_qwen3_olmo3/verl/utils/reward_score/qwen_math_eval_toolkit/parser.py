import random
import regex
import re
import sympy
from typing import TypeVar, Iterable, List, Union, Any, Dict
from word2number import w2n


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except Exception:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _fix_sqrt(string):
    _string = re.sub(r"\\sqrt(\w+)", r"\\sqrt{\1}", string)
    return _string


def convert_word_number(text: str) -> str:
    try:
        text = str(w2n.word_to_num(text))
    except Exception:
        pass
    return text


unit_texts = [
    "east",
    "degree",
    "mph",
    "kmph",
    "ft",
    "m sqaure",
    " m east",
    "sq m",
    "deg",
    "mile",
    "q .",
    "monkey",
    "prime",
    "ratio",
    "profit of rs",
    "rd",
    "o",
    "gm",
    "p . m",
    "lb",
    "tile",
    "per",
    "dm",
    "lt",
    "gain",
    "ab",
    "way",
    "west",
    "a .",
    "b .",
    "c .",
    "d .",
    "e .",
    "f .",
    "g .",
    "h .",
    "t",
    "a",
    "h",
    "no change",
    "men",
    "soldier",
    "pie",
    "bc",
    "excess",
    "st",
    "inches",
    "noon",
    "percent",
    "by",
    "gal",
    "kmh",
    "c",
    "acre",
    "rise",
    "a . m",
    "th",
    "π r 2",
    "sq",
    "mark",
    "l",
    "toy",
    "coin",
    "sq . m",
    "gallon",
    "° f",
    "profit",
    "minw",
    "yr",
    "women",
    "feet",
    "am",
    "pm",
    "hr",
    "cu cm",
    "square",
    "v â € ™",
    "are",
    "rupee",
    "rounds",
    "cubic",
    "cc",
    "mtr",
    "s",
    "ohm",
    "number",
    "kmph",
    "day",
    "hour",
    "minute",
    "min",
    "second",
    "man",
    "woman",
    "sec",
    "cube",
    "mt",
    "sq inch",
    "mp",
    "∏ cm ³",
    "hectare",
    "more",
    "sec",
    "unit",
    "cu . m",
    "cm 2",
    "rs .",
    "rs",
    "kg",
    "g",
    "month",
    "km",
    "m",
    "cm",
    "mm",
    "apple",
    "liter",
    "loss",
    "yard",
    "pure",
    "year",
    "increase",
    "decrease",
    "d",
    "less",
    "Surface",
    "litre",
    "pi sq m",
    "s .",
    "metre",
    "meter",
    "inch",
]

unit_texts.extend([t + "s" for t in unit_texts])


def strip_string(string, skip_unit=False):
    string = str(string).strip()
    string = string.replace("\n", "")
    string = string.rstrip(".")
    string = string.replace("\\!", "")

    string = re.sub(r"\\begin\{array\}\{.*?\}", r"\\begin{pmatrix}", string)
    string = re.sub(r"\\end\{array\}", r"\\end{pmatrix}", string)
    string = string.replace("bmatrix", "pmatrix")

    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = (
        string.replace("\\neq", "\\ne")
        .replace("\\leq", "\\le")
        .replace("\\geq", "\\ge")
    )

    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("\\{", "{")
    string = string.replace("\\}", "}")

    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        string = _string

    if not skip_unit:
        for _ in range(2):
            for unit_text in unit_texts:
                _string = re.sub(r"(^|\W)" + unit_text + r"($|\W)", r"\1\2", string)
                if _string != "":
                    string = _string

    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    string = string.replace("\\$", "")
    string = string.replace("$", "")
    string = string.replace("\\(", "").replace("\\)", "")

    string = convert_word_number(string)

    string = re.sub(r"\\text\{(.*?)\}", r"\1", string)
    for key in ["x=", "y=", "z=", "x\\in", "y\\in", "z\\in", "x\\to", "y\\to", "z\\to"]:
        string = string.replace(key, "")
    string = string.replace("\\emptyset", r"{}")
    string = string.replace("(-\\infty,\\infty)", "\\mathbb{R}")

    string = string.replace("\\%", "")
    string = string.replace("\%", "")
    string = string.replace("%", "")

    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    if (
        string.startswith("{")
        and string.endswith("}")
        and string.isalnum()
        or string.startswith("(")
        and string.endswith(")")
        and string.isalnum()
        or string.startswith("[")
        and string.endswith("]")
        and string.isalnum()
    ):
        string = string[1:-1]

    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    string = re.sub(r"\\mbox{.*?}", "", string)

    string.replace("'", "")
    string.replace('"', "")

    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    string = re.sub(r"(\d+)\.0*([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0*$", r"\1", string)

    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    string = _fix_a_slash_b(string)

    return string


def extract_last_boxed(text):
    """
    Extract content from the last \boxed command in LaTeX text。
    Return the complete \boxed{...} segment; return None if not found.
    """
    pattern = r'\\boxed\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}'
    matches = list(re.finditer(pattern, text))
    if matches:
        return matches[-1].group(0)
    return None


def extract_answer(pred_str, data_name, use_last_number=True):
    """
    Extract final answer from model output for math-style datasets.
    This is a trimmed version focused on the 'math' path.
    """
    pred_str = pred_str.replace("\u043a\u0438", "")

    if data_name in ["mmlu_stem", "sat_math", "aqua", "gaokao2023"]:
        return choice_answer_clean(pred_str)

    if "final answer is $" in pred_str and "$. I hope" in pred_str:
        tmp = pred_str.split("final answer is $", 1)[1]
        pred = tmp.split("$. I hope", 1)[0].strip()
    elif "boxed" in pred_str:
        ans = pred_str.split("boxed")[-1]
        if len(ans) == 0:
            return ""
        if ans[0] == "{":
            stack = 1
            a = ""
            for c in ans[1:]:
                if c == "{":
                    stack += 1
                    a += c
                elif c == "}":
                    stack -= 1
                    if stack == 0:
                        break
                    a += c
                else:
                    a += c
        else:
            a = ans.split("$")[0].strip()
        pred = a
    elif "he answer is" in pred_str:
        pred = pred_str.split("he answer is")[-1].strip()
    elif "final answer is" in pred_str:
        pred = pred_str.split("final answer is")[-1].strip()
    elif "answer is" in pred_str:
        pred = pred_str.split("answer is")[1].strip().split("\n\n")[0].strip()
    else:
        if use_last_number:
            pattern = "-?\d*\.?\d+"
            pred_nums = re.findall(pattern, pred_str.replace(",", ""))
            if len(pred_nums) >= 1:
                pred = pred_nums[-1]
            else:
                pred = ""
        else:
            pred = ""

    if (
        data_name in ["sat_math", "aqua"]
        or "mmlu" in data_name
    ):
        tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
        if tmp:
            pred = tmp[-1]
        else:
            pred = pred.strip().strip(".")

    pred = re.sub(r"\n\s*", "", pred)
    if pred != "" and pred[0] == ":":
        pred = pred[1:]
    if pred != "" and pred[-1] == ".":
        pred = pred[:-1]
    if pred != "" and pred[-1] == "/":
        pred = pred[:-1]
    pred = strip_string(pred, skip_unit=data_name in ["carp_en", "minerva_math"])
    return pred


def choice_answer_clean(pred: str):
    pred = pred.strip("\n")
    ICL = False
    for trigger in ("choice is", "answer is"):
        if pred.count(trigger) > 1:
            ICL = True
    if ICL:
        pred = pred.split("\n\n")[0]

    preds = re.split("|".join(("choice is", "answer is")), pred)
    if len(preds) > 1:
        answer_flag = True
        pred = preds[-1]
    else:
        answer_flag = False

    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ").lstrip(":")

    tmp = re.findall(r"\b(A|B|C|D|E)\b", pred.upper())
    if tmp:
        pred = tmp
    else:
        pred = [pred.strip().strip(".")]

    if len(pred) == 0:
        pred = ""
    else:
        if answer_flag:
            pred = pred[0]
        else:
            pred = pred[-1]

    pred = pred.rstrip(".").rstrip("/")

    return pred



