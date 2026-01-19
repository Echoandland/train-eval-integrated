"""
Qwen-based solution classifier for grouping similar solutions (Stage-1: Qwen; Stage-2/3: OpenAI GPT).

This is a lightly adapted copy of the implementation from simpleRL-reason-mine,
kept API-compatible so that creativity-style grouping code can be reused.
"""

import logging
import os
from typing import List, Tuple

from openai import OpenAI


# ===== Qwen API config (DashScope OpenAI-compatible) =====
BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
API_KEY = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DASHSCOPE_API_KEY to your Alibaba Cloud Model Studio API key.")

# Qwen client（fixed for Stage 1）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

#，
model_name = "qwen2.5-72b-instruct"
hf_token = ""


#===== OpenAI GPT client（ Stage 2 & 3，lazy initialization）=====
gpt_client = None
GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "o3-mini")  # default to o3-mini, cheaper


def get_openai_client():
    """Lazy initialization of OpenAI client (for GPT stages)."""
    global gpt_client
    if gpt_client is None:
        try:
            gpt_client = OpenAI()  # read OPENAI_API_KEY
            logging.info("OpenAI client initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI client: {e}")
            gpt_client = False
    return gpt_client if gpt_client is not False else None


def retry_call(func, max_retries: int = 3):
    """Simple retry wrapper for API calls."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
    return None


def get_qwen_response(model: str, tokenizer, messages, temp: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 4096):
    """Call Qwen via DashScope/OpenAI compatible API; signature kept consistent with old version."""

    def _do():
        base = os.getenv("QWEN_BASE_URL", "")
        use_dashscope = "dashscope" in base

        kwargs = dict(model=model, messages=messages)
        if use_dashscope:
            kwargs["max_completion_tokens"] = max_new_tokens
        else:
            kwargs["max_tokens"] = max_new_tokens
            # if sampling control needed，can enable：
            # kwargs["temperature"] = temp
            # kwargs["top_p"] = top_p

        resp = client.chat.completions.create(**kwargs)
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("Qwen empty output")
        return text

    return retry_call(_do)


def _messages_to_responses_input(messages):
    """
    Adapt Chat-style messages to Responses API:
    - If only 1 user message and content is plain string => return that string directly
    - Otherwise construct with input_text content blocks
    """
    try:
        if (
            isinstance(messages, list)
            and len(messages) == 1
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "user"
            and isinstance(messages[0].get("content"), str)
        ):
            return messages[0]["content"]
        return {"input_text": str(messages)}
    except Exception:
        return str(messages)


def get_gpt_response(model: str, messages, temp: float = 0.7, top_p: float = 0.9, max_new_tokens: int = 1024):
    """
    OpenAI GPT call（for Stage 2 / 3）。
    为了兼容 o 系列，使用 Responses API 风格的call。
    """
    _client = get_openai_client()
    if _client is None:
        raise RuntimeError("OpenAI client not available - cannot perform GPT calls.")

    def _do():
        payload = _messages_to_responses_input(messages)
        resp = _client.responses.create(
            model=model,
            input=payload,
            max_output_tokens=max_new_tokens,
        )
        #output_text output
        text = getattr(resp, "output_text", None)
        if text is None:
            text = ""
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "output_text":
                    part = getattr(item, "content", [""])[0]
                    text += getattr(part, "text", "") or ""
        return (text or "").strip()

    return retry_call(_do)


class Entropy_Calculation_Classify_By_Qwen:
    """Encapsulates the 3-stage classification pipeline (Qwen → GPT → GPT)."""

    #classifier API call， reward
    MAX_SOLUTION_CHARS_FOR_CLASSIFIER = 30000  # max characters per solution
    MAX_TOTAL_CHARS_FOR_CLASSIFIER = 120000    # max total input characters

    def __init__(self, model: str, solutions: List[str]):
        self.solutions = solutions
        self.stage1_cates = ""
        self.stage2_cates = ""
        self.categories = []
        self.model = model
        self.tokenizer = None  # kept for backward compatibility

    def _truncate_for_classifier(self, text: str, max_chars: int) -> str:
        """仅for classifier call的truncate，保留开头和结尾。"""
        if len(text) <= max_chars:
            return text
        #60% 30%
        head_len = int(max_chars * 0.6)
        tail_len = int(max_chars * 0.3)
        return text[:head_len] + "\n\n... [truncated] ...\n\n" + text[-tail_len:]

    def analyze_solutions_with_qwen(self, n_solutions: int) -> str:
        try:
            #===== Stage 1（Qwen）：grouping =====
            #note： classifier API，original solutions
            str_solutions = ""
            index = 1
            for solution in self.solutions[:n_solutions]:
                #solution
                truncated = self._truncate_for_classifier(solution, self.MAX_SOLUTION_CHARS_FOR_CLASSIFIER)
                str_solutions += f"Solution{index}:" + truncated + "\n"
                index += 1
            
            #length，
            if len(str_solutions) > self.MAX_TOTAL_CHARS_FOR_CLASSIFIER:
                avg_chars = (self.MAX_TOTAL_CHARS_FOR_CLASSIFIER - 2000) // n_solutions
                str_solutions = ""
                index = 1
                for solution in self.solutions[:n_solutions]:
                    truncated = self._truncate_for_classifier(solution, avg_chars)
                    str_solutions += f"Solution{index}:" + truncated + "\n"
                    index += 1

            user_input = (
                "Here are several solutions to the same question:" + str_solutions
                + "Please analyze and determine how these solutions can be grouped based on the methods they use. "
                "Your classification criteria must remain strictly high-level. Place solutions in different categories only when their overarching strategies are completely distinct; differences limited to sub-steps or implementation details do not count as high-level distinctions."
                "Before you begin grouping, clearly state the classification criteria you will follow. "
                "In your response, focus on explaining your reasoning and clearly state which solution indices should be grouped together. "
                "Note that if all solutions use entirely different approaches, each should be placed in its own distinct group. "
                "In your grouping, each solution should be assigned to exactly one of the groups. Make sure to carefully check the total number of solutions."
            )
            messages = [{"role": "user", "content": user_input}]
            response = get_qwen_response(self.model, self.tokenizer, messages)

            response = (
                response.replace("<|im_end|>", "")
                .replace("<|endoftext|>", "")
                .replace("<|im_start|>", "")
                .strip()
            )

            self.categories = response
            self.stage1_cates = self.categories

            #===== Stage 2（GPT）：extract {group: "Solution i, Solution j"} =====
            user_input = (
                "extract the category groups from the following text: " + self.categories
                + ' return the solution with categories like this format (for example, {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}), '
                'without any other text, and only use expressions like "Solution 1", "Solution 2"...to represent each solution, '
                "follow the example I give you. Make sure to carefully check the total number of solutions."
            )
            messages = [{"role": "user", "content": user_input}]
            response = get_gpt_response(GPT_MODEL, messages)
            self.categories = response
            self.stage2_cates = self.categories

            #===== Stage 3（GPT）：will mapping [1,2,2,...] =====
            user_input = (
                f"Convert this dictionary mapping to a list of {n_solutions} integers.\n\n"
                f"Input mapping: {self.categories}\n\n"
                f"Task: Create a list where position i contains the category number of Solution (i+1).\n"
                f"- List must have exactly {n_solutions} elements\n"
                f"- Use only the category numbers that appear in the mapping\n"
                f"- Order matters: [category_of_solution_1, category_of_solution_2, ...]\n\n"
                "Format: Return only the Python list, no explanation.\n\n"
                "Example:\n"
                'Input: {1: "Solution 1, Solution 5", 2: "Solution 3, Solution 4", 3: "Solution 2"}\n'
                "Output: [1, 3, 2, 2, 1]"
            )
            messages = [{"role": "user", "content": user_input}]
            response = get_gpt_response(GPT_MODEL, messages)
            self.categories = response
            return self.categories
        except Exception as e:
            print(f"Error in analyze_solutions_with_qwen: {e}")
            self.categories = str([1] * n_solutions)
            return self.categories


def get_categories(solutions: List[str], model: str = model_name) -> Tuple[str, str, str]:
    """
    Get category grouping for a list of solutions.

    Returns:
        Tuple of (categories_string, stage1_categories, stage2_categories)
    """
    try:
        ent = Entropy_Calculation_Classify_By_Qwen(model, solutions)
        cats = ent.analyze_solutions_with_qwen(len(solutions))
        return cats, ent.stage1_cates, ent.stage2_cates
    except Exception as e:
        print(f"Error in get_categories: {e}")
        fallback_categories = str(list(range(1, len(solutions) + 1)))
        return fallback_categories, "", ""


def parse_category_list(category_string: str) -> List[int]:
    """Parse the category string returned by GPT-4/Qwen into a list of integers."""
    try:
        cleaned = category_string.strip()
        if cleaned.startswith("[") and cleaned.endswith("]"):
            return list(map(int, eval(cleaned)))  # noqa: S307 - trusted LLM output in training context
        import re

        numbers = re.findall(r"\d+", cleaned)
        return [int(x) for x in numbers]
    except Exception as e:
        logging.error(f"Failed to parse category string '{category_string}': {e}")
        return []



