"""
Solution classification using OpenAI (e.g., o3) for grouping similar solutions.

This mirrors the API of the simpleRL-reason-mine implementation so that
creativity-style GRPO weighting code can be reused in RLVR without changes.
"""

import logging
from typing import List, Tuple

from openai import OpenAI


client = None  # lazy init


def get_openai_client():
    """Lazy initialization of OpenAI client."""
    global client
    if client is None:
        try:
            client = OpenAI()
            logging.info("OpenAI client initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI client: {e}")
            client = False
    return client if client is not False else None


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


class Entropy_Calculation_Classify_By_GPT4:
    """Three-stage grouping using an OpenAI chat model (default: o3)."""

    def __init__(self, solutions: List[str]):
        self.solutions = solutions
        self.stage1_cates = ""
        self.stage2_cates = ""
        self.categories = []
        self.model = "o3"

    def analyze_solutions_with_gpt4o(self, n_solutions: int) -> str:
        openai_client = get_openai_client()
        if openai_client is None:
            print("OpenAI client not available - cannot perform solution classification")
            self.categories = str([1] * n_solutions)
            return self.categories

        try:
            #===== Stage 1：grouping =====
            str_solutions = ""
            index = 1
            for solution in self.solutions[:n_solutions]:
                str_solutions += f"Solution{index}:" + solution + "\n"
                index += 1

            user_input = (
                "Here are several solutions to the same question:" + str_solutions
                + "Please analyze and determine how these solutions can be grouped based on the methods they use. "
                "Your classification criteria must remain strictly high-level. Place solutions in different categories only when their overarching strategies are completely distinct; differences limited to sub-steps or implementation details do not count as high-level distinctions."
                "Before you begin grouping, clearly state the classification criteria you will follow. "
                "In your response, focus on explaining your reasoning and clearly state which solution indices should be grouped together. "
                "Note that if all solutions use entirely different approaches, each should be placed in its own distinct group. "
                "In your grouping, each solution should be assigned to exactly one of the groups. Make sure to carefully check the total number of solutions"
            )
            messages = [{"role": "user", "content": user_input}]
            resp = retry_call(lambda: openai_client.chat.completions.create(model=self.model, messages=messages))
            self.categories = resp.choices[0].message.content
            self.stage1_cates = self.categories

            #===== Stage 2：extract {1: "Solution 1, Solution 2", ...} =====
            user_input = (
                "extract the category groups from the following text: " + self.categories
                + ' return the solution with categories like this format (for example, {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}), '
                'without any other text, and only use expressions like "Solution 1", "Solution 2"...to represent each solution, '
                "follow the example I give you. Make sure to carefully check the total number of solutions."
            )
            messages = [{"role": "user", "content": user_input}]
            resp = retry_call(lambda: openai_client.chat.completions.create(model=self.model, messages=messages))
            self.categories = resp.choices[0].message.content
            self.stage2_cates = self.categories

            #===== Stage 3： mapping [1,2,2,...] =====
            user_input = (
                "You will be given a mapping from category ids to the solution indices:\n"
                f"{self.categories}\n\n"
                "Task: Produce a list L of length N, where N is the number of solutions, and L[i] "
                "is the category id of the (i+1)-th solution. The order of elements in L must "
                "match the original order of the solutions. Use exactly the ids that appear in the mapping.\n\n"
                "Output format: a Python list of integers, e.g., [1, 3, 2, 2, 1]. "
                "Return ONLY the list, with no extra text.\n\n"
                "Example:\n"
                'Mapping: {1: "Solution 1, Solution 5", 2: "Solution 3, Solution 4", 3: "Solution 2"}\n'
                "Output: [1, 3, 2, 2, 1]"
            )
            messages = [{"role": "user", "content": user_input}]
            resp = retry_call(lambda: openai_client.chat.completions.create(model=self.model, messages=messages))
            self.categories = resp.choices[0].message.content
            return self.categories
        except Exception as e:
            print(f"Error in analyze_solutions_with_gpt4o: {e}")
            self.categories = str([1] * n_solutions)
            return self.categories


def get_categories(solutions: List[str], model: str = "o3") -> Tuple[str, str, str]:
    """
    Get category grouping for a list of solutions.

    Returns:
        Tuple of (categories_string, stage1_categories, stage2_categories)
    """
    test_client = get_openai_client()
    if test_client is None:
        print("Warning: OpenAI client not initialized. Cannot perform solution classification.")
        return str(list(range(1, len(solutions) + 1))), "", ""

    try:
        ent = Entropy_Calculation_Classify_By_GPT4(solutions)
        ent.model = model
        cats = ent.analyze_solutions_with_gpt4o(len(solutions))
        return cats, ent.stage1_cates, ent.stage2_cates
    except Exception as e:
        print(f"Error in get_categories: {e}")
        fallback_categories = str(list(range(1, len(solutions) + 1)))
        return fallback_categories, "", ""


def parse_category_list(category_string: str) -> List[int]:
    """Parse the category string returned by GPT-4 into a list of integers."""
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



