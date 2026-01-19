# """
# Solution classification using GPT-4 for grouping similar mathematical solutions.
# """
# import logging
# import os
# from typing import List, Tuple
# import json
# from openai import OpenAI

# # ===== Qwen API config (DashScope OpenAI-compatible) =====
# BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
# API_KEY  = os.getenv("DASHSCOPE_API_KEY")
# if not API_KEY:
#     raise RuntimeError("Please set DASHSCOPE_API_KEY to your Alibaba Cloud Model Studio API key.")

# # Qwen client（fixed for Stage 1）
# client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# # （keep original variable name，but no longer using local model）
# model_name = 'qwen2.5-72b-instruct'  # if unavailable，can switch to 'qwen-plus' / 'qwen-max'
# hf_token = ''             # no longer used，placeholder onlyto avoid other logic changes

## ===== OpenAI GPT client（ Stage 2 & 3，lazy initialization）=====
# gpt_client = None
# GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "o3-mini")  # default too3-mini，cheaper

# def get_openai_client():
#     """Lazy initialization of OpenAI client (for GPT)."""
#     global gpt_client
#     if gpt_client is None:
#         try:
#             gpt_client = OpenAI()  # read OPENAI_API_KEY
#             logging.info("OpenAI client initialized successfully")
#         except Exception as e:
#             logging.warning(f"Failed to initialize OpenAI client: {e}")
#             gpt_client = False  # Use False to indicate failed initialization
#     return gpt_client if gpt_client is not False else None


# def retry_call(func, max_retries=3):
#     """Simple retry wrapper for API calls"""
#     for attempt in range(max_retries):
#         try:
#             return func()
#         except Exception as e:
#             if attempt == max_retries - 1:
#                 raise e
#             logging.warning(f"Attempt {attempt + 1} failed: {e}")
#     return None


# def get_qwen_response(model, tokenizer, messages, temp=0.7, top_p=0.9, max_new_tokens=512):
#"""API-only call；signature kept as-is（ tokenizer ），to ensure other logic unchanged（ Stage 1）"""
#     def _do():
## dashscope compatible interface Qwen ，sampling parameters（temperature/top_p）will throw unsupported_parameter
#         resp = client.chat.completions.create(
#             model=model,
#             messages=messages,
## DashScope max_completion_tokens， max_tokens
#             max_completion_tokens=max_new_tokens,
#             # don't pass temperature / top_p，use model default sampling
#         )
#         return (resp.choices[0].message.content or "").strip()
#     return retry_call(_do)


# def get_gpt_response(model, messages, temp=0.7, top_p=0.9, max_new_tokens=512):
#"""OpenAI GPT call（ Stage 2 / Stage 3）—— Responses API support o3"""
#     _client = get_openai_client()
#     if _client is None:
#         raise RuntimeError("OpenAI client not available - cannot perform GPT calls.")

#     def _do():
#         # o3 use/goes through Responses API，and don't pass temperature/top_p
#         kwargs = {
#             "model": model,
#"input": messages, # messages passed as-is，ensure prompt completely unchanged
#             "max_output_tokens": max_new_tokens,
#             # optional：reasoning effort（not required）
#             # "reasoning": {"effort": "medium"},
#         }

## if in the future you want to reuse this for o （such as gpt-4o），add sampling parameters as needed：
#         # if not model.lower().startswith("o"):
#         #     kwargs.update({"temperature": temp, "top_p": top_p})

#         resp = _client.responses.create(**kwargs)

## read：prefer convenience attribute，otherwise fallback to iteration
#         text = getattr(resp, "output_text", None)
#         if text is None:
#             text = "".join(
#                 (item.content[0].text if getattr(item, "content", None) else "")
#                 for item in getattr(resp, "output", []) or []
#                 if getattr(item, "type", "") == "output_text"
#             )
#         return (text or "").strip()

#     return retry_call(_do)


# class Entropy_Calculation_Classify_By_Qwen:
#     def __init__(self, model_name, solutions):
#         self.solutions = solutions
#         self.stage1_cates = ""
#         self.stage2_cates = ""
#         self.categories = []
#         # completely remove local/HFloading logic；to keep other code structure unchanged，still keep attribute
#         self.model = None
#         self.tokenizer = None
#         self.model_name = model_name

#     def analyze_solutions_with_qwen(self, n_solutions):
#         try:
#             # ===== consistent with original script：Solution{index}:（no space） =====
#             str_solutions = ''
#             index = 1
#             for solution in self.solutions[:n_solutions]:
#                 str_solutions += f"Solution{index}:" + solution + "\n"
#                 index += 1

#             # ===== Stage 1（Qwen）：completely restore your original prompt concatenation and content =====
#             user_input = (
#                 "Here are several solutions to the same question:" + str_solutions +
#                 "Please analyze and determine how these solutions can be grouped based on the methods they use. "
#                 "Your classification criteria must remain strictly high-level. Place solutions in different categories only when their overarching strategies are completely distinct; differences limited to sub-steps or implementation details do not count as high-level distinctions."
#                 "Before you begin grouping, clearly state the classification criteria you will follow. "
#                 "In your response, focus on explaining your reasoning and clearly state which solution indices should be grouped together. "
#                 "Note that if all solutions use entirely different approaches, each should be placed in its own distinct group. "
#                 "In your grouping, each solution should be assigned to exactly one of the groups. Make sure to carefully check the total number of solutions."
#                 "Here is an Example Answer: \nHigh-level method used\n\nGroup 1 – Pure trigonometric-identity manipulation \n• Solution 1 \n• Solution 2 \n• Solution 3 \n\nAll three start from the given tangent"
#                 " (or sine) values, work only with standard trig identities (addition, double-angle, Pythagorean) and arrive at β = π⁄2 – 2α without introducing additional geometric constructs.\n\nGroup"
#                 " 2 – Classical Euclidean geometry / Ptolemy in a cyclic quadrilateral \n• Solution 4 \n\nThis solution interprets the numbers as side lengths of right triangles, embeds them in a cyclic"
#                 " quadrilateral, applies Ptolemy’s theorem and angle chasing; no explicit trigonometric identities are used. \n\nGroup 3 – Complex-number (arg) technique \n• Solution 5 \n\nHere the"
#                 " vectors 4+3i and 24+7i are treated as complex numbers; their arguments are manipulated multiplicatively to extract a relation ship between angles, which is conceptually different from"
#                 " both the purely trigonometric and the purely synthetic-geometric approaches.\n\nThus every solution belongs to exactly one of three distinct groups:\n• Group 1: 1, 2, 3 \n• Group 2:"
#                 " 4 \n• Group 3: 5"
#                 ""
#             )
#             messages = [{"role": "user", "content": user_input}]
#             response = get_qwen_response(self.model_name, self.tokenizer, messages)

## —— only in Stage1 internal“in-place cleanup”，does not affect subsequent Stage2/3 prompt and logic ——
#             response = response.replace('<|im_end|>', '').replace('<|endoftext|>', '').replace('<|im_start|>', '').strip()

#             self.categories = response
#             self.stage1_cates = self.categories

#             # ===== Stage 2（GPT/o3）：keep original prompt unchanged =====
#             user_input = (
#                 "extract the category groups from the following text: " + self.categories + 
#                 ' return the solution with categories like this format (for example, {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}), '
#                 'without any other text, and only use expressions like "Solution 1", "Solution 2"...to represent each solution, '
#                 'follow the example I give you. Make sure to carefully check the total number of solutions.'
#             )
#             messages = [{"role": "user", "content": user_input}]
#             response = get_gpt_response(GPT_MODEL, messages)
#             self.categories = response
#             self.stage2_cates = self.categories

#             # ===== Stage 3（GPT/o3）：optimizepromptto reduceempty output =====
#             # originalprompt（kept for backup）：
#             # user_input = (
#             #     "You will be given a mapping from category ids to the solution indices:\n"
#             #     f"{self.categories}\n\n"
#             #     "Task: Produce a list L of length N, where N is the number of solutions, and L[i] "
#             #     "is the category id of the (i+1)-th solution. The order of elements in L must "
#             #     "match the original order of the solutions. Use exactly the ids that appear in the mapping.\n\n"
#             #     "Output format: a Python list of integers, e.g., [1, 3, 2, 2, 1]. "
#             #     "Return ONLY the list, with no extra text.\n\n"
#             #     "Example:\n"
#             #     'Mapping: {1: "Solution 1, Solution 5", 2: "Solution 3, Solution 4", 3: "Solution 2"}\n'
#             #     "Output: [1, 3, 2, 2, 1]"
#             # )
            
#             # new optimizedprompt：
#             user_input = (
#                 f"Convert this dictionary mapping to a list of {n_solutions} integers.\n\n"
#                 f"Input mapping: {self.categories}\n\n"
#                 f"Task: Create a list where position i contains the category number of Solution (i+1).\n"
#                 f"- List must have exactly {n_solutions} elements\n"
#                 f"- Use only the category numbers that appear in the mapping\n"
#                 f"- Order matters: [category_of_solution_1, category_of_solution_2, ...]\n\n"
#                 "Format: Return only the Python list, no explanation.\n\n"
#                 "Example:\n"
#                 'Input: {1: "Solution 1, Solution 5", 2: "Solution 3, Solution 4", 3: "Solution 2"}\n'
#                 "Output: [1, 3, 2, 2, 1]"
#             )
#             messages = [{"role": "user", "content": user_input}]
#             response = get_gpt_response(GPT_MODEL, messages)

#             # ===== validation print messages kept consistent =====
#             try:
#                 import re
#                 list_pattern = r'\[[^\[\]]*\]'
#                 list_match = re.search(list_pattern, response)
#                 if list_match:
#                     print(f"Stage3 validation passed: Found list format")
#                 else:
#                     print(f"ERROR: Stage3 output is not in list format!")
#                     print(f"Raw response: {response}")

#                 if list_match:
#                     try:
#                         parsed_list = eval(list_match.group(0))
#                         if len(parsed_list) != n_solutions:
#                             print(f"ERROR: List length {len(parsed_list)} does not match expected {n_solutions}")
#                         elif not all(isinstance(x, int) for x in parsed_list):
#                             print(f"ERROR: List contains non-integer elements: {parsed_list}")
#                         else:
#                             print(f"Stage3 content validation passed: {parsed_list}")
#                     except Exception as parse_error:
#                         print(f"ERROR: Failed to parse list content: {parse_error}")
#             except Exception as e:
#                 print(f"ERROR: Failed to validate stage3 output: {e}")
#                 print(f"Raw response: {response}")

#             self.categories = response
#             return self.categories
#         except Exception as e:
#             print(f"Error in analyze_solutions_with_qwen: {e}")
#             self.categories = str([1] * n_solutions)
#             return self.categories


# def get_categories(solutions, model=model_name) -> Tuple[str, str, str]:
#     """
#     Get category grouping for a list of solutions.

#     Returns:
#         Tuple of (categories_string, stage1_categories, stage2_categories)
#     """
#     try:
#         ent = Entropy_Calculation_Classify_By_Qwen(model, solutions)
#         cats = ent.analyze_solutions_with_qwen(len(solutions))
#         return cats, ent.stage1_cates, ent.stage2_cates
#     except Exception as e:
#         print(f"Error in get_categories: {e}")
#         fallback_categories = str(list(range(1, len(solutions) + 1)))
#         return fallback_categories, "", ""


# def parse_category_list(category_string: str) -> List[int]:
#     """Parse the category string returned by GPT-4 into a list of integers."""
#     try:
#         cleaned = category_string.strip()
#         if cleaned.startswith('[') and cleaned.endswith(']'):
#             return eval(cleaned)
#         else:
#             import re
#             numbers = re.findall(r'\d+', cleaned)
#             return [int(x) for x in numbers]
#     except Exception as e:
#         logging.error(f"Failed to parse category string '{category_string}': {e}")
#         return []


# def classify_solutions_by_models(solutions, models):
#     """
#     returnformat：
#     {
#         "o1":      {"stage1_cates": "...", "stage2_cates": "...", "cats": "..."},
#         "o3":      {...},
#         "gpt-4o":  {...}
#     }
#     """
#     results = {}
#     for m in models:
#         cats, stage1_cates, stage2_cates = get_categories(solutions, m)
#         results[m] = {
#             "stage1_cates": stage1_cates,
#             "stage2_cates": stage2_cates,
#             "cats":         cats
#         }
#     return results


## ===== readdata =====
# with open('/home/zhiyuan/yc/simpleRL-reason/verl/utils/trial_10.json', "r", encoding="utf-8") as f:
#     problems = json.load(f)

## note：the model name here must be“API available”，such as 'qwen2.5-72b-instruct' / 'qwen-plus'
# models = ['qwen2.5-72b-instruct']

# overall = {}
# for idx, item in enumerate(problems, 1):
#     key = f"Q{idx}"
#     overall[key] = classify_solutions_by_models(item["solutions"], models)
#     print(f"{key} done → {overall[key]}")

# output = '/home/zhiyuan/yc/simpleRL-reason/verl/utils/testing_yc_222.json'
# with open(output, "w", encoding="utf-8") as f:
#     json.dump(overall, f, ensure_ascii=False, indent=2)
# print(f"\nall done，results saved to {output}")
"""
Solution classification using GPT-4 for grouping similar mathematical solutions.
"""
import logging
import os
from typing import List, Tuple
import json
from openai import OpenAI

# ===== Qwen API config (DashScope OpenAI-compatible) =====
BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")
API_KEY  = os.getenv("DASHSCOPE_API_KEY")
if not API_KEY:
    raise RuntimeError("Please set DASHSCOPE_API_KEY to your Alibaba Cloud Model Studio API key.")

# Qwen client（fixed for Stage 1）
client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

# （keep original variable name，but no longer using local model）
model_name = 'qwen2.5-72b-instruct'  # if unavailable, can switch to 'qwen-plus' / 'qwen-max'
hf_token = ''             # no longer used, placeholder to avoid other logic changes

#===== OpenAI GPT client（ Stage 2 & 3，lazy initialization）=====
gpt_client = None
GPT_MODEL = os.getenv("OPENAI_GPT_MODEL", "o3-mini")  # default to o3-mini, cheaper

def get_openai_client():
    """Lazy initialization of OpenAI client (for GPT)."""
    global gpt_client
    if gpt_client is None:
        try:
            gpt_client = OpenAI()  # read OPENAI_API_KEY
            logging.info("OpenAI client initialized successfully")
        except Exception as e:
            logging.warning(f"Failed to initialize OpenAI client: {e}")
            gpt_client = False  # Use False to indicate failed initialization
    return gpt_client if gpt_client is not False else None


def retry_call(func, max_retries=3):
    """Simple retry wrapper for API calls"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            logging.warning(f"Attempt {attempt + 1} failed: {e}")
    return None


# def get_qwen_response(model, tokenizer, messages, temp=0.7, top_p=0.9, max_new_tokens=1024):
#"""API-only call；signature kept as-is（ tokenizer ），to ensure other logic unchanged（ Stage 1）"""
#     def _do():
## dashscope compatible interface Qwen ，sampling parameters（temperature/top_p）will throw unsupported_parameter
#         resp = client.chat.completions.create(
#             model=model,
#             messages=messages,
## DashScope max_completion_tokens， max_tokens
#             max_completion_tokens=max_new_tokens,
#             # don't pass temperature / top_p，use model default sampling
#         )
#         text = (resp.choices[0].message.content or "").strip()
#         if not text:
## empty output → retry
#             raise RuntimeError("Qwen empty output")
#         return text
#     return retry_call(_do)
def get_qwen_response(model, tokenizer, messages, temp=0.7, top_p=0.9, max_new_tokens=4096):
    """API-only call; signature kept as-is (including tokenizer) to ensure other logic unchanged (for Stage 1)"""
    def _do():
        base = os.getenv("QWEN_BASE_URL", "")
        use_dashscope = ("dashscope" in base)

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


#===== Responses API /output（ prompt）=====

def _messages_to_responses_input(messages):
    """
    Adapt Chat-style messages to Responses API:
    - If only 1 user message and content is plain string => return that string directly
    - Otherwise construct with input_text content blocks
    """
    try:
        if (
            isinstance(messages, list) and
            len(messages) == 1 and
            isinstance(messages[0], dict) and
            messages[0].get("role") == "user" and
            isinstance(messages[0].get("content"), str)
        ):
            return messages[0]["content"]
    except Exception:
        pass

    converted = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        converted.append({
            "role": role,
            "content": [{"type": "input_text", "text": str(content)}]
        })
    return converted


def _extract_text_from_responses(resp) -> str:
    """
    Robust extraction of Responses API text output:
    - Prefer resp.output_text
    - Otherwise iterate resp.output[*].content[*].text
    """
    text = getattr(resp, "output_text", None)
    if text:
        return text.strip()

    try:
        output = getattr(resp, "output", None) or []
        buf = []
        for item in output:
            content = getattr(item, "content", None) or []
            for c in content:
                t = getattr(c, "text", None)
                if t:
                    buf.append(t)
        joined = "\n".join(buf).strip()
        if joined:
            return joined
    except Exception:
        pass
    return ""


def get_gpt_response(model, messages, temp=0.7, top_p=0.9, max_new_tokens=2048):
    """OpenAI GPT call（for Stage 2 / Stage 3）——用 Responses API supports o3；empty输出则retry"""
    _client = get_openai_client()
    if _client is None:
        raise RuntimeError("OpenAI client not available - cannot perform GPT calls.")

    def _do():
        # o3 use/goes through Responses API，and don't pass temperature/top_p
        safe_input = _messages_to_responses_input(messages)
        resp = _client.responses.create(
            model=model,
            input=safe_input,                 # 你的 messages 原样转换post-传入，保证 prompt 完全不变
            max_output_tokens=max_new_tokens,
        )
        text = _extract_text_from_responses(resp)
        if not text:
            # will“empty output”treat as failure，throw exception to trigger retry
            try:
                logging.warning("Empty output from Responses API; raw: %s",
                                resp.model_dump_json()[:2000])
            except Exception:
                pass
            raise RuntimeError("Empty output from Responses API")
        return text

    return retry_call(_do)


class Entropy_Calculation_Classify_By_Qwen:
    def __init__(self, model_name, solutions):
        self.solutions = solutions
        self.stage1_cates = ""
        self.stage2_cates = ""
        self.categories = []
        # completely remove local/HFloading logic；to keep other code structure unchanged，still keep attribute
        self.model = None
        self.tokenizer = None
        self.model_name = model_name

    def analyze_solutions_with_qwen(self, n_solutions):
        try:
            # ===== consistent with original script：Solution{index}:（no space） =====
            str_solutions = ''
            index = 1
            for solution in self.solutions[:n_solutions]:
                str_solutions += f"Solution{index}:" + solution + "\n"
                index += 1

            # ===== Stage 1（Qwen）：completely restore your original prompt concatenation and content =====
            user_input = (
                "Here are several solutions to the same question:" + str_solutions +
                "Please analyze and determine how these solutions can be grouped based on the methods they use. "
                "Your classification criteria must remain strictly high-level. Place solutions in different categories only when their overarching strategies are completely distinct; differences limited to sub-steps or implementation details do not count as high-level distinctions."
                "Before you begin grouping, clearly state the classification criteria you will follow. "
                "In your response, focus on explaining your reasoning and clearly state which solution indices should be grouped together. "
                "Note that if all solutions use entirely different approaches, each should be placed in its own distinct group. "
                "In your grouping, each solution should be assigned to exactly one of the groups. Make sure to carefully check the total number of solutions."
                "Here is an Example Answer: \nHigh-level method used\n\nGroup 1 – Pure trigonometric-identity manipulation \n• Solution 1 \n• Solution 2 \n• Solution 3 \n\nAll three start from the given tangent"
                " (or sine) values, work only with standard trig identities (addition, double-angle, Pythagorean) and arrive at β = π⁄2 – 2α without introducing additional geometric constructs.\n\nGroup"
                " 2 – Classical Euclidean geometry / Ptolemy in a cyclic quadrilateral \n• Solution 4 \n\nThis solution interprets the numbers as side lengths of right triangles, embeds them in a cyclic"
                " quadrilateral, applies Ptolemy’s theorem and angle chasing; no explicit trigonometric identities are used. \n\nGroup 3 – Complex-number (arg) technique \n• Solution 5 \n\nHere the"
                " vectors 4+3i and 24+7i are treated as complex numbers; their arguments are manipulated multiplicatively to extract a relation ship between angles, which is conceptually different from"
                " both the purely trigonometric and the purely synthetic-geometric approaches.\n\nThus every solution belongs to exactly one of three distinct groups:\n• Group 1: 1, 2, 3 \n• Group 2:"
                " 4 \n• Group 3: 5"
                ""
            )
            messages = [{"role": "user", "content": user_input}]
            response = get_qwen_response(self.model_name, self.tokenizer, messages)

            #—— only in Stage1 internal“in-place cleanup”，does not affect subsequent Stage2/3 prompt and logic ——
            response = response.replace('<|im_end|>', '').replace('<|endoftext|>', '').replace('<|im_start|>', '').strip()

            self.categories = response
            self.stage1_cates = self.categories

            # ===== Stage 2（GPT/o3）：keep original prompt unchanged =====
            user_input = (
                "extract the category groups from the following text: " + self.categories + 
                ' return the solution with categories like this format (for example, {1: "Solution 1, Solution 2", 2: "Solution 3, Solution 4", 3: "Solution 5"}), '
                'without any other text, and only use expressions like "Solution 1", "Solution 2"...to represent each solution, '
                'follow the example I give you. Make sure to carefully check the total number of solutions.'
            )
            messages = [{"role": "user", "content": user_input}]
            response = get_gpt_response(GPT_MODEL, messages)
            self.categories = response
            self.stage2_cates = self.categories

            #===== Stage 3（GPT/o3）：optimizepromptto reduceempty output（unchanged）=====
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

            # ===== validation print messages kept consistent =====
            try:
                import re
                list_pattern = r'\[[^\[\]]*\]'
                list_match = re.search(list_pattern, response)
                if list_match:
                    print(f"Stage3 validation passed: Found list format")
                else:
                    print(f"ERROR: Stage3 output is not in list format!")
                    print(f"Raw response: {response}")

                if list_match:
                    try:
                        parsed_list = eval(list_match.group(0))
                        if len(parsed_list) != n_solutions:
                            print(f"ERROR: List length {len(parsed_list)} does not match expected {n_solutions}")
                        elif not all(isinstance(x, int) for x in parsed_list):
                            print(f"ERROR: List contains non-integer elements: {parsed_list}")
                        else:
                            print(f"Stage3 content validation passed: {parsed_list}")
                    except Exception as parse_error:
                        print(f"ERROR: Failed to parse list content: {parse_error}")
            except Exception as e:
                print(f"ERROR: Failed to validate stage3 output: {e}")
                print(f"Raw response: {response}")

            self.categories = response
            return self.categories
        except Exception as e:
            print(f"Error in analyze_solutions_with_qwen: {e}")
            self.categories = str([1] * n_solutions)
            return self.categories


def get_categories(solutions, model=model_name) -> Tuple[str, str, str]:
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
        fallback_categories = str([1] * len(solutions))  # all-1 list
        return fallback_categories, "", ""


def parse_category_list(category_string: str) -> List[int]:
    """Parse the category string returned by GPT-4 into a list of integers."""
    try:
        cleaned = category_string.strip()
        if cleaned.startswith('[') and cleaned.endswith(']'):
            return eval(cleaned)
        else:
            import re
            numbers = re.findall(r'\d+', cleaned)
            return [int(x) for x in numbers]
    except Exception as e:
        logging.error(f"Failed to parse category string '{category_string}': {e}")
        return [1]  # return单个1，训练器会checklength不match并使用fallback


def classify_solutions_by_models(solutions, models):
    """
    Return format:
    {
        "o1":      {"stage1_cates": "...", "stage2_cates": "...", "cats": "..."},
        "o3":      {...},
        "gpt-4o":  {...}
    }
    """
    results = {}
    for m in models:
        cats, stage1_cates, stage2_cates = get_categories(solutions, m)
        results[m] = {
            "stage1_cates": stage1_cates,
            "stage2_cates": stage2_cates,
            "cats":         cats
        }
    return results


## ===== readdata =====
# with open('/home/zhiyuan/yc/simpleRL-reason/verl/utils/trial_10.json', "r", encoding="utf-8") as f:
#     problems = json.load(f)

## note：the model name here must be“API available”，such as 'qwen2.5-72b-instruct' / 'qwen-plus'
# models = ['qwen2.5-72b-instruct']

# overall = {}
# for idx, item in enumerate(problems, 1):
#     key = f"Q{idx}"
#     overall[key] = classify_solutions_by_models(item["solutions"], models)
#     print(f"{key} done → {overall[key]}")

# output = '/home/zhiyuan/yc/simpleRL-reason/verl/utils/testing_yc_444.json'
# with open(output, "w", encoding="utf-8") as f:
#     json.dump(overall, f, ensure_ascii=False, indent=2)
# print(f"\nall done，results saved to {output}")
