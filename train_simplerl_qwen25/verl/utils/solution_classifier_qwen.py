"""
Solution classification using GPT-4 for grouping similar mathematical solutions.
"""
import logging
import os
from typing import List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# model_name = 'Qwen/Qwen3-8B'
model_name = 'Qwen/Qwen2.5-72B-Instruct'
hf_token = os.environ.get("HF_TOKEN", "")
# model = AutoModelForCausalLM.from_pretrained(model_name, 
#                                              torch_dtype=torch.bfloat16,
#                                              trust_remote_code=True)
# tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# ===== vLLM switches (default to vLLM) =====
USE_VLLM = os.getenv("USE_VLLM", "1") == "1"
VLLM_BASE = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")

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

def get_qwen_response(model, tokenizer, messages, temp=0.7, top_p=0.9, max_new_tokens=512):
    if USE_VLLM:
        from openai import OpenAI
        client = OpenAI(base_url=VLLM_BASE, api_key="EMPTY")
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temp,
            top_p=top_p,
            max_tokens=max_new_tokens
        )
        return (resp.choices[0].message.content or "").strip()

    # ===== HF fallback path (enabled when USE_VLLM=0) =====
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs,
                             max_new_tokens=max_new_tokens,
                             temperature=temp,
                             do_sample=True,
                             top_p=top_p)
    result = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]).strip()
    if result.endswith('<|im_end|>'):
        result = result[:-10]
    if result.endswith('<|endoftext|>'):
        result = result[:-13]
    return result.strip()

class Entropy_Calculation_Classify_By_Qwen:
    def __init__(self, model_name, solutions):
        self.solutions = solutions
        self.stage1_cates = ""
        self.stage2_cates = ""
        self.categories = []
        if USE_VLLM:
            self.model = None
            self.tokenizer = None
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, 
                                                 torch_dtype=torch.bfloat16,
                                                 trust_remote_code=True)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    def analyze_solutions_with_qwen(self, n_solutions):
        # Get Qwen model at the start of the function
        try:
            str_solutions = ''
            index = 1
            for solution in self.solutions[:n_solutions]:
                str_solutions += f"Solution{index}:" + solution + "\n"
                # str_solutions += solution + "\n"
                index += 1
            #：grouping
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
            response = retry_call(lambda: get_qwen_response(self.model, self.tokenizer, messages))
            self.categories = response
            self.stage1_cates = self.categories

            #：extractformatgrouping
            user_input = (
                "TASK: Extract ONLY the grouping information from the analysis below. "
                "Do NOT provide any analysis, reasoning, or explanation.\n\n"
                "Analysis text:\n" + self.categories + "\n\n"
                "REQUIRED OUTPUT FORMAT: A Python dictionary mapping group numbers to solution lists.\n"
                "Example: {1: \"Solution 1, Solution 2\", 2: \"Solution 3, Solution 4\", 3: \"Solution 5\"}\n\n"
                "IMPORTANT:\n"
                "- Return ONLY the dictionary, no other text\n"
                "- Use exactly the format \"Solution X\" for each solution\n"
                "- Ensure all solutions are included exactly once\n"
                "- Double-check the total count matches the number of input solutions"
            )
            messages = [{"role": "user", "content": user_input}]
            response = retry_call(lambda: get_qwen_response(self.model, self.tokenizer, messages))
            
            #verifystage2outputwhetherdictionary format，extract
            # try:
            #     import re
            ## finddictionary formatcontent
            #     dict_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
            #     dict_match = re.search(dict_pattern, response)
            #     if dict_match:
            #         response = dict_match.group(0)
            #         print(f"Stage2 extracted dictionary: {response}")
            #     else:
            #         print(f"Warning: Stage2 output is not in dictionary format: {response[:100]}...")
            ## dictionary format，trygeneratedefaultgrouping
            #         response = '{1: "' + ', '.join([f"Solution {i+1}" for i in range(n_solutions)]) + '"}'
            # except Exception as e:
            #     print(f"Warning: Failed to validate stage2 output: {e}")
            #     response = '{1: "' + ', '.join([f"Solution {i+1}" for i in range(n_solutions)]) + '"}'
            
            self.categories = response
            self.stage2_cates = self.categories

            #：extract，format [1,1,2,2,3]
            # user_input = (
            #     "extract the categories from the following text: " + self.categories + 
            #     'return the solution with categories like this list (for example, [1,1,2,2,3]), without any other text. '
            #     'Note the number of elements in the list should be exactly the same as the number of solutions. '
            #     'Note that the order of the numbers in the list must match the original order in which the solutions were given; e.g., if the mapping is {1: "Solution 1, Solution 5", 2: "Solution 3, Solution 4", 3: "Solution 2"}, the corresponding list should be [1, 3, 2, 2, 1].Its important to note that the order of the numbers in the list must match the original order in which the solutions were given.'
            #     'return the categories like this list (for example, [1,3,2,2,1]), without any other text. '
            # )
            user_input = (
            "TASK: Convert the solution-to-category mapping into an ordered list.\n\n"
            f"Mapping: {self.categories}\n\n"
            "INSTRUCTIONS:\n"
            "1. Create a list L where L[i] = category_id of Solution (i+1)\n"
            "2. The list length must equal the total number of solutions\n"
            "3. List order must match the original solution order (Solution 1, Solution 2, ...)\n\n"
            "OUTPUT: YOUR RESPONSE SHOULD ONLY BE a list of integers. No other explanations.\n\n"
            "Example:\n"
            'Input: {1: "Solution 1, Solution 5", 2: "Solution 3, Solution 4", 3: "Solution 2"}\n'
            "YOUR RESPONSE: [1, 3, 2, 2, 1]"
            # "You will be given a mapping from category ids to the solution indices:\n"
            # f"{self.categories}\n\n"
            # "Task: Produce a list L of length N, where N is the number of solutions, and L[i] "
            # "is the category id of the (i+1)-th solution. The order of elements in L must "
            # "match the original order of the solutions. Use exactly the ids that appear in the mapping.\n\n"
            # "Output format: a Python list of integers, e.g., [1, 3, 2, 2, 1]. "
            # "Return ONLY the list, with no extra text.\n\n"
            # "Example:\n"
            # 'Mapping: {1: "Solution 1, Solution 5", 2: "Solution 3, Solution 4", 3: "Solution 2"}\n'
            # "Output: [1, 3, 2, 2, 1]"
        )
            messages = [{"role": "user", "content": user_input}]
            response = retry_call(lambda: get_qwen_response(self.model, self.tokenizer, messages))
            
            #verifystage3outputwhetherlist format，
            try:
                import re
                #findlist formatcontent
                list_pattern = r'\[[^\[\]]*\]'
                list_match = re.search(list_pattern, response)
                if list_match:
                    print(f"Stage3 validation passed: Found list format")
                else:
                    print(f"ERROR: Stage3 output is not in list format!")
                    print(f"Raw response: {response}")
                    
                #additional check：，verifycontentwhetherreasonable
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


def get_categories(solutions, model=model_name):
    """
    Get category grouping for a list of solutions.
    
    Args:
        solutions: List of solution strings
        model: Model to use for classification
        
    Returns:
        Tuple of (stage1_categories, stage2_categories, categories_string)
    """
    # Check if OpenAI client is available
    try:
        ent = Entropy_Calculation_Classify_By_Qwen(model, solutions)
        # ent.model = model
        cats = ent.analyze_solutions_with_qwen(len(solutions))
        return ent.stage1_cates, ent.stage2_cates, cats
    except Exception as e:
        print(f"Error in get_categories: {e}")
        # Return sequential categories as fallback
        fallback_categories = str(list(range(1, len(solutions) + 1)))
        return fallback_categories, "", ""


def parse_category_list(category_string: str) -> List[int]:
    """
    Parse the category string returned by GPT-4 into a list of integers.
    
    Args:
        category_string: String containing category assignments like "[1,1,2,2,3]"
        
    Returns:
        List of category indices
    """
    try:
        # Clean the string and evaluate it as a Python list
        cleaned = category_string.strip()
        if cleaned.startswith('[') and cleaned.endswith(']'):
            return eval(cleaned)
        else:
            # Try to extract numbers from the string
            import re
            numbers = re.findall(r'\d+', cleaned)
            return [int(x) for x in numbers]
    except Exception as e:
        logging.error(f"Failed to parse category string '{category_string}': {e}")
        return [] 

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
        stage1_cates, stage2_cates, cats = get_categories(solutions, m)
        results[m] = {                 # create dict first then assign
            "stage1_cates": stage1_cates,
            "stage2_cates": stage2_cates,
            "cats":         cats
        }
    return results

# readdata
import json
with open('/home/zhiyuan/yc/simpleRL-reason/verl/utils/trial_10.json', "r", encoding="utf-8") as f:
    problems = json.load(f)

# models = ['Qwen/Qwen2.5-72B-Instruct', 'Qwen/Qwen2-72B-Instruct']
models = ['Qwen/Qwen2.5-Math-72B-Instruct']
overall = {}
for idx, item in enumerate(problems, 1):
    key = f"Q{idx}"
    overall[key] = classify_solutions_by_models(item["solutions"], models)
    print(f"{key} done → {overall[key]}")

def clean_qwen_tokens_in_dict(obj):
    """Recursively clean Qwen model markers from dict"""
    if isinstance(obj, dict):
        return {k: clean_qwen_tokens_in_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_qwen_tokens_in_dict(item) for item in obj]
    elif isinstance(obj, str):
        #removeQwenmarker/tag
        obj = obj.replace('<|im_end|>', '').replace('<|endoftext|>', '').replace('<|im_start|>', '')
        return obj.strip()
    else:
        return obj

# cleanoutputdata
overall = clean_qwen_tokens_in_dict(overall)

output = '/home/zhiyuan/yc/simpleRL-reason/verl/utils/testing_yc.json'
with open(output, "w", encoding="utf-8") as f:
    json.dump(overall, f, ensure_ascii=False, indent=2)
print(f"\nall done，results saved to {output}")