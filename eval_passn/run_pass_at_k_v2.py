import os
import argparse
import time
import json
import subprocess
import requests
import signal
import sys
import atexit
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer
from math_verify import parse, verify
from huggingface_hub import snapshot_download

#─── OpenAI SDK（optional， LLM Judge）────────────────────────────────────────
try:
    import openai as _openai_sdk
except ImportError:
    _openai_sdk = None

#─── clean ─────────────────────────────────────────────────
_running_ports = []  # track currently running vLLM ports
_model_name_for_cleanup = None  # current model name for cleanup

#─── ────────────────────────────────────────────────────────────────
#default（ --model ）
model_name = "Echoandland/qwen3-8b-dapo-high-entropy-step8"
num_runs = 100  # number of runs, i.e., maximum k value

#data：(dataset_name, split, key_for_problem, key_for_solution)
datasets = [
    ("zhiyuanhucs/aime24_25", "train", "problem", "solution"),
]

#System prompt ( build_prompt)
SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

# Qwen3 default to thinking mode，keep original prompt format
prompt_templates = {
    "qwen": (
        "<|im_start|>system\n"
        "Please reason step by step, and put your final answer within \\boxed{{}}."
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "{problem}"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
}
prompt_style = "qwen"

def is_olmo_model(model_name: str) -> bool:
    """detect if this is an Olmo model"""
    model_lower = model_name.lower()
    return "olmo" in model_lower

def build_prompt(tokenizer, problem: str) -> str:
    """
    使用官方 apply_chat_template 构建 prompt
    适for OLMo 等需要使用官方 chat template 的模型
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": problem},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

# (translated comment)
gpu_per_model = 1               # GPU count per model
base_port = 8024               # starting port number
num_workers = 32               # concurrent request thread count（reduce concurrency for single GPU to avoid timeout）

# HTTP request timeout & retries (vLLM can take long for long generations)
request_timeout_s = 900
request_retries = 2
request_retry_backoff_s = 2

# (translated comment)
temperature = 1
max_new_tokens = 20480

# (translated comment)
cache_dir = "cache_pass_at_k"
results_dir = "results_pass_at_k"
detailed_results_dir = "detailed_results_pass_at_k"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(detailed_results_dir, exist_ok=True)

#─── LLM Judge ─────────────────────────────────────────────────────

def _extract_boxed_answer(text: str) -> str:
    """从模型输出中提取 \\boxed{...} 内的答案"""
    #\boxed{...}，support
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()  # return last boxed answer
    return ""


def _build_llm_judge_messages(predicted_answer: str, gold_answer: str, full_response: str = "") -> list:
    """构建通用数学/科学问答的 LLM Judge prompt"""
    system = (
        "You are a strict grader for mathematical and scientific answers. "
        "Return EXACTLY one lowercase character: 'y' if the predicted answer is equivalent to the correct answer, 'n' otherwise. "
        "No spaces, no punctuation, no explanations. If uncertain, answer 'n'."
    )
    user = f"""TASK
Judge whether the predicted answer is mathematically/semantically equivalent to the correct answer.

SCORING POLICY — answer 'y' (EQUIVALENT) if ANY apply:
• Exact match (after normalization of whitespace/formatting)
• Mathematically equivalent expressions: x^2 + 2x + 1 ≡ (x+1)^2; 1/2 ≡ 0.5; π/4 ≡ 45°
• Equivalent symbolic representations: Z ⊕ Z ≡ Z+Z; ℤ₂ ≡ Z/2Z
• Same numerical value with different notation: 2.5 ≡ 5/2 ≡ 2½
• Equivalent set/group notation: {{1,2,3}} ≡ {{3,2,1}}
• Case-insensitive for text answers

Answer 'n' (NOT equivalent) if ANY apply:
• Different numerical values
• Different mathematical objects/structures
• Missing key components
• Partially correct but incomplete

Predicted answer: {predicted_answer}
Correct answer: {gold_answer}

Output exactly one character: y or n.
"""
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _yn_to_bool(text: str) -> bool:
    """convert LLM's y/n response to bool"""
    t = (text or "").strip().lower()
    if t.startswith("y"):
        return True
    return False


def llm_judge_single(
    response_text: str,
    gold_answer: str,
    *,
    api_key: str = None,
    model: str = "gpt-4o-mini",
    base_url: str = None,
    max_tokens: int = 16,
    temperature: float = 0.0,
) -> bool:
    """
    use LLM to judge if prediction is correct
    return True/False
    """
    if _openai_sdk is None:
        print("[LLM Judge] OpenAI SDK not installed; cannot run LLM judge.")
        return False
    
    #extract boxed
    predicted = _extract_boxed_answer(response_text)
    if not predicted:
        #boxed，trycontent
        lines = [ln.strip() for ln in response_text.strip().splitlines() if ln.strip()]
        predicted = lines[-1] if lines else response_text[:500]
    
    messages = _build_llm_judge_messages(predicted, gold_answer, response_text)
    
    try:
        client = _openai_sdk.OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        text = resp.choices[0].message.content if resp.choices else ""
        return _yn_to_bool(text)
    except Exception as e:
        print(f"[LLM Judge] API call failed: {type(e).__name__}: {e}")
        return False


def llm_judge_batch(
    tasks: list,  # list of (response_text, gold_answer)
    *,
    api_key: str = None,
    model: str = "gpt-4o-mini",
    base_url: str = None,
    max_workers: int = 16,
) -> list:
    """
    批量 LLM Judge，return list of bool
    """
    results = [False] * len(tasks)
    
    def _judge_one(idx, response_text, gold_answer):
        return idx, llm_judge_single(
            response_text, gold_answer,
            api_key=api_key, model=model, base_url=base_url
        )
    
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {
            exe.submit(_judge_one, i, t[0], t[1]): i
            for i, t in enumerate(tasks)
        }
        for fut in tqdm(as_completed(futures), total=len(tasks), desc="LLM Judge"):
            try:
                idx, is_correct = fut.result()
                results[idx] = is_correct
            except Exception as e:
                print(f"[LLM Judge] Task failed: {e}")
    
    return results


#─── ────────────────────────────────────────────────────────────────

def _load_any_dataset(ds_name, split):
    """Load either a HF Hub dataset (by name) or a local file (json/jsonl/csv/parquet)."""
    if os.path.isfile(ds_name):
        ext = os.path.splitext(ds_name)[1].lower()
        if ext in [".jsonl", ".json"]:
            return load_dataset("json", data_files={"train": ds_name}, split="train")
        if ext == ".csv":
            return load_dataset("csv", data_files={"train": ds_name}, split="train")
        if ext == ".parquet":
            return load_dataset("parquet", data_files={"train": ds_name}, split="train")
        raise ValueError(f"Unsupported local dataset file extension: {ext} ({ds_name})")
    return load_dataset(ds_name, split=split)


def _make_dataset_tag(ds_name, split):
    """Create a short, filesystem-safe tag for filenames and summaries."""
    try:
        if os.path.isfile(ds_name):
            base = os.path.splitext(os.path.basename(ds_name))[0]
        else:
            base = ds_name.split("/")[-1]
    except Exception:
        base = str(ds_name).replace("/", "_")
    return f"{base}_{split}"


def _normalize_dataset_value(v):
    """Normalize dataset fields to a plain string (handles list/dict values)."""
    if v is None:
        return ""
    # Common case for some datasets: answers are stored as a list (e.g. final_answer=["42"])
    if isinstance(v, (list, tuple)):
        if len(v) == 0:
            return ""
        if len(v) == 1:
            return _normalize_dataset_value(v[0])
        return "\n".join(_normalize_dataset_value(x) for x in v)
    if isinstance(v, dict):
        try:
            return json.dumps(v, ensure_ascii=False)
        except Exception:
            return str(v)
    return str(v)

def start_model_server(model_name, ports=None, gpu_memory_utilization=0.9):
    """Launch vLLM server for a given model and wait until it's ready."""
    script = "run_Ngpus.sh"
    if ports is None:
        ports = [base_port + i for i in range(gpu_per_model)]
        ports_csv = ",".join(str(p) for p in ports)
    else:
        assert len(ports) == gpu_per_model, f"len(ports) ({len(ports)}) must equal gpu_per_model ({gpu_per_model})"
        ports_csv = ",".join(str(p) for p in ports)

    cmd = ["bash", script, model_name, str(gpu_per_model), ports_csv, str(gpu_memory_utilization)]
    subprocess.run(cmd, check=True)
    
    def _wait_port(host: str, port: int, timeout_s: int = 600):
        import socket
        deadline = time.time() + timeout_s
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"vLLM port not ready: {host}:{port} within {timeout_s}s")
            try:
                with socket.create_connection((host, port), timeout=2):
                    return
            except OSError:
                time.sleep(2)

    for port in ports:
        _wait_port("127.0.0.1", int(port), timeout_s=600)
    time.sleep(30)  # waitserver完全pre-热（torch.compile等）
    return ports


def _load_tokenizer_safely(model_path: str, model_name_hint: str | None = None) -> AutoTokenizer:
    """Load tokenizer from the *actual* model path whenever possible."""
    last_err = None
    for cand in [model_path, model_name_hint]:
        if not cand:
            continue
        try:
            return AutoTokenizer.from_pretrained(cand, trust_remote_code=True)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load tokenizer from model_path={model_path!r} (hint={model_name_hint!r}). "
        f"Last error: {type(last_err).__name__}: {last_err}"
    )


def stop_model_server():
    """Deprecated: kept for backward compatibility."""
    # vLLM has multiple entrypoints; we may use the OpenAI-compatible server.
    subprocess.run('pkill -f "python -m vllm.entrypoints.api_server"', shell=True)
    subprocess.run('pkill -f "python -m vllm.entrypoints.openai.api_server"', shell=True)


def emergency_cleanup():
    """emergency cleanup function, called on abnormal program exit"""
    global _running_ports, _model_name_for_cleanup
    if _running_ports:
        print(f"\n=== Emergency cleanup: shutting down vLLM servers (Port: {_running_ports}) ===")
        stop_model_server_safe(_running_ports, _model_name_for_cleanup)
        _running_ports.clear()


def signal_handler(signum, frame):
    """signal handler for Ctrl+C and other interrupt signals"""
    print(f"\n=== Received signal {signum}，cleaning up and exiting ===")
    emergency_cleanup()
    print("Cleanup complete, program exiting")
    sys.exit(1)


def stop_model_server_safe(ports=None, model_name=None):
    """Stop vLLM server processes on specific ports, or all if no ports specified."""
    if ports is None:
        print("Warning: will terminate all vLLM service processes")
        subprocess.run('pkill -f "python -m vllm.entrypoints.api_server"', shell=True)
        subprocess.run('pkill -f "python -m vllm.entrypoints.openai.api_server"', shell=True)
        return

    def _kill_vllm_by_port(port: int, model_name_filter: str | None = None):
        try:
            ps = subprocess.run(
                ["ps", "-eo", "pid,args"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as e:
            print(f"Unable to list processes (ps) to clean port {port}: {e}")
            return

        pids = []
        needle_port = f"--port {port}"
        for line in ps.stdout.splitlines():
            if ("vllm.entrypoints.api_server" not in line) and ("vllm.entrypoints.openai.api_server" not in line):
                continue
            if needle_port not in line:
                continue
            if model_name_filter:
                if (model_name_filter not in line) and (model_name_filter.replace("/", "_") not in line):
                    continue
            try:
                pid = int(line.strip().split(None, 1)[0])
                pids.append(pid)
            except Exception:
                continue

        if not pids:
            print(f"Port {port} no vLLM api_server process found")
            return
        for pid in pids:
            print(f"Terminating port {port} process PID: {pid}")
            subprocess.run(["kill", str(pid)], check=False)

    print(f"Terminating port {ports} vLLM service")
    for port in ports:
        try:
            _kill_vllm_by_port(int(port), model_name_filter=model_name)
        except Exception as e:
            print(f"Terminating port {port} error terminating process on: {e}")

    time.sleep(2)

    for port in ports:
        import socket
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
                print(f"Warning: port {port} still has processes running")
        except OSError:
            print(f"Port {port} successfully released")


def list_vllm_servers():
    """List all currently running vLLM servers."""
    print("=== Currently running vLLM servers ===")
    try:
        # Match both legacy api_server and OpenAI-compatible openai.api_server
        result = subprocess.run('pgrep -f "python -m vllm.entrypoints.*api_server"', 
                              shell=True, capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if not pid:
                    continue
                ps_result = subprocess.run(f"ps -p {pid} -o pid,args --no-headers", 
                                         shell=True, capture_output=True, text=True)
                if ps_result.returncode == 0:
                    process_info = ps_result.stdout.strip()
                    print(f"PID {pid}: {process_info}")
                    cmd_line = ps_result.stdout.strip()
                    if "--port" in cmd_line:
                        parts = cmd_line.split("--port")
                        if len(parts) > 1:
                            port_part = parts[1].split()[0]
                            print(f"  -> Port: {port_part}")
        else:
            print("No running vLLM servers found")
    except Exception as e:
        print(f"Error listing vLLM servers: {e}")
    print("=" * 40)


def query_prompts(prompts, ports, eos_ids, temp=None, top_p=None, model_name="default"):
    """Send prompts to the running vLLM OpenAI-compatible server(s) and collect outputs.
    
    Uses /v1/completions endpoint (OpenAI compatible format).
    """
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    #temp， temperature
    actual_temp = temp if temp is not None else temperature

    def _query_one(prompt, port):
        #OpenAI /v1/completions
        payload = {
            "model": model_name,
            "prompt": prompt,
            "n": 1,
            "temperature": actual_temp,
            "max_tokens": max_new_tokens,
            "stop_token_ids": eos_ids,
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        last_err = None
        for attempt in range(request_retries + 1):
            try:
                resp = requests.post(
                    f"http://localhost:{port}/v1/completions",
                    json=payload,
                    timeout=request_timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                #OpenAI formatresponse
                return data["choices"][0]["text"]
            except Exception as e:
                last_err = e
                if attempt < request_retries:
                    time.sleep(request_retry_backoff_s * (attempt + 1))
                    continue
                raise last_err

    generated = [""] * len(prompts)
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        future_to_idx = {
            exe.submit(_query_one, prompts[i], ports[i % len(ports)]): i
            for i in range(len(prompts))
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                generated[idx] = fut.result()
            except Exception as e:
                print(f"[ERROR] prompt #{idx}: {e}")
    return generated


def query_prompts_chat(problems, ports, model_name="default", temp=0.6, top_p=0.95, stop_token_ids=None):
    """Send problems to vLLM using /v1/chat/completions endpoint (recommended for OLMo3-Instruct).
    
    This follows the official vLLM recommendation for OLMo3-Instruct:
    - Use /v1/chat/completions with messages array (not /v1/completions with formatted prompt)
    - Use temperature=0.6, top_p=0.95 (Ai2 recommended values)
    
    Args:
        problems: List of problem strings (user content)
        ports: List of vLLM server ports
        model_name: Model name for the API
        temp: Temperature (default 0.6, Ai2 recommended)
        top_p: Top-p sampling (default 0.95, Ai2 recommended)
    
    Returns:
        List of generated responses
    """
    if isinstance(stop_token_ids, int):
        stop_token_ids = [stop_token_ids]

    def _query_one_chat(problem, port):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": problem},
        ]
        payload = {
            "model": model_name,
            "messages": messages,
            "temperature": temp,
            "top_p": top_p,
            "max_tokens": max_new_tokens,
        }
        # vLLM OpenAI server often supports stop_token_ids for both /v1/completions and /v1/chat/completions.
        # For OLMo3, relying on EOS generation alone can lead to extremely long, repetitive outputs.
        if stop_token_ids:
            payload["stop_token_ids"] = stop_token_ids
        last_err = None
        for attempt in range(request_retries + 1):
            try:
                resp = requests.post(
                    f"http://localhost:{port}/v1/chat/completions",
                    json=payload,
                    timeout=request_timeout_s,
                )
                # Compatibility fallback: some vLLM versions may reject unknown fields on chat endpoint.
                if resp.status_code == 400 and ("stop_token_ids" in payload):
                    try:
                        body = resp.text or ""
                    except Exception:
                        body = ""
                    # Retry once without stop_token_ids if the server complains.
                    if ("stop_token_ids" in body) or ("extra fields" in body.lower()) or ("unknown" in body.lower()):
                        payload2 = dict(payload)
                        payload2.pop("stop_token_ids", None)
                        resp = requests.post(
                            f"http://localhost:{port}/v1/chat/completions",
                            json=payload2,
                            timeout=request_timeout_s,
                        )
                resp.raise_for_status()
                data = resp.json()
                #Chat completions formatresponse
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_err = e
                if attempt < request_retries:
                    time.sleep(request_retry_backoff_s * (attempt + 1))
                    continue
                raise last_err

    generated = [""] * len(problems)
    with ThreadPoolExecutor(max_workers=num_workers) as exe:
        future_to_idx = {
            exe.submit(_query_one_chat, problems[i], ports[i % len(ports)]): i
            for i in range(len(problems))
        }
        for fut in as_completed(future_to_idx):
            idx = future_to_idx[fut]
            try:
                generated[idx] = fut.result()
            except Exception as e:
                print(f"[ERROR] problem #{idx}: {e}")
    return generated


def load_detailed_results(detailed_file):
    """Load detailed results file, return completed run count and results"""
    if not os.path.exists(detailed_file):
        return 0, []
    
    with open(detailed_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    completed_runs = data.get('completed_runs', 0)
    results = data.get('results', [])
    return completed_runs, results


def save_detailed_results(detailed_file, completed_runs, results, model_name, ds_name, split, total_problems):
    """save detailed results to file"""
    os.makedirs(os.path.dirname(detailed_file), exist_ok=True)
    
    data = {
        'model': model_name,
        'dataset': f"{ds_name}/{split}",
        'total_problems': total_problems,
        'completed_runs': completed_runs,
        'target_runs': num_runs,
        'results': results
    }
    
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calculate_pass_at_k(detailed_results, k):
    """calculate based on detailed results pass@k accuracy"""
    if not detailed_results:
        return {"correct": 0, "total": 0, "accuracy": 0.0}
    
    total = len(detailed_results)
    correct = 0
    
    for problem_result in detailed_results:
        runs_to_check = min(k, len(problem_result['runs']))
        if any(problem_result['runs'][i]['is_correct'] for i in range(runs_to_check)):
            correct += 1
    
    accuracy = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }


#─── ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(emergency_cleanup)
    
    parser = argparse.ArgumentParser(description="pass@k runner with vLLM")
    parser.add_argument("--model", type=str, default=None, help="Model name or path (overrides config)")
    parser.add_argument("--ports", type=str, default=None, help="Comma-separated explicit ports, length must equal gpu_per_model")
    parser.add_argument("--port", type=int, default=None, help="Single explicit port (only valid when gpu_per_model == 1)")
    parser.add_argument("--gpu-per-model", type=int, default=None, help="Override gpu_per_model")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9, help="GPU memory utilization for vLLM (default: 0.9)")
    parser.add_argument("--base-port", type=int, default=None, help="Override base_port when no explicit ports provided")
    parser.add_argument("--no-check-existing", action="store_true", help="Do not check existing detailed/final results; run inference fresh")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation; compute pass@k only from existing is_correct flags")
    parser.add_argument("--force-validate", action="store_true", help="Force re-validation: overwrite existing is_correct flags (useful when switching to LLM judge).")
    parser.add_argument("--filename-ts", action="store_true", help="Include run timestamp in cache and detailed result filenames")
    parser.add_argument("--resume-ts", type=str, default=None, help="Resume from a previous run by specifying its timestamp (e.g., 20251227_142113)")
    parser.add_argument("--detailed-file", type=str, default=None, help="Path to an existing detailed results JSON file")
    #（）
    parser.add_argument("--run-offset", type=int, default=0, help="Global run index offset for this job (default: 0). Use with --num-runs for parallel chunks.")
    parser.add_argument("--job-id", type=str, default=None, help="Job identifier to make detailed/results filenames unique when running in parallel.")
    #LLM Judge
    parser.add_argument("--use-llm-judge", action="store_true", help="Use LLM (e.g., GPT-4o-mini) for answer verification instead of math_verify")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="LLM model for judging (default: gpt-4o-mini)")
    parser.add_argument("--judge-api-key", type=str, default=None, help="OpenAI API key for LLM judge (or set OPENAI_API_KEY env)")
    parser.add_argument("--judge-base-url", type=str, default=None, help="Base URL for LLM judge API (for compatible APIs)")
    parser.add_argument("--judge-workers", type=int, default=16, help="Number of parallel workers for LLM judge (default: 16)")
    #data
    parser.add_argument("--dataset-path", type=str, default=None, help="Override dataset path (local file or HF hub name)")
    parser.add_argument("--dataset-split", type=str, default="train", help="Dataset split (default: train)")
    parser.add_argument("--dataset-problem-key", type=str, default="problem", help="Column name for problem (default: problem)")
    parser.add_argument("--dataset-solution-key", type=str, default="solution", help="Column name for solution (default: solution)")
    parser.add_argument("--max-problems", type=int, default=None, help="Limit number of problems for a quick smoke test (e.g., 20)")
    parser.add_argument("--problem-offset", type=int, default=0, help="Start index when using --max-problems (default: 0)")
    parser.add_argument(
        "--olmo-endpoint",
        type=str,
        default="chat",
        choices=["chat", "completions"],
        help="For OLMo models only: choose vLLM endpoint. 'chat' uses /v1/chat/completions; 'completions' uses /v1/completions with apply_chat_template prompt.",
    )
    parser.add_argument("--num-runs", type=int, default=None, help="Override number of runs (default: 100)")
    args, _ = parser.parse_known_args()

    # Apply overrides
    if args.model is not None:
        model_name = args.model
    if args.gpu_per_model is not None:
        gpu_per_model = args.gpu_per_model
    if args.base_port is not None:
        base_port = args.base_port
    if args.num_runs is not None:
        num_runs = args.num_runs

    run_offset = int(args.run_offset or 0)
    job_id = args.job_id
    job_id_safe = None
    if job_id:
        job_id_safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(job_id)).strip("_")
        if not job_id_safe:
            job_id_safe = None

    #（detailed/results）
    _suffix_parts = []
    if job_id_safe:
        _suffix_parts.append(f"job{job_id_safe}")
    if run_offset:
        _suffix_parts.append(f"offset{run_offset}")
    file_suffix = ("_" + "_".join(_suffix_parts)) if _suffix_parts else ""
    
    #data
    if args.dataset_path is not None:
        datasets = [(args.dataset_path, args.dataset_split, args.dataset_problem_key, args.dataset_solution_key)]
    
    #LLM Judge
    use_llm_judge = args.use_llm_judge
    force_validate = bool(args.force_validate)
    judge_kwargs = {
        "api_key": args.judge_api_key or os.environ.get("OPENAI_API_KEY"),
        "model": args.judge_model,
        "base_url": args.judge_base_url,
        "max_workers": args.judge_workers,
    }
    if use_llm_judge and not judge_kwargs["api_key"]:
        print("[WARNING] --use-llm-judge enabled but no API key provided. Set --judge-api-key or OPENAI_API_KEY env.")

    #support
    if args.resume_ts:
        run_ts = args.resume_ts
        print(f"[info] Resuming from timestamp: {run_ts}")
    else:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
    explicit_ports = None
    if args.ports:
        explicit_ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
        assert len(explicit_ports) == gpu_per_model, f"--ports count ({len(explicit_ports)}) must equal gpu_per_model ({gpu_per_model})"
    elif args.port is not None:
        if gpu_per_model != 1:
            raise ValueError("--port is only allowed when gpu_per_model == 1; use --ports for multiple GPUs")
        explicit_ports = [args.port]

    #verify
    if args.detailed_file is not None:
        print("=== Using specified detailed results file, proceeding directly to validation and statistics ===")
        detailed_file = args.detailed_file
        if not os.path.exists(detailed_file):
            raise FileNotFoundError(f"详细resultsfilenot exists: {detailed_file}")

        with open(detailed_file, 'r', encoding='utf-8') as f:
            _data = json.load(f)

        file_model = _data.get('model', model_name)
        dataset_str = _data.get('dataset', 'unknown/unknown')
        try:
            ds_name, split = dataset_str.rsplit('/', 1)
        except ValueError:
            ds_name, split = dataset_str, 'unknown'

        detailed_results = _data.get('results', [])
        completed_runs = _data.get('completed_runs', 0)
        total_problems = _data.get('total_problems', len(detailed_results))

        do_validate = not args.skip_validate
        if do_validate:
            try:
                if use_llm_judge:
                    print(f"正在使用 LLM Judge ({args.judge_model}) verifyprediction result...")
                    #verify
                    tasks_to_judge = []
                    task_indices = []  # (problem_idx, run_idx)
                    for p_idx, problem_result in enumerate(detailed_results):
                        gold_sol_str = problem_result.get('gold_solution', '')
                        for r_idx, run_result in enumerate(problem_result.get('runs', [])):
                            if force_validate or run_result.get('is_correct') is None:
                                tasks_to_judge.append((run_result.get('raw_output', ''), gold_sol_str))
                                task_indices.append((p_idx, r_idx))
                    
                    if tasks_to_judge:
                        print(f"共 {len(tasks_to_judge)} 条需要verify")
                        results = llm_judge_batch(
                            tasks_to_judge,
                            api_key=judge_kwargs["api_key"],
                            model=judge_kwargs["model"],
                            base_url=judge_kwargs["base_url"],
                            max_workers=judge_kwargs["max_workers"],
                        )
                        for (p_idx, r_idx), is_correct in zip(task_indices, results):
                            detailed_results[p_idx]['runs'][r_idx]['is_correct'] = is_correct
                            detailed_results[p_idx]['runs'][r_idx]['judge_method'] = 'llm'
                else:
                    print("正在使用 math_verify verifyprediction result...")
                    validated_count = 0
                    with tqdm(detailed_results, desc="validation progress") as pbar:
                        for problem_result in pbar:
                            gold_sol_str = problem_result.get('gold_solution', '')
                            try:
                                gold_parsed = parse(gold_sol_str)
                            except Exception:
                                gold_parsed = []
                            for run_result in problem_result.get('runs', []):
                                if force_validate or run_result.get('is_correct') is None:
                                    try:
                                        pred_parsed = parse(run_result.get('raw_output', ''))
                                        run_result['is_correct'] = verify(pred_parsed, gold_parsed)
                                        run_result['judge_method'] = 'math_verify'
                                    except Exception as e:
                                        print(f"validation error: {e}")
                                        run_result['is_correct'] = False
                            validated_count += 1
                            #verify 100 save
                            if validated_count % 100 == 0:
                                save_detailed_results(detailed_file, completed_runs, detailed_results, file_model, ds_name, split, total_problems)
            except Exception as e:
                print(f"[WARNING] validation phase出错: {type(e).__name__}: {e}")
                print("[WARNING] current state saved，can retry validation later")
        else:
            print("跳过validation phase（--skip-validate）。将基于已有 is_correct 字段calculate pass@k。")

        save_detailed_results(detailed_file, completed_runs, detailed_results, file_model, ds_name, split, total_problems)

        pass_at_k_results = {}
        print("calculating pass@k results...")
        with tqdm(range(1, completed_runs + 1), desc="calculatepass@k") as pbar:
            for k in pbar:
                result = calculate_pass_at_k(detailed_results, k)
                pass_at_k_results[f"pass@{k}"] = result
                if k in [1, 5, 10, 20, 50, 100, 200] or k == completed_runs or k % 50 == 0:
                    pbar.set_postfix({"pass@{k}".format(k=k): f"{result['accuracy']:.4f}"})

        print("key pass@k results:")
        key_k_values = [1, 5, 10, 20, 50, 100, 200, completed_runs]
        key_k_values = [k for k in key_k_values if k <= completed_runs]
        key_k_values = sorted(set(key_k_values))
        for k in key_k_values:
            result = pass_at_k_results[f"pass@{k}"]
            print(f"  pass@{k}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")

        length_stats = {}
        for run_idx in range(completed_runs):
            lengths = [problem['runs'][run_idx]['response_length']
                       for problem in detailed_results
                       if run_idx < len(problem.get('runs', []))]
            if lengths:
                length_stats[f"run_{run_idx}"] = {
                    "avg": sum(lengths) / len(lengths),
                    "min": min(lengths),
                    "max": max(lengths)
                }

        final_summary = {}
        final_summary[f"{ds_name}_{split}"] = {
            "model": file_model,
            "completed_runs": completed_runs,
            "target_runs": _data.get('target_runs', completed_runs),
            "total_problems": total_problems,
            "pass_at_k": pass_at_k_results,
            "length_stats": length_stats,
            "detailed_results_file": detailed_file,
        }

        dataset_name_tag = _make_dataset_tag(ds_name, split)
        out_file = os.path.join(results_dir, f"pass_at_k_{run_ts}_{file_model.replace('/', '_')}_{dataset_name_tag}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

        print(f"\n=== complete！results已保存到 {out_file} ===")
        print(f"详细results保exists {detailed_file} 中")
        print(f"本次runcontains的数据集: {list(final_summary.keys())}")
        raise SystemExit(0)

    #vLLM
    list_vllm_servers()

    #data
    all_data = {}
    for ds_name, split, prob_key, sol_key in datasets:
        ds_tag = _make_dataset_tag(ds_name, split)
        print(f"Loading dataset: {ds_name}/{split} -> tag: {ds_tag}")
        ds = _load_any_dataset(ds_name, split)
        problems = [_normalize_dataset_value(d[prob_key]) for d in ds]
        gold_solutions_text = [_normalize_dataset_value(d[sol_key]) for d in ds]
        if args.max_problems is not None:
            off = int(args.problem_offset or 0)
            n = int(args.max_problems)
            if off < 0:
                off = 0
            if n < 0:
                n = 0
            problems = problems[off:off + n]
            gold_solutions_text = gold_solutions_text[off:off + n]
            print(f"[info] limiting problems: offset={off}, max_problems={n}, actual={len(problems)}")
        all_data[(ds_name, split)] = {"problems": problems, "gold_solutions_text": gold_solutions_text}

    check_existing = (not args.no_check_existing)
    need_inference = True if not check_existing else False
    if check_existing:
        for (ds_name, split), data in all_data.items():
            ds_tag = _make_dataset_tag(ds_name, split)
            tag_prefix = f"{run_ts}_" if args.filename_ts else ""
            detailed_file = os.path.join(detailed_results_dir, f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}{file_suffix}_detailed.json")
            completed_runs, _ = load_detailed_results(detailed_file)
            if completed_runs < num_runs:
                need_inference = True
                break

    try:
        #whether Olmo ， build_prompt
        use_build_prompt = is_olmo_model(model_name)
        if use_build_prompt:
            print(f"[info] 检测到 Olmo 模型，将使用 tokenizer.apply_chat_template() 构建 prompt")
        
        if need_inference:
            print(f"=== start vLLM server for {model_name} ===")
            _root = model_name if os.path.isdir(model_name) else snapshot_download(model_name)
            _actor_hf = os.path.join(_root, "actor", "huggingface")
            _actor     = os.path.join(_root, "actor")
            if os.path.isdir(_actor_hf):
                _model_path = _actor_hf
            elif os.path.isdir(_actor):
                _model_path = _actor
            else:
                _model_path = _root
            tokenizer = _load_tokenizer_safely(_model_path, model_name)
            _gen_config_path = os.path.join(_model_path, "generation_config.json")
            if os.path.exists(_gen_config_path):
                with open(_gen_config_path, "r") as f:
                    _gen_config = json.load(f)
                eos_ids = _gen_config.get("eos_token_id", tokenizer.eos_token_id)
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
            else:
                eos_ids = [tokenizer.eos_token_id]
            print(f"[info] eos_token_ids: {eos_ids}")
            ports = start_model_server(_model_path, ports=explicit_ports, gpu_memory_utilization=args.gpu_memory_utilization)
            print(f"[info] using model path: {_model_path}")

            _running_ports = ports.copy() if ports else []
            _model_name_for_cleanup = model_name
        else:
            print("=== all inference completed，skipping inference phase ===\n")
            tokenizer = None
            eos_ids = None
            ports = None

        for (ds_name, split), data in all_data.items():
            ds_tag = _make_dataset_tag(ds_name, split)
            print(f"=== handle数据集: {ds_name}/{split} (tag: {ds_tag}) ===")
            
            tag_prefix = f"{run_ts}_" if args.filename_ts else ""
            detailed_file = os.path.join(detailed_results_dir, f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}{file_suffix}_detailed.json")
            if check_existing:
                completed_runs, detailed_results = load_detailed_results(detailed_file)
            else:
                completed_runs, detailed_results = 0, []
            
            problems = data["problems"]
            gold_solutions_text = data["gold_solutions_text"]
            
            if not detailed_results:
                detailed_results = []
                for i, (problem, gold_sol_text) in enumerate(zip(problems, gold_solutions_text)):
                    detailed_results.append({
                        'problem_id': i,
                        'problem': problem,
                        'gold_solution': str(gold_sol_text),
                        'runs': []
                    })
            
            #OLMo temperature
            model_temperature = temperature
            if use_build_prompt:
                #OLMo will /v1/chat/completions， query_prompts_chat
                if args.olmo_endpoint == "chat":
                    print(f"[info] OLMo 模型将使用 /v1/chat/completions (temp=0.6, top_p=0.95)")
                else:
                    print(f"[info] OLMo 模型将使用 /v1/completions + apply_chat_template prompt (temp=0.6, top_p=0.95)")
            
            if completed_runs < num_runs and need_inference:
                start_global = run_offset + completed_runs
                end_global = run_offset + num_runs - 1
                print(f"继续从第 {completed_runs + 1} 次runstart... (global run {start_global}..{end_global})")
                
                with tqdm(range(completed_runs, num_runs), desc=f"推理进度 {ds_name}/{split}") as pbar:
                    for run_idx in pbar:
                        global_run_id = run_offset + run_idx
                        pbar.set_description(f"推理进度 {ds_name}/{split} (local {run_idx+1}/{num_runs}, global run {global_run_id})")
                        
                        tag_prefix = f"{run_ts}_" if args.filename_ts else ""
                        cache_file = os.path.join(cache_dir, f"{tag_prefix}{model_name.replace('/', '_')}_run{global_run_id}_{ds_tag}.jsonl")
                        
                        if check_existing and os.path.exists(cache_file):
                            gen_texts = []
                            with open(cache_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    s = line.strip()
                                    if not s:
                                        continue
                                    #： JSONL，（output）
                                    if s.lstrip().startswith("{"):
                                        try:
                                            obj = json.loads(s)
                                            gen_texts.append(obj.get("response") or obj.get("raw_output") or "")
                                        except Exception:
                                            gen_texts.append(s)
                                    else:
                                        gen_texts.append(s)
                            if len(gen_texts) != len(problems):
                                print(f"[WARNING] cache 行数不match：{cache_file} got {len(gen_texts)} lines, expected {len(problems)}. 将ignore cache 并re-推理该 run。")
                                gen_texts = None
                        else:
                            gen_texts = None

                        if gen_texts is None:
                            # (translated comment)
                            if use_build_prompt and tokenizer is not None:
                                #OLMo ：optional /v1/chat/completions /v1/completions
                                #sampling parameters（temp=0.6, top_p=0.95）， stop
                                if args.olmo_endpoint == "chat":
                                    print(f"[info] OLMo 模型使用 /v1/chat/completions, temp=0.6, top_p=0.95")
                                    gen_texts = query_prompts_chat(
                                        problems,
                                        ports,
                                        model_name=_model_path,
                                        temp=0.6,
                                        top_p=0.95,
                                        stop_token_ids=eos_ids,
                                    )
                                    #generate prompt（/）
                                    prompts = [build_prompt(tokenizer, p) for p in problems]
                                else:
                                    print(f"[info] OLMo 模型使用 /v1/completions, temp=0.6, top_p=0.95, stop_token_ids={eos_ids}")
                                    prompts = [build_prompt(tokenizer, p) for p in problems]
                                    gen_texts = query_prompts(
                                        prompts,
                                        ports,
                                        eos_ids,
                                        temp=0.6,
                                        top_p=0.95,
                                        model_name=_model_path,
                                    )
                            else:
                                #： /v1/completions
                                prompts = [prompt_templates[prompt_style].format(problem=p) for p in problems]
                                gen_texts = query_prompts(prompts, ports, eos_ids, temp=model_temperature, model_name=_model_path)
                            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                            with open(cache_file, "w", encoding="utf-8") as f:
                                for p, r in zip(prompts, gen_texts):
                                    f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
                        
                        for i, gen_text in enumerate(gen_texts):
                            #prompt
                            if use_build_prompt and tokenizer is not None:
                                prompt_for_result = build_prompt(tokenizer, problems[i])
                            else:
                                prompt_for_result = prompt_templates[prompt_style].format(problem=problems[i])
                            run_result = {
                                'run_id': global_run_id,
                                'prompt': prompt_for_result,
                                'raw_output': gen_text,
                                'response_length': len(gen_text),
                                'is_correct': None
                            }
                            detailed_results[i]['runs'].append(run_result)
                        
                        completed_runs += 1
                        
                        if (run_idx + 1) % 10 == 0:
                            save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                                model_name, ds_name, split, len(problems))
                
                save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                    model_name, ds_name, split, len(problems))
                print(f"推理complete，共complete {completed_runs} 次run")

    except KeyboardInterrupt:
        print(f"\n=== program interrupted by user (Ctrl+C) ===")
        #save
        for (ds_name, split), data in all_data.items():
            ds_tag = _make_dataset_tag(ds_name, split)
            tag_prefix = f"{run_ts}_" if args.filename_ts else ""
            #note： file_suffix， job detailed
            detailed_file = os.path.join(
                detailed_results_dir,
                f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}{file_suffix}_detailed.json",
            )
            if 'detailed_results' in dir() and detailed_results:
                print(f"保存interrupt前的推理results到: {detailed_file}")
                save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                    model_name, ds_name, split, len(data["problems"]))
        emergency_cleanup()
        print("Cleanup complete, program exiting")
        sys.exit(1)
    except Exception as e:
        print(f"\n=== program exception occurred ===")
        print(f"errortype: {type(e).__name__}")
        print(f"error message: {str(e)}")
        #save
        for (ds_name, split), data in all_data.items():
            ds_tag = _make_dataset_tag(ds_name, split)
            tag_prefix = f"{run_ts}_" if args.filename_ts else ""
            #note： file_suffix， job detailed
            detailed_file = os.path.join(
                detailed_results_dir,
                f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}{file_suffix}_detailed.json",
            )
            if 'detailed_results' in dir() and detailed_results:
                print(f"保存exception前的推理results到: {detailed_file}")
                save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                    model_name, ds_name, split, len(data["problems"]))
        emergency_cleanup()
        print("Cleanup complete, program exiting")
        sys.exit(1)
    finally:
        if need_inference:
            print("=== shutting down vLLM server ===")
            ports_to_stop = [base_port + i for i in range(gpu_per_model)]
            stop_model_server_safe(ports_to_stop, model_name=None)
            _running_ports.clear()

    print("\n=== starting calculation pass@k results（validation phase）===")
    do_validate = not args.skip_validate
    
    out_file = os.path.join(results_dir, f"final_pass_at_k_results_{model_name.replace('/', '_')}{file_suffix}.json")
    if check_existing and os.path.exists(out_file):
        print(f"发现已有resultsfile，将append新的数据集results: {out_file}")
        with open(out_file, 'r', encoding='utf-8') as f:
            final_summary = json.load(f)
    else:
        prefix = "create新的resultsfile" if check_existing else "跳过read已有results，create新的resultsfile"
        print(f"{prefix}: {out_file}")
        final_summary = {}

    for (ds_name, split), data in all_data.items():
        ds_tag = _make_dataset_tag(ds_name, split)
        print(f"\ncalculateresults - 数据集: {ds_name}/{split} (tag: {ds_tag})")
        
        tag_prefix = f"{run_ts}_" if args.filename_ts else ""
        detailed_file = os.path.join(detailed_results_dir, f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}{file_suffix}_detailed.json")
        completed_runs, detailed_results = load_detailed_results(detailed_file)
        
        #verifysave（verify）
        save_detailed_results(detailed_file, completed_runs, detailed_results, 
                            model_name, ds_name, split, len(data["problems"]))
        print(f"[checkpoint] 推理results已保存到: {detailed_file}")
        
        if do_validate:
            try:
                if use_llm_judge:
                    print(f"正在使用 LLM Judge ({args.judge_model}) verifyprediction result...")
                    #verify
                    tasks_to_judge = []
                    task_indices = []  # (problem_idx, run_idx)
                    for p_idx, problem_result in enumerate(detailed_results):
                        gold_sol_str = problem_result.get('gold_solution', '')
                        for r_idx, run_result in enumerate(problem_result.get('runs', [])):
                            if force_validate or run_result.get('is_correct') is None:
                                tasks_to_judge.append((run_result.get('raw_output', ''), gold_sol_str))
                                task_indices.append((p_idx, r_idx))
                    
                    if tasks_to_judge:
                        print(f"共 {len(tasks_to_judge)} 条需要verify")
                        results = llm_judge_batch(
                            tasks_to_judge,
                            api_key=judge_kwargs["api_key"],
                            model=judge_kwargs["model"],
                            base_url=judge_kwargs["base_url"],
                            max_workers=judge_kwargs["max_workers"],
                        )
                        for (p_idx, r_idx), is_correct in zip(task_indices, results):
                            detailed_results[p_idx]['runs'][r_idx]['is_correct'] = is_correct
                            detailed_results[p_idx]['runs'][r_idx]['judge_method'] = 'llm'
                        #verifysave
                        save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                            model_name, ds_name, split, len(data["problems"]))
                        print(f"[checkpoint] validation results saved")
                else:
                    print("正在使用 math_verify verifyprediction result...")
                    validated_count = 0
                    with tqdm(detailed_results, desc="validation progress") as pbar:
                        for problem_result in pbar:
                            gold_sol_str = problem_result['gold_solution']
                            try:
                                gold_parsed = parse(gold_sol_str)
                            except Exception:
                                gold_parsed = []
                            for run_result in problem_result['runs']:
                                if force_validate or run_result['is_correct'] is None:
                                    try:
                                        pred_parsed = parse(run_result['raw_output'])
                                        run_result['is_correct'] = verify(pred_parsed, gold_parsed)
                                        run_result['judge_method'] = 'math_verify'
                                    except Exception as e:
                                        print(f"validation error: {e}")
                                        run_result['is_correct'] = False
                            validated_count += 1
                            #verify 100 save
                            if validated_count % 100 == 0:
                                save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                                    model_name, ds_name, split, len(data["problems"]))
                    #verifysave
                    save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                        model_name, ds_name, split, len(data["problems"]))
                    print(f"[checkpoint] validation results saved")
            except Exception as e:
                print(f"[WARNING] validation phase出错: {type(e).__name__}: {e}")
                print("[WARNING] 推理results已保存，可稍post-使用 --skip-validate 跳过verify或使用 --use-llm-judge 切换verify方式")
                #verifysave
                save_detailed_results(detailed_file, completed_runs, detailed_results, 
                                    model_name, ds_name, split, len(data["problems"]))
        else:
            print("跳过validation phase（--skip-validate）。将基于已有 is_correct 字段calculate pass@k。")
        
        pass_at_k_results = {}
        
        print(f"calculating pass@k results...")
        with tqdm(range(1, completed_runs + 1), desc="calculatepass@k") as pbar:
            for k in pbar:
                result = calculate_pass_at_k(detailed_results, k)
                pass_at_k_results[f"pass@{k}"] = result
                
                if k in [1, 5, 10, 20, 50, 100, 200] or k == completed_runs or k % 50 == 0:
                    pbar.set_postfix({"pass@{k}".format(k=k): f"{result['accuracy']:.4f}"})
        
        print("key pass@k results:")
        key_k_values = [1, 5, 10, 20, 50, 100, 200, completed_runs]
        key_k_values = [k for k in key_k_values if k <= completed_runs]
        key_k_values = sorted(set(key_k_values))
        
        for k in key_k_values:
            result = pass_at_k_results[f"pass@{k}"]
            print(f"  pass@{k}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}")
        
        length_stats = {}
        for run_idx in range(completed_runs):
            lengths = [detailed_results[i]['runs'][run_idx]['response_length'] 
                      for i in range(len(detailed_results)) 
                      if run_idx < len(detailed_results[i]['runs'])]
            if lengths:
                length_stats[f"run_{run_idx}"] = {
                    "avg": sum(lengths) / len(lengths),
                    "min": min(lengths),
                    "max": max(lengths)
                }
        
        final_summary[f"{ds_name}_{split}"] = {
            "model": model_name,
            "completed_runs": completed_runs,
            "target_runs": num_runs,
            "total_problems": len(data["problems"]),
            "pass_at_k": pass_at_k_results,
            "length_stats": length_stats,
            "detailed_results_file": detailed_file,
        }

        dataset_names = "_".join([_make_dataset_tag(ds_name, split) for ds_name, split, _, _ in datasets])
        out_file = os.path.join(results_dir, f"pass_at_k_{run_ts}_{model_name.replace('/', '_')}_{dataset_names}{file_suffix}.json")
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== complete！results已保存到 {out_file} ===")
    print(f"详细results保exists {detailed_results_dir} directory中")
    print(f"本次runcontains的数据集: {list(final_summary.keys())}")

