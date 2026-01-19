
import os
import argparse
import time
import json
import subprocess
import requests
import signal
import sys
import atexit
import glob
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Use OlympiadBench MathJudger for validation (local copy under eval_passn_)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "OlympiadBench", "inference", "code")))
try:
    from math_judger import MathJudger
except Exception as _e:
    MathJudger = None

#─── clean ─────────────────────────────────────────────────
_running_ports = []  # track currently running vLLM ports
_model_name_for_cleanup = None  # current model name for cleanup

#─── ────────────────────────────────────────────────────────────────
model_name = "Echoandland/olmo3-7b-grpo-purerl-creativity-step28"
num_runs = 100  # number of runs, i.e., maximum k value

#OlympiadBench : question / final_answer (list[str]) / error (str|null)
datasets = [
    ("data/OE_TO_physics_en_COMP.jsonl", "train", "question", "final_answer", "error")
]

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

# (translated comment)
gpu_per_model = 1               # GPU count per model
base_port = 8024                # starting port number
# NOTE: 1024 threads is too aggressive for a single vLLM server; it increases timeouts.
num_workers = 64                # concurrent request thread count (inference phase)

# HTTP request timeout & retries (vLLM can take >120s for long generations)
request_timeout_s = 900
request_retries = 2
request_retry_backoff_s = 2

# (translated comment)
temperature = 1
#Qwen3 thinking mode: 20480
max_new_tokens = 20480

# (translated comment)
cache_dir = "cache_pass_at_k"
results_dir = "results_pass_at_k"
detailed_results_dir = "detailed_results_pass_at_k"
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)
os.makedirs(detailed_results_dir, exist_ok=True)

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

def start_model_server(model_name, ports=None):
    """Launch vLLM server for a given model and wait until it's ready."""
    script = "run_Ngpus.sh"  # use the same script uniformly
    if ports is None:
        ports = [base_port + i for i in range(gpu_per_model)]
        # IMPORTANT: pass explicit ports to run_Ngpus.sh so that --base-port is respected
        ports_csv = ",".join(str(p) for p in ports)
    else:
        assert len(ports) == gpu_per_model, f"len(ports) ({len(ports)}) must equal gpu_per_model ({gpu_per_model})"
        ports_csv = ",".join(str(p) for p in ports)

    cmd = ["bash", script, model_name, str(gpu_per_model)]
    cmd.append(ports_csv)
    subprocess.run(cmd, check=True)
    #（ lsof；）
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
    time.sleep(5)
    return ports


def _load_tokenizer_safely(model_path: str, model_name_hint: str | None = None) -> AutoTokenizer:
    """
    Load tokenizer from the *actual* model path whenever possible.
    This keeps eos/special tokens consistent with the served model (important for Qwen3).
    """
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
    subprocess.run('pkill -f "python -m vllm.entrypoints.api_server"', shell=True)

def emergency_cleanup():
    """emergency cleanup function, called on abnormal program exit"""
    global _running_ports, _model_name_for_cleanup
    if _running_ports:
        print(f"\n=== Emergency cleanup: shutting down vLLM servers (Port: {_running_ports}) ===", flush=True)
        stop_model_server_safe(_running_ports, _model_name_for_cleanup)
        _running_ports.clear()

def signal_handler(signum, frame):
    """signal handler for Ctrl+C and other interrupt signals"""
    print(f"\n=== Received signal {signum}，cleaning up and exiting ===", flush=True)
    emergency_cleanup()
    print("Cleanup complete, program exiting", flush=True)
    sys.exit(1)

def stop_model_server_safe(ports=None, model_name=None):
    """Stop vLLM server processes on specific ports, or all if no ports specified."""
    if ports is None:
        print("Warning: will terminate all vLLM service processes", flush=True)
        subprocess.run('pkill -f "python -m vllm.entrypoints.api_server"', shell=True)
        return

    def _kill_vllm_by_port(port: int, model_name_filter: str | None = None):
        """Kill vLLM api_server processes that were launched with a given --port.
        Do NOT rely on lsof (may be missing in minimal containers).
        """
        try:
            ps = subprocess.run(
                ["ps", "-eo", "pid,args"],
                capture_output=True,
                text=True,
                check=True,
            )
        except Exception as e:
            print(f"Unable to list processes (ps) to clean port {port}: {e}", flush=True)
            return

        pids = []
        needle_port = f"--port {port}"
        for line in ps.stdout.splitlines():
            if "vllm.entrypoints.api_server" not in line:
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
            print(f"Port {port} no vLLM api_server process found", flush=True)
            return
        for pid in pids:
            print(f"Terminating port {port} process PID: {pid}", flush=True)
            subprocess.run(["kill", str(pid)], check=False)

    print(f"Terminating port {ports} vLLM service", flush=True)
    for port in ports:
        try:
            _kill_vllm_by_port(int(port), model_name_filter=model_name)
        except Exception as e:
            print(f"Terminating port {port} error terminating process on: {e}", flush=True)

    time.sleep(2)

    for port in ports:
        import socket
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1):
                print(f"Warning: port {port} still has processes running", flush=True)
        except OSError:
            print(f"Port {port} successfully released", flush=True)

def list_vllm_servers():
    """List all currently running vLLM servers."""
    print("=== Currently running vLLM servers ===", flush=True)
    try:
        result = subprocess.run('pgrep -f "python -m vllm.entrypoints.api_server"', 
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
                    print(f"PID {pid}: {process_info}", flush=True)
                    cmd_line = ps_result.stdout.strip()
                    if "--port" in cmd_line:
                        parts = cmd_line.split("--port")
                        if len(parts) > 1:
                            port_part = parts[1].split()[0]
                            print(f"  -> Port: {port_part}", flush=True)
        else:
            print("No running vLLM servers found", flush=True)
    except Exception as e:
        print(f"Error listing vLLM servers: {e}", flush=True)
    print("=" * 40, flush=True)

def query_prompts(prompts, ports, eos_ids):
    """Send prompts to the running vLLM server(s) and collect outputs.
    
    Args:
        eos_ids: int or list of ints for stop token ids
    """
    #eos_ids
    if isinstance(eos_ids, int):
        eos_ids = [eos_ids]
    default_args = {
        "n": 1,
        "temperature": temperature,
        "max_tokens": max_new_tokens,
        "stop_token_ids": eos_ids,
    }

    def _query_one(prompt, port):
        payload = {**default_args, "prompt": prompt}
        last_err = None
        for attempt in range(request_retries + 1):
            try:
                resp = requests.post(
                    f"http://localhost:{port}/generate",
                    json=payload,
                    timeout=request_timeout_s,
                )
                resp.raise_for_status()
                data = resp.json()
                return data["text"][0][len(prompt):]
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
                print(f"[ERROR] prompt #{idx}: {e}", flush=True)
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


def _find_best_existing_detailed_file(detailed_dir: str, model_name: str, ds_tag: str):
    """
    If the expected (non-timestamped) detailed file doesn't exist, try to find a best
    previous timestamped detailed file and resume from it.
    """
    model_tag = model_name.replace("/", "_")
    pat = os.path.join(detailed_dir, f"*{model_tag}_{ds_tag}_detailed.json")
    candidates = glob.glob(pat)
    best = None
    best_runs = -1
    for p in candidates:
        try:
            cr, _ = load_detailed_results(p)
            if cr > best_runs:
                best_runs = cr
                best = p
        except Exception:
            continue
    return best, best_runs


def _find_best_existing_cache_file(cache_dir: str, model_name: str, run_idx: int, ds_tag: str):
    """Find a cache file for this run_idx even if it was created with --filename-ts."""
    model_tag = model_name.replace("/", "_")
    pat = os.path.join(cache_dir, f"*{model_tag}_run{run_idx}_{ds_tag}.jsonl")
    cands = glob.glob(pat)
    if not cands:
        return None
    # choose newest
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def save_detailed_results(detailed_file, completed_runs, results, model_name, ds_name, split, total_problems):
    """Save detailed results to file (atomic, compact, prefer orjson)."""
    os.makedirs(os.path.dirname(detailed_file), exist_ok=True)
    data = {
        'model': model_name,
        'dataset': f"{ds_name}/{split}",
        'total_problems': total_problems,
        'completed_runs': completed_runs,
        'target_runs': num_runs,
        'results': results
    }
    tmp = detailed_file + ".tmp"
    try:
        import orjson
        with open(tmp, "wb") as f:
            f.write(orjson.dumps(data))
    except Exception:
        with open(tmp, "w", encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            f.flush()
    os.replace(tmp, detailed_file)

def _pool_worker_init():
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)

#── ： pickle ────────────────────────────────────────────
def _judge_subproc_entry(q, gold_text, pred_text, precision):
    try:
        from math_judger import MathJudger
        j = MathJudger()
        if precision is None:
            ok = bool(j.judge(gold_text, pred_text))
        else:
            ok = bool(j.judge(gold_text, pred_text, precision))
        q.put(("ok", ok, None))
    except Exception as e:
        q.put(("error", False, str(e)[:500]))

#=== + verify ===
def _judge_worker(gold_text, pred_text, precision, timeout_s):
    """
    在“判分 worker 进程”里再start一个短命child进程跑真正的 judge，
    join(timeout_s) timeout则 terminate，保证每个 task 都会按时return。
    """
    #prefer fork（Linux available，、 pickling）， spawn
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    ctx = mp.get_context(start_method)

    q = ctx.Queue(1)
    p = ctx.Process(target=_judge_subproc_entry, args=(q, gold_text, pred_text, precision), daemon=True)
    p.start()
    p.join(timeout_s)

    if p.is_alive():
        try:
            p.terminate()
            p.join(2)
        finally:
            return {"ok": False, "status": "timeout", "err": None}

    try:
        status, ok, err = q.get_nowait()
    except Exception:
        return {"ok": False, "status": "error", "err": "no result from judge subprocess"}
    return {"ok": bool(ok), "status": status, "err": err}

def validate_results_parallel(
    detailed_results,
    timeout_s=30,
    workers=1,
    desc="validation progress",
    batch_size=2000,
    max_send_chars=None,
    force_revalidate=False,
):
    """
    multi-process parallel scoring by question/run（batch submission）：
    - judgment range：
        - Default:is_correct 为 None，或 judge_status ∈ {'error','collect-timeout'}
        - force_revalidate=True：force re-judge all entries（overwrite existing is_correct）
    - timeout: is_correct=False 且 judge_status='timeout'
    - Exception: is_correct=False 且 judge_status='error' 并记录 judge_error
    - normal: judge_status='ok'
    """
    if max_send_chars is None:
        max_send_chars = int(os.getenv("JUDGE_MAX_CHARS", "20000"))

    tasks = []  # (pi, ri, gold_text, pred_text, precision)
    for pi, problem_result in enumerate(detailed_results):
        # Prefer explicit gold answer (e.g., injected from Physics JSONL final_answer); fall back to gold_solution.
        gold_src = problem_result.get('gold_answer', None)
        if gold_src is None:
            gold_src = problem_result.get('gold_solution', '')
        if isinstance(gold_src, (list, tuple)):
            gold_text = ",".join(str(x) for x in gold_src if str(x).strip())
        else:
            gold_text = str(gold_src)
        precision = problem_result.get('precision', None)
        for ri, run_result in enumerate(problem_result.get('runs', [])):
            need_judge = (
                bool(force_revalidate)
                or (run_result.get('is_correct') is None)
                or (run_result.get('judge_status') in ('error', 'collect-timeout'))
            )
            if need_judge:
                pred_text = (run_result.get('raw_output', '') or '')
                # FIX: If output is too long, try to extract boxed answer from the end first
                # This preserves answers that appear at the end of long outputs
                if max_send_chars and len(pred_text) > max_send_chars:
                    # Try to find boxed answer in the last portion
                    last_portion = pred_text[-max_send_chars:]
                    # Check if there's a boxed answer in the last portion
                    import re
                    boxed_in_last = re.search(r'\\boxed\{([^}]+)\}', last_portion)
                    if boxed_in_last:
                        # If boxed answer is in last portion, use last portion
                        pred_text = last_portion
                    else:
                        # Otherwise, check if there's boxed in the full text (might be truncated)
                        boxed_in_full = re.search(r'\\boxed\{([^}]+)\}', pred_text)
                        if boxed_in_full:
                            # Extract a window around the boxed answer
                            boxed_start = max(0, boxed_in_full.start() - 1000)
                            boxed_end = min(len(pred_text), boxed_in_full.end() + 1000)
                            pred_text = pred_text[boxed_start:boxed_end]
                        else:
                            # No boxed found, use last portion (MathJudger will try to extract from last line)
                            pred_text = last_portion
                tasks.append((pi, ri, gold_text, pred_text, precision))

    if not tasks:
        return 0  # no scoring needed

    #：prefer fork
    ctx_name = "fork" if "fork" in mp.get_all_start_methods() else "spawn"

    updated = 0
    total = len(tasks)
    with tqdm(total=total, desc=desc) as pbar:
        for start in range(0, total, batch_size):
            chunk = tasks[start:start+batch_size]
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=mp.get_context(ctx_name),
                initializer=_pool_worker_init
            ) as ex:
                future_to_idx = {}
                for (pi, ri, g, p, pr) in chunk:
                    fut = ex.submit(_judge_worker, g, p, pr, timeout_s)
                    future_to_idx[fut] = (pi, ri)

                import concurrent.futures as _cf
                for fut in as_completed(future_to_idx):
                    pi, ri = future_to_idx[fut]
                    try:
                        res = fut.result()
                    except _cf.TimeoutError:
                        res = {"ok": False, "status": "collect-timeout", "err": None}
                    except Exception as e:
                        res = {"ok": False, "status": "error", "err": str(e)[:500]}

                    ok = bool(res.get("ok", False))
                    status = res.get("status", "error")
                    err = res.get("err")

                    run = detailed_results[pi]['runs'][ri]
                    run['is_correct'] = ok
                    run['judge_status'] = status
                    run['judge_method'] = 'olympiadbench'
                    if err:
                        run['judge_error'] = err
                    else:
                        run.pop('judge_error', None)

                    updated += 1
                    pbar.update(1)

    return updated


def _load_precision_list_from_physics_jsonl(dataset_path: str, error_key: str = "error"):
    """
    Load per-problem precision/tolerance list from a local Physics JSONL dataset.
    Mirrors the parsing logic used in the main dataset-loading path:
      - error can be None
      - error can be float-like string
      - error can be comma-separated list of floats (for multi-answer)
    Returns: list[None|float|list[float]]
    """
    precisions = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            err_val = obj.get(error_key, None) if isinstance(obj, dict) else None
            precision_parsed = None
            if err_val:
                if isinstance(err_val, str) and "," in err_val:
                    try:
                        precision_parsed = [float(x) if x else 1e-8 for x in err_val.split(",")]
                    except Exception:
                        precision_parsed = None
                else:
                    try:
                        precision_parsed = float(err_val)
                    except Exception:
                        precision_parsed = None
            precisions.append(precision_parsed)
    return precisions


def _load_final_answer_list_from_physics_jsonl(dataset_path: str, answer_key: str = "final_answer"):
    """
    Load per-problem final answers from a local Physics JSONL dataset.
    Returns: list[str]
    - final_answer can be list[str] or str
    - for multi-answer, we join with comma to match MathJudger's comma-splitting behavior
    """
    answers = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            val = obj.get(answer_key, "") if isinstance(obj, dict) else ""
            if isinstance(val, (list, tuple)):
                # Keep original LaTeX pieces; MathJudger will extract $...$ or \boxed{...}
                answers.append(",".join(str(x) for x in val if str(x).strip()))
            else:
                answers.append(str(val))
    return answers

def calculate_pass_at_k(detailed_results, k):
    """calculate based on detailed results pass@k accuracy"""
    if not detailed_results:
        return {"correct": 0, "total": 0, "accuracy": 0.0}
    total = len(detailed_results)
    correct = 0
    for problem_result in detailed_results:
        runs_to_check = min(k, len(problem_result['runs']))
        #None False
        if any(bool(problem_result['runs'][i].get('is_correct')) for i in range(runs_to_check)):
            correct += 1
    accuracy = correct / total if total > 0 else 0.0
    return {
        "correct": correct,
        "total": total,
        "accuracy": accuracy
    }

#─── ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)   # Ctrl+C
    signal.signal(signal.SIGTERM, signal_handler)  # termination signal
    atexit.register(emergency_cleanup)  # cleanup on normal program exit
    
    parser = argparse.ArgumentParser(description="pass@k runner with vLLM (OlympiadBench validation)")
    parser.add_argument("--ports", type=str, default=None, help="Comma-separated explicit ports, length must equal gpu_per_model")
    parser.add_argument("--port", type=int, default=None, help="Single explicit port (only valid when gpu_per_model == 1)")
    parser.add_argument("--gpu-per-model", type=int, default=None, help="Override gpu_per_model")
    parser.add_argument("--base-port", type=int, default=None, help="Override base_port when no explicit ports provided")
    parser.add_argument("--no-check-existing", action="store_true", help="Do not check existing detailed/final results; run inference fresh and rebuild summaries in this session")
    parser.add_argument("--skip-validate", action="store_true", help="Skip validation; compute pass@k only from existing is_correct flags")
    parser.add_argument("--force-validate", action="store_true", help="Force re-validation: overwrite existing is_correct flags (also honored by env FORCE_REVALIDATE=1).")
    parser.add_argument("--filename-ts", action="store_true", help="Include run timestamp in cache and detailed result filenames")
    parser.add_argument("--detailed-file", type=str, default=None, help="Path to an existing detailed results JSON file to validate and compute pass@k directly, skipping inference")
    parser.add_argument("--validate-workers", type=int, default=None, help="parallel validationvalidation process count（default=CPUhalf）")
    parser.add_argument("--validate-timeout", type=int, default=30, help="single item scoring timeout时seconds数（timeout记为False并标注timeout）")
    parser.add_argument("--validate-batch-size", type=int, default=2000, help="batch size for validation task submission")
    # When validating an existing detailed file that came from another runner (e.g. run_pass_at_k_v2.py),
    # we may not have per-problem 'precision' saved. This option lets us inject precision from the
    # original Physics JSONL's 'error' field before running MathJudger.
    parser.add_argument("--precision-dataset-path", type=str, default=None, help="Local physics JSONL path to read per-problem precision from (uses --precision-error-key).")
    parser.add_argument("--precision-error-key", type=str, default="error", help="Field name in physics JSONL for tolerance/error (default: error).")
    parser.add_argument("--force-precision", action="store_true", help="Overwrite existing problem_result['precision'] when injecting from dataset.")
    args, _ = parser.parse_known_args()

    if args.gpu_per_model is not None:
        gpu_per_model = args.gpu_per_model
    if args.base_port is not None:
        base_port = args.base_port

    run_ts = time.strftime("%Y%m%d_%H%M%S")
    validate_workers = args.validate_workers or max(1, (os.cpu_count() or 2)//2)
    validate_timeout  = args.validate_timeout
    validate_batch_size = max(100, args.validate_batch_size)
    _env_force = str(os.getenv("FORCE_REVALIDATE", "")).strip().lower()
    force_revalidate = bool(args.force_validate) or (_env_force not in ("", "0", "false", "no", "off"))

    explicit_ports = None
    if args.ports:
        explicit_ports = [int(p.strip()) for p in args.ports.split(",") if p.strip()]
        assert len(explicit_ports) == gpu_per_model, f"--ports count ({len(explicit_ports)}) must equal gpu_per_model ({gpu_per_model})"
    elif args.port is not None:
        if gpu_per_model != 1:
            raise ValueError("--port is only allowed when gpu_per_model == 1; use --ports for multiple GPUs")
        explicit_ports = [args.port]

    #verify：，verifystatistics，
    if args.detailed_file is not None:
        print("=== Using specified detailed results file, proceeding directly to validation and statistics (OlympiadBench rules) ===", flush=True)
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
            if MathJudger is None:
                raise ImportError("无法导入 OlympiadBench MathJudger，请checkpath: OlympiadBench/inference/code/math_judger.py")

            # Inject per-problem precision if missing (useful for Physics JSONL where tolerance is stored in 'error')
            want_inject = bool(args.precision_dataset_path)
            if not want_inject:
                # Best-effort auto-detect: if dataset_str looks like a local jsonl path (data/...jsonl/train)
                # and at least one problem lacks 'precision', try to use it.
                any_missing_prec = any(('precision' not in pr) or (pr.get('precision') is None) for pr in detailed_results[:5])
                if any_missing_prec and isinstance(dataset_str, str) and dataset_str.endswith("/train"):
                    cand = dataset_str.rsplit("/", 1)[0]
                    if cand.endswith(".jsonl") and os.path.exists(cand):
                        args.precision_dataset_path = cand
                        want_inject = True

            if want_inject and args.precision_dataset_path:
                ds_path = args.precision_dataset_path
                if not os.path.isabs(ds_path):
                    ds_path = os.path.join(os.path.dirname(__file__), ds_path)
                if os.path.exists(ds_path):
                    precisions = _load_precision_list_from_physics_jsonl(ds_path, error_key=args.precision_error_key)
                    gold_answers = _load_final_answer_list_from_physics_jsonl(ds_path, answer_key="final_answer")
                    # Attach to each problem by problem_id (fallback to index)
                    for idx, pr in enumerate(detailed_results):
                        pid = pr.get("problem_id", idx)
                        if not isinstance(pid, int) or pid < 0 or pid >= len(precisions):
                            pid = idx
                        if args.force_precision or ("precision" not in pr) or (pr.get("precision") is None):
                            pr["precision"] = precisions[pid]
                        # Inject gold answers if missing; needed when validating files produced by other runners
                        if ("gold_answer" not in pr) or (pr.get("gold_answer") in (None, "")):
                            if pid < len(gold_answers):
                                pr["gold_answer"] = gold_answers[pid]
                    print(f"[info] injected per-problem precision from: {ds_path}", flush=True)
                else:
                    print(f"[WARNING] precision dataset path not found: {ds_path} (skip precision injection)", flush=True)

            print(f"正在按 OlympiadBench 规则verifyprediction result（parallel={validate_workers}，timeout={validate_timeout}s，批size={validate_batch_size}）...", flush=True)
            _n = validate_results_parallel(
                detailed_results,
                timeout_s=validate_timeout,
                workers=validate_workers,
                desc="validation progress (parallel+timeout)",
                batch_size=validate_batch_size,
                force_revalidate=force_revalidate,
            )
            print(f"已判分/update条目: {_n}", flush=True)
        else:
            print("跳过validation phase（--skip-validate）。将基于已有 is_correct 字段calculate pass@k。", flush=True)

        save_detailed_results(detailed_file, completed_runs, detailed_results, file_model, ds_name, split, total_problems)

        pass_at_k_results = {}
        print("calculating pass@k results...", flush=True)
        with tqdm(range(1, completed_runs + 1), desc="calculatepass@k") as pbar:
            for k in pbar:
                result = calculate_pass_at_k(detailed_results, k)
                pass_at_k_results[f"pass@{k}"] = result
                if k in [1, 5, 10, 20, 50, 100, 200] or k == completed_runs or k % 50 == 0:
                    pbar.set_postfix({"pass@{k}".format(k=k): f"{result['accuracy']:.4f}"})

        print("key pass@k results:", flush=True)
        key_k_values = [1, 5, 10, 20, 50, 100, 200, completed_runs]
        key_k_values = [k for k in key_k_values if k <= completed_runs]
        key_k_values = sorted(set(key_k_values))
        for k in key_k_values:
            result = pass_at_k_results[f"pass@{k}"]
            print(f"  pass@{k}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}", flush=True)

        final_summary = {}
        final_summary[f"{ds_name}_{split}"] = {
            "model": file_model,
            "completed_runs": completed_runs,
            "target_runs": _data.get('target_runs', completed_runs),
            "total_problems": total_problems,
            "pass_at_k": pass_at_k_results,
            "length_stats": {},
            "detailed_results_file": detailed_file,
        }

        dataset_name_tag = _make_dataset_tag(ds_name, split)
        out_file = os.path.join(results_dir, f"pass_at_k_{run_ts}_{file_model.replace('/', '_')}_{dataset_name_tag}.json")
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

        print(f"\n=== complete！results已保存到 {out_file} ===", flush=True)
        print(f"详细results保exists {detailed_file} 中", flush=True)
        print(f"本次runcontains的数据集: {list(final_summary.keys())}", flush=True)
        raise SystemExit(0)

    list_vllm_servers()

    #data
    all_data = {}
    for ds_name, split, prob_key, sol_key, err_key in datasets:
        ds_tag = _make_dataset_tag(ds_name, split)
        print(f"Loading dataset: {ds_name}/{split} -> tag: {ds_tag}", flush=True)
        ds = _load_any_dataset(ds_name, split)
        problems = [d[prob_key] for d in ds]
        gold_solutions_text = []
        per_problem_precision = []
        for d in ds:
            gold_val = d[sol_key]
            if isinstance(gold_val, list) and gold_val:
                gold_solutions_text.append(gold_val[0])
            else:
                gold_solutions_text.append(str(gold_val))
            err_val = d.get(err_key, None) if isinstance(d, dict) else None
            precision_parsed = None
            if err_val:
                if isinstance(err_val, str) and "," in err_val:
                    try:
                        precision_parsed = [float(x) if x else 1e-8 for x in err_val.split(",")]
                    except Exception:
                        precision_parsed = None
                else:
                    try:
                        precision_parsed = float(err_val)
                    except Exception:
                        precision_parsed = None
            per_problem_precision.append(precision_parsed)
        all_data[(ds_name, split)] = {
            "problems": problems,
            "gold_solutions_text": gold_solutions_text,
            "precisions": per_problem_precision,
        }

    check_existing = (not args.no_check_existing)
    need_inference = True if not check_existing else False
    if check_existing:
        for (ds_name, split), data in all_data.items():
            ds_tag = _make_dataset_tag(ds_name, split)
            tag_prefix = f"{run_ts}_" if args.filename_ts else ""
            detailed_file = os.path.join(detailed_results_dir, f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}_detailed.json")
            completed_runs, _ = load_detailed_results(detailed_file)
            if completed_runs < num_runs:
                need_inference = True
                break

    try:
        if need_inference:
            print(f"=== start vLLM server for {model_name} ===", flush=True)
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
            #prefer generation_config eos_token_id（）， fallback tokenizer
            _gen_config_path = os.path.join(_model_path, "generation_config.json")
            if os.path.exists(_gen_config_path):
                with open(_gen_config_path, "r") as f:
                    _gen_config = json.load(f)
                eos_ids = _gen_config.get("eos_token_id", tokenizer.eos_token_id)
                if isinstance(eos_ids, int):
                    eos_ids = [eos_ids]
            else:
                eos_ids = [tokenizer.eos_token_id]
            print(f"[info] eos_token_ids: {eos_ids}", flush=True)
            ports = start_model_server(_model_path, ports=explicit_ports)
            print(f"[info] using model path: {_model_path}", flush=True)
            _running_ports = ports.copy() if ports else []
            _model_name_for_cleanup = model_name
        else:
            print("=== all inference completed，skipping inference phase ===\n", flush=True)
            tokenizer = None
            eos_ids = None
            ports = None

        #data
        for (ds_name, split), data in all_data.items():
            ds_tag = _make_dataset_tag(ds_name, split)
            print(f"=== handle数据集: {ds_name}/{split} (tag: {ds_tag}) ===", flush=True)
            
            tag_prefix = f"{run_ts}_" if args.filename_ts else ""
            detailed_file = os.path.join(detailed_results_dir, f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}_detailed.json")
            if check_existing:
                completed_runs, detailed_results = load_detailed_results(detailed_file)
                # Fallback: resume from an older timestamped detailed file if non-ts is missing/empty
                if completed_runs == 0 and not detailed_results and (not args.filename_ts):
                    alt, alt_runs = _find_best_existing_detailed_file(detailed_results_dir, model_name, ds_tag)
                    if alt and alt_runs > 0:
                        print(f"[resume] using previous detailed file: {alt} (completed_runs={alt_runs})", flush=True)
                        completed_runs, detailed_results = load_detailed_results(alt)
            else:
                completed_runs, detailed_results = 0, []
            
            problems = data["problems"]
            gold_solutions_text = data["gold_solutions_text"]
            precisions_list = data["precisions"]
            
            # (translated comment)
            if not detailed_results:
                detailed_results = []
                for i, (problem, gold_sol_text) in enumerate(zip(problems, gold_solutions_text)):
                    detailed_results.append({
                        'problem_id': i,
                        'problem': problem,
                        'gold_solution': gold_sol_text,
                        'precision': precisions_list[i],
                        'runs': []
                    })
            
            # (translated comment)
            if completed_runs < num_runs and need_inference:
                print(f"继续从第 {completed_runs + 1} 次runstart...", flush=True)
                
                with tqdm(range(completed_runs, num_runs), desc=f"推理进度 {ds_name}/{split}") as pbar:
                    for run_idx in pbar:
                        pbar.set_description(f"推理进度 {ds_name}/{split} (第{run_idx+1}/{num_runs}次)")
                        
                        tag_prefix = f"{run_ts}_" if args.filename_ts else ""
                        cache_file = os.path.join(cache_dir, f"{tag_prefix}{model_name.replace('/', '_')}_run{run_idx}_{ds_tag}.jsonl")
                        
                        if check_existing and os.path.exists(cache_file):
                            gen_texts = []
                            with open(cache_file, "r", encoding="utf-8") as f:
                                for line in f:
                                    obj = json.loads(line)
                                    gen_texts.append(obj["response"])
                        else:
                            # Always build prompts (needed for caching even when reusing older cache)
                            prompts = [prompt_templates[prompt_style].format(problem=p) for p in problems]

                            # Fallback: if non-ts cache is missing, try to reuse a timestamped cache from previous runs
                            if check_existing and (not args.filename_ts) and (not os.path.exists(cache_file)):
                                alt_cache = _find_best_existing_cache_file(cache_dir, model_name, run_idx, ds_tag)
                                if alt_cache and os.path.exists(alt_cache):
                                    gen_texts = []
                                    with open(alt_cache, "r", encoding="utf-8") as f:
                                        for line in f:
                                            obj = json.loads(line)
                                            gen_texts.append(obj["response"])
                                else:
                                    gen_texts = query_prompts(prompts, ports, eos_ids)
                            else:
                                gen_texts = query_prompts(prompts, ports, eos_ids)
                            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                            with open(cache_file, "w", encoding="utf-8") as f:
                                for p, r in zip(prompts, gen_texts):
                                    f.write(json.dumps({"prompt": p, "response": r}, ensure_ascii=False) + "\n")
                        
                        for i, gen_text in enumerate(gen_texts):
                            run_result = {
                                'run_id': run_idx,
                                'prompt': prompt_templates[prompt_style].format(problem=problems[i]),
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
                print(f"推理complete，共complete {completed_runs} 次run", flush=True)

    except KeyboardInterrupt:
        print(f"\n=== program interrupted by user (Ctrl+C) ===", flush=True)
        emergency_cleanup()
        print("Cleanup complete, program exiting", flush=True)
        sys.exit(1)
    except Exception as e:
        print(f"\n=== program exception occurred ===", flush=True)
        print(f"errortype: {type(e).__name__}", flush=True)
        print(f"error message: {str(e)}", flush=True)
        emergency_cleanup()
        print("Cleanup complete, program exiting", flush=True)
        sys.exit(1)
    finally:
        if need_inference:
            print("=== shutting down vLLM server ===", flush=True)
            ports_to_stop = [base_port + i for i in range(gpu_per_model)]
            stop_model_server_safe(ports_to_stop, model_name=None)
            _running_ports.clear()

    print("\n=== starting calculation pass@k results（validation phase，OlympiadBench）===", flush=True)
    do_validate = not args.skip_validate
    
    out_file = os.path.join(results_dir, f"final_pass_at_k_results_{model_name.replace('/', '_')}.json")
    if check_existing and os.path.exists(out_file):
        print(f"发现已有resultsfile，将append新的数据集results: {out_file}", flush=True)
        with open(out_file, 'r', encoding='utf-8') as f:
            final_summary = json.load(f)
    else:
        prefix = "create新的resultsfile" if check_existing else "跳过read已有results，create新的resultsfile"
        print(f"{prefix}: {out_file}", flush=True)
        final_summary = {}

    if do_validate and MathJudger is None:
        raise ImportError("无法导入 OlympiadBench MathJudger，请checkpath: OlympiadBench/inference/code/math_judger.py")
    judger = MathJudger() if do_validate else None

    for (ds_name, split), data in all_data.items():
        ds_tag = _make_dataset_tag(ds_name, split)
        print(f"\ncalculateresults - 数据集: {ds_name}/{split} (tag: {ds_tag})", flush=True)
        
        tag_prefix = f"{run_ts}_" if args.filename_ts else ""
        detailed_file = os.path.join(detailed_results_dir, f"{tag_prefix}{model_name.replace('/', '_')}_{ds_tag}_detailed.json")
        completed_runs, detailed_results = load_detailed_results(detailed_file)
        
        if do_validate:
            if MathJudger is None:
                raise ImportError("无法导入 OlympiadBench MathJudger，请checkpath: OlympiadBench/inference/code/math_judger.py")
            print(f"正在verifyprediction result（parallel={validate_workers}，timeout={validate_timeout}s，批size={validate_batch_size}）...", flush=True)
            _n = validate_results_parallel(
                detailed_results,
                timeout_s=validate_timeout,
                workers=validate_workers,
                desc="validation progress (parallel+timeout)",
                batch_size=validate_batch_size,
                force_revalidate=force_revalidate,
            )
            print(f"已判分/update条目: {_n}", flush=True)
        else:
            print("跳过validation phase（--skip-validate）。将基于已有 is_correct 字段calculate pass@k。", flush=True)
        
        save_detailed_results(detailed_file, completed_runs, detailed_results, 
                            model_name, ds_name, split, len(data["problems"]))
        
        pass_at_k_results = {}
        
        print(f"calculating pass@k results...", flush=True)
        with tqdm(range(1, completed_runs + 1), desc="calculatepass@k") as pbar:
            for k in pbar:
                result = calculate_pass_at_k(detailed_results, k)
                pass_at_k_results[f"pass@{k}"] = result
                if k in [1, 5, 10, 20, 50, 100, 200] or k == completed_runs or k % 50 == 0:
                    pbar.set_postfix({"pass@{k}".format(k=k): f"{result['accuracy']:.4f}"})
        
        print("key pass@k results:", flush=True)
        key_k_values = [1, 5, 10, 20, 50, 100, 200, completed_runs]
        key_k_values = [k for k in key_k_values if k <= completed_runs]
        key_k_values = sorted(set(key_k_values))
        
        for k in key_k_values:
            result = pass_at_k_results[f"pass@{k}"]
            print(f"  pass@{k}: {result['correct']}/{result['total']} = {result['accuracy']:.4f}", flush=True)
        
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

        dataset_names = "_".join([_make_dataset_tag(ds_name, split) for ds_name, split, _, _, _ in datasets])
        out_file = os.path.join(results_dir, f"pass_at_k_{run_ts}_{model_name.replace('/', '_')}_{dataset_names}.json")
        
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== complete！results已保存到 {out_file} ===", flush=True)
    print(f"详细results保exists {detailed_results_dir} directory中", flush=True)
    print(f"本次runcontains的数据集: {list(final_summary.keys())}", flush=True)
