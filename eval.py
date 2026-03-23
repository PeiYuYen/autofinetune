"""
eval.py — Fixed evaluation oracle for AutoFinetune.

DO NOT MODIFY this file. It is the ground truth evaluation harness.
The agent may only modify finetune.py.

Evaluates a finetuned model on three benchmarks:
  1. IFEval  — instruction following (prompt-level strict accuracy)
  2. MATH-500 — mathematical reasoning (exact match)
  3. HumanEval+ — code generation (pass@1, via EvalPlus)

Usage:
  python eval.py --model_path <path_to_adapter_or_merged>
  python eval.py --model_path <path> --fast   # subsampled fast eval
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import torch

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Default composite score weights (equal weighting)
W_IFEVAL = 1 / 3
W_MATH = 1 / 3
W_HUMANEVAL = 1 / 3

# Fast eval limits (subsampled for speed during search)
# NOTE: leaderboard_math_hard is a GROUP of 7 subtasks — --limit applies per subtask.
# FAST_MATH_LIMIT=7 → 7 subtasks × 7 samples = 49 total (comparable to IFEval's 50).
FAST_IFEVAL_LIMIT = 50
FAST_MATH_LIMIT = 7
FAST_HUMANEVAL_LIMIT = 20

# Base model for loading adapters
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def run_lm_eval(model_path: str, task: str, num_fewshot: int = 0,
                limit: int | None = None, batch_size: str = "4") -> dict:
    """Run lm-evaluation-harness and return the results dict."""
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True,dtype=bfloat16",
        "--tasks", task,
        "--num_fewshot", str(num_fewshot),
        "--batch_size", batch_size,
        "--output_path", f"/tmp/lm_eval_{task}",
        "--log_samples",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if result.returncode != 0:
        print(f"lm_eval failed for {task}:", file=sys.stderr)
        print(result.stderr[-2000:], file=sys.stderr)
        return {}

    # Parse results from output directory
    output_dir = f"/tmp/lm_eval_{task}"
    results_file = None
    for root, dirs, files in os.walk(output_dir):
        for f in sorted(files, reverse=True):  # newest first
            if f.startswith("results") and f.endswith(".json"):
                results_file = os.path.join(root, f)
                break
        if results_file:
            break

    if results_file and os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    return {}


def eval_ifeval(model_path: str, limit: int | None = None) -> float:
    """Evaluate IFEval prompt-level strict accuracy. Returns score in [0, 1]."""
    results = run_lm_eval(model_path, "leaderboard_ifeval", num_fewshot=0, limit=limit)
    if not results:
        return 0.0
    try:
        # lm-eval-harness stores IFEval results under this key
        r = results.get("results", {})
        for key in r:
            if "ifeval" in key.lower():
                metrics = r[key]
                # Try prompt-level strict accuracy first
                for metric_key in ["prompt_level_strict_acc", "prompt_level_strict_acc,none"]:
                    if metric_key in metrics:
                        return float(metrics[metric_key])
                # Fallback to any accuracy metric
                for metric_key in metrics:
                    if "acc" in metric_key and "stderr" not in metric_key:
                        return float(metrics[metric_key])
    except (KeyError, TypeError, ValueError) as e:
        print(f"Warning: could not parse IFEval results: {e}", file=sys.stderr)
    return 0.0


def eval_math500(model_path: str, limit: int | None = None) -> float:
    """Evaluate MATH-500 exact match. Returns score in [0, 1]."""
    results = run_lm_eval(
        model_path, "leaderboard_math_hard",
        num_fewshot=0, limit=limit
    )
    if not results:
        return 0.0
    try:
        r = results.get("results", {})
        for key in r:
            if "math" in key.lower():
                metrics = r[key]
                for metric_key in ["exact_match,none", "exact_match", "acc,none", "acc"]:
                    if metric_key in metrics:
                        return float(metrics[metric_key])
    except (KeyError, TypeError, ValueError) as e:
        print(f"Warning: could not parse MATH results: {e}", file=sys.stderr)
    return 0.0


def eval_humaneval(model_path: str, limit: int | None = None) -> float:
    """Evaluate HumanEval+ pass@1 via EvalPlus. Returns score in [0, 1].

    EvalPlus uses a two-step flow: codegen (generate solutions) → evaluate (run tests).
    The --id_range flag on codegen limits which problems to solve for fast eval.

    Bug fixes vs original:
    1. evalplus.evaluate asserts all 164 problems must be present — we pad missing ones
       with stub solutions ("pass") so the assertion passes. pass@1 is then computed
       only on the actually-generated problems (not the stubs) to avoid score dilution.
    2. evalplus prints "pass@1:\\t0.XXX" (not "="), and the "humaneval+" label is on a
       separate header line — we read _eval_results.json directly instead of parsing stdout.
    """
    import tempfile
    import re
    root_dir = tempfile.mkdtemp(prefix="evalplus_")

    try:
        # Step 1: Generate code solutions
        codegen_cmd = [
            sys.executable, "-m", "evalplus.codegen",
            model_path, "humaneval",
            "--root", root_dir,
            "--backend", "hf",
            "--greedy",
            "--trust_remote_code",
            "--n_samples", "1",
            "--attn_implementation", "eager",
        ]
        if limit is not None:
            # Fire parses list args as "[0,N]" not "0 N" (the latter makes id_range=int → TypeError)
            codegen_cmd.extend(["--id_range", f"[0,{limit}]"])

        result = subprocess.run(codegen_cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"EvalPlus codegen failed:", file=sys.stderr)
            print((result.stdout + result.stderr)[-2000:], file=sys.stderr)
            return 0.0

        # Find the generated samples file (.jsonl)
        samples_path = None
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".jsonl"):
                    samples_path = os.path.join(dirpath, f)
                    break
            if samples_path:
                break

        if not samples_path:
            print("Warning: No EvalPlus samples found after codegen", file=sys.stderr)
            return 0.0

        # Track which problems were actually generated (needed to compute honest pass@1
        # on the evaluated subset, not the padded stubs).
        generated_ids: set[str] = set()
        with open(samples_path) as f:
            for line in f:
                if line.strip():
                    generated_ids.add(json.loads(line)["task_id"])

        # Bug fix 1: evalplus.evaluate requires all 164 problems in the samples file.
        # When using --id_range, only a subset is generated.  Pad with stub solutions
        # ("pass") for missing problems so the assertion inside evaluate() passes.
        if limit is not None and len(generated_ids) < 164:
            from evalplus.data import get_human_eval_plus
            all_problems = get_human_eval_plus()
            with open(samples_path, "a") as f:
                for task_id, prob in all_problems.items():
                    if task_id not in generated_ids:
                        stub = prob["prompt"] + "    pass\n"
                        f.write(json.dumps({"task_id": task_id, "solution": stub}) + "\n")

        # Step 2: Evaluate solutions
        eval_cmd = [
            sys.executable, "-m", "evalplus.evaluate",
            "humaneval",
            "--samples", samples_path,
            "--i_just_wanna_run",
        ]

        subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)

        # Bug fix 2: evalplus prints "pass@1:\t0.XXX" (tab-separated, not "=") and the
        # "humaneval+" header is on a DIFFERENT line from the score.  Read the JSON result
        # file directly instead of parsing stdout.
        eval_results_path = samples_path.replace(".jsonl", "_eval_results.json")
        if os.path.exists(eval_results_path):
            with open(eval_results_path) as f:
                eval_data = json.load(f)

            task_results = eval_data.get("eval", {})

            # Compute pass@1 only on the actually-generated problems (not stubs).
            # This avoids diluting the score by 164/limit when limit < 164.
            relevant = {
                tid: task_results[tid]
                for tid in generated_ids
                if tid in task_results
            }
            if relevant:
                n_pass = sum(
                    1 for res in relevant.values()
                    if res and res[0]["base_status"] == "pass" and res[0]["plus_status"] == "pass"
                )
                return n_pass / len(relevant)

        # Fallback: parse stdout with regex (handles "pass@1:\t0.XXX" and ANSI codes)
        for line in (result.stdout + result.stderr).split("\n"):
            if "pass@1" in line.lower():
                match = re.search(r'pass@1[:\s\t]+([0-9.]+)', line, re.IGNORECASE)
                if match:
                    try:
                        return float(match.group(1))
                    except ValueError:
                        continue
    except subprocess.TimeoutExpired:
        print("Warning: HumanEval evaluation timed out", file=sys.stderr)
    except Exception as e:
        print(f"Warning: HumanEval evaluation failed: {e}", file=sys.stderr)
    finally:
        import shutil
        shutil.rmtree(root_dir, ignore_errors=True)
    return 0.0


def compute_composite(ifeval: float, math: float, humaneval: float,
                       w_ifeval: float = W_IFEVAL, w_math: float = W_MATH,
                       w_humaneval: float = W_HUMANEVAL) -> float:
    """Compute weighted composite score. All inputs and output in [0, 1]."""
    return w_ifeval * ifeval + w_math * math + w_humaneval * humaneval


def evaluate(model_path: str, fast: bool = False,
             w_ifeval: float = W_IFEVAL, w_math: float = W_MATH,
             w_humaneval: float = W_HUMANEVAL) -> dict:
    """
    Run all benchmarks and return results dict.

    Args:
        model_path: Path to merged model or adapter directory
        fast: If True, subsample MATH and HumanEval for speed
        w_ifeval, w_math, w_humaneval: Composite score weights
    """
    t_start = time.time()
    peak_vram_before = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    # Reset VRAM tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    ifeval_limit = FAST_IFEVAL_LIMIT if fast else None
    math_limit = FAST_MATH_LIMIT if fast else None
    humaneval_limit = FAST_HUMANEVAL_LIMIT if fast else None

    print(f"[eval] Starting evaluation (fast={fast})")
    print(f"[eval] Model: {model_path}")

    # Run benchmarks
    t0 = time.time()
    ifeval_score = eval_ifeval(model_path, limit=ifeval_limit)
    t_ifeval = time.time() - t0
    print(f"[eval] IFEval: {ifeval_score:.4f} ({t_ifeval:.0f}s)")

    t0 = time.time()
    math_score = eval_math500(model_path, limit=math_limit)
    t_math = time.time() - t0
    print(f"[eval] MATH-500: {math_score:.4f} ({t_math:.0f}s)")

    t0 = time.time()
    humaneval_score = eval_humaneval(model_path, limit=humaneval_limit)
    t_humaneval = time.time() - t0
    print(f"[eval] HumanEval+: {humaneval_score:.4f} ({t_humaneval:.0f}s)")

    composite = compute_composite(
        ifeval_score, math_score, humaneval_score,
        w_ifeval, w_math, w_humaneval
    )

    t_total = time.time() - t_start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

    results = {
        "composite_score": composite,
        "ifeval_strict": ifeval_score,
        "math_500_em": math_score,
        "humaneval_pass1": humaneval_score,
        "eval_seconds": t_total,
        "peak_vram_mb": peak_vram_mb,
        "fast_mode": fast,
    }

    # Print in AutoResearch-compatible format
    print("---")
    print(f"composite_score:  {composite:.6f}")
    print(f"ifeval_strict:    {ifeval_score:.6f}")
    print(f"math_500_em:      {math_score:.6f}")
    print(f"humaneval_pass1:  {humaneval_score:.6f}")
    print(f"eval_seconds:     {t_total:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate finetuned model on IFEval + MATH-500 + HumanEval+")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to merged model or adapter directory")
    parser.add_argument("--fast", action="store_true",
                        help="Use subsampled benchmarks for faster evaluation")
    parser.add_argument("--w_ifeval", type=float, default=W_IFEVAL,
                        help="Weight for IFEval in composite score")
    parser.add_argument("--w_math", type=float, default=W_MATH,
                        help="Weight for MATH-500 in composite score")
    parser.add_argument("--w_humaneval", type=float, default=W_HUMANEVAL,
                        help="Weight for HumanEval+ in composite score")
    args = parser.parse_args()

    results = evaluate(
        model_path=args.model_path,
        fast=args.fast,
        w_ifeval=args.w_ifeval,
        w_math=args.w_math,
        w_humaneval=args.w_humaneval,
    )

    # Save results to JSON (use model dir if it's a local path, else cwd)
    model_dir = os.path.dirname(args.model_path)
    if model_dir and os.path.isdir(model_dir):
        output_file = os.path.join(model_dir, "eval_results.json")
    else:
        output_file = "eval_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[eval] Results saved to {output_file}")
