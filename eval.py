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
FAST_MATH_LIMIT = 100
FAST_HUMANEVAL_LIMIT = 30
# IFEval always runs in full (541 prompts, already fast enough)

# Base model for loading adapters
DEFAULT_BASE_MODEL = "Qwen/Qwen3-8B"


def run_lm_eval(model_path: str, task: str, num_fewshot: int = 0,
                limit: int | None = None, batch_size: str = "auto") -> dict:
    """Run lm-evaluation-harness and return the results dict."""
    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_path},trust_remote_code=True",
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
        for f in files:
            if f == "results.json":
                results_file = os.path.join(root, f)
                break
        if results_file:
            break

    if results_file and os.path.exists(results_file):
        with open(results_file) as f:
            return json.load(f)
    return {}


def eval_ifeval(model_path: str) -> float:
    """Evaluate IFEval prompt-level strict accuracy. Returns score in [0, 1]."""
    results = run_lm_eval(model_path, "leaderboard_ifeval", num_fewshot=0)
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
    """
    import tempfile
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
            codegen_cmd.extend(["--id_range", "0", str(limit)])

        result = subprocess.run(codegen_cmd, capture_output=True, text=True, timeout=1800)
        if result.returncode != 0:
            print(f"EvalPlus codegen failed:", file=sys.stderr)
            print((result.stdout + result.stderr)[-2000:], file=sys.stderr)
            return 0.0

        # Find the generated samples directory
        samples_dir = None
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for f in filenames:
                if f.endswith(".jsonl"):
                    samples_dir = os.path.join(dirpath, f)
                    break
            if samples_dir:
                break

        if not samples_dir:
            print("Warning: No EvalPlus samples found after codegen", file=sys.stderr)
            return 0.0

        # Step 2: Evaluate solutions
        eval_cmd = [
            sys.executable, "-m", "evalplus.evaluate",
            "humaneval",
            "--samples", samples_dir,
            "--i_just_wanna_run",
        ]

        result = subprocess.run(eval_cmd, capture_output=True, text=True, timeout=600)
        output = result.stdout + result.stderr

        # Parse pass@1 from output
        for line in output.split("\n"):
            if "pass@1" in line.lower() and "humaneval+" in line.lower():
                parts = line.split("=")
                if len(parts) >= 2:
                    try:
                        return float(parts[-1].strip())
                    except ValueError:
                        continue
            elif "pass@1" in line.lower():
                parts = line.split("=")
                if len(parts) >= 2:
                    try:
                        return float(parts[-1].strip())
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

    math_limit = FAST_MATH_LIMIT if fast else None
    humaneval_limit = FAST_HUMANEVAL_LIMIT if fast else None

    print(f"[eval] Starting evaluation (fast={fast})")
    print(f"[eval] Model: {model_path}")

    # Run benchmarks
    t0 = time.time()
    ifeval_score = eval_ifeval(model_path)
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
