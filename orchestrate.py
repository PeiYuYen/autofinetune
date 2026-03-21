"""
orchestrate.py — Main experiment loop for AutoFinetune.

Manages the git-based experiment loop:
  agent modifies finetune.py → commit → run finetune → run eval
  → keep/revert based on composite score → log to results.tsv

Usage:
  python orchestrate.py --mode agent --iterations 50
  python orchestrate.py --mode agent --iterations 50 --fast
  python orchestrate.py --mode optuna --trials 50
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).parent.resolve()
FINETUNE_SCRIPT = REPO_DIR / "finetune.py"
EVAL_SCRIPT = REPO_DIR / "eval.py"
RESULTS_TSV = REPO_DIR / "results.tsv"
OUTPUT_BASE = REPO_DIR / "output"
RUN_LOG = REPO_DIR / "run.log"

# Timeout for finetune + eval combined (seconds)
FINETUNE_TIMEOUT = 1800   # 30 min max for finetune
EVAL_TIMEOUT = 1800       # 30 min max for eval

# TSV header
TSV_HEADER = "commit\tcomposite\tifeval\tmath\thumaneval\tvram_gb\tstatus\tdescription\n"


def git_short_hash() -> str:
    """Get current short git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        capture_output=True, text=True, cwd=REPO_DIR
    )
    return result.stdout.strip()


def git_commit(message: str) -> str:
    """Stage finetune.py and commit. Returns short hash."""
    subprocess.run(["git", "add", "finetune.py"], cwd=REPO_DIR, check=True)
    subprocess.run(
        ["git", "commit", "-m", message],
        cwd=REPO_DIR, check=True
    )
    return git_short_hash()


def git_revert():
    """Revert to previous commit (undo last experiment)."""
    subprocess.run(
        ["git", "reset", "--hard", "HEAD~1"],
        cwd=REPO_DIR, check=True
    )


def init_results_tsv():
    """Initialize results.tsv with header if it doesn't exist."""
    if not RESULTS_TSV.exists():
        with open(RESULTS_TSV, "w") as f:
            f.write(TSV_HEADER)


def append_result(commit: str, composite: float, ifeval: float, math: float,
                  humaneval: float, vram_gb: float, status: str, description: str):
    """Append a result row to results.tsv."""
    row = (
        f"{commit}\t{composite:.6f}\t{ifeval:.6f}\t{math:.6f}\t"
        f"{humaneval:.6f}\t{vram_gb:.1f}\t{status}\t{description}\n"
    )
    with open(RESULTS_TSV, "a") as f:
        f.write(row)


def get_best_composite() -> float:
    """Get the best composite score from results.tsv."""
    if not RESULTS_TSV.exists():
        return -1.0
    best = -1.0
    with open(RESULTS_TSV) as f:
        for line in f:
            if line.startswith("commit"):
                continue
            parts = line.strip().split("\t")
            if len(parts) >= 7 and parts[6] == "keep":
                try:
                    score = float(parts[1])
                    best = max(best, score)
                except ValueError:
                    continue
    return best


def run_finetune(output_dir: str) -> dict | None:
    """Run finetune.py and parse results. Returns dict or None on failure."""
    cmd = [sys.executable, str(FINETUNE_SCRIPT), "--output_dir", output_dir]

    print(f"[orchestrate] Running finetune → {output_dir}")
    with open(RUN_LOG, "w") as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=REPO_DIR, timeout=FINETUNE_TIMEOUT
        )

    if result.returncode != 0:
        print("[orchestrate] Finetune CRASHED")
        # Print last 50 lines of log for debugging
        with open(RUN_LOG) as f:
            lines = f.readlines()
            print("".join(lines[-50:]))
        return None

    # Parse results from log
    results = {}
    with open(RUN_LOG) as f:
        for line in f:
            line = line.strip()
            if ":" in line and line.startswith(("training_seconds:", "peak_vram_mb:",
                                                 "num_steps:", "train_loss:", "merged_path:")):
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val

    if "merged_path" not in results:
        results["merged_path"] = os.path.join(output_dir, "merged")

    return results


def run_eval(model_path: str, fast: bool = False) -> dict | None:
    """Run eval.py and parse results. Returns dict or None on failure."""
    cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model_path", model_path,
    ]
    if fast:
        cmd.append("--fast")

    eval_log = REPO_DIR / "eval.log"
    print(f"[orchestrate] Running eval (fast={fast}) on {model_path}")

    with open(eval_log, "w") as log_f:
        result = subprocess.run(
            cmd, stdout=log_f, stderr=subprocess.STDOUT,
            cwd=REPO_DIR, timeout=EVAL_TIMEOUT
        )

    if result.returncode != 0:
        print("[orchestrate] Eval CRASHED")
        with open(eval_log) as f:
            lines = f.readlines()
            print("".join(lines[-50:]))
        return None

    # Parse results from log
    results = {}
    with open(eval_log) as f:
        in_results = False
        for line in f:
            line = line.strip()
            if line == "---":
                in_results = True
                continue
            if in_results and ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    results[key] = float(val)
                except ValueError:
                    results[key] = val

    return results


def run_single_experiment(run_id: int, description: str,
                          fast_eval: bool = True) -> dict:
    """
    Run a single finetune + eval experiment.

    Returns dict with all results and status.
    """
    output_dir = str(OUTPUT_BASE / f"run_{run_id:04d}")
    os.makedirs(output_dir, exist_ok=True)

    # Commit current finetune.py changes
    try:
        commit_hash = git_commit(f"exp {run_id}: {description}")
    except subprocess.CalledProcessError:
        # Nothing to commit (no changes)
        commit_hash = git_short_hash()

    best_composite = get_best_composite()

    # Run finetune
    try:
        train_results = run_finetune(output_dir)
    except subprocess.TimeoutExpired:
        print(f"[orchestrate] Finetune TIMEOUT for run {run_id}")
        append_result(commit_hash, 0.0, 0.0, 0.0, 0.0, 0.0, "crash",
                      f"{description} (timeout)")
        git_revert()
        return {"status": "crash", "reason": "timeout"}

    if train_results is None:
        append_result(commit_hash, 0.0, 0.0, 0.0, 0.0, 0.0, "crash",
                      f"{description} (crash)")
        git_revert()
        return {"status": "crash", "reason": "finetune_crash"}

    # Run eval
    merged_path = train_results.get("merged_path", os.path.join(output_dir, "merged"))
    try:
        eval_results = run_eval(merged_path, fast=fast_eval)
    except subprocess.TimeoutExpired:
        print(f"[orchestrate] Eval TIMEOUT for run {run_id}")
        vram_gb = train_results.get("peak_vram_mb", 0) / 1024
        append_result(commit_hash, 0.0, 0.0, 0.0, 0.0, vram_gb, "crash",
                      f"{description} (eval timeout)")
        git_revert()
        return {"status": "crash", "reason": "eval_timeout"}

    if eval_results is None:
        vram_gb = train_results.get("peak_vram_mb", 0) / 1024
        append_result(commit_hash, 0.0, 0.0, 0.0, 0.0, vram_gb, "crash",
                      f"{description} (eval crash)")
        git_revert()
        return {"status": "crash", "reason": "eval_crash"}

    # Extract scores
    composite = eval_results.get("composite_score", 0.0)
    ifeval = eval_results.get("ifeval_strict", 0.0)
    math_score = eval_results.get("math_500_em", 0.0)
    humaneval = eval_results.get("humaneval_pass1", 0.0)
    vram_gb = max(
        train_results.get("peak_vram_mb", 0),
        eval_results.get("peak_vram_mb", 0)
    ) / 1024

    # Keep or discard
    if composite > best_composite:
        status = "keep"
        print(f"[orchestrate] KEEP — composite {composite:.6f} > {best_composite:.6f}")
    else:
        status = "discard"
        print(f"[orchestrate] DISCARD — composite {composite:.6f} <= {best_composite:.6f}")
        git_revert()

    append_result(commit_hash, composite, ifeval, math_score, humaneval,
                  vram_gb, status, description)

    # Cleanup merged model to save disk (keep adapter only)
    if os.path.exists(merged_path):
        shutil.rmtree(merged_path, ignore_errors=True)

    return {
        "status": status,
        "composite": composite,
        "ifeval": ifeval,
        "math": math_score,
        "humaneval": humaneval,
        "vram_gb": vram_gb,
        "train_loss": train_results.get("train_loss", 0),
        "num_steps": train_results.get("num_steps", 0),
    }


def read_results_history() -> str:
    """Read results.tsv and return as string for agent context."""
    if not RESULTS_TSV.exists():
        return "No results yet."
    with open(RESULTS_TSV) as f:
        return f.read()


def read_finetune_py() -> str:
    """Read current finetune.py content."""
    with open(FINETUNE_SCRIPT) as f:
        return f.read()


# ---------------------------------------------------------------------------
# Agent mode: LLM agent modifies finetune.py
# ---------------------------------------------------------------------------

def run_agent_mode(iterations: int, fast_eval: bool = True):
    """
    Run the agent loop. The agent is invoked externally (e.g., via Claude API).
    This function provides the infrastructure for the loop.

    In practice, the LLM agent (Claude) runs this loop by:
    1. Reading finetune.py and results.tsv
    2. Modifying finetune.py with a new experiment idea
    3. Calling run_single_experiment()
    4. Repeating
    """
    init_results_tsv()

    print(f"[orchestrate] Agent mode: {iterations} iterations, fast_eval={fast_eval}")
    print(f"[orchestrate] Results: {RESULTS_TSV}")
    print(f"[orchestrate] Finetune script: {FINETUNE_SCRIPT}")
    print()

    # Run baseline first
    print("=" * 60)
    print("[orchestrate] Running baseline (unmodified finetune.py)")
    print("=" * 60)
    result = run_single_experiment(0, "baseline", fast_eval=fast_eval)
    print(f"[orchestrate] Baseline result: {json.dumps(result, indent=2)}")
    print()

    print(f"[orchestrate] Agent loop ready. The LLM agent should now modify finetune.py")
    print(f"[orchestrate] and call: python orchestrate.py --run-one <run_id> '<description>'")
    print(f"[orchestrate] Current best composite: {get_best_composite():.6f}")


# ---------------------------------------------------------------------------
# Single experiment mode (called by agent per iteration)
# ---------------------------------------------------------------------------

def run_one(run_id: int, description: str, fast_eval: bool = True):
    """Run a single experiment (called by the agent)."""
    init_results_tsv()
    result = run_single_experiment(run_id, description, fast_eval=fast_eval)
    print(f"\n[orchestrate] Experiment {run_id} result:")
    print(json.dumps(result, indent=2))
    print(f"[orchestrate] Current best composite: {get_best_composite():.6f}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoFinetune experiment orchestrator")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Agent mode
    agent_parser = subparsers.add_parser("agent", help="Initialize agent mode")
    agent_parser.add_argument("--iterations", type=int, default=50)
    agent_parser.add_argument("--fast", action="store_true", help="Use fast eval")

    # Single experiment
    one_parser = subparsers.add_parser("run-one", help="Run a single experiment")
    one_parser.add_argument("run_id", type=int)
    one_parser.add_argument("description", type=str)
    one_parser.add_argument("--fast", action="store_true", help="Use fast eval")
    one_parser.add_argument("--full", action="store_true", help="Use full eval (override fast)")

    # Status
    status_parser = subparsers.add_parser("status", help="Show current experiment status")

    args = parser.parse_args()

    if args.command == "agent":
        run_agent_mode(iterations=args.iterations, fast_eval=args.fast)
    elif args.command == "run-one":
        fast = args.fast and not args.full
        run_one(args.run_id, args.description, fast_eval=fast)
    elif args.command == "status":
        init_results_tsv()
        print(f"Best composite: {get_best_composite():.6f}")
        print(f"\nResults history:")
        print(read_results_history())
    else:
        parser.print_help()
