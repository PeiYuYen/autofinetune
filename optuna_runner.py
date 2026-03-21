"""
optuna_runner.py — Optuna TPE baseline for AutoFinetune.

Searches over NUMERIC hyperparameters only (no dataset/prompt/module selection).
This demonstrates the agent's semantic advantage over algorithmic HPO.

Usage:
  python optuna_runner.py --trials 50 --fast
  python optuna_runner.py --trials 50 --study-name my_study
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import optuna

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO_DIR = Path(__file__).parent.resolve()
FINETUNE_TEMPLATE = REPO_DIR / "finetune.py"
EVAL_SCRIPT = REPO_DIR / "eval.py"
OUTPUT_BASE = REPO_DIR / "output" / "optuna"
RESULTS_FILE = REPO_DIR / "optuna_results.json"

# Fixed values (not searched by Optuna — this is the agent's advantage)
FIXED_BASE_MODEL = "Qwen/Qwen3-8B"
FIXED_DATASET = '("teknium/OpenHermes-2.5", None, "train", 1.0)'
FIXED_TARGET_MODULES = '["q_proj", "k_proj", "v_proj", "o_proj"]'
FIXED_DATASET_SAMPLE_SIZE = 10000


def generate_finetune_py(trial: optuna.Trial) -> tuple[str, dict]:
    """
    Generate a finetune.py with Optuna-suggested hyperparameters.
    Returns (file_content, params_dict).
    """
    # Suggest hyperparameters
    lora_rank = trial.suggest_categorical("lora_rank", [4, 8, 16, 32, 64])
    lora_alpha = lora_rank * 2
    lora_dropout = trial.suggest_float("lora_dropout", 0.0, 0.1, step=0.05)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    lr_scheduler = trial.suggest_categorical("lr_scheduler", ["linear", "cosine", "constant"])
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.1, step=0.01)
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
    per_device_batch_size = trial.suggest_categorical("per_device_batch_size", [2, 4, 8])
    gradient_accumulation = trial.suggest_categorical("gradient_accumulation_steps", [2, 4, 8])
    max_seq_length = trial.suggest_categorical("max_seq_length", [1024, 2048])

    params = {
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "learning_rate": learning_rate,
        "lr_scheduler": lr_scheduler,
        "warmup_ratio": warmup_ratio,
        "weight_decay": weight_decay,
        "per_device_batch_size": per_device_batch_size,
        "gradient_accumulation_steps": gradient_accumulation,
        "max_seq_length": max_seq_length,
    }

    # Read template and substitute values
    with open(FINETUNE_TEMPLATE) as f:
        content = f.read()

    # Replace hyperparameters using regex
    replacements = {
        r"LORA_RANK = \d+": f"LORA_RANK = {lora_rank}",
        r"LORA_ALPHA = \d+": f"LORA_ALPHA = {lora_alpha}",
        r"LORA_DROPOUT = [\d.]+": f"LORA_DROPOUT = {lora_dropout}",
        r"LEARNING_RATE = [\d.e\-]+": f"LEARNING_RATE = {learning_rate}",
        r'LR_SCHEDULER = "[^"]*"': f'LR_SCHEDULER = "{lr_scheduler}"',
        r"WARMUP_RATIO = [\d.]+": f"WARMUP_RATIO = {warmup_ratio}",
        r"WEIGHT_DECAY = [\d.]+": f"WEIGHT_DECAY = {weight_decay}",
        r"PER_DEVICE_BATCH_SIZE = \d+": f"PER_DEVICE_BATCH_SIZE = {per_device_batch_size}",
        r"GRADIENT_ACCUMULATION_STEPS = \d+": f"GRADIENT_ACCUMULATION_STEPS = {gradient_accumulation}",
        r"MAX_SEQ_LENGTH = \d+": f"MAX_SEQ_LENGTH = {max_seq_length}",
    }

    for pattern, replacement in replacements.items():
        content = re.sub(pattern, replacement, content)

    return content, params


def run_trial(content: str, trial_id: int, fast_eval: bool = True) -> dict | None:
    """Run a single Optuna trial: write finetune.py, train, eval."""
    output_dir = str(OUTPUT_BASE / f"trial_{trial_id:04d}")
    os.makedirs(output_dir, exist_ok=True)

    # Write modified finetune.py to a temp location
    trial_finetune = os.path.join(output_dir, "finetune.py")
    with open(trial_finetune, "w") as f:
        f.write(content)

    # Run finetune
    cmd = [sys.executable, trial_finetune, "--output_dir", output_dir]
    run_log = os.path.join(output_dir, "run.log")

    try:
        with open(run_log, "w") as log_f:
            result = subprocess.run(
                cmd, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=1800
            )
        if result.returncode != 0:
            return None
    except subprocess.TimeoutExpired:
        return None

    # Find merged model path
    merged_path = os.path.join(output_dir, "merged")
    if not os.path.exists(merged_path):
        return None

    # Run eval
    eval_cmd = [
        sys.executable, str(EVAL_SCRIPT),
        "--model_path", merged_path,
    ]
    if fast_eval:
        eval_cmd.append("--fast")

    eval_log = os.path.join(output_dir, "eval.log")
    try:
        with open(eval_log, "w") as log_f:
            result = subprocess.run(
                eval_cmd, stdout=log_f, stderr=subprocess.STDOUT,
                timeout=1800
            )
        if result.returncode != 0:
            return None
    except subprocess.TimeoutExpired:
        return None

    # Parse eval results
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
                try:
                    results[key.strip()] = float(val.strip())
                except ValueError:
                    pass

    # Cleanup merged model
    import shutil
    if os.path.exists(merged_path):
        shutil.rmtree(merged_path, ignore_errors=True)

    return results


def objective(trial: optuna.Trial, fast_eval: bool = True) -> float:
    """Optuna objective: maximize composite score."""
    content, params = generate_finetune_py(trial)

    print(f"\n[optuna] Trial {trial.number}: {params}")

    results = run_trial(content, trial.number, fast_eval=fast_eval)

    if results is None:
        print(f"[optuna] Trial {trial.number}: FAILED")
        return 0.0  # Return worst score for failed trials

    composite = results.get("composite_score", 0.0)
    ifeval = results.get("ifeval_strict", 0.0)
    math_score = results.get("math_500_em", 0.0)
    humaneval = results.get("humaneval_pass1", 0.0)

    print(f"[optuna] Trial {trial.number}: composite={composite:.4f} "
          f"(ifeval={ifeval:.4f}, math={math_score:.4f}, humaneval={humaneval:.4f})")

    # Store per-benchmark scores as user attributes
    trial.set_user_attr("ifeval_strict", ifeval)
    trial.set_user_attr("math_500_em", math_score)
    trial.set_user_attr("humaneval_pass1", humaneval)

    return composite


def main():
    parser = argparse.ArgumentParser(description="Optuna HPO baseline for AutoFinetune")
    parser.add_argument("--trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--study-name", type=str, default="autofinetune_optuna")
    parser.add_argument("--fast", action="store_true", help="Use fast eval")
    parser.add_argument("--storage", type=str, default=None,
                        help="Optuna storage URL (default: in-memory)")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=args.storage,
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    print(f"[optuna] Starting {args.trials} trials (fast={args.fast})")
    print(f"[optuna] Study: {args.study_name}")
    print(f"[optuna] Search space: lora_rank, lora_alpha, lora_dropout, "
          f"learning_rate, lr_scheduler, warmup_ratio, weight_decay, "
          f"batch_size, grad_accum, max_seq_length")
    print(f"[optuna] FIXED: dataset=OpenHermes-2.5, target_modules=qkvo, "
          f"model={FIXED_BASE_MODEL}")
    print()

    study.optimize(
        lambda trial: objective(trial, fast_eval=args.fast),
        n_trials=args.trials,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("[optuna] OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best composite score: {study.best_value:.6f}")
    print(f"Best params: {json.dumps(study.best_params, indent=2)}")

    if study.best_trial.user_attrs:
        print(f"Best IFEval: {study.best_trial.user_attrs.get('ifeval_strict', 'N/A')}")
        print(f"Best MATH: {study.best_trial.user_attrs.get('math_500_em', 'N/A')}")
        print(f"Best HumanEval: {study.best_trial.user_attrs.get('humaneval_pass1', 'N/A')}")

    # Save all results
    results = []
    for trial in study.trials:
        results.append({
            "number": trial.number,
            "value": trial.value,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "state": str(trial.state),
        })

    with open(RESULTS_FILE, "w") as f:
        json.dump({
            "study_name": args.study_name,
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
            "best_params": study.best_params,
            "trials": results,
        }, f, indent=2)

    print(f"\n[optuna] Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
