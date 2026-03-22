# AutoFinetune

Autonomous QLoRA finetuning optimization for Qwen3-8B, adapted from [AutoResearch](https://github.com/karpathy/autoresearch) (Karpathy, 2026).

The idea: give an AI agent a finetuning setup and let it experiment autonomously. It modifies the config, finetunes for ~20 minutes, evaluates on three benchmarks, keeps improvements, reverts failures, and repeats.

## How it works

```
agent modifies finetune.py
    → git commit
    → finetune Qwen3-8B with QLoRA 4-bit (~20 min)
    → eval on IFEval + MATH-500 + HumanEval+ (~16 min)
    → composite_score improved? keep : git reset --hard HEAD~1
    → log to results.tsv
```

**Metric:** `composite_score` = mean(IFEval, MATH-500, HumanEval+), range [0, 1], higher is better.

## Project structure

```
finetune.py          — LoRA config, datasets, hyperparams (agent modifies this)
eval.py              — evaluation oracle: IFEval + MATH-500 + HumanEval+ (read-only)
orchestrate.py       — git-based experiment loop infrastructure (read-only)
optuna_runner.py     — Optuna TPE baseline for numeric HPO
program_finetune.md  — agent instructions and strategy guide
environment.yml      — mamba environment spec
USER_GUIDELINE.md    — manual operation guide
```

## Quick start

```bash
# 1. Create environment
mamba env create -f environment.yml
mamba activate autofinetune
pip install torch==2.9.1+cu128 --index-url https://download.pytorch.org/whl/cu128

# 2. Run baseline + enter agent loop
python orchestrate.py agent --fast

# 3. Or run a single experiment manually
python orchestrate.py run-one 1 "add MetaMathQA for math improvement" --fast
```

## Running the agent

Point your agent at `program_finetune.md`:

```
Have a look at program_finetune.md and let's kick off a new finetuning experiment.
```

The agent modifies only `finetune.py` — choosing datasets, prompt formats, LoRA targets, and hyperparameters. Each experiment takes ~36 minutes. The key advantage over the Optuna baseline (`optuna_runner.py`) is semantic decisions: which datasets to mix, how to format prompts, which model layers to adapt.

## The three benchmarks

| Benchmark | Measures | Typical 7B score |
|-----------|----------|-----------------|
| IFEval (prompt-level strict) | Instruction following | 50–70% |
| MATH-500 (exact match) | Mathematical reasoning | 30–50% |
| HumanEval+ (pass@1) | Code generation | 40–60% |

## Requirements

- NVIDIA GPU with ≥24GB VRAM (tested on RTX 3090 Ti)
- Python 3.10, mamba

## License

MIT
