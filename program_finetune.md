# AutoFinetune

This is an experiment to have an LLM agent autonomously optimize LLM finetuning configurations, adapted from AutoResearch (Karpathy, 2026).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar22`). The branch `autofinetune/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autofinetune/<tag>` from current master.
3. **Read the in-scope files**: The repo has these key files:
   - `finetune.py` — **the file you modify**. LoRA config, dataset selection, prompt format, training hyperparameters.
   - `eval.py` — fixed evaluation oracle. Do NOT modify. Runs IFEval + MATH-500 + HumanEval+.
   - `orchestrate.py` — experiment loop infrastructure. Do NOT modify.
   - `program_finetune.md` — this file. Your instructions.
4. **Verify environment**: Check that the required packages are installed (`pip list | grep -E "unsloth|trl|peft|lm.eval|evalplus"`).
5. **Initialize results.tsv**: Create with header:
   ```
   commit	composite	ifeval	math	humaneval	vram_gb	status	description
   ```
6. **Run baseline**: Run the first experiment with unmodified finetune.py to establish ground truth.

## The Three Benchmarks

Your composite score is the average of three benchmarks (all in [0, 1], higher is better):

1. **IFEval** (prompt-level strict accuracy): Tests instruction following. Example: "Write a response in exactly 3 paragraphs." Measures whether the model precisely follows formatting/structural constraints.

2. **MATH-500** (exact match): Tests mathematical reasoning. 500 competition-level math problems. A 7B model typically scores 30-50%. Improving this requires math-focused training data and reasoning capabilities.

3. **HumanEval+** (pass@1): Tests code generation. 164 Python programming problems with extended test cases (harder than original HumanEval). Requires code-focused training data.

**Composite score** = (IFEval + MATH-500 + HumanEval+) / 3

## What You CAN Modify

**Only `finetune.py`** — but everything in it is fair game:

### Numeric hyperparameters
- `LORA_RANK` (4, 8, 16, 32, 64)
- `LORA_ALPHA` (typically 2× rank, but you can experiment)
- `LORA_DROPOUT` (0.0 to 0.1)
- `LEARNING_RATE` (1e-5 to 5e-4)
- `LR_SCHEDULER` ("linear", "cosine", "constant")
- `WARMUP_RATIO`, `WEIGHT_DECAY`
- `PER_DEVICE_BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`
- `MAX_SEQ_LENGTH` (1024, 2048)
- `MAX_STEPS` (controls training duration within time budget)

### Semantic decisions (YOUR ADVANTAGE over Optuna)
- **Dataset selection**: Change `DATASET_CONFIG` to use different datasets or mix multiple:
  - `teknium/OpenHermes-2.5` — 1M general instruction-following examples
  - `meta-math/MetaMathQA` — 395K math reasoning (helps MATH-500)
  - `m-a-p/CodeFeedback-Filtered-Instruction` — 157K code instructions (helps HumanEval)
  - `Open-Orca/SlimOrca` — 518K general reasoning
  - `TIGER-Lab/MathInstruct` — 262K diverse math
  - You can use any open dataset on HuggingFace Hub
- **Dataset mixing ratio**: Adjust weights in DATASET_CONFIG to balance benchmarks
- **Prompt format**: Modify `format_prompt()` to change how data is formatted (ChatML, Alpaca, custom)
- **LoRA target modules**: Change which layers get adapted: `["q_proj", "v_proj"]` vs `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]`
- **Sample size**: `DATASET_SAMPLE_SIZE` controls total training examples

## What You CANNOT Modify

- `eval.py` — fixed evaluation oracle
- `orchestrate.py` — experiment infrastructure
- Base model — always `Qwen/Qwen3-8B` with QLoRA 4-bit
- Cannot install new packages

## Running Experiments

Each experiment:
1. Modify `finetune.py` with your idea
2. `git commit -am "exp N: <description>"`
3. `python orchestrate.py run-one <N> "<description>" --fast`
4. Read the output for composite score and per-benchmark scores
5. If composite improved → commit stays (keep)
6. If composite equal or worse → orchestrate.py auto-reverts (discard)

Extract results: `grep "^composite_score:\|^ifeval_strict:\|^math_500_em:\|^humaneval_pass1:" eval.log`

## Logging Results

`orchestrate.py` automatically logs to `results.tsv` (tab-separated):

```
commit	composite	ifeval	math	humaneval	vram_gb	status	description
a1b2c3d	0.452300	0.520000	0.380000	0.457000	18.1	keep	baseline
b2c3d4e	0.481200	0.530000	0.420000	0.493000	18.3	keep	add MetaMathQA to mix
c3d4e5f	0.440100	0.510000	0.370000	0.439000	18.0	discard	increase rank to 64
```

## Strategy Guide

Think about WHY each change might help specific benchmarks:

- **MATH-500 is low?** → Add MetaMathQA or MathInstruct to dataset mix, increase their weight
- **HumanEval is low?** → Add CodeFeedback, consider code-specific prompt format
- **IFEval is low?** → Ensure training data has diverse formatting constraints, use ChatML format
- **All scores are mediocre?** → Try bigger LoRA rank (more capacity), different LR
- **Training loss still high?** → More steps, lower LR, bigger batch
- **Overfitting?** → Fewer steps, add dropout, reduce rank

**KEY INSIGHT**: Optuna can only search numeric parameters. YOUR advantage is making semantic decisions — choosing the right datasets for the right benchmarks, designing prompt formats, selecting which model layers to adapt. Use this advantage.

## The Experiment Loop

LOOP FOREVER:

1. Read `results.tsv` — understand what's been tried, what worked, what didn't
2. Analyze per-benchmark scores — identify which benchmark needs the most improvement
3. Form a hypothesis: "If I [change X], then [benchmark Y] should improve because [reason Z]"
4. Implement the change in `finetune.py`
5. Run the experiment
6. Record results and reasoning
7. If the hypothesis was wrong, think about why and adjust

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human. The human might be asleep. You are autonomous. If you run out of ideas, think harder — re-read the benchmark descriptions, try combining previous near-misses, try more radical changes. The loop runs until manually interrupted.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds complexity is not worth it. Removing something and getting equal or better results is a great outcome.

**Timeout**: Each experiment takes ~36 minutes (20 min finetune + 16 min fast eval). If a run exceeds 60 minutes total, kill it.
