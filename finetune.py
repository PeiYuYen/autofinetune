"""
finetune.py — The agent's modification target for AutoFinetune.

This is the ONLY file the agent modifies. Everything is fair game:
LoRA config, dataset selection, prompt format, training hyperparameters.

Usage:
  python finetune.py
  python finetune.py --output_dir ./output/run_001
"""

import argparse
import json
import math
import os
import time

import torch
from datasets import load_dataset, concatenate_datasets
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer, SFTConfig

# ---------------------------------------------------------------------------
# Hyperparameters (agent modifies this section)
# ---------------------------------------------------------------------------

# Base model
BASE_MODEL = "Qwen/Qwen3-8B"

# LoRA configuration
LORA_RANK = 16
LORA_ALPHA = 32  # typically 2 * rank
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

# Training hyperparameters
LEARNING_RATE = 2e-4
LR_SCHEDULER = "cosine"      # "linear", "cosine", "constant"
WARMUP_RATIO = 0.03
WEIGHT_DECAY = 0.01
NUM_TRAIN_EPOCHS = 1
MAX_STEPS = 200               # overrides epochs if set; -1 to disable
PER_DEVICE_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 4
MAX_SEQ_LENGTH = 2048
GRADIENT_CHECKPOINTING = True
OPTIM = "paged_adamw_8bit"    # "adamw_torch", "paged_adamw_8bit", "paged_adamw_32bit"

# Dataset configuration
# Each entry: (dataset_name, subset, split, weight)
# Weight controls mixing ratio when multiple datasets are used
DATASET_CONFIG = [
    ("teknium/OpenHermes-2.5", None, "train", 1.0),
]
DATASET_SAMPLE_SIZE = 10000   # total samples across all datasets (None = use all)

# Time budget (seconds) — hard limit for training wall-clock time
TIME_BUDGET = 1200  # 20 minutes

# ---------------------------------------------------------------------------
# Prompt formatting (agent can modify this function)
# ---------------------------------------------------------------------------

def format_prompt(example: dict) -> str:
    """
    Format a dataset example into a training prompt string.
    The agent can modify this to change prompt templates.

    Supports common dataset formats:
    - conversations: [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]
    - instruction/input/output format
    - messages: [{"role": "user", "content": "..."}, ...]
    """
    # ChatML format (compatible with Qwen models)
    if "conversations" in example and example["conversations"]:
        parts = []
        for turn in example["conversations"]:
            role = turn.get("from", turn.get("role", "user"))
            content = turn.get("value", turn.get("content", ""))
            if role in ("human", "user"):
                parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role in ("gpt", "assistant"):
                parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "system":
                parts.append(f"<|im_start|>system\n{content}<|im_end|>")
        return "\n".join(parts)

    if "messages" in example and example["messages"]:
        parts = []
        for msg in example["messages"]:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        return "\n".join(parts)

    # Instruction/input/output format
    instruction = example.get("instruction", "")
    inp = example.get("input", "")
    output = example.get("output", "")

    if inp:
        user_content = f"{instruction}\n\n{inp}"
    else:
        user_content = instruction

    return (
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{output}<|im_end|>"
    )


# ---------------------------------------------------------------------------
# Dataset loading and mixing
# ---------------------------------------------------------------------------

def load_and_mix_datasets() -> list[str]:
    """Load datasets according to DATASET_CONFIG and return formatted texts."""
    all_texts = []

    total_weight = sum(w for _, _, _, w in DATASET_CONFIG)

    for ds_name, subset, split, weight in DATASET_CONFIG:
        print(f"[finetune] Loading {ds_name} (subset={subset}, split={split}, weight={weight})")
        if subset:
            ds = load_dataset(ds_name, subset, split=split)
        else:
            ds = load_dataset(ds_name, split=split)

        # Calculate how many samples from this dataset
        if DATASET_SAMPLE_SIZE is not None:
            n_samples = int(DATASET_SAMPLE_SIZE * (weight / total_weight))
            n_samples = min(n_samples, len(ds))
            ds = ds.shuffle(seed=42).select(range(n_samples))

        # Format each example
        texts = []
        for example in ds:
            try:
                text = format_prompt(example)
                if text and len(text.strip()) > 10:
                    texts.append(text)
            except Exception:
                continue

        all_texts.extend(texts)
        print(f"[finetune] Loaded {len(texts)} examples from {ds_name}")

    print(f"[finetune] Total training examples: {len(all_texts)}")
    return all_texts


# ---------------------------------------------------------------------------
# Main training logic
# ---------------------------------------------------------------------------

def train(output_dir: str = "./output/latest"):
    t_start = time.time()

    # Quantization config for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print(f"[finetune] Loading base model: {BASE_MODEL}")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA configuration
    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and format dataset
    train_texts = load_and_mix_datasets()

    # Create dataset from texts
    from datasets import Dataset
    train_dataset = Dataset.from_dict({"text": train_texts})

    # Training arguments
    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LR_SCHEDULER,
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        optim=OPTIM,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        max_length=MAX_SEQ_LENGTH,
        dataset_text_field="text",
        report_to="none",
        seed=42,
    )

    # Time-limited training callback
    class TimeBudgetCallback:
        """Stop training when time budget is exceeded."""
        def __init__(self, budget_seconds):
            self.budget = budget_seconds
            self.start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.start_time
            if elapsed >= self.budget:
                print(f"[finetune] Time budget reached ({elapsed:.0f}s >= {self.budget}s)")
                control.should_training_stop = True

    from transformers import TrainerCallback
    class TimeBudgetTrainerCallback(TrainerCallback):
        def __init__(self, budget_seconds):
            self.budget = budget_seconds
            self.start_time = time.time()

        def on_step_end(self, args, state, control, **kwargs):
            elapsed = time.time() - self.start_time
            if elapsed >= self.budget:
                print(f"[finetune] Time budget reached ({elapsed:.0f}s >= {self.budget}s)")
                control.should_training_stop = True

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        callbacks=[TimeBudgetTrainerCallback(TIME_BUDGET)],
    )

    # Train
    print(f"[finetune] Starting training (budget={TIME_BUDGET}s, max_steps={MAX_STEPS})")
    t_train_start = time.time()
    train_result = trainer.train()
    t_train_end = time.time()
    training_seconds = t_train_end - t_train_start

    # Save adapter
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"[finetune] Adapter saved to {adapter_path}")

    # Merge and save for evaluation
    merged_path = os.path.join(output_dir, "merged")
    print(f"[finetune] Merging adapter weights...")
    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(merged_path)
    tokenizer.save_pretrained(merged_path)
    print(f"[finetune] Merged model saved to {merged_path}")

    # Summary
    t_end = time.time()
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad) if hasattr(model, 'parameters') else 0
    num_steps = train_result.global_step if hasattr(train_result, 'global_step') else MAX_STEPS
    train_loss = train_result.training_loss if hasattr(train_result, 'training_loss') else 0.0

    print("---")
    print(f"training_seconds: {training_seconds:.1f}")
    print(f"total_seconds:    {t_end - t_start:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {num_steps}")
    print(f"train_loss:       {train_loss:.6f}")
    print(f"trainable_params: {num_params_trainable:,}")
    print(f"lora_rank:        {LORA_RANK}")
    print(f"learning_rate:    {LEARNING_RATE}")
    print(f"merged_path:      {merged_path}")

    return {
        "training_seconds": training_seconds,
        "total_seconds": t_end - t_start,
        "peak_vram_mb": peak_vram_mb,
        "num_steps": num_steps,
        "train_loss": train_loss,
        "merged_path": merged_path,
        "adapter_path": adapter_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune LLM with QLoRA")
    parser.add_argument("--output_dir", type=str, default="./output/latest",
                        help="Directory to save adapter and merged model")
    args = parser.parse_args()

    results = train(output_dir=args.output_dir)

    # Save training metadata
    meta_path = os.path.join(args.output_dir, "train_meta.json")
    with open(meta_path, "w") as f:
        json.dump(results, f, indent=2)
