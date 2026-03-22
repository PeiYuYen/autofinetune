# AutoFinetune — Docker image for autonomous QLoRA finetuning
# Requirements: NVIDIA GPU ≥24GB VRAM, CUDA 12.x driver, nvidia-container-toolkit

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/workspace/.cache/huggingface

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN pip install --upgrade pip

# PyTorch (CUDA 12.8)
RUN pip install torch==2.9.1+cu128 triton==3.5.1 \
    --index-url https://download.pytorch.org/whl/cu128

# AutoFinetune core
RUN pip install \
    transformers==5.3.0 \
    peft==0.18.1 \
    trl==0.29.1 \
    accelerate==1.13.0 \
    bitsandbytes==0.49.2 \
    datasets==4.8.3 \
    sentencepiece==0.2.1 \
    tokenizers==0.22.2 \
    safetensors==0.7.0 \
    "huggingface-hub>=0.27.0"

# Evaluation
RUN pip install \
    lm-eval==0.4.11 \
    evalplus==0.3.1

# HPO + utilities
RUN pip install \
    optuna==4.8.0 \
    numpy==2.2.6 \
    pandas==2.3.3 \
    matplotlib==3.10.8 \
    pyarrow==23.0.1 \
    requests==2.32.5

WORKDIR /workspace

# Copy project files
COPY finetune.py eval.py orchestrate.py optuna_runner.py program_finetune.md ./
COPY USER_GUIDELINE.md README.md environment.yml ./

# Mount points:
#   /workspace/output     — training output (adapters)
#   /workspace/.cache     — HuggingFace model/dataset cache
VOLUME ["/workspace/output", "/workspace/.cache"]

# Default: run baseline + agent loop
CMD ["python", "orchestrate.py", "agent", "--fast"]
