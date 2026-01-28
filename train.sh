#!/bin/bash
# USM-AA pretraining script - DeepSpeed version

# ============ Configuration ============
HOST_GPU_NUM=8

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src"

# Config file path (DeepSpeed config is included in yaml)
CONFIG_FILE="${SCRIPT_DIR}/yaml/config.yaml"

# ============ DeepSpeed Training ============
deepspeed \
    --num_gpus=${HOST_GPU_NUM} \
    trainer.py \
    --config ${CONFIG_FILE}

