#!/bin/bash
# USM-AA pretraining script
# Standalone implementation without tgnn dependency

# ============ Configuration ============
HOST_GPU_NUM=2

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src"

# Config file path
CONFIG_FILE="${SCRIPT_DIR}/yaml/config.yaml"

# Pretrained weights path (optional, leave empty to train from scratch)
PRETRAINED_WEIGHTS="${SCRIPT_DIR}/weight/model.safetensors"

# ============ Training Command ============

# Build command arguments
CMD_ARGS="--config ${CONFIG_FILE}"

# Add pretrained weights if exists
if [ -n "${PRETRAINED_WEIGHTS}" ] && [ -f "${PRETRAINED_WEIGHTS}" ]; then
    CMD_ARGS="${CMD_ARGS} --pretrained ${PRETRAINED_WEIGHTS}"
    echo "Loading pretrained weights from: ${PRETRAINED_WEIGHTS}"
else
    echo "No pretrained weights, training from scratch"
fi

# Multi-GPU training
torchrun \
    --standalone \
    --nproc_per_node=${HOST_GPU_NUM} \
    trainer.py \
    ${CMD_ARGS}


