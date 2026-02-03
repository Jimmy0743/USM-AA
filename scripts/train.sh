#!/bin/bash
# USM-AA training script - DeepSpeed version
# Usage: bash scripts/train.sh [config.yaml]

# ============ Configuration ============
HOST_GPU_NUM=4

MASTER_PORT=$((RANDOM % 500 + 29500))

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Config file path (use argument or default)
CONFIG_FILE="${1:-${PROJECT_ROOT}/configs/usmaa_100m_finetune.yaml}"

# Resolve relative config path
if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

echo "Using config: ${CONFIG_FILE}"

# ============ DeepSpeed Training ============
deepspeed \
    --num_gpus=${HOST_GPU_NUM} \
    --master_port=${MASTER_PORT} \
    scripts/train_net.py \
    --config ${CONFIG_FILE}

