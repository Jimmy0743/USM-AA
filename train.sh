#!/bin/bash
# USM-AA pretraining script - DeepSpeed version

# ============ Configuration ============
HOST_GPU_NUM=1

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Add project root to PYTHONPATH
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Config file path (DeepSpeed config is included in yaml)
CONFIG_FILE="${SCRIPT_DIR}/configs/usm_aa_100m_pretrain.yaml"

# ============ DeepSpeed Training ============
deepspeed \
    --num_gpus=${HOST_GPU_NUM} \
    scripts/train_net.py \
    --config ${CONFIG_FILE}

