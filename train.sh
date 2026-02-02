#!/bin/bash
# USM-AA Training Script (DeepSpeed)

# ============ Configuration ============

#8 for pretrain and 4 for finetune
HOST_GPU_NUM=1

# Choose config: usmaa_100m_pretrain (pretrain) or usmaa_100m_finetune (finetune)
CONFIG_NAME="usmaa_100m_pretrain"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src"

CONFIG_FILE="${SCRIPT_DIR}/config/${CONFIG_NAME}.yaml"

# ============ DeepSpeed Training ============
deepspeed \
    --num_gpus=${HOST_GPU_NUM} \
    trainer.py \
    --mode train \
    --config ${CONFIG_FILE}

