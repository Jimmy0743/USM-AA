#!/bin/bash
# USM-AA pretraining script - DeepSpeed version

# ============ Configuration ============
HOST_GPU_NUM=8

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/src"

# Config file paths
CONFIG_FILE="${SCRIPT_DIR}/yaml/config_deepspeed.yaml"
DS_CONFIG="${SCRIPT_DIR}/yaml/ds_config.json"

# ============ DeepSpeed Training ============
deepspeed \
    --num_gpus=${HOST_GPU_NUM} \
    trainer_deepspeed.py \
    --config ${CONFIG_FILE} \
    --deepspeed ${DS_CONFIG}

# Optional: specify master_port to avoid conflicts
# deepspeed \
#     --num_gpus=${HOST_GPU_NUM} \
#     --master_port 29501 \
#     trainer_deepspeed.py \
#     --config ${CONFIG_FILE} \
#     --deepspeed ${DS_CONFIG}

# Optional: override config with command line arguments
# deepspeed \
#     --num_gpus=${HOST_GPU_NUM} \
#     trainer_deepspeed.py \
#     --config ${CONFIG_FILE} \
#     --deepspeed ${DS_CONFIG} \
#     --batch_size 160 \
#     --model_type usm-aa-1b
