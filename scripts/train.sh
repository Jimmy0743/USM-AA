#!/bin/bash
# USM-AA training script - DeepSpeed version
# Usage: bash scripts/train.sh [config.yaml] [--eval] [--resume]

# 8 GPUs for pretrain / 4 GPUs for finetune
HOST_GPU_NUM=4
MASTER_PORT=$((RANDOM % 500 + 29500))

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Parse arguments
CONFIG_FILE=""
EXTRA_ARGS=""

for arg in "$@"; do
    if [[ "${arg}" == --* ]]; then
        # Collect extra arguments (--eval, --resume, etc.)
        EXTRA_ARGS="${EXTRA_ARGS} ${arg}"
    elif [[ -z "${CONFIG_FILE}" ]]; then
        # First non-flag argument is config file
        CONFIG_FILE="${arg}"
    fi
done

# Default config if not specified
CONFIG_FILE="${CONFIG_FILE:-${PROJECT_ROOT}/configs/usmaa_100m_finetune.yaml}"

# Resolve relative config path
if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${PROJECT_ROOT}/${CONFIG_FILE}"
fi

# ============ Logging Setup ============
# Extract output_dir from config file
OUTPUT_DIR=$(grep -E '^\s*output_dir:' "${CONFIG_FILE}" | head -1 | sed 's/.*output_dir:\s*//' | tr -d '"' | tr -d "'")

# Resolve relative output_dir path
if [[ -n "${OUTPUT_DIR}" && "${OUTPUT_DIR}" != /* ]]; then
    CONFIG_DIR="$(dirname "${CONFIG_FILE}")"
    OUTPUT_DIR="$(cd "${CONFIG_DIR}" && cd "$(dirname "${OUTPUT_DIR}")" && pwd)/$(basename "${OUTPUT_DIR}")"
fi

# Create output directory and set log file
if [[ -n "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
    LOG_FILE="${OUTPUT_DIR}/train.log"
else
    LOG_FILE=""
fi


# ============ Print Info ============
echo "=========================================="
echo "  USM-AA Training"
echo "=========================================="
echo "Config:     ${CONFIG_FILE}"
echo "Output:     ${OUTPUT_DIR:-N/A}"
echo "Log file:   ${LOG_FILE:-N/A}"
echo "GPUs:       ${HOST_GPU_NUM}"
echo "Extra args: ${EXTRA_ARGS:-None}"
echo "=========================================="

# ============ DeepSpeed Training ============
if [[ -n "${LOG_FILE}" ]]; then
    deepspeed \
        --num_gpus=${HOST_GPU_NUM} \
        --master_port=${MASTER_PORT} \
        ${PROJECT_ROOT}/scripts/train_net.py \
        --config ${CONFIG_FILE} \
        ${EXTRA_ARGS} \
        2>&1 | tee -a "${LOG_FILE}"
else
    deepspeed \
        --num_gpus=${HOST_GPU_NUM} \
        --master_port=${MASTER_PORT} \
        ${PROJECT_ROOT}/scripts/train_net.py \
        --config ${CONFIG_FILE} \
        ${EXTRA_ARGS}
fi

