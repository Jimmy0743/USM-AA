#!/bin/bash
# Prepare pretrain datasets for USM-AA (process downloaded data)
# Usage: bash prepare_pre_data.sh [ref|1kg|dna|all]

set -e

# ============ Configuration ============

# Reference genome (required for alignment)
REF_FILE="../data/reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"

# Base data directory (relative to scripts/)
DATA_DIR="../data"

# [ref] Reference genome preparation
REF_INPUT="${DATA_DIR}/reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"
REF_OUTPUT_DIR="${DATA_DIR}"

# [dna] DNA alignment data preparation
DNA_INPUT="${DATA_DIR}/dna/train.txt"
DNA_OUTPUT_DIR="${DATA_DIR}/dna"
DNA_PRESET="sr"      # sr: short read, map-ont: ONT, map-hifi: PacBio HiFi
DNA_BEST_N=10        # Number of best alignments to keep

# [1kg] 1000 Genomes data preparation
KG_VCF_DIR="${DATA_DIR}/1kg"
KG_OUTPUT_DIR="${DATA_DIR}"
KG_CHUNK_SIZE=4      # Number of samples per task
KG_BATCH_SIZE=512    # Variants per document

# =======================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Function to resolve relative path
resolve_path() {
    local path="$1"
    if [[ "${path}" != /* ]]; then
        path="$(cd "${SCRIPT_DIR}" && mkdir -p "${path}" 2>/dev/null; cd "${path}" && pwd)"
    fi
    echo "${path}"
}

prepare_ref() {
    local output_dir
    output_dir=$(resolve_path "${REF_OUTPUT_DIR}")
    
    echo ""
    echo "=========================================="
    echo "  Preparing Reference Genome"
    echo "=========================================="
    echo "Input:  ${REF_INPUT}"
    echo "Output: ${output_dir}/ref.fasta"
    echo ""
    
    python "${SCRIPT_DIR}/prepare_ref.py" \
        --input "${REF_INPUT}" \
        --output_dir "${output_dir}"
    
    echo "Reference genome prepared!"
}

prepare_dna() {
    local output_dir
    local input_file
    output_dir=$(resolve_path "${DNA_OUTPUT_DIR}")
    input_file="${SCRIPT_DIR}/${DNA_INPUT}"
    
    echo ""
    echo "=========================================="
    echo "  Preparing DNA Alignment Data"
    echo "=========================================="
    echo "Input:  ${input_file}"
    echo "Ref:    ${REF_FILE}"
    echo "Output: ${output_dir}/dna.jsonl"
    echo "Preset: ${DNA_PRESET}, Best N: ${DNA_BEST_N}"
    echo ""
    
    python "${SCRIPT_DIR}/prepare_dna.py" \
        --input "${input_file}" \
        --output "${output_dir}/dna.jsonl" \
        --ref "${REF_FILE}" \
        --preset "${DNA_PRESET}" \
        --best_n "${DNA_BEST_N}"
    
    echo "DNA alignment data prepared!"
}

prepare_1kg() {
    local output_dir
    output_dir=$(resolve_path "${KG_OUTPUT_DIR}")
    
    echo ""
    echo "=========================================="
    echo "  Preparing 1000 Genomes Data"
    echo "=========================================="
    echo "Input:  ${KG_VCF_DIR}"
    echo "Output: ${output_dir}/1kg.jsonl"
    echo "Chunk:  ${KG_CHUNK_SIZE}, Batch: ${KG_BATCH_SIZE}"
    echo ""
    
    python "${SCRIPT_DIR}/prepare_1kg.py" \
        --vcf_dir "${KG_VCF_DIR}" \
        --output_dir "${output_dir}" \
        --chunk_size "${KG_CHUNK_SIZE}" \
        --batch_size "${KG_BATCH_SIZE}"
    
    echo "1000 Genomes data prepared!"
}

show_help() {
    echo "USM-AA Pretrain Data Preparation Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  ref   Prepare reference genome (GRCh38)"
    echo "  dna   Prepare DNA alignment data"
    echo "  1kg   Prepare 1000 Genomes variant data"
    echo "  all   Prepare all datasets"
    echo "  help  Show this help message"
    echo ""
    echo "Note: Run download_pre_data.sh first to download datasets."
    echo ""
}

case "${1:-help}" in
    ref)
        prepare_ref
        ;;
    dna)
        prepare_dna
        ;;
    1kg)
        prepare_1kg
        ;;
    all)
        prepare_ref
        prepare_dna
        prepare_1kg
        echo ""
        echo "=========================================="
        echo "  All Pretrain Data Prepared!"
        echo "=========================================="
        ;;
    help|--help|-h)
        show_help
        exit 0
        ;;
    *)
        echo "Unknown command: $1"
        show_help
        exit 1
        ;;
esac

echo ""
echo "Done!"
