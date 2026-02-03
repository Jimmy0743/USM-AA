#!/bin/bash
# Data preparation script for USM-AA pretraining
# Usage: bash prepare_data.sh [ref|dna|1kg|all]

set -e

# ============ Configuration ============

# [ref] Reference genome preparation
REF_INPUT="/path/to/grch38.fasta"
REF_OUTPUT_DIR="../data"

# [dna] DNA alignment data preparation
DNA_INPUT="/path/to/dna_sequences.txt"
DNA_REF="/path/to/grch38.fasta"
DNA_OUTPUT_DIR="../data"
DNA_PRESET="sr"      # sr: short read, map-ont: ONT, map-hifi: PacBio HiFi
DNA_BEST_N=10        # Number of best alignments to keep

# [1kg] 1000 Genomes data preparation
KG_VCF_DIR="/path/to/1kgenomes/vcf"
KG_OUTPUT_DIR="../data"
KG_CHUNK_SIZE=4      # Number of samples per task
KG_BATCH_SIZE=512    # Variants per document

# =======================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
    echo "  📦 Preparing Reference Genome"
    echo "=========================================="
    echo "Input:  ${REF_INPUT}"
    echo "Output: ${output_dir}/ref.fasta"
    echo ""
    
    python "${SCRIPT_DIR}/prepare_ref.py" \
        --input "${REF_INPUT}" \
        --output_dir "${output_dir}"
    
    echo "✅ Reference genome prepared!"
}

prepare_dna() {
    local output_dir
    output_dir=$(resolve_path "${DNA_OUTPUT_DIR}")
    
    echo ""
    echo "=========================================="
    echo "  🧬 Preparing DNA Alignment Data"
    echo "=========================================="
    echo "Input:  ${DNA_INPUT}"
    echo "Ref:    ${DNA_REF}"
    echo "Output: ${output_dir}/dna.jsonl"
    echo "Preset: ${DNA_PRESET}, Best N: ${DNA_BEST_N}"
    echo ""
    
    python "${SCRIPT_DIR}/prepare_dna.py" \
        --input "${DNA_INPUT}" \
        --output "${output_dir}/dna.jsonl" \
        --ref "${DNA_REF}" \
        --preset "${DNA_PRESET}" \
        --best_n "${DNA_BEST_N}"
    
    echo "✅ DNA alignment data prepared!"
}

prepare_1kg() {
    local output_dir
    output_dir=$(resolve_path "${KG_OUTPUT_DIR}")
    
    echo ""
    echo "=========================================="
    echo "  🌍 Preparing 1000 Genomes Data"
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
    
    echo "✅ 1000 Genomes data prepared!"
}

show_help() {
    echo "USM-AA Data Preparation Script"
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
    echo "Configuration (edit at the top of this script):"
    echo ""
    echo "  [ref] Reference genome:"
    echo "    REF_INPUT        Input FASTA file"
    echo "    REF_OUTPUT_DIR   Output directory"
    echo ""
    echo "  [dna] DNA alignment:"
    echo "    DNA_INPUT        Input sequences file"
    echo "    DNA_REF          Reference genome for alignment"
    echo "    DNA_OUTPUT_DIR   Output directory"
    echo "    DNA_PRESET       Minimap2 preset (sr/map-ont/map-hifi)"
    echo "    DNA_BEST_N       Number of best alignments"
    echo ""
    echo "  [1kg] 1000 Genomes:"
    echo "    KG_VCF_DIR       VCF files directory"
    echo "    KG_OUTPUT_DIR    Output directory"
    echo "    KG_CHUNK_SIZE    Samples per task"
    echo "    KG_BATCH_SIZE    Variants per document"
    echo ""
}

# Parse command line arguments
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
        echo "  ✅ All Data Prepared Successfully!"
        echo "=========================================="
        ;;
    help|--help|-h)
        show_help
        exit 0
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "Done!"
