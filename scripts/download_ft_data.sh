#!/bin/bash
# Download finetune datasets for USM-AA
# Usage: bash download_finetunedata.sh [gb|gue|nt|all]

set -e

# ============ Configuration ============

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Base data directory (absolute path)
DATA_DIR="${PROJECT_ROOT}/data"

# [gb] Genomic Benchmarks dataset
GB_DATA_DIR="${DATA_DIR}/genomic_benchmarks"

# [gue] GUE dataset
GUE_DATA_DIR="${DATA_DIR}/GUE"

# [nt] Nucleotide Transformer benchmark (requires manual download)
NT_DATA_DIR="${DATA_DIR}/nt_benchmark"

# =======================================

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

download_gb() {
    echo ""
    echo "=========================================="
    echo "  Downloading Genomic Benchmarks Dataset"
    echo "=========================================="
    echo "Output: ${GB_DATA_DIR}"
    echo ""
    
    python "${SCRIPT_DIR}/prepare_gb.py" \
        --data_dir "${GB_DATA_DIR}" \
        --ref "" \
        --mode download
    
    echo "Genomic Benchmarks downloaded!"
}

download_gue() {
    echo ""
    echo "=========================================="
    echo "  Downloading GUE Dataset"
    echo "=========================================="
    echo "Output: ${GUE_DATA_DIR}"
    echo ""
    
    mkdir -p "${DATA_DIR}"
    cd "${DATA_DIR}"
    
    # Download from Google Drive using gdown
    FILE_ID="1uOrwlf07qGQuruXqGXWMpPn8avBoW7T-"
    echo "Downloading GUE_v2.zip from Google Drive..."
    pip install gdown -q
    gdown "${FILE_ID}"
    
    echo "Extracting GUE_v2.zip..."
    unzip -o GUE_v2.zip
    rm -f GUE_v2.zip
    
    echo "GUE dataset downloaded!"
}

download_nt() {
    echo ""
    echo "=========================================="
    echo "  Downloading NT Benchmark Dataset"
    echo "=========================================="
    echo "Output: ${NT_DATA_DIR}"
    echo ""
    
    python "${SCRIPT_DIR}/prepare_nt.py" \
        --data_dir "${NT_DATA_DIR}" \
        --mode download
    
    echo "NT Benchmark downloaded!"
}

show_help() {
    echo "USM-AA Finetune Data Download Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  gb    Download Genomic Benchmarks from HuggingFace"
    echo "  gue   Show GUE dataset download instructions"
    echo "  nt    Show NT benchmark download instructions"
    echo "  all   Download/show all datasets"
    echo "  help  Show this help message"
    echo ""
}

case "${1:-help}" in
    gb)
        download_gb
        ;;
    gue)
        download_gue
        ;;
    nt)
        download_nt
        ;;
    all)
        download_gb
        download_gue
        download_nt
        echo ""
        echo "=========================================="
        echo "  Download Complete!"
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
