#!/bin/bash
# Process finetune datasets for USM-AA (add alignment info)
# Usage: bash prepare_finetunedata.sh [gb|gue|nt|all]

set -e

# ============ Configuration ============

# Reference genome (required for alignment)
REF_FILE="../data/reference/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna"

# Base data directory (relative to scripts/)
DATA_DIR="../data"

# [gb] Genomic Benchmarks dataset
GB_DATA_DIR="${DATA_DIR}/genomic_benchmarks"
GB_BEST_N=10         # Number of best alignments to keep
GB_N_FOLDS=10        # Number of cross-validation folds
GB_SEED=42           # Random seed for CV splits

# [gue] GUE dataset
GUE_DATA_DIR="${DATA_DIR}/GUE"
GUE_BEST_N=10        # Number of best alignments to keep

# [nt] Nucleotide Transformer benchmark
NT_DATA_DIR="${DATA_DIR}/nt_benchmark"
NT_BEST_N=10         # Number of best alignments to keep
NT_N_FOLDS=10        # Number of cross-validation folds
NT_SEED=42           # Random seed for CV splits

# =======================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

prepare_gb() {
    echo ""
    echo "=========================================="
    echo "  Processing Genomic Benchmarks Dataset"
    echo "=========================================="
    echo "Data dir: ${GB_DATA_DIR}"
    echo "Ref file: ${REF_FILE}"
    echo ""
    
    echo "[1/2] Adding alignment information..."
    python "${SCRIPT_DIR}/prepare_gb.py" \
        --data_dir "${GB_DATA_DIR}" \
        --ref "${REF_FILE}" \
        --best_n "${GB_BEST_N}" \
        --mode prepare
    
    echo "[2/2] Creating cross-validation splits..."
    python "${SCRIPT_DIR}/prepare_gb.py" \
        --data_dir "${GB_DATA_DIR}" \
        --ref "${REF_FILE}" \
        --best_n "${GB_BEST_N}" \
        --n_folds "${GB_N_FOLDS}" \
        --seed "${GB_SEED}" \
        --mode cv
    
    echo "Genomic Benchmarks processed!"
}

prepare_gue() {
    echo ""
    echo "=========================================="
    echo "  Processing GUE Dataset"
    echo "=========================================="
    echo "Data dir: ${GUE_DATA_DIR}"
    echo "Ref file: ${REF_FILE}"
    echo ""
    
    echo "[1/2] Processing GUE tasks..."
    python "${SCRIPT_DIR}/prepare_gue.py" \
        --data_dir "${GUE_DATA_DIR}" \
        --ref "${REF_FILE}" \
        --best_n "${GUE_BEST_N}" \
        --mode gue
    
    echo "[2/2] Processing EPI tasks..."
    python "${SCRIPT_DIR}/prepare_gue.py" \
        --data_dir "${GUE_DATA_DIR}" \
        --ref "${REF_FILE}" \
        --best_n "${GUE_BEST_N}" \
        --mode epi
    
    echo "GUE dataset processed!"
}

prepare_nt() {
    echo ""
    echo "=========================================="
    echo "  Processing NT Benchmark Dataset"
    echo "=========================================="
    echo "Data dir: ${NT_DATA_DIR}"
    echo "Ref file: ${REF_FILE}"
    echo ""
    
    echo "[1/2] Adding alignment information..."
    python "${SCRIPT_DIR}/prepare_nt.py" \
        --data_dir "${NT_DATA_DIR}" \
        --ref "${REF_FILE}" \
        --best_n "${NT_BEST_N}" \
        --mode prepare
    
    echo "[2/2] Creating cross-validation splits..."
    python "${SCRIPT_DIR}/prepare_nt.py" \
        --data_dir "${NT_DATA_DIR}" \
        --ref "${REF_FILE}" \
        --best_n "${NT_BEST_N}" \
        --n_folds "${NT_N_FOLDS}" \
        --seed "${NT_SEED}" \
        --mode cv
    
    echo "NT Benchmark processed!"
}

analyze_gb() {
    python "${SCRIPT_DIR}/prepare_gb.py" \
        --data_dir "${GB_DATA_DIR}" --ref "${REF_FILE}" --mode analyze
}

analyze_gue() {
    python "${SCRIPT_DIR}/prepare_gue.py" \
        --data_dir "${GUE_DATA_DIR}" --ref "${REF_FILE}" --mode analyze
}

analyze_nt() {
    python "${SCRIPT_DIR}/prepare_nt.py" \
        --data_dir "${NT_DATA_DIR}" --ref "${REF_FILE}" --mode analyze
}

show_help() {
    echo "USM-AA Finetune Data Processing Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  gb          Process Genomic Benchmarks dataset"
    echo "  gue         Process GUE dataset"
    echo "  nt          Process NT benchmark dataset"
    echo "  all         Process all datasets"
    echo "  gb-analyze  Analyze Genomic Benchmarks"
    echo "  gue-analyze Analyze GUE dataset"
    echo "  nt-analyze  Analyze NT benchmark"
    echo "  analyze     Analyze all datasets"
    echo "  help        Show this help message"
    echo ""
    echo "Note: Run download_finetunedata.sh first to download datasets."
    echo ""
}

case "${1:-help}" in
    gb)
        prepare_gb
        ;;
    gue)
        prepare_gue
        ;;
    nt)
        prepare_nt
        ;;
    all)
        prepare_gb
        prepare_gue
        prepare_nt
        echo ""
        echo "=========================================="
        echo "  All Datasets Processed!"
        echo "=========================================="
        ;;
    gb-analyze)
        analyze_gb
        ;;
    gue-analyze)
        analyze_gue
        ;;
    nt-analyze)
        analyze_nt
        ;;
    analyze)
        analyze_gb
        analyze_gue
        analyze_nt
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
