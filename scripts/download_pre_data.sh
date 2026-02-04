#!/bin/bash
# Download pretrain datasets for USM-AA
# Usage: bash download_pre_data.sh [ref|1kg|all]

set -e

# ============ Configuration ============

# Base data directory (relative to scripts/)
DATA_DIR="../data"

# [ref] Reference genome
REF_URL="https://ftp.ncbi.nlm.nih.gov/genomes/all/GCA/000/001/405/GCA_000001405.15_GRCh38/seqs_for_alignment_pipelines.ucsc_ids/GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz"
REF_DATA_DIR="${DATA_DIR}/reference"

# [1kg] 1000 Genomes Project
KG_BASE_URL="http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/working/20220422_3202_phased_SNV_INDEL_SV"
KG_DATA_DIR="${DATA_DIR}/1kg"

# [dna] Multi-species DNA (DNABERT-2 pretrain data)
DNA_FILE_ID="1dSXJfwGpDSJ59ry9KAp8SugQLK35V83f"
DNA_DATA_DIR="${DATA_DIR}/dna"

# =======================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

download_ref() {
    echo ""
    echo "=========================================="
    echo "  Downloading Reference Genome (GRCh38)"
    echo "=========================================="
    echo "Output: ${REF_DATA_DIR}"
    echo ""
    
    mkdir -p "${REF_DATA_DIR}"
    cd "${REF_DATA_DIR}"
    
    if [ -f "GCA_000001405.15_GRCh38_no_alt_analysis_set.fna" ]; then
        echo "Reference genome already exists, skipping download."
    else
        echo "Downloading GRCh38 reference genome..."
        wget -c "${REF_URL}" -O GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
        
        echo "Extracting..."
        gunzip -k GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
        rm -f GCA_000001405.15_GRCh38_no_alt_analysis_set.fna.gz
    fi
    
    echo "Reference genome downloaded!"
}

download_1kg() {
    echo ""
    echo "=========================================="
    echo "  Downloading 1000 Genomes VCF Files"
    echo "=========================================="
    echo "Output: ${KG_DATA_DIR}"
    echo ""
    
    mkdir -p "${KG_DATA_DIR}"
    cd "${KG_DATA_DIR}"
    
    # Download chromosome VCF files (chr1-22, chrX)
    for chr in {1..22} X; do
        vcf_file="1kGP_high_coverage_Illumina.chr${chr}.filtered.SNV_INDEL_SV_phased_panel.vcf.gz"
        if [ -f "${vcf_file}" ]; then
            echo "chr${chr} already exists, skipping..."
            continue
        fi
        echo "Downloading chr${chr}..."
        wget -c "${KG_BASE_URL}/${vcf_file}" -O "${vcf_file}"
        wget -c "${KG_BASE_URL}/${vcf_file}.tbi" -O "${vcf_file}.tbi"
    done
    
    echo "1000 Genomes VCF files downloaded!"
}

download_dna() {
    echo ""
    echo "=========================================="
    echo "  Downloading Multi-species DNA Data"
    echo "=========================================="
    echo "Output: ${DNA_DATA_DIR}"
    echo ""
    
    mkdir -p "${DNA_DATA_DIR}"
    cd "${DNA_DATA_DIR}"
    
    if [ -d "dnabert_2_pretrain" ]; then
        echo "DNA data already exists, skipping download."
    else
        echo "Downloading dnabert_2_pretrain.zip from Google Drive..."
        pip install gdown -q
        gdown "${DNA_FILE_ID}"
        
        echo "Extracting..."
        unzip -o dnabert_2_pretrain.zip
        rm -f dnabert_2_pretrain.zip
    fi
    
    echo "Multi-species DNA data downloaded!"
}

show_help() {
    echo "USM-AA Pretrain Data Download Script"
    echo ""
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  ref   Download GRCh38 reference genome from NCBI"
    echo "  1kg   Download 1000 Genomes phased VCF files"
    echo "  dna   Download multi-species DNA data from Google Drive"
    echo "  all   Download all pretrain datasets"
    echo "  help  Show this help message"
    echo ""
}

case "${1:-help}" in
    ref)
        download_ref
        ;;
    1kg)
        download_1kg
        ;;
    dna)
        download_dna
        ;;
    all)
        download_ref
        download_1kg
        download_dna
        echo ""
        echo "=========================================="
        echo "  All Pretrain Data Downloaded!"
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
