#!/bin/bash

# USM-AA Embedding Extraction Script

# Configuration
CONFIG="config/usmaa_100m_pretrain.yaml"
CHECKPOINT="path/to/safetensor"
DEVICE="cuda"

# Example DNA sequence
SEQUENCE="ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"

# ============ Alignment Information ============
# Each 6mer token needs corresponding alignment info
# Format: comma-separated values, one per token (including BOS/EOS)
# 
# For a sequence of length 60bp:
#   - Tokenized: [BOS] + 10 tokens (60/6=10) + [EOS] = 12 tokens total
#   - Each array should have 12 values
#
# chr_ids:      Chromosome ID per token (1-22, 23=X, 24=Y, 0=unknown)
# positions:    Genomic position (bp) per token
# strand_ids:   Strand per token (0=forward, 1=reverse)
# cigar_ids:    CIGAR operation per token
#               CIGAR_OPS = ["M", "I", "D", "N", "S", "H", "P", "=", "X", "B"]
#               0=M (match), 7== (equal/match), default should be 7
#
# Example for chr1, positions 1000-1054, forward strand:
CHR_IDS="1,1,1,1,1,1,1,1,1,1,1,1"
POSITIONS="0,1000,1006,1012,1018,1024,1030,1036,1042,1048,1054,248956422"
STRAND_IDS="0,0,0,0,0,0,0,0,0,0,0,0"
CIGAR_IDS="7,7,7,7,7,7,7,7,7,7,7,7"

# Run embedding extraction
cd "$(dirname "$0")"

python3 src/embedding.py \
        --config "$CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --device "$DEVICE" \
        --sequence "$SEQUENCE" \
        --chr_ids "$CHR_IDS" \
        --positions "$POSITIONS" \
        --strand_ids "$STRAND_IDS" \
        --cigar_ids "$CIGAR_IDS"

