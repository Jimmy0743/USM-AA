# Copyright (c) 2025, Tencent Inc. All rights reserved.
# USM-AA: Alignment-Aware DNA Foundation Model
# Standalone implementation without tgnn dependency

from .usmaa import (
    USMAAConfig,
    USMAAModel,
    USMAAForSequenceClassification,
    USMAAPreTrainedModel,
    RMSNorm,
    TransformerBlock,
    MultiHeadAttention,
    MLP,
    AttentionPooling,
    precompute_freqs_cis,
)

from .tokenizer import DNAKmerTokenizer, build_tokenizer, reverse_complement

from .dataset import (
    CIGAR_OPS,
    CHROM2ID,
    CHROM_LENGTHS,
    print_rank_0,
    collate_dense_tensors,
    get_random_mask,
    MMapIndexedJsonlDataset,
    MMapIndexedFastaDataset,
    PretrainDataset,
    Discrete1KGPretrainDataset,
    DNAPretrainDataset,
    RefseqPretrainDataset,
    USMAAPretrainDataset,
)

__all__ = [
    # Model
    "USMAAConfig",
    "USMAAModel",
    "USMAAForSequenceClassification",
    "USMAAPreTrainedModel",
    "precompute_freqs_cis",
    # Tokenizer
    "DNAKmerTokenizer",
    "build_tokenizer",
    "reverse_complement",
    # Constants
    "CIGAR_OPS",
    "CHROM2ID",
    "CHROM_LENGTHS",
    # Utils
    "print_rank_0",
    "collate_dense_tensors",
    "get_random_mask",
    # Dataset
    "MMapIndexedJsonlDataset",
    "MMapIndexedFastaDataset",
    "PretrainDataset",
    "Discrete1KGPretrainDataset",
    "DNAPretrainDataset",
    "RefseqPretrainDataset",
    "USMAAPretrainDataset",
]
