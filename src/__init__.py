# USM-AA: Alignment-Aware DNA Foundation Model
from .model.modeling_usmaa import (
    USMAAModel,
    USMAAForSequenceClassification,
    USMAAPreTrainedModel,
    precompute_freqs_cis,
)
from .model.configuration_usmaa import USMAAConfig

from .tokenizer import build_tokenizer
from .tokenizer.kmer_tokenizer import DNAKmerTokenizer

from .data.dataset import (
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
from .data.dataset.pretrain_dataset import CIGAR_OPS, CHROM2ID, CHROM_LENGTHS, print_rank_0

# Evaluator
from .evaluator import ClassificationEvaluator

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
    # Evaluator
    "ClassificationEvaluator",
]
