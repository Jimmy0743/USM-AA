import os
import random
import numpy as np
import pysam
import torch
from torch.utils.data import Dataset

from src.tokenizer import DNAKmerTokenizer
from .jsonl_dataset import MMapIndexedJsonlDataset
from .utils import collate_dense_tensors
from .dense_dataset import CIGAR_OPS, reverse_complement, print_rank_0
from .denseaa_dataset import DenseAADataset, CHROM2ID, CHROM_LENGTHS


class NonContiguousDataset(DenseAADataset):
    """Non-Contiguous (Sparse) Dataset - sparse sampling for matched regions.
    
    This dataset uses sparse sampling for MATCH/EQUAL regions in alignments,
    which can significantly reduce sequence length while preserving important
    information like insertions, deletions, and mismatches.
    
    Useful for:
    - Long sequences that exceed max_len
    - Tasks where sparse representation is beneficial
    - Reducing computational cost while maintaining accuracy
    
    Data format (JSONL): Same as DenseAADataset
    
    For cross-validation (with fold_id):
        {"sequence": "...", "label": 0, "alignments": [...], "fold_id": 0}
    """

    def __init__(
        self,
        filename: str,
        tokenizer: DNAKmerTokenizer,
        split: str = "train",
        max_len: int = 1024,
        chr_id: int = 0,
        rc_aug: bool = True,
        sparse_k: int = 4,
        dense_prob: float = 0.4,
        use_alignment_prob: float = 0.8,
        cv_val_folds: list = None
    ):
        """
        Args:
            filename: Path to JSONL file
            tokenizer: DNAKmerTokenizer instance
            split: Dataset split (train/val/test)
            max_len: Maximum sequence length
            chr_id: Default chromosome ID for sequences without alignment
            rc_aug: Whether to use reverse complement augmentation
            sparse_k: Sampling interval for MATCH regions (keep every k-th token)
            dense_prob: Probability of using dense (vs sparse) representation during training
            use_alignment_prob: Probability of using alignment info during training
            cv_val_folds: List of fold IDs to use as validation set (for CV mode)
        """
        super().__init__(filename, tokenizer, split, max_len, chr_id, rc_aug, use_alignment_prob, cv_val_folds)
        self.sparse_k = sparse_k
        self.dense_prob = dense_prob

        # Statistics on sparse vs dense lengths
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            sparse_lens, dense_lens = [], []
            count = 0
            for i in range(len(self)):
                data = self._get_item_by_idx(i)
                if data.get("alignments"):
                    inputs_sparse = self.get_sparse_aa_inputs(data["alignments"][0])
                    inputs_dense = self.get_dense_aa_inputs(data["alignments"][0])
                    sparse_lens.append(len(inputs_sparse[0]))
                    dense_lens.append(len(inputs_dense[0]))
                    count += 1
                    if count >= 100:
                        break
            if sparse_lens:
                print(f"NonContiguousDataset: sparse_k={sparse_k}, "
                      f"avg sparse: {np.mean(sparse_lens):.1f}, "
                      f"avg dense: {np.mean(dense_lens):.1f}, "
                      f"compression: {np.mean(sparse_lens)/np.mean(dense_lens):.2f}")

    def get_sparse_aa_inputs(self, alignment: dict):
        """Get sparse AA input tensors from alignment.
        
        Sparse sampling strategy:
        - MATCH/EQUAL regions: keep every k-th token + start/end tokens
        - INSERTION regions: keep all tokens (important for variants)
        - DELETION regions: keep start and end tokens
        - Other operations: keep all tokens
        
        Args:
            alignment: Alignment dict with keys [chrom, strand, cigars, positions]
            
        Returns:
            List of tensors: [token_ids, strand_ids, chr_ids, position_ids, cigar_ids]
        """
        chrom = alignment["chrom"]
        chr_id = CHROM2ID.get(chrom, 0)
        strand = alignment["strand"]

        assert len(alignment["cigars"]) == len(alignment["positions"]), \
            f"{len(alignment['cigars'])} != {len(alignment['positions'])}"

        token_ids = [self.tokenizer.bos]
        cigar_ids = [self.cigar_equal_id]
        positions = [0]

        # Add start motif for match/equal
        if alignment["cigars"][0][1] in (pysam.CEQUAL, pysam.CMATCH):
            length, op, motif = alignment["cigars"][0]
            pos = alignment["positions"][0]
            kmer = min(length, self.kmer)
            token_ids.append(self.tokenizer.piece_to_id(motif[:kmer]))
            positions.append(pos)
            cigar_ids.append(op)

        k = self.sparse_k
        for cigar, pos in zip(alignment["cigars"], alignment["positions"]):
            length, op, motif = cigar

            if op in (pysam.CEQUAL, pysam.CMATCH):
                # Sparse sampling for match regions
                if k is None or k < 1:
                    continue
                motif_tokens = self.tokenizer(motif, to_tensor=False)
                for i, t in enumerate(motif_tokens):
                    if i % k == 0:
                        token_ids.append(t)
                        positions.append(pos + i * self.kmer)
                        cigar_ids.append(op)

            elif op in (pysam.CDEL,):
                # For deletions, add start and end motifs
                kmer = min(length, self.kmer)
                token_ids.append(self.tokenizer.piece_to_id(motif[:kmer]))
                positions.append(pos)
                cigar_ids.append(op)
                token_ids.append(self.tokenizer.piece_to_id(motif[-kmer:]))
                positions.append(pos + length - kmer)
                cigar_ids.append(op)

            else:
                # For other operations (INS, etc.), add all tokens
                motif_tokens = self.tokenizer(motif, to_tensor=False)
                for i, t in enumerate(motif_tokens):
                    token_ids.append(t)
                    positions.append(pos + i * self.kmer)
                    cigar_ids.append(op)

        # Add end motif for match/equal
        if alignment["cigars"][-1][1] in (pysam.CEQUAL, pysam.CMATCH):
            length, op, motif = alignment["cigars"][-1]
            pos = alignment["positions"][-1]
            kmer = min(length, self.kmer)
            token_ids.append(self.tokenizer.piece_to_id(motif[-kmer:]))
            positions.append(pos + length - kmer)
            cigar_ids.append(op)

        # Add EOS
        token_ids.append(self.tokenizer.eos)
        cigar_ids.append(self.cigar_equal_id)
        positions.append(CHROM_LENGTHS.get(chrom, 1000000) - 1)

        token_len = len(token_ids)
        strand_ids = torch.tensor([strand] * token_len, dtype=torch.long)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        chr_ids = torch.tensor([chr_id] * token_len, dtype=torch.long)
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor(cigar_ids, dtype=torch.long)
        return [token_ids, strand_ids, chr_ids, position_ids, cigar_ids]

    def __getitem__(self, index):
        data = self._get_item_by_idx(index)
        label = data["label"]
        alignments = data.get("alignments", [])

        if not alignments:
            # No alignment, use dense input
            if self.training and self.rc_aug and random.random() < 0.5:
                inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
            else:
                inputs = self.get_dense_inputs(data["sequence"], strand=0)
        else:
            if self.training:
                p = random.random()
                if p < (1 - self.use_alignment_prob) / 2:
                    # Use raw sequence with reverse complement
                    inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
                elif p < (1 - self.use_alignment_prob):
                    # Use raw sequence without reverse complement
                    inputs = self.get_dense_inputs(data["sequence"], strand=0)
                elif p < (1 - self.use_alignment_prob) + self.use_alignment_prob * self.dense_prob:
                    # Use dense AA input
                    weights = [max(align["mapq"], 10) for align in alignments]
                    alignment = random.choices(alignments, weights=weights, k=1)[0]
                    inputs = self.get_dense_aa_inputs(alignment)
                else:
                    # Use sparse AA input
                    weights = [max(align["mapq"], 10) for align in alignments]
                    alignment = random.choices(alignments, weights=weights, k=1)[0]
                    inputs = self.get_sparse_aa_inputs(alignment)
            else:
                # During eval/test, always use sparse representation
                inputs = self.get_sparse_aa_inputs(alignments[0])

        # Clip if exceeds max_len
        if len(inputs[0]) > self.max_len:
            inputs = self.clip_tensors(inputs)

        label = torch.tensor(label, dtype=torch.long)
        return inputs, label
