
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from src.tokenizer import DNAKmerTokenizer
from .jsonl_dataset import MMapIndexedJsonlDataset
from .utils import collate_dense_tensors


def print_rank_0(msg):
    """Print only on rank 0"""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(msg)


def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


# CIGAR operations
CIGAR_OPS = ["M", "I", "D", "N", "S", "H", "P", "=", "X"]


class DenseDataset(Dataset):
    """Dense Dataset - sequences without alignment information.
    
    Used for classification/regression finetuning tasks where
    only raw DNA sequences are available (no alignment info).
    
    Data format (JSONL):
        {"sequence": "ATCGATCG...", "label": 0}
        
    For cross-validation (with fold_id):
        {"sequence": "ATCGATCG...", "label": 0, "fold_id": 0}
    """

    def __init__(
        self,
        filename: str,
        tokenizer: DNAKmerTokenizer,
        split: str = "train",
        max_len: int = 1024,
        chr_id: int = 0,
        rc_aug: bool = True,
        cv_val_folds: list = None
    ):
        """
        Args:
            filename: Path to JSONL file
            tokenizer: DNAKmerTokenizer instance
            split: Dataset split (train/val/test)
            max_len: Maximum sequence length
            chr_id: Chromosome ID (default 0 for unknown)
            rc_aug: Whether to use reverse complement augmentation during training
            cv_val_folds: List of fold IDs to use as validation set (for CV mode)
                         If provided, train split uses all folds except these,
                         val split uses only these folds.
        """
        super().__init__()
        self._full_ds = MMapIndexedJsonlDataset(filename, verbose=False)
        
        # Filter by fold_id for cross-validation
        self.indices = None
        if cv_val_folds is not None:
            self.indices = []
            for i, data in enumerate(self._full_ds):
                fold_id = data.get("fold_id", -1)
                if split == "val":
                    # Validation: only include val_folds
                    if fold_id in cv_val_folds:
                        self.indices.append(i)
                else:
                    # Train: exclude val_folds
                    if fold_id not in cv_val_folds:
                        self.indices.append(i)
            print_rank_0(f"DenseDataset CV mode: split={split}, val_folds={cv_val_folds}, "
                        f"filtered {len(self.indices)}/{len(self._full_ds)}")
        
        print_rank_0(f"DenseDataset split: {split}\t{len(self)}\t{filename}")

        self.max_len = max_len
        self.training = split in ("train", "trainval")
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad
        self.chr_id = chr_id
        self.kmer = tokenizer.kmer
        self.cigar_equal_id = CIGAR_OPS.index("=")
        self.cigar_pad_id = CIGAR_OPS.index("P")
        self.rc_aug = rc_aug

    @property
    def ds(self):
        """Access dataset with optional index filtering for CV."""
        return self
    
    def _get_item_by_idx(self, idx):
        """Get item by actual index (accounting for CV filtering)."""
        if self.indices is not None:
            return self._full_ds[self.indices[idx]]
        return self._full_ds[idx]

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self._full_ds)

    def clip_tensors(self, tensors):
        """Clip tensors to max_len with random start during training."""
        if len(tensors[0]) <= self.max_len:
            return tensors
        start = 0
        if self.training:
            start = np.random.randint(0, len(tensors[0]) - self.max_len)
        return [t[start: start + self.max_len] for t in tensors]

    def get_dense_inputs(self, seq: str, strand: int = 0):
        """Get dense input tensors from sequence.
        
        Args:
            seq: DNA sequence string
            strand: Strand indicator (0=forward, 1=reverse)
            
        Returns:
            List of tensors: [token_ids, strand_ids, chr_ids, position_ids, cigar_ids]
        """
        start = 0
        end = len(seq) - 1
        token_ids = self.tokenizer(seq, bos=True, eos=True, to_tensor=True)
        token_len = len(token_ids)
        chr_ids = torch.tensor([self.chr_id] * token_len, dtype=torch.long)
        positions = [0] + list(range(start, start + self.kmer * (token_len - 2), self.kmer)) + [end]
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor([self.cigar_equal_id] * token_len, dtype=torch.long)
        strand_ids = torch.tensor([strand] * token_len, dtype=torch.long)
        return [token_ids, strand_ids, chr_ids, position_ids, cigar_ids]

    def __getitem__(self, index):
        data = self._get_item_by_idx(index)
        label = data["label"]

        # Reverse complement augmentation during training
        if self.training and self.rc_aug and random.random() < 0.5:
            inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
        else:
            inputs = self.get_dense_inputs(data["sequence"], strand=0)

        # Clip if exceeds max_len
        if len(inputs[0]) > self.max_len:
            inputs = self.clip_tensors(inputs)

        label = torch.tensor(label, dtype=torch.long)
        return inputs, label

    def collate_fn(self, batch):
        """Collate function for DataLoader.
        
        Returns:
            data: Dict with keys [token_ids, strand_ids, chr_ids, position_ids, cigar_ids, seq_mask]
            targets: Dict with key [labels]
        """
        inputs, labels = zip(*batch)
        token_ids, strand_ids, chr_ids, position_ids, cigar_ids = zip(*inputs)

        seq_masks = [torch.ones_like(t, dtype=torch.bool) for t in token_ids]
        seq_masks = collate_dense_tensors(seq_masks, False)
        chr_ids = collate_dense_tensors(chr_ids, 0)
        position_ids = collate_dense_tensors(position_ids, 0)
        cigar_ids = collate_dense_tensors(cigar_ids, self.cigar_pad_id)
        strand_ids = collate_dense_tensors(strand_ids, 0)
        token_ids = collate_dense_tensors(token_ids, self.pad_id)
        labels = torch.stack(labels, dim=0)

        data = {
            "token_ids": token_ids,
            "strand_ids": strand_ids,
            "chr_ids": chr_ids,
            "position_ids": position_ids,
            "cigar_ids": cigar_ids,
            "seq_mask": seq_masks
        }
        return data, {"labels": labels}
