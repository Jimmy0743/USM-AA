import os
import random
import pysam
import torch
from torch.utils.data import Dataset

from src.tokenizer import DNAKmerTokenizer
from src.bio.molecular_constants import CHROM2ID, CHROM_LENGTHS
from .jsonl_dataset import MMapIndexedJsonlDataset
from .utils import collate_dense_tensors
from .dense_dataset import DenseDataset, CIGAR_OPS, reverse_complement, print_rank_0


class DenseAADataset(DenseDataset):
    """Dense AA Dataset - sequences with alignment information.
    
    Used for classification/regression finetuning tasks where
    alignment information is available.
    
    Data format (JSONL):
        {
            "sequence": "ATCGATCG...", 
            "label": 0,
            "alignments": [
                {
                    "chrom": "chr1",
                    "strand": 0,
                    "mapq": 60,
                    "cigars": [[length, op, motif], ...],
                    "positions": [pos1, pos2, ...]
                },
                ...
            ]
        }
        
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
            use_alignment_prob: Probability of using alignment info during training
            cv_val_folds: List of fold IDs to use as validation set (for CV mode)
        """
        super().__init__(filename, tokenizer, split, max_len, chr_id, rc_aug, cv_val_folds)
        self.use_alignment_prob = use_alignment_prob

        # Count samples with alignments
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            num_alignments = sum(1 for i in range(len(self)) 
                               if len(self._get_item_by_idx(i).get("alignments", [])) > 0)
            print(f"DenseAADataset: alignments: {num_alignments}, total: {len(self)}, "
                  f"ratio: {num_alignments / len(self):.4f}")

    def get_dense_aa_inputs(self, alignment: dict):
        """Get dense AA input tensors from alignment.
        
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

        for cigar, pos in zip(alignment["cigars"], alignment["positions"]):
            length, op, motif = cigar
            motif_tokens = self.tokenizer(motif, to_tensor=False)
            for i, t in enumerate(motif_tokens):
                token_ids.append(t)
                positions.append(pos + i * self.kmer)
                cigar_ids.append(op)

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
            # Has alignment
            if self.training:
                # During training, randomly choose between dense and AA input
                if random.random() < self.use_alignment_prob:
                    # Use alignment info
                    weights = [max(align["mapq"], 10) for align in alignments]
                    alignment = random.choices(alignments, weights=weights, k=1)[0]
                    inputs = self.get_dense_aa_inputs(alignment)
                else:
                    # Use raw sequence
                    if self.rc_aug and random.random() < 0.5:
                        inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
                    else:
                        inputs = self.get_dense_inputs(data["sequence"], strand=0)
            else:
                # During eval/test, always use the best alignment
                alignment = alignments[0]
                inputs = self.get_dense_aa_inputs(alignment)

        # Clip if exceeds max_len
        if len(inputs[0]) > self.max_len:
            inputs = self.clip_tensors(inputs)

        label = torch.tensor(label, dtype=torch.long)
        return inputs, label

    def collate_fn(self, batch):
        """Collate function for DataLoader with first batch debug printing.
        
        Returns:
            data: Dict with keys [token_ids, strand_ids, chr_ids, position_ids, cigar_ids, seq_mask]
            targets: Dict with key [labels]
        """
        inputs, labels = zip(*batch)
        token_ids, strand_ids, chr_ids, position_ids, cigar_ids = zip(*inputs)

        # Print first batch debug info (first 2 samples)
        rank = int(os.environ.get("RANK", 0))


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
