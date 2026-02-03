import os
import random
import pysam
import torch
from torch.utils.data import Dataset

from src.tokenizer import DNAKmerTokenizer
from .jsonl_dataset import MMapIndexedJsonlDataset
from .utils import collate_dense_tensors
from .dense_dataset import DenseDataset, CIGAR_OPS, reverse_complement, print_rank_0


# Chromosome to ID mapping
CHROM2ID = {f"chr{i}": i for i in range(1, 23)}
CHROM2ID.update({"chrX": 23, "chrY": 24, "chrM": 25})
# Also support without 'chr' prefix
CHROM2ID.update({str(i): i for i in range(1, 23)})
CHROM2ID.update({"X": 23, "Y": 24, "M": 25})

# Chromosome lengths (GRCh38)
CHROM_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
    "chrM": 16569
}


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
    """

    def __init__(
        self,
        filename: str,
        tokenizer: DNAKmerTokenizer,
        split: str = "train",
        max_len: int = 1024,
        chr_id: int = 0,
        rc_aug: bool = True,
        use_alignment_prob: float = 0.8
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
        """
        super().__init__(filename, tokenizer, split, max_len, chr_id, rc_aug)
        self.use_alignment_prob = use_alignment_prob

        # Count samples with alignments
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            num_alignments = sum(1 for data in self.ds if len(data.get("alignments", [])) > 0)
            print(f"DenseAADataset: alignments: {num_alignments}, total: {len(self.ds)}, "
                  f"ratio: {num_alignments / len(self.ds):.4f}")

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
        data = self.ds[index]
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
