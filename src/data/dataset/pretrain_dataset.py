import os
import random
from itertools import accumulate
import numpy as np
import pysam
import torch
from torch.utils.data import Dataset
from src.tokenizer import DNAKmerTokenizer, build_tokenizer
from .jsonl_dataset import MMapIndexedJsonlDataset
from .fasta_dataset import MMapIndexedFastaDataset
from .utils import collate_dense_tensors, get_random_mask


def print_rank_0(msg):
    """Print only on rank 0"""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(msg)


# CIGAR operations
CIGAR_OPS = ["M", "I", "D", "N", "S", "H", "P", "=", "X"]

# Chromosome to ID mapping
CHROM2ID = {f"chr{i}": i for i in range(1, 23)}
CHROM2ID.update({"chrX": 23, "chrY": 24, "chrM": 25})

# Chromosome lengths (approximate, GRCh38)
CHROM_LENGTHS = {
    "chr1": 248956422, "chr2": 242193529, "chr3": 198295559, "chr4": 190214555,
    "chr5": 181538259, "chr6": 170805979, "chr7": 159345973, "chr8": 145138636,
    "chr9": 138394717, "chr10": 133797422, "chr11": 135086622, "chr12": 133275309,
    "chr13": 114364328, "chr14": 107043718, "chr15": 101991189, "chr16": 90338345,
    "chr17": 83257441, "chr18": 80373285, "chr19": 58617616, "chr20": 64444167,
    "chr21": 46709983, "chr22": 50818468, "chrX": 156040895, "chrY": 57227415,
    "chrM": 16569
}


class PretrainDataset(Dataset):
    """Base class for pretraining datasets."""

    def __init__(self, tokenizer: DNAKmerTokenizer, seq_len: int = 1024,
                 mask_ratio: float = 0.15, rc_ratio: float = 0.15):
        super().__init__()
        self.seq_len = seq_len
        self.mask_ratio = mask_ratio
        self.tokenizer = tokenizer
        self.kmer = tokenizer.kmer
        self.mask_id = tokenizer.mask
        self.pad_id = tokenizer.pad
        self.cigar_equal_id = CIGAR_OPS.index("=")
        self.cigar_pad_id = CIGAR_OPS.index("P")

        # Tokens to exclude from masking
        self.filter_ids = [tokenizer.bos_id, tokenizer.eos_id]
        for i in range(1, tokenizer.kmer + 1):
            self.filter_ids.append(tokenizer.piece_to_id("N" * i))
        self.rc_ratio = rc_ratio

    def reverse_complement_masking(self, token_ids: torch.Tensor, mask: torch.Tensor):
        """Reverse complement and flip mask."""
        rc_token_ids = [self.tokenizer.get_reverse_complement_token(t) for t in token_ids.tolist()[::-1]]
        rc_token_ids = torch.tensor(rc_token_ids, dtype=torch.long)
        rc_mask = torch.flip(mask, dims=[0])
        return rc_token_ids, rc_mask

    def collate_fn(self, batches):
        """Batch collation function."""
        inputs, targets = zip(*batches)
        token_ids, chr_ids, position_ids, strand_ids, cigar_ids = zip(*inputs)

        seq_masks = [torch.ones_like(t, dtype=torch.bool) for t in token_ids]
        seq_masks = collate_dense_tensors(seq_masks, False)

        token_ids = collate_dense_tensors(token_ids, self.pad_id)
        chr_ids = collate_dense_tensors(chr_ids, 0)
        position_ids = collate_dense_tensors(position_ids, 0)
        strand_ids = collate_dense_tensors(strand_ids, 0)
        cigar_ids = collate_dense_tensors(cigar_ids, self.cigar_pad_id)

        labels, masks = zip(*targets)
        labels = collate_dense_tensors(labels, self.pad_id)
        masks = collate_dense_tensors(masks, False)

        data = {
            "token_ids": token_ids,
            "chr_ids": chr_ids,
            "position_ids": position_ids,
            "strand_ids": strand_ids,
            "cigar_ids": cigar_ids,
            "seq_mask": seq_masks
        }
        return data, {"labels": labels, "mask": masks}


class Discrete1KGPretrainDataset(PretrainDataset):
    """1000 Genomes discrete variant dataset."""

    def __init__(self, filename: str, tokenizer: DNAKmerTokenizer,
                 seq_len: int = 1024, mask_ratio: float = 0.15):
        super().__init__(tokenizer, mask_ratio=mask_ratio, seq_len=seq_len)
        assert filename.endswith(".jsonl"), f"{filename} must be a jsonl file"
        self.ds = MMapIndexedJsonlDataset(filename)
        self.num_samples = len(self.ds)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        index = random.randint(0, len(self.ds) - 1)
        data = self.ds[index]
        chrom = data["chrom"]
        variants = self._get_random_variants(data["variants"])
        chr_id = CHROM2ID[chrom]

        token_ids = [self.tokenizer.bos]
        cigar_ids = [self.cigar_equal_id]
        positions = [0]

        for var in variants:
            motif = var["seq"]
            cigar = var["cigar"]
            op = CIGAR_OPS.index(cigar) if isinstance(cigar, str) else cigar
            pos = var["position"]

            motif_tokens = self.tokenizer(motif, to_tensor=False)
            for i, t in enumerate(motif_tokens):
                token_ids.append(t)
                positions.append(pos + i * self.kmer)
                cigar_ids.append(op)

            if len(token_ids) >= self.seq_len - 1:
                break

        token_ids.append(self.tokenizer.eos)
        cigar_ids.append(self.cigar_equal_id)
        positions.append(CHROM_LENGTHS[chrom] - 1)

        token_len = len(token_ids)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        chr_ids = torch.tensor([chr_id] * token_len, dtype=torch.long)
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor(cigar_ids, dtype=torch.long)
        strand_ids = torch.tensor([0] * token_len, dtype=torch.long)

        bert_mask = get_random_mask(token_ids, mask_ratio=self.mask_ratio, filter_ids=self.filter_ids)
        masked_token_ids = token_ids.masked_fill(bert_mask, self.mask_id)

        return (masked_token_ids, chr_ids, position_ids, strand_ids, cigar_ids), (token_ids, bert_mask)

    def _get_random_variants(self, variants):
        if len(variants) <= self.seq_len:
            start = 0
        else:
            start = random.randint(0, len(variants) - self.seq_len)
        return variants[start:start + self.seq_len]


class DNAPretrainDataset(PretrainDataset):
    """Multi-species DNA pretraining dataset."""

    def __init__(self, filename: str, tokenizer: DNAKmerTokenizer, chr_id: int = 0,
                 mask_ratio: float = 0.15, seq_len: int = 1024, rc_ratio: float = 0.3):
        super().__init__(tokenizer, mask_ratio=mask_ratio, rc_ratio=rc_ratio, seq_len=seq_len)
        self.ds = MMapIndexedJsonlDataset(filename, verbose=False)
        self.chr_id = chr_id

    def __len__(self):
        return len(self.ds)

    def get_dense_input(self, seq: str):
        """Dense sequence input."""
        start = 0
        end = len(seq) - 1
        token_ids = self.tokenizer(seq, to_tensor=True, bos=True, eos=True)
        token_len = len(token_ids)
        chr_ids = torch.tensor([self.chr_id] * token_len, dtype=torch.long)
        positions = [0] + list(range(start, start + self.kmer * (token_len - 2), self.kmer)) + [end]
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor([self.cigar_equal_id] * token_len, dtype=torch.long)
        strand_ids = torch.tensor([0] * token_len, dtype=torch.long)
        return token_ids, strand_ids, chr_ids, position_ids, cigar_ids

    def get_dense_aa_input(self, alignment: dict):
        """Dense input with alignment information."""
        chrom = alignment["chrom"]
        chr_id = CHROM2ID[chrom]
        strand = alignment["strand"]

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
        positions.append(CHROM_LENGTHS[chrom] - 1)

        token_len = len(token_ids)
        strand_ids = torch.tensor([strand] * token_len, dtype=torch.long)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        chr_ids = torch.tensor([chr_id] * token_len, dtype=torch.long)
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor(cigar_ids, dtype=torch.long)
        return token_ids, strand_ids, chr_ids, position_ids, cigar_ids

    def get_sparse_aa_input(self, alignment: dict):
        """Sparse input with alignment information."""
        chrom = alignment["chrom"]
        chr_id = CHROM2ID[chrom]
        strand = alignment["strand"]

        token_ids = [self.tokenizer.bos]
        cigar_ids = [self.cigar_equal_id]
        positions = [0]

        # Add starting motif
        if alignment["cigars"][0][1] in (pysam.CEQUAL, pysam.CMATCH):
            length, op, motif = alignment["cigars"][0]
            pos = alignment["positions"][0]
            kmer = min(length, self.kmer)
            token_ids.append(self.tokenizer.piece_to_id(motif[:kmer]))
            positions.append(pos)
            cigar_ids.append(op)

        k = random.randint(0, 32)
        for cigar, pos in zip(alignment["cigars"], alignment["positions"]):
            length, op, motif = cigar
            if op in (pysam.CEQUAL, pysam.CMATCH):
                if k is None or k < 1:
                    continue
                motif_tokens = self.tokenizer(motif, to_tensor=False)
                for i, t in enumerate(motif_tokens):
                    if i % k == 0:
                        token_ids.append(t)
                        positions.append(pos + i * self.kmer)
                        cigar_ids.append(op)
            elif op in (pysam.CDEL,):
                kmer = min(length, self.kmer)
                token_ids.append(self.tokenizer.piece_to_id(motif[:kmer]))
                positions.append(pos)
                cigar_ids.append(op)
                token_ids.append(self.tokenizer.piece_to_id(motif[-kmer:]))
                positions.append(pos + length - kmer)
                cigar_ids.append(op)
            else:
                motif_tokens = self.tokenizer(motif, to_tensor=False)
                for i, t in enumerate(motif_tokens):
                    token_ids.append(t)
                    positions.append(pos + i * self.kmer)
                    cigar_ids.append(op)

        # Add ending motif
        if alignment["cigars"][-1][1] in (pysam.CEQUAL, pysam.CMATCH):
            length, op, motif = alignment["cigars"][-1]
            pos = alignment["positions"][-1]
            kmer = min(length, self.kmer)
            token_ids.append(self.tokenizer.piece_to_id(motif[-kmer:]))
            positions.append(pos + length - kmer)
            cigar_ids.append(op)

        token_ids.append(self.tokenizer.eos)
        cigar_ids.append(self.cigar_equal_id)
        positions.append(CHROM_LENGTHS[chrom] - 1)

        token_len = len(token_ids)
        strand_ids = torch.tensor([strand] * token_len, dtype=torch.long)
        token_ids = torch.tensor(token_ids, dtype=torch.long)
        chr_ids = torch.tensor([chr_id] * token_len, dtype=torch.long)
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor(cigar_ids, dtype=torch.long)
        return token_ids, strand_ids, chr_ids, position_ids, cigar_ids

    def __getitem__(self, idx):
        data = self.ds[idx]
        seq = data["seq"]
        alignments = data.get("alignments", None)
        random_reverse = True

        if not alignments:
            inputs = self.get_dense_input(seq)
        else:
            random_reverse = False
            weights = [max(align["mapq"], 10) for align in alignments]
            alignment = random.choices(alignments, weights=weights, k=1)[0]
            if random.random() < 0.5:
                inputs = self.get_dense_aa_input(alignment)
            else:
                inputs = self.get_sparse_aa_input(alignment)

        token_ids, strand_ids, chr_ids, position_ids, cigar_ids = inputs

        if random_reverse and (random.random() < self.rc_ratio):
            strand_ids = torch.ones_like(strand_ids, dtype=torch.long)

        bert_mask = get_random_mask(token_ids, mask_ratio=self.mask_ratio, filter_ids=self.filter_ids)
        masked_token_ids = token_ids.masked_fill(bert_mask, self.mask_id)

        if strand_ids[0].item() != 0:
            token_ids, bert_mask = self.reverse_complement_masking(token_ids, bert_mask)

        return (masked_token_ids, chr_ids, position_ids, strand_ids, cigar_ids), (token_ids, bert_mask)


class RefseqPretrainDataset(PretrainDataset):
    """Reference genome pretraining dataset."""

    def __init__(self, filename: str, tokenizer: DNAKmerTokenizer, seq_len: int = 1024,
                 mask_ratio: float = 0.15, rc_ratio: float = 0.3, sample_ratio: float = 0.15,
                 split: str = "train"):
        super().__init__(tokenizer, mask_ratio=mask_ratio, rc_ratio=rc_ratio, seq_len=seq_len)
        self.filename = filename

        lengths = np.array([len(data["seq"]) for data in MMapIndexedFastaDataset(filename, verbose=False)])
        self.num_chrs = len(lengths)
        self.num_samples = int(np.sum(lengths) // (seq_len * self.kmer))
        self.lengths = lengths / np.sum(lengths)
        self.sample_ratio = sample_ratio
        self.training = split in ["train", "trainval"]
        self.ref_seqs = {}

    def get_ref_seq(self, chr_id: int):
        if chr_id not in self.ref_seqs:
            self.ref_seqs[chr_id] = MMapIndexedFastaDataset(self.filename, verbose=False)[chr_id]
        return self.ref_seqs[chr_id]

    def __len__(self):
        return self.num_samples

    def get_random_continue_seq(self):
        """Random contiguous sequence."""
        while True:
            chr_id = random.choices(range(self.num_chrs), weights=self.lengths, k=1)[0]
            seq = self.get_ref_seq(chr_id)["seq"]
            start = random.randint(0, len(seq) - self.seq_len * self.kmer - 1)
            ref_seq = seq[start:start + self.seq_len * self.kmer]
            if ref_seq.count("N") / len(ref_seq) < 0.25:
                break

        token_ids = self.tokenizer(ref_seq, to_tensor=True, bos=True, eos=True)
        token_len = len(token_ids)
        chr_ids = [chr_id + 1] * token_len
        positions = [0] + list(range(start, start + self.kmer * (token_len - 2), self.kmer)) + [
            int(self.lengths[chr_id] * 3e9) - 1]
        cigar_ids = [self.cigar_equal_id] * token_len

        chr_ids = torch.tensor(chr_ids, dtype=torch.long)
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor(cigar_ids, dtype=torch.long)
        return token_ids, chr_ids, position_ids, cigar_ids

    def get_random_discrete_seq(self):
        """Random discrete sequence."""
        chr_id = random.choices(range(self.num_chrs), weights=self.lengths, k=1)[0]
        seq = self.get_ref_seq(chr_id)["seq"]
        positions = random.sample(range(0, len(seq) - self.kmer), self.seq_len)
        positions = sorted(positions)

        kmer = 6 if not self.training else random.randint(1, self.kmer)
        motifs = [seq[p:p + kmer] for p in positions]

        token_ids = self.tokenizer(motifs, to_tensor=True, bos=True, eos=True)
        token_len = len(token_ids)
        chr_ids = [chr_id + 1] * token_len
        positions = [0] + positions + [int(self.lengths[chr_id] * 3e9) - 1]
        cigar_ids = [self.cigar_equal_id] * token_len

        chr_ids = torch.tensor(chr_ids, dtype=torch.long)
        position_ids = torch.tensor(positions, dtype=torch.float32)
        cigar_ids = torch.tensor(cigar_ids, dtype=torch.long)
        return token_ids, chr_ids, position_ids, cigar_ids

    def __getitem__(self, idx):
        if random.random() < self.sample_ratio:
            token_ids, chr_ids, position_ids, cigar_ids = self.get_random_discrete_seq()
        else:
            token_ids, chr_ids, position_ids, cigar_ids = self.get_random_continue_seq()

        bert_mask = get_random_mask(token_ids, mask_ratio=self.mask_ratio, filter_ids=self.filter_ids)
        masked_token_ids = token_ids.masked_fill(bert_mask, self.mask_id)
        strand_ids = torch.tensor([0] * len(token_ids), dtype=torch.long)

        if random.random() < self.rc_ratio:
            strand_ids = torch.ones_like(strand_ids, dtype=torch.long)
            token_ids, bert_mask = self.reverse_complement_masking(token_ids, bert_mask)

        return (masked_token_ids, chr_ids, position_ids, strand_ids, cigar_ids), (token_ids, bert_mask)


class USMAAPretrainDataset(Dataset):
    """USM-AA pretraining dataset with weighted sampling from multiple sources."""

    def __init__(
            self,
            ref_file: str,
            dna_file: str = None,
            n1kg_file: str = None,
            tokenizer: DNAKmerTokenizer = None,
            seq_len: int = 256,
            mask_ratio: float = 0.15,
            rc_ratio: float = 0.3,
            sample_weights: dict = None
    ):
        """
        Args:
            ref_file: Reference genome FASTA file path
            dna_file: Multi-species DNA JSONL file path (optional)
            n1kg_file: 1000 Genomes JSONL file path (optional)
            tokenizer: DNAKmerTokenizer instance
            seq_len: Sequence length
            mask_ratio: Mask ratio
            rc_ratio: Reverse complement ratio
            sample_weights: Sampling weights for each data source
                           e.g., {"ref": 1.0, "dna": 0.5, "n1kg": 0.3}
        """
        super().__init__()
        if tokenizer is None:
            tokenizer = build_tokenizer(kmer=6, dynamic=True)

        self.tokenizer = tokenizer
        self.datasets = []
        self.dataset_names = []
        self.weights = []

        # Default weights
        if sample_weights is None:
            sample_weights = {"ref": 1.0, "dna": 1.0, "n1kg": 1.0}

        # Build datasets
        if ref_file and os.path.exists(ref_file):
            ds = RefseqPretrainDataset(
                ref_file, tokenizer,
                mask_ratio=mask_ratio,
                rc_ratio=rc_ratio,
                seq_len=seq_len
            )
            self.datasets.append(ds)
            self.dataset_names.append("ref")
            self.weights.append(sample_weights.get("ref", 1.0))
            print_rank_0(f"RefseqPretrainDataset: {len(ds)} samples, weight={sample_weights.get('ref', 1.0)}")

        if dna_file and os.path.exists(dna_file):
            ds = DNAPretrainDataset(
                dna_file, tokenizer,
                mask_ratio=mask_ratio,
                rc_ratio=rc_ratio,
                seq_len=seq_len
            )
            self.datasets.append(ds)
            self.dataset_names.append("dna")
            self.weights.append(sample_weights.get("dna", 1.0))
            print_rank_0(f"DNAPretrainDataset: {len(ds)} samples, weight={sample_weights.get('dna', 1.0)}")

        if n1kg_file and os.path.exists(n1kg_file):
            ds = Discrete1KGPretrainDataset(
                n1kg_file, tokenizer,
                mask_ratio=mask_ratio,
                seq_len=seq_len
            )
            self.datasets.append(ds)
            self.dataset_names.append("n1kg")
            self.weights.append(sample_weights.get("n1kg", 1.0))
            print_rank_0(f"Discrete1KGPretrainDataset: {len(ds)} samples, weight={sample_weights.get('n1kg', 1.0)}")

        assert len(self.datasets) > 0, "No valid dataset found"

        # Compute dataset sizes and cumulative indices
        self.dataset_sizes = [len(ds) for ds in self.datasets]
        self.cumulative_sizes = list(accumulate(self.dataset_sizes))
        self.total_size = self.cumulative_sizes[-1]

        # Normalize weights to probabilities
        total_weight = sum(self.weights)
        self.probs = [w / total_weight for w in self.weights]

        print_rank_0(f"Total datasets: {len(self.datasets)}, Total samples: {self.total_size}")
        print_rank_0(f"Sampling probabilities: {dict(zip(self.dataset_names, self.probs))}")

    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        # Weighted random sampling: choose dataset based on probabilities
        dataset_idx = random.choices(range(len(self.datasets)), weights=self.probs, k=1)[0]
        ds = self.datasets[dataset_idx]
        # Random sample from chosen dataset
        sample_idx = random.randint(0, len(ds) - 1)
        return ds[sample_idx]

    def collate_fn(self, batch):
        return self.datasets[0].collate_fn(batch)
