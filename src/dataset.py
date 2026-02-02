# Pretraining Dataset
import json
import os
import random
import re
import struct
from functools import lru_cache
from itertools import accumulate
from typing import List, Optional, Union

import numpy as np
import pysam
import torch
from torch.utils.data import ConcatDataset, Dataset
from tqdm import tqdm

# Support both direct execution and module import
try:
    from .tokenizer import DNAKmerTokenizer, build_tokenizer
except ImportError:
    from tokenizer import DNAKmerTokenizer, build_tokenizer

# ============ Constants ============

CIGAR_OPS = ["M", "I", "D", "N", "S", "H", "P", "=", "X", "B"]

CHROM2ID = {
    "chr1": 1, "1": 1, "chr2": 2, "2": 2, "chr3": 3, "3": 3,
    "chr4": 4, "4": 4, "chr5": 5, "5": 5, "chr6": 6, "6": 6,
    "chr7": 7, "7": 7, "chr8": 8, "8": 8, "chr9": 9, "9": 9,
    "chr10": 10, "10": 10, "chr11": 11, "11": 11, "chr12": 12, "12": 12,
    "chr13": 13, "13": 13, "chr14": 14, "14": 14, "chr15": 15, "15": 15,
    "chr16": 16, "16": 16, "chr17": 17, "17": 17, "chr18": 18, "18": 18,
    "chr19": 19, "19": 19, "chr20": 20, "20": 20, "chr21": 21, "21": 21,
    "chr22": 22, "22": 22, "chrX": 23, "X": 23, "chrY": 24, "Y": 24,
    "chrM": 25, "M": 25
}

CHROM_LENGTHS = {
    'chr1': 248956422, 'chr2': 242193529, 'chr3': 198295559, 'chr4': 190214555,
    'chr5': 181538259, 'chr6': 170805979, 'chr7': 159345973, 'chr8': 145138636,
    'chr9': 138394717, 'chr10': 133797422, 'chr11': 135086622, 'chr12': 133275309,
    'chr13': 114364328, 'chr14': 107043718, 'chr15': 101991189, 'chr16': 90338345,
    'chr17': 83257441, 'chr18': 80373285, 'chr19': 58617616, 'chr20': 64444167,
    'chr21': 46709983, 'chr22': 50818468, 'chrX': 156040895, 'chrY': 57227415
}


# ============ Utility Functions ============

def print_rank_0(msg: str):
    """Print only on rank 0."""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(msg, flush=True)


def collate_dense_tensors(tensors: List[torch.Tensor], pad_value=0) -> torch.Tensor:
    """Pad tensors of different lengths to the same length and stack."""
    max_len = max(t.size(0) for t in tensors)
    batch_size = len(tensors)
    
    if tensors[0].dim() == 1:
        result = torch.full((batch_size, max_len), pad_value, dtype=tensors[0].dtype)
    else:
        result = torch.full((batch_size, max_len, *tensors[0].shape[1:]), pad_value, dtype=tensors[0].dtype)
    
    for i, t in enumerate(tensors):
        result[i, :t.size(0)] = t
    
    return result


def get_random_mask(tokens: torch.Tensor, mask_ratio: float = 0.15, 
                    filter_ids: Optional[List[int]] = None) -> torch.Tensor:
    """Generate random mask."""
    mask = torch.lt(torch.rand_like(tokens, dtype=torch.float32), mask_ratio)
    if filter_ids is not None:
        filter_ids = torch.tensor(list(set(filter_ids)), dtype=torch.long)
        fmask = (tokens.unsqueeze(-1) == filter_ids).any(dim=-1)
        mask = mask & (~fmask)
    return mask


# ============ Indexed Dataset Base Classes ============

class MMIndex:
    """Memory-mapped index."""
    _HDR_MAGIC = b'MMIDIDX\x00\x00'

    dtypes = {
        1: np.uint8, 2: np.int8, 3: np.int16, 4: np.int32,
        5: np.int64, 6: np.float32, 7: np.double, 8: np.uint16
    }

    @classmethod
    def code(cls, dtype):
        for k, v in cls.dtypes.items():
            if v == dtype:
                return k
        raise ValueError(f"Unknown dtype: {dtype}")

    @classmethod
    def is_index(cls, filename: str) -> bool:
        if not os.path.exists(filename):
            return False
        with open(filename, "rb") as f:
            magic = f.read(len(cls._HDR_MAGIC))
        return magic == cls._HDR_MAGIC

    @classmethod
    def writer(cls, path: str, dtype):
        class _Writer:
            def __enter__(self):
                self._file = open(path, 'wb')
                return self

            def write(self, sizes, doc_idx=None):
                if doc_idx is None:
                    doc_idx = np.arange(len(sizes), dtype=np.int64)
                else:
                    doc_idx = np.array(doc_idx, dtype=np.int64)
                
                # Write header
                self._file.write(cls._HDR_MAGIC)
                self._file.write(struct.pack('<Q', 1))
                self._file.write(struct.pack('<B', cls.code(dtype)))
                self._file.write(struct.pack('<Q', len(sizes)))
                self._file.write(struct.pack('<Q', len(doc_idx)))
                
                # Write sizes
                sizes32 = np.array(sizes, dtype=np.int32)
                self._file.write(sizes32.tobytes(order='C'))
                
                # Write pointers
                pointers = np.array(sizes, dtype=np.int64)
                pointers *= dtype().itemsize
                np.cumsum(pointers, axis=0, out=pointers)
                pointers = np.concatenate([[0], pointers[:-1]]) if len(pointers) > 0 else pointers
                self._file.write(pointers.tobytes(order='C'))
                
                # Write doc_idx
                self._file.write(doc_idx.tobytes(order='C'))

            def __exit__(self, *args):
                self._file.close()

        return _Writer()

    def __init__(self, path: str, skip_warmup: bool = True, verbose: bool = False):
        self.verbose = verbose
        self._do_init(path, skip_warmup)

    def _do_init(self, path: str, skip_warmup: bool):
        with open(path, 'rb') as f:
            magic = f.read(len(self._HDR_MAGIC))
            assert magic == self._HDR_MAGIC, "Invalid index file"
            
            version = struct.unpack('<Q', f.read(8))
            assert version == (1,)
            
            dtype_code = struct.unpack('<B', f.read(1))[0]
            self._dtype = self.dtypes[dtype_code]
            self._dtype_size = self._dtype().itemsize
            self._len = struct.unpack('<Q', f.read(8))[0]
            self._doc_count = struct.unpack('<Q', f.read(8))[0]
            offset = f.tell()

        self._bin_buffer_mmap = np.memmap(path, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)
        
        self._sizes = np.frombuffer(self._bin_buffer, dtype=np.int32, count=self._len, offset=offset)
        self._pointers = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._len,
                                       offset=offset + self._sizes.nbytes)
        self._doc_idx = np.frombuffer(self._bin_buffer, dtype=np.int64, count=self._doc_count,
                                      offset=offset + self._sizes.nbytes + self._pointers.nbytes)
        self.buffer_size = np.sum(self._sizes, dtype=np.int64)

    def __del__(self):
        if hasattr(self, '_bin_buffer_mmap') and self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

    @property
    def dtype(self):
        return self._dtype

    @property
    def sizes(self):
        return self._sizes

    @property
    def pointers(self):
        return self._pointers

    @lru_cache(maxsize=8)
    def __getitem__(self, i):
        return self._pointers[i], self._sizes[i]

    def __len__(self):
        return self._len


class JsonlIndex(MMIndex):
    """JSONL file index."""
    _HDR_MAGIC = b'JSONIDX\x00\x00'

    @classmethod
    def build_index(cls, filename: str):
        """Build JSONL index."""
        assert filename.endswith("jsonl"), f"file is not jsonl: {filename}"
        
        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            for line in tqdm(f, desc="Building index"):
                sizes.append(len(line.encode('utf-8')))

        idx_filename = f"{filename}.idx"
        with cls.writer(idx_filename, np.uint8) as index:
            index.write(sizes)
        
        # Sync timestamps
        import shutil
        bin_stat = os.stat(filename)
        os.utime(idx_filename, (bin_stat.st_atime, bin_stat.st_mtime))


class FastaIndex(MMIndex):
    """FASTA file index."""
    _HDR_MAGIC = b'FASTAIDX\x00\x00'

    @classmethod
    def build_index(cls, filename: str):
        """Build FASTA index."""
        sizes = []
        with open(filename, 'r', encoding="utf-8") as f:
            fasta_lines = []
            for line in tqdm(f, desc="Building index"):
                if line.startswith(">") and fasta_lines:
                    length = len("".join(fasta_lines).encode('utf-8'))
                    sizes.append(length)
                    fasta_lines = []
                fasta_lines.append(line)
            
            if fasta_lines:
                length = len("".join(fasta_lines).encode('utf-8'))
                sizes.append(length)

        idx_filename = f"{filename}.idx"
        with cls.writer(idx_filename, np.uint8) as index:
            index.write(sizes)
        
        bin_stat = os.stat(filename)
        os.utime(idx_filename, (bin_stat.st_atime, bin_stat.st_mtime))


class MMapIndexedDataset(Dataset):
    """Memory-mapped indexed dataset base class."""

    def __init__(self, path: str, skip_warmup: bool = True, verbose: bool = False):
        super().__init__()
        self._path = path
        self.verbose = verbose
        assert os.path.isfile(path), f"File not found: {path}"
        self._do_init(path, skip_warmup)

    @property
    def index_file(self):
        return f"{self._path}.idx"

    @property
    def bin_file(self):
        return self._path

    def _do_init(self, path: str, skip_warmup: bool = True):
        self._index = self.read_index(self.index_file, skip_warmup)
        self._bin_buffer_mmap = np.memmap(self.bin_file, mode='r', order='C')
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def read_index(self, path: str, skip_warmup: bool = True):
        raise NotImplementedError

    def __del__(self):
        if hasattr(self, '_bin_buffer_mmap') and self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
            del self._bin_buffer_mmap

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx):
        ptr, size = self._index[idx]
        return np.frombuffer(self._bin_buffer, dtype=self._index.dtype, count=size, offset=ptr)


class MMapIndexedJsonlDataset(MMapIndexedDataset):
    """Memory-mapped JSONL dataset."""

    def read_index(self, path: str, skip_warmup: bool = True):
        if not os.path.isfile(path) or not JsonlIndex.is_index(path):
            print_rank_0(f"Building index: {path}")
            JsonlIndex.build_index(self.bin_file)
        return JsonlIndex(path, skip_warmup=skip_warmup, verbose=self.verbose)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return json.loads(data.tobytes())


class MMapIndexedFastaDataset(MMapIndexedDataset):
    """Memory-mapped FASTA dataset."""

    def read_index(self, path: str, skip_warmup: bool = True):
        if not os.path.isfile(path) or not FastaIndex.is_index(path):
            print_rank_0(f"Building index: {path}")
            FastaIndex.build_index(self.bin_file)
        return FastaIndex(path, skip_warmup=skip_warmup, verbose=self.verbose)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        fasta_line = data.tobytes().decode("utf-8")
        title, *lines = fasta_line.split("\n")
        title = title[1:].rstrip()
        seq = "".join(lines)
        seq_id = title.split(None, 1)[0]
        return {"id": seq_id, "seq": seq}


# ============ Pretraining Datasets ============

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
        
        print_rank_0(f"Loading reference genome index from: {filename}")
        fasta_dataset = MMapIndexedFastaDataset(filename, verbose=False)
        print_rank_0(f"Computing chromosome lengths...")
        lengths = np.array([len(fasta_dataset[i]["seq"]) for i in tqdm(range(len(fasta_dataset)), desc="Indexing chromosomes", disable=int(os.environ.get("RANK", 0)) != 0)])
        self.num_chrs = len(lengths)
        self.num_samples = int(np.sum(lengths) // (seq_len * self.kmer))
        self.lengths = lengths / np.sum(lengths)
        self.sample_ratio = sample_ratio
        self.training = split in ["train", "trainval"]
        self.ref_seqs = {}
        print_rank_0(f"RefseqPretrainDataset initialized: {self.num_chrs} chromosomes, {self.num_samples} samples")

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
        positions = [0] + list(range(start, start + self.kmer * (token_len - 2), self.kmer)) + [int(self.lengths[chr_id] * 3e9) - 1]
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


# ============ GUE Classification Datasets ============

def reverse_complement(seq: str) -> str:
    """Return reverse complement of DNA sequence."""
    complement = {'A': 'T', 'T': 'A', 'G': 'C', 'C': 'G', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(seq.upper()))


class DenseDataset(Dataset):
    """GUE Dense Dataset - sequences without alignment information."""

    def __init__(self, filename: str, tokenizer: DNAKmerTokenizer,
                 split: str = "train", max_len: int = 1024, chr_id: int = 0):
        super().__init__()
        self.ds = MMapIndexedJsonlDataset(filename, verbose=False)
        print_rank_0(f"DenseDataset split: {split}\t{len(self.ds)}\t{filename}")
        
        self.max_len = max_len
        self.training = split in ("train",)
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad
        self.chr_id = chr_id
        self.kmer = tokenizer.kmer
        self.cigar_equal_id = CIGAR_OPS.index("=")
        self.cigar_pad_id = CIGAR_OPS.index("P")

    def __len__(self):
        return len(self.ds)

    def clip_tensors(self, tensors):
        """Clip tensors to max_len with random start during training."""
        if len(tensors[0]) <= self.max_len:
            return tensors
        start = 0
        if self.training:
            start = np.random.randint(0, len(tensors[0]) - self.max_len)
        return [t[start: start + self.max_len] for t in tensors]

    def get_dense_inputs(self, seq: str, strand: int = 0):
        """Get dense input tensors from sequence."""
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
        data = self.ds[index]
        label = data["label"]
        
        if self.training and random.random() < 0.5:
            inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
        else:
            inputs = self.get_dense_inputs(data["sequence"], strand=0)

        if len(inputs[0]) > self.max_len:
            inputs = self.clip_tensors(inputs)

        label = torch.tensor(label, dtype=torch.long)
        return inputs, label

    def collate_fn(self, batch):
        """Collate function for DataLoader."""
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


class DenseAADataset(DenseDataset):
    """GUE Dense AA Dataset - sequences with alignment information."""

    def __init__(self, filename: str, tokenizer: DNAKmerTokenizer,
                 split: str = "train", max_len: int = 1024, chr_id: int = 0):
        super().__init__(filename, tokenizer, split, max_len, chr_id)
        
        # Count samples with alignments
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            num_alignments = sum(1 for data in self.ds if len(data.get("alignments", [])) > 0)
            print(f"alignments: {num_alignments}, total: {len(self.ds)}, ratio: {num_alignments / len(self.ds):.4f}")

    def get_dense_aa_inputs(self, alignment: dict):
        """Get dense AA input tensors from alignment."""
        chrom = alignment["chrom"]
        chr_id = CHROM2ID[chrom]
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
        positions.append(CHROM_LENGTHS[chrom] - 1)
        
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
            if self.training and random.random() < 0.5:
                inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
            else:
                inputs = self.get_dense_inputs(data["sequence"], strand=0)
        else:
            # Use alignment
            if self.training:
                weights = [max(align["mapq"], 10) for align in alignments]
                alignment = random.choices(alignments, weights=weights, k=1)[0]
            else:
                alignment = alignments[0]
            inputs = self.get_dense_aa_inputs(alignment)

        if len(inputs[0]) > self.max_len:
            inputs = self.clip_tensors(inputs)

        label = torch.tensor(label, dtype=torch.long)
        return inputs, label


class NonContiguousDataset(DenseAADataset):
    """GUE Non-Contiguous (Sparse) Dataset - sparse sampling for matched regions."""

    def __init__(self, filename: str, tokenizer: DNAKmerTokenizer,
                 split: str = "train", max_len: int = 1024, chr_id: int = 0,
                 sparse_k: int = 4):
        super().__init__(filename, tokenizer, split, max_len, chr_id)
        self.sparse_k = sparse_k
        
        # Statistics on sparse vs dense lengths
        rank = int(os.environ.get("RANK", 0))
        if rank == 0:
            sparse_lens, dense_lens = [], []
            for data in self.ds:
                if data.get("alignments"):
                    inputs_sparse = self.get_sparse_aa_inputs(data["alignments"][0])
                    inputs_dense = self.get_dense_aa_inputs(data["alignments"][0])
                    sparse_lens.append(len(inputs_sparse[0]))
                    dense_lens.append(len(inputs_dense[0]))
                    if len(sparse_lens) >= 100:
                        break
            if sparse_lens:
                print(f"✅ sparse_k={sparse_k}, avg sparse: {np.mean(sparse_lens):.1f}, "
                      f"avg dense: {np.mean(dense_lens):.1f}, ratio: {np.mean(sparse_lens)/np.mean(dense_lens):.2f}")

    def get_sparse_aa_inputs(self, alignment: dict):
        """Get sparse AA input tensors from alignment."""
        chrom = alignment["chrom"]
        chr_id = CHROM2ID[chrom]
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
                # For other operations, add all tokens
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
        positions.append(CHROM_LENGTHS[chrom] - 1)
        
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
            if self.training and random.random() < 0.5:
                inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
            else:
                inputs = self.get_dense_inputs(data["sequence"], strand=0)
        else:
            if self.training:
                p = random.random()
                if 0 <= p < 0.1:
                    inputs = self.get_dense_inputs(reverse_complement(data["sequence"]), strand=1)
                elif 0.1 <= p < 0.2:
                    inputs = self.get_dense_inputs(data["sequence"], strand=0)
                elif 0.2 <= p < 0.6:
                    weights = [max(align["mapq"], 10) for align in alignments]
                    alignment = random.choices(alignments, weights=weights, k=1)[0]
                    inputs = self.get_dense_aa_inputs(alignment)
                else:
                    weights = [max(align["mapq"], 10) for align in alignments]
                    alignment = random.choices(alignments, weights=weights, k=1)[0]
                    inputs = self.get_sparse_aa_inputs(alignment)
            else:
                inputs = self.get_sparse_aa_inputs(alignments[0])

        if len(inputs[0]) > self.max_len:
            inputs = self.clip_tensors(inputs)

        label = torch.tensor(label, dtype=torch.long)
        return inputs, label


if __name__ == "__main__":
    pass
