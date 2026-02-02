# DNA Kmer Tokenizer
import itertools as it
import re
from typing import Optional, Union, List

import numpy as np
import torch

# Reverse complement mapping
COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}


def reverse_complement(seq: str) -> str:
    """Compute reverse complement of a sequence."""
    return ''.join(COMPLEMENT.get(b, 'N') for b in reversed(seq))


class DNAKmerTokenizer:
    """DNA K-mer Tokenizer
    
    Supports dynamic kmer (1-6mer) encoding with special tokens:
    - <unk>: Unknown token
    - <s>: Sequence start (BOS)
    - </s>: Sequence end (EOS)
    - <pad>: Padding
    - <mask>: Mask token (for MLM pretraining)
    """
    bases = ["N", "A", "C", "G", "T"]

    def __init__(
        self,
        kmer: int = 6,
        dynamic: bool = True,
        special_tokens: List[str] = None,
        with_special_tokens: bool = True
    ):
        """
        Args:
            kmer: Maximum k-mer length
            dynamic: Whether to support dynamic kmer (1 to kmer)
            special_tokens: Additional special token list
            with_special_tokens: Whether to include basic special tokens
        """
        # Basic special tokens
        if with_special_tokens:
            self.special_tokens = ["<unk>", "<s>", "</s>"]
            self.unk_id = 0
            self.bos_id = 1
            self.eos_id = 2
        else:
            self.special_tokens = []
            self.unk_id = None
            self.bos_id = None
            self.eos_id = None

        self.dynamic = dynamic
        
        # Add extra special tokens
        if special_tokens is None:
            special_tokens = []
        for stoken in special_tokens:
            assert stoken.startswith("<") and stoken.endswith(">"), \
                f"special token must be in format <token>, got: {stoken}"
            if stoken not in self.special_tokens:
                self.special_tokens.append(stoken)

        # Set special token ids
        if "<pad>" in self.special_tokens:
            self.pad_id = self.special_tokens.index("<pad>")
        else:
            self.pad_id = self.eos_id

        if "<mask>" in self.special_tokens:
            self.mask_id = self.special_tokens.index("<mask>")
        else:
            self.mask_id = None

        # Build vocabulary
        self.kmer = kmer
        self.kmers = self._get_base_kmers(self.kmer, dynamic=self.dynamic)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.special_tokens + self.kmers)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        
        # Reverse complement mapping cache
        self._rc_token_map = None

    def _get_base_kmers(self, k: int, dynamic: bool = True) -> List[str]:
        """Generate all possible k-mers."""
        kmers = []
        start_k = 1 if dynamic else k
        for i in range(start_k, k + 1):
            kmers += list(it.product(self.bases, repeat=i))
        return [''.join(m) for m in kmers]

    def __len__(self) -> int:
        return len(self.token_to_idx)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_idx)

    @property
    def vocab(self) -> dict:
        return self.token_to_idx

    @property
    def bos(self) -> int:
        return self.bos_id

    @property
    def eos(self) -> int:
        return self.eos_id

    @property
    def pad(self) -> int:
        return self.pad_id

    @property
    def mask(self) -> int:
        if self.mask_id is not None:
            return self.mask_id
        raise ValueError("mask token not defined")

    @property
    def unk(self) -> int:
        return self.unk_id

    def piece_to_id(self, token: str) -> int:
        """Convert token to id."""
        return self.token_to_idx.get(token, self.unk_id)

    def id_to_piece(self, token_id: int) -> str:
        """Convert id to token."""
        assert 0 <= token_id < len(self), f"token id {token_id} out of range [0, {len(self)})"
        return self.idx_to_token[token_id]

    def _get_rc_piece(self, piece: str) -> str:
        """Get reverse complement of a token."""
        if piece.startswith("<") or piece.endswith(">"):
            return piece
        return reverse_complement(piece)

    def _get_rc_token_map(self) -> dict:
        """Build reverse complement mapping."""
        if self._rc_token_map is None:
            self._rc_token_map = {}
            for piece in self.vocab:
                rc_piece = self._get_rc_piece(piece)
                self._rc_token_map[self.piece_to_id(piece)] = self.piece_to_id(rc_piece)
        return self._rc_token_map

    def get_reverse_complement_token(self, token: int) -> int:
        """Get reverse complement token id for a given token."""
        rc_map = self._get_rc_token_map()
        return rc_map[token]

    def tokenize(self, text: str) -> List[str]:
        """Split sequence into tokens."""
        tokens = re.split(r"[<>]", text)
        new_tokens = []
        n = self.kmer
        for t in tokens:
            if not t:
                continue
            if f"<{t}>" in self.special_tokens:
                new_tokens.append(f"<{t}>")
            else:
                # Split by kmer
                chunks = [t[i:i + n] for i in range(0, len(t), n)]
                new_tokens += chunks
        return new_tokens

    def _encode_piece(self, token) -> int:
        """Encode a single piece."""
        if isinstance(token, (list, tuple, np.ndarray)):
            return [self._encode_piece(t) for t in token]
        return self.piece_to_id(token)

    def encode(
        self,
        text: Union[str, List[str]],
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
        pad: bool = False,
        to_tensor: bool = True,
        device: Optional[torch.device] = None
    ) -> Union[torch.Tensor, List[int]]:
        """Encode sequence.
        
        Args:
            text: DNA sequence string or token list
            bos: Whether to add BOS token
            eos: Whether to add EOS token
            max_length: Maximum length (-1 means no limit)
            pad: Whether to pad to max_length
            to_tensor: Whether to return tensor
            device: Tensor device
            
        Returns:
            Token ids
        """
        if not self.dynamic:
            if isinstance(text, str):
                assert len(text) % self.kmer == 0, \
                    f"sequence length {len(text)} is not divisible by {self.kmer}"
        
        # Tokenize
        if isinstance(text, str):
            pieces = self.tokenize(text)
        else:
            pieces = text
        
        # Encode
        tokens = self._encode_piece(pieces)
        
        # Add special tokens
        if bos:
            tokens = [self.bos_id] + tokens
        if eos:
            tokens = tokens + [self.eos_id]
        
        # Truncate
        if max_length > 0:
            tokens = tokens[:max_length]
        
        # Pad
        if pad and max_length > 0 and len(tokens) < max_length:
            tokens += [self.pad_id] * (max_length - len(tokens))
        
        if to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
        
        return tokens

    def decode(self, token_ids: Union[torch.Tensor, np.ndarray, List[int]]) -> str:
        """Decode token ids to sequence."""
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()
        
        seq = "".join([
            self.id_to_piece(tid) 
            for tid in token_ids 
            if tid not in (self.bos_id, self.eos_id, self.pad_id)
        ])
        return seq

    def __call__(
        self,
        text: Union[str, List[str]],
        bos: bool = False,
        eos: bool = False,
        max_length: int = -1,
        to_tensor: bool = True,
        **kwargs
    ) -> Union[torch.Tensor, List[int]]:
        """Call encode method."""
        return self.encode(text, bos=bos, eos=eos, max_length=max_length, to_tensor=to_tensor, **kwargs)


def build_tokenizer(
    kmer: int = 6,
    dynamic: bool = True,
    special_tokens: List[str] = None
) -> DNAKmerTokenizer:
    """Build tokenizer.
    
    Args:
        kmer: K-mer length
        dynamic: Whether to support dynamic kmer
        special_tokens: Special tokens list. If None, use default tokens.
        
    Returns:
        DNAKmerTokenizer instance
    """
    if special_tokens is None:
        # Default special tokens
        special_tokens = [
            "<pad>", "<mask>",
            "<R1>", "<R2>", "<R3>", "<R4>", "<R5>", "<R6>",
            "<R7>", "<R8>", "<R9>", "<R10>",
            "<R11>", "<R12>", "<R13>", "<R14>",
            "<R15>", "<R16>", "<R17>",
        ]
    
    return DNAKmerTokenizer(
        kmer=kmer,
        dynamic=dynamic,
        special_tokens=special_tokens,
        with_special_tokens=True
    )

if __name__ == "__main__":
    main()