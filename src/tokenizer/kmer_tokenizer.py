# -*- coding: utf-8 -*-
# Copyright (c) 2023, Tencent Inc. All rights reserved.
# Author: chenchenqin
# Data: 2023/5/18 11:00
import itertools as it
import re
import sys
from enum import Enum
from typing import Optional, Union
from Bio.Seq import Seq
import numpy as np
import torch

from src.config import configurable
from .base_tokenizer import AbstractTokenizer
from .build import TOKENIZER_REGISTRY


@TOKENIZER_REGISTRY.register()
class KmerTokenizer(AbstractTokenizer):
    # N is unkown base in reference seqence
    bases = ["N", "A", "C", "G", "T"]

    @classmethod
    def from_config(cls, cfg):
        return {
            "kmer": cfg.tokenizer.kmer,
            "dynamic": cfg.tokenizer.get("dynamic", True),
            "special_tokens": cfg.tokenizer.special_tokens,
            "with_special_tokens": cfg.tokenizer.get("with_special_tokens", True)
        }

    @configurable
    def __init__(self, kmer=3, dynamic=True, special_tokens=None, with_special_tokens=True):
        super().__init__("KmerTokenizer")
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
        if special_tokens is None:
            special_tokens = []

        for stoken in special_tokens:
            assert stoken[0] == "<" and stoken[-1] == ">", f"only support special token: <token>"
            if stoken in self.special_tokens:
                print(f"token: {stoken} already in special token, skip!", file=sys.stderr, flush=True)
                continue
            self.special_tokens.append(stoken)

        if "<pad>" in self.special_tokens:
            self.pad_id = self.special_tokens.index("<pad>")
        else:
            self.pad_id = self.eos_id

        if "<mask>" in self.special_tokens:
            self.mask_id = self.special_tokens.index("<mask>")

        self.kmer = kmer
        self.kmers = self.get_base_kmers(self.kmer, dynamic=self.dynamic)
        self.token_to_idx = {token: idx for idx, token in enumerate(self.special_tokens + self.kmers)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}
        self.rc_token_map = None

    def get_rc_piece(self, piece):
        if piece.startswith("<") or piece.endswith(">"):
            return piece
        return str(Seq(piece).reverse_complement())

    def get_rc_token_map(self):
        if not self.rc_token_map:
            self.rc_token_map = {}
            for piece in self.vocab:
                rc_piece = self.get_rc_piece(piece)
                self.rc_token_map[self.piece_to_id(piece)] = self.piece_to_id(rc_piece)
        return self.rc_token_map

    def get_reverse_complement_token(self, token):
        rc_map = self.get_rc_token_map()
        return rc_map[token]

    def piece_to_id(self, token: str):
        return self.token_to_idx.get(token, self.unk_id)

    def id_to_piece(self, token_id):
        assert 0 <= token_id < len(self), f"out of range of token id {token_id}, max {len(self)}"
        return self.idx_to_token[token_id]

    @property
    def mask(self):
        if hasattr(self, "mask_id"):
            return self.mask_id
        else:
            raise f"not set mask id"

    @property
    def bos(self):
        return self.bos_id

    @property
    def eos(self):
        return self.eos_id

    @property
    def pad(self):
        return self.pad_id

    @property
    def vocab_size(self):
        return len(self.token_to_idx)

    @property
    def vocab(self):
        return self.token_to_idx

    @classmethod
    def get_vocab_size(cls, k, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        reserved_tokens = list(reserved_tokens)
        n_bases = len(cls.bases)
        n_base_kmer = sum([n_bases ** k for k in range(1, k + 1)])

        return n_base_kmer + len(reserved_tokens)

    def get_base_kmers(self, k, dynamic=True):
        kmers = []
        for i in range(1 if dynamic else k, k + 1):
            kmers += list(it.product(self.bases, repeat=i))

        return [''.join(m) for m in kmers]

    def _encode(self, token):
        if not isinstance(token, (list, tuple, np.ndarray)):
            return self.piece_to_id(token)

        return [self._encode(t) for t in token]

    def tokenize(self, text):
        tokens = re.split(r"[<>]", text)
        new_tokens = []
        n = self.kmer
        for t in tokens:
            if not t:
                continue

            if f"<{t}>" in self.special_tokens:
                new_tokens.append(f"<{t}>")
            else:
                seq = t
                # split kmers
                chunks = [seq[i:i + n] for i in range(0, len(seq), n)]
                new_tokens += chunks
        return new_tokens

    def encode(
            self,
            text: str,
            bos: bool = False,
            eos: bool = False,
            max_length: int = -1,
            pad: bool = False,
            device: Optional[torch.device] = None,
            to_tensor=True
    ) -> torch.Tensor:
        if not self.dynamic:
            assert len(text) % self.kmer == 0, f"input sequence length {len(text)} is not divisible by {self.kmer}mer"
        if isinstance(text, str):
            pieces = self.tokenize(text)
        else:
            assert isinstance(text, (list, tuple, np.ndarray)), f"only support list of pieces input"
            pieces = text

        tokens = self._encode(pieces)
        if bos:
            tokens = [self.bos_id] + tokens

        if eos:
            tokens = tokens + [self.eos_id]

        if max_length > 0:
            tokens = tokens[:max_length]

        if pad and len(tokens) < max_length:
            tokens += [self.pad] * (max_length - len(tokens))

        if to_tensor:
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)

        return tokens

    def decode(self, token_ids: Union[torch.Tensor, np.ndarray]) -> str:
        if isinstance(token_ids, (torch.Tensor, np.ndarray)):
            token_ids = token_ids.tolist()

        seq = "".join([self.id_to_piece(tid) for tid in token_ids if tid not in (self.bos, self.eos)])
        return seq

    @property
    def unk(self):
        return self.unk_id


DNAKmerTokenizer = KmerTokenizer
TOKENIZER_REGISTRY.register("DNAKmerTokenizer", DNAKmerTokenizer)
