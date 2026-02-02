# Copyright (c) 2026, Tencent Inc. All rights reserved.
# Data: 2026/1/28 20:51
# Author: chenchenqin
from .dataset import (
    RefseqPretrainDataset,
    DNAPretrainDataset,
    Discrete1KGPretrainDataset,
    USMAAPretrainDataset,
)
from .dataset.pretrain_dataset import print_rank_0


def build_dataset(args, tokenizer):
    """Build dataset based on config"""
    if args.dataset_type == 'refseq':
        print_rank_0(f"Using RefseqPretrainDataset")
        dataset = RefseqPretrainDataset(
            args.ref_file, tokenizer,
            mask_ratio=args.mask_ratio,
            rc_ratio=args.rc_ratio,
            seq_len=args.seq_len
        )
    elif args.dataset_type == 'dna':
        print_rank_0(f"Using DNAPretrainDataset")
        dataset = DNAPretrainDataset(
            args.dna_file, tokenizer,
            mask_ratio=args.mask_ratio,
            rc_ratio=args.rc_ratio,
            seq_len=args.seq_len
        )
    elif args.dataset_type == '1kg':
        print_rank_0(f"Using Discrete1KGPretrainDataset")
        dataset = Discrete1KGPretrainDataset(
            args.n1kg_file, tokenizer,
            mask_ratio=args.mask_ratio,
            seq_len=args.seq_len
        )
    else:  # combined
        print_rank_0(f"Using USMAAPretrainDataset (combined)")
        dataset = USMAAPretrainDataset(
            ref_file=args.ref_file,
            dna_file=args.dna_file,
            n1kg_file=args.n1kg_file,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            mask_ratio=args.mask_ratio,
            rc_ratio=args.rc_ratio,
            sample_weights=args.sample_weights
        )

    print_rank_0(f"Dataset size: {len(dataset)}")
    return dataset
