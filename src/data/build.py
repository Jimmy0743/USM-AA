
from .dataset import (
    RefseqPretrainDataset,
    DNAPretrainDataset,
    Discrete1KGPretrainDataset,
    USMAAPretrainDataset,
    DenseDataset,
    DenseAADataset,
    NonContiguousDataset,
)
from .dataset.pretrain_dataset import print_rank_0


def get_cv_val_folds(cv: int, seed: int) -> list:
    """Get validation fold IDs based on CV folds and seed.
    
    For 5-fold CV:
        seed 0 -> val folds [0, 1]
        seed 1 -> val folds [2, 3]
        seed 2 -> val fold [4]
    
    For 10-fold CV:
        seed 0 -> val fold [0]
        seed 1 -> val fold [1]
        ...
        seed 9 -> val fold [9]
    
    Args:
        cv: Number of CV folds (5 or 10)
        seed: Random seed (0, 1, 2 for 5-fold; 0-9 for 10-fold)
        
    Returns:
        List of fold IDs to use as validation set
    """
    if cv == 5:
        # 5-fold: seed 0 -> [0,1], seed 1 -> [2,3], seed 2 -> [4]
        if seed == 0:
            return [0, 1]
        elif seed == 1:
            return [2, 3]
        elif seed == 2:
            return [4]
        else:
            raise ValueError(f"Invalid seed {seed} for 5-fold CV. Must be 0, 1, or 2.")
    elif cv == 10:
        # 10-fold: seed directly maps to fold
        if 0 <= seed <= 9:
            return [seed]
        else:
            raise ValueError(f"Invalid seed {seed} for 10-fold CV. Must be 0-9.")
    else:
        raise ValueError(f"Unsupported CV folds: {cv}. Must be 5 or 10.")


def build_dataset(args, tokenizer):
    """Build dataset based on config
    
    args.train is always a dict:
    - For USMAAPretrainDataset: {ref_file: path, dna_file: path, n1kg_file: path}
    - For RefseqPretrainDataset: {ref_file: path}
    - For DNAPretrainDataset: {dna_file: path}
    - For Discrete1KGPretrainDataset: {n1kg_file: path}
    """
    if args.dataset_type == 'RefseqPretrainDataset':
        print_rank_0(f"Using RefseqPretrainDataset")
        ref_file = args.train.get('ref_file')
        dataset = RefseqPretrainDataset(
            ref_file, tokenizer,
            mask_ratio=args.mask_ratio,
            rc_ratio=args.rc_ratio,
            seq_len=args.seq_len
        )
    elif args.dataset_type == 'DNAPretrainDataset':
        print_rank_0(f"Using DNAPretrainDataset")
        dna_file = args.train.get('dna_file')
        dataset = DNAPretrainDataset(
            dna_file, tokenizer,
            mask_ratio=args.mask_ratio,
            rc_ratio=args.rc_ratio,
            seq_len=args.seq_len
        )
    elif args.dataset_type == 'Discrete1KGPretrainDataset':
        print_rank_0(f"Using Discrete1KGPretrainDataset")
        n1kg_file = args.train.get('n1kg_file')
        dataset = Discrete1KGPretrainDataset(
            n1kg_file, tokenizer,
            mask_ratio=args.mask_ratio,
            seq_len=args.seq_len
        )
    elif args.dataset_type == 'USMAAPretrainDataset':
        print_rank_0(f"Using USMAAPretrainDataset")
        # args.train is a dict containing ref_file, dna_file, n1kg_file
        dataset = USMAAPretrainDataset(
            train=args.train,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            mask_ratio=args.mask_ratio,
            rc_ratio=args.rc_ratio,
            sample_weights=args.sample_weights
        )
    else:
        raise ValueError(f"Unknown pretrain dataset_type: {args.dataset_type}")

    print_rank_0(f"Dataset size: {len(dataset)}")
    return dataset


def build_finetune_dataset(args, tokenizer, split="train"):
    """Build finetune dataset based on config.
    
    Args:
        args: Config with attributes:
            - dataset_type: 'DenseDataset', 'DenseAADataset', or 'NonContiguousDataset'
            - train: path to training file
            - dev: path to dev file (None if not available)
            - test: path to test file (None if not available)
            - max_len: Maximum sequence length
            - sparse_k: Sparse sampling interval (for non_contiguous)
            - cv: Number of CV folds (5 or 10, None for no CV)
            - cv_seed: Seed for CV split selection
        tokenizer: DNAKmerTokenizer instance
        split: Dataset split ('train', 'val', 'test')
        
    Returns:
        Dataset instance
    """
    # Get file path based on split
    if split == "train":
        filename = args.train
    elif split == "val":
        filename = args.dev
    else:
        filename = args.test
    
    # Handle cross-validation
    cv = getattr(args, 'cv', None)
    cv_seed = getattr(args, 'cv_seed', 0)
    val_folds = None
    
    if cv is not None:
        val_folds = get_cv_val_folds(cv, cv_seed)
        print_rank_0(f"Cross-validation: {cv}-fold, seed={cv_seed}, val_folds={val_folds}")
        # For CV, train and val both use the train file (cv.jsonl)
        # train: all folds except val_folds
        # val: val_folds only
        if split in ["train", "val"]:
            filename = args.train
    
    max_len = getattr(args, 'max_len', 1024)
    rc_aug = getattr(args, 'rc_aug', True)
    
    if args.dataset_type == 'DenseDataset':
        print_rank_0(f"Using DenseDataset for {split}")
        dataset = DenseDataset(
            filename=filename,
            tokenizer=tokenizer,
            split=split,
            max_len=max_len,
            rc_aug=rc_aug,
            cv_val_folds=val_folds if cv else None
        )
    elif args.dataset_type == 'DenseAADataset':
        print_rank_0(f"Using DenseAADataset for {split}")
        use_alignment_prob = getattr(args, 'use_alignment_prob', 0.8)
        dataset = DenseAADataset(
            filename=filename,
            tokenizer=tokenizer,
            split=split,
            max_len=max_len,
            rc_aug=rc_aug,
            use_alignment_prob=use_alignment_prob,
            cv_val_folds=val_folds if cv else None
        )
    elif args.dataset_type == 'NonContiguousDataset':
        print_rank_0(f"Using NonContiguousDataset for {split}")
        sparse_k = getattr(args, 'sparse_k', 4)
        dense_prob = getattr(args, 'dense_prob', 0.4)
        use_alignment_prob = getattr(args, 'use_alignment_prob', 0.8)
        dataset = NonContiguousDataset(
            filename=filename,
            tokenizer=tokenizer,
            split=split,
            max_len=max_len,
            rc_aug=rc_aug,
            sparse_k=sparse_k,
            dense_prob=dense_prob,
            use_alignment_prob=use_alignment_prob,
            cv_val_folds=val_folds if cv else None
        )
    else:
        raise ValueError(f"Unknown finetune dataset_type: {args.dataset_type}")
    
    print_rank_0(f"{split} dataset size: {len(dataset)}")
    return dataset
