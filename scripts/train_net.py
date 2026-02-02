import argparse
import os
import time
import yaml
import deepspeed
import torch
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Local imports
from src.model.build import build_model, print_model_info
from src.tokenizer.build import build_tokenizer
from src.data.build import build_dataset
from src.engine.deepspeed import MLMLoss, train_epoch, save_checkpoint, load_checkpoint, print_rank_0


def get_args():
    parser = argparse.ArgumentParser(description="USM-AA Pretraining with DeepSpeed")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Data arguments
    parser.add_argument("--dataset_type", type=str, default=None,
                        choices=["refseq", "dna", "1kg", "combined"],
                        help="Dataset type: refseq, dna, 1kg, or combined")
    parser.add_argument("--ref_file", type=str, default=None, help="Reference genome FASTA file")
    parser.add_argument("--dna_file", type=str, default=None, help="Multi-species DNA JSONL file")
    parser.add_argument("--n1kg_file", type=str, default=None, help="1KG variants JSONL file")
    parser.add_argument("--seq_len", type=int, default=None, help="Sequence length")

    # Model arguments
    parser.add_argument("--model_type", type=str, default=None,
                        choices=["usm-aa-100m", "usm-aa-500m", "usm-aa-1b"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")

    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping")
    parser.add_argument("--mask_ratio", type=float, default=None, help="MLM mask ratio")
    parser.add_argument("--rc_ratio", type=float, default=None, help="Reverse complement ratio")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")

    # System arguments
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log_freq", type=int, default=None, help="Log frequency")
    parser.add_argument("--save_freq", type=int, default=None, help="Save checkpoint frequency")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument("--activation_checkpoint", action="store_true", help="Use activation checkpointing")

    # DeepSpeed arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    parser.add_argument("--deepspeed", type=str, default=None, help="DeepSpeed config file path")

    args = parser.parse_args()

    # Load defaults from yaml config file
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

        # Data config
        data_cfg = cfg.get('data', {})
        if args.dataset_type is None:
            args.dataset_type = data_cfg.get('dataset_type', 'refseq')
        if args.ref_file is None:
            args.ref_file = data_cfg.get('ref_file')
        if args.dna_file is None:
            args.dna_file = data_cfg.get('dna_file')
        if args.n1kg_file is None:
            args.n1kg_file = data_cfg.get('n1kg_file')
        if args.seq_len is None:
            args.seq_len = data_cfg.get('seq_len', 256)
        args.sample_weights = data_cfg.get('sample_weights', {"ref": 1.0})

        # Model config
        model_cfg = cfg.get('model', {})
        if args.model_type is None:
            args.model_type = model_cfg.get('type', 'usm-aa-100m')

        # Training config
        train_cfg = cfg.get('training', {})
        if args.batch_size is None:
            args.batch_size = train_cfg.get('batch_size', 72)
        if args.num_epochs is None:
            args.num_epochs = train_cfg.get('num_epochs', 20)
        if args.lr is None:
            args.lr = train_cfg.get('learning_rate', 1e-4)
        if args.weight_decay is None:
            args.weight_decay = train_cfg.get('weight_decay', 0.1)
        if args.warmup_steps is None:
            args.warmup_steps = train_cfg.get('warmup_steps', 10000)
        if args.grad_clip is None:
            args.grad_clip = train_cfg.get('grad_clip', 1.0)
        if args.mask_ratio is None:
            args.mask_ratio = train_cfg.get('mask_ratio', 0.15)
        if args.rc_ratio is None:
            args.rc_ratio = train_cfg.get('rc_ratio', 0.3)
        if args.gradient_accumulation_steps == 1:
            args.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)

        # DataLoader config
        loader_cfg = cfg.get('dataloader', {})
        if args.num_workers is None:
            args.num_workers = loader_cfg.get('num_workers', 8)

        # Precision config
        prec_cfg = cfg.get('precision', {})
        if not args.bf16:
            args.bf16 = prec_cfg.get('bf16', False)
        if not args.fp16:
            args.fp16 = prec_cfg.get('fp16', False)
        if not args.activation_checkpoint:
            args.activation_checkpoint = prec_cfg.get('activation_checkpoint', False)

        # Logging config
        log_cfg = cfg.get('logging', {})
        if args.output_dir is None:
            args.output_dir = log_cfg.get('output_dir', './output')
        if args.log_freq is None:
            args.log_freq = log_cfg.get('log_freq', 10)
        if args.save_freq is None:
            args.save_freq = log_cfg.get('save_freq', 5000)

    # Set defaults (if no config file and not specified via command line)
    if args.dataset_type is None:
        args.dataset_type = 'refseq'
    if args.seq_len is None:
        args.seq_len = 256
    if not hasattr(args, 'sample_weights'):
        args.sample_weights = {"ref": 1.0}
    if args.model_type is None:
        args.model_type = 'usm-aa-100m'
    if args.batch_size is None:
        args.batch_size = 72
    if args.num_epochs is None:
        args.num_epochs = 20
    if args.lr is None:
        args.lr = 1e-4
    if args.weight_decay is None:
        args.weight_decay = 0.1
    if args.warmup_steps is None:
        args.warmup_steps = 10000
    if args.grad_clip is None:
        args.grad_clip = 1.0
    if args.mask_ratio is None:
        args.mask_ratio = 0.15
    if args.rc_ratio is None:
        args.rc_ratio = 0.3
    if args.num_workers is None:
        args.num_workers = 8
    if args.output_dir is None:
        args.output_dir = './output'
    if args.log_freq is None:
        args.log_freq = 10
    if args.save_freq is None:
        args.save_freq = 5000

    # Validate required arguments
    if args.dataset_type in ['refseq', 'combined'] and args.ref_file is None:
        raise ValueError("--ref_file is required for refseq/combined dataset")
    if args.dataset_type == 'dna' and args.dna_file is None:
        raise ValueError("--dna_file is required for dna dataset")
    if args.dataset_type == '1kg' and args.n1kg_file is None:
        raise ValueError("--n1kg_file is required for 1kg dataset")

    return args


def main():
    args = get_args()

    # Load yaml config for deepspeed settings
    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    # DeepSpeed initialize distributed
    deepspeed.init_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    # Create output directory
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Build tokenizer
    tokenizer = build_tokenizer(kmer=6, dynamic=True)
    print_rank_0(f"Tokenizer vocab size: {len(tokenizer)}")

    # Build dataset
    dataset = build_dataset(args, tokenizer)

    # DataLoader
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        collate_fn=dataset.collate_fn,
        pin_memory=True,
        prefetch_factor=2
    )

    # Build model using build_model module
    model, config = build_model(
        vocab_size=len(tokenizer),
        model_type=args.model_type,
    )
    print_model_info(model, config)

    # Compute total_steps
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # DeepSpeed config - load from yaml if available
    ds_config_from_yaml = cfg.get('deepspeed', {})

    ds_config = {
        "train_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": ds_config_from_yaml.get("gradient_clipping", args.grad_clip),
        "prescale_gradients": ds_config_from_yaml.get("prescale_gradients", False),
        "steps_per_print": ds_config_from_yaml.get("steps_per_print", args.log_freq),
        "wall_clock_breakdown": ds_config_from_yaml.get("wall_clock_breakdown", False),
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": args.weight_decay
            }
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": args.warmup_steps,
                "total_num_steps": total_steps
            }
        },
        "zero_optimization": ds_config_from_yaml.get("zero_optimization", {
            "stage": 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 5e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "contiguous_gradients": True
        }),
        "bfloat16": ds_config_from_yaml.get("bfloat16", {"enabled": args.bf16}),
        "fp16": ds_config_from_yaml.get("fp16", {"enabled": args.fp16}),
        "activation_checkpointing": {
            "partition_activations": args.activation_checkpoint,
            "cpu_checkpointing": False,
            "contiguous_memory_optimization": args.activation_checkpoint
        },
    }

    # Add tensorboard if configured
    if ds_config_from_yaml.get("tensorboard", {}).get("enabled", False):
        ds_config["tensorboard"] = ds_config_from_yaml["tensorboard"]

    # Add data_sampling if configured
    if ds_config_from_yaml.get("data_sampling", {}).get("enabled", False):
        ds_config["data_sampling"] = ds_config_from_yaml["data_sampling"]

    # Add zero_allow_untested_optimizer if configured
    if "zero_allow_untested_optimizer" in ds_config_from_yaml:
        ds_config["zero_allow_untested_optimizer"] = ds_config_from_yaml["zero_allow_untested_optimizer"]

    # If external DeepSpeed config file provided, load and merge (override auto values)
    if args.deepspeed and os.path.exists(args.deepspeed):
        print_rank_0(f"Loading DeepSpeed config from: {args.deepspeed}")
        import json
        with open(args.deepspeed, 'r') as f:
            external_config = json.load(f)

        # Override with external config, but keep our computed values (replace "auto")
        def merge_config(base, override):
            """Recursively merge configs, skip 'auto' values"""
            for key, value in override.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    merge_config(base[key], value)
                elif value != "auto":
                    base[key] = value
            return base

        ds_config = merge_config(ds_config, external_config)

    # Initialize DeepSpeed
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )

    # Loss function
    criterion = MLMLoss(vocab_size=len(tokenizer))

    # Load checkpoint
    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        start_epoch, global_step = load_checkpoint(model_engine, args)
        print_rank_0(f"Resumed from epoch {start_epoch}, step {global_step}")

    # TensorBoard
    writer = None
    if rank == 0:
        writer = SummaryWriter(Path(args.output_dir) / "tensorboard")

    # Training loop
    zero_stage = ds_config.get("zero_optimization", {}).get("stage", 1)
    print_rank_0(f"Starting DeepSpeed training: {args.num_epochs} epochs, {len(dataloader)} batches/epoch")
    print_rank_0(f"ZeRO stage: {zero_stage}, BF16: {args.bf16}, Activation Checkpoint: {args.activation_checkpoint}")

    for epoch in range(start_epoch, args.num_epochs):
        sampler.set_epoch(epoch)

        avg_loss, avg_acc, global_step, epoch_time = train_epoch(
            model_engine, dataloader, criterion, device, epoch, args, writer, global_step
        )

        print_rank_0(
            f"Epoch {epoch} completed | "
            f"Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f} | "
            f"Time: {epoch_time / 60:.1f}min"
        )

        # Save epoch checkpoint
        save_checkpoint(model_engine, epoch + 1, global_step, args)

    if writer is not None:
        writer.close()

    print_rank_0("DeepSpeed training completed!")


if __name__ == "__main__":
    main()
