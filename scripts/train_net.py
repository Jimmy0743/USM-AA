"""USM-AA training script with DeepSpeed.

This script provides the main training loop for USM-AA models, supporting
both pretraining (MLM) and fine-tuning (classification) tasks. It uses
DeepSpeed for distributed training with ZeRO optimization.

"""
import argparse
import os
import yaml
import deepspeed
import torch
from pathlib import Path
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from src.model.build import build_model, load_pretrained_weights, print_model_info
from src.tokenizer.build import build_tokenizer
from src.data.build import build_dataset, build_finetune_dataset
from src.loss import build_loss
from src.engine.deepspeed import (
    train_epoch,
    train_epoch_finetune,
    save_checkpoint,
    load_checkpoint,
    print_rank_0
)
from src.evaluator import build_evaluator


def get_args() -> argparse.Namespace:
    """Parse command line arguments and load configuration.
    
    Returns:
        Namespace containing all configuration parameters.
        
    Raises:
        ValueError: If required configuration parameters are missing.
    """
    parser = argparse.ArgumentParser(description="USM-AA Training with DeepSpeed")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file"
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluation only mode (skip training)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed config file path (optional, overrides yaml)"
    )

    args = parser.parse_args()

    # Load configuration from YAML
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Get config file directory for resolving relative paths
    config_dir = os.path.dirname(os.path.abspath(args.config))

    # Dataset configuration
    dataset_cfg = cfg.get('dataset', {})
    args.dataset_type = dataset_cfg.get('type')
    args.seq_len = dataset_cfg.get('seq_len')
    args.sample_weights = dataset_cfg.get('sample_weights', {"ref": 1.0})
    args.max_len = dataset_cfg.get('max_len')
    args.num_classes = dataset_cfg.get('num_classes')
    
    # Cross-validation configuration
    # cv: number of folds (5 for 5-fold, 10 for 10-fold, None for no CV)
    # seed: random seed, determines which fold(s) to use as validation
    #   - 5-fold: seed 0 -> val folds [0,1], seed 1 -> val folds [2,3], seed 2 -> val fold [4]
    #   - 10-fold: seed 0 -> val fold [0], seed 1 -> val fold [1], etc.
    args.cv = dataset_cfg.get('cv', None)
    args.cv_seed = dataset_cfg.get('seed', 0)
    
    # Parse data paths based on dataset_type
    # - DenseDataset | DenseAADataset | NonContiguousDataset: use data_dir + files format
    # - Others (pretrain): use train dict format {ref_file, dna_file, n1kg_file}
    if args.dataset_type in ['DenseDataset', 'DenseAADataset', 'NonContiguousDataset']:
        data_dir = dataset_cfg.get('data_dir')
        # Resolve relative data_dir path
        if data_dir and not os.path.isabs(data_dir):
            data_dir = os.path.normpath(os.path.join(config_dir, data_dir))
        files = dataset_cfg.get('files', {})
        if data_dir and files:
            args.train = os.path.join(data_dir, files.get('train')) if files.get('train') else None
            args.dev = os.path.join(data_dir, files.get('dev')) if files.get('dev') else None
            args.test = os.path.join(data_dir, files.get('test')) if files.get('test') else None
        else:
            raise ValueError(
                "data_dir and files are required for DenseDataset/DenseAADataset/NonContiguousDataset"
            )
    else:
        train_cfg_data = dataset_cfg.get('train', {})
        # Resolve relative paths in train dict
        if isinstance(train_cfg_data, dict):
            for key in ['ref_file', 'dna_file', 'n1kg_file']:
                if train_cfg_data.get(key) and not os.path.isabs(train_cfg_data[key]):
                    train_cfg_data[key] = os.path.normpath(
                        os.path.join(config_dir, train_cfg_data[key])
                    )
        args.train = train_cfg_data
        args.dev = None
        args.test = None

    # Model configuration
    model_cfg = cfg.get('model', {})
    args.model_type = model_cfg.get('type')
    args.checkpoint = model_cfg.get('checkpoint')
    # Resolve relative checkpoint path
    if args.checkpoint and not os.path.isabs(args.checkpoint):
        args.checkpoint = os.path.normpath(os.path.join(config_dir, args.checkpoint))
    args.model_name = model_cfg.get('name')

    # Loss configuration
    args.loss = cfg.get('loss', None)

    # Training configuration
    train_cfg = cfg.get('training', {})
    args.batch_size = train_cfg.get('batch_size')
    args.num_epochs = train_cfg.get('num_epochs')
    args.lr = train_cfg.get('learning_rate')
    args.weight_decay = train_cfg.get('weight_decay')
    args.warmup_steps = train_cfg.get('warmup_steps')
    args.warmup_ratio = train_cfg.get('warmup_ratio')
    args.grad_clip = train_cfg.get('grad_clip', 1.0)
    args.mask_ratio = train_cfg.get('mask_ratio')
    args.rc_ratio = train_cfg.get('rc_ratio')
    args.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
    args.early_stopping_patience = train_cfg.get('early_stopping_patience', None)

    # DataLoader configuration
    loader_cfg = cfg.get('dataloader', {})
    args.num_workers = loader_cfg.get('num_workers', 8)

    # Precision configuration
    prec_cfg = cfg.get('precision', {})
    args.bf16 = prec_cfg.get('bf16', False)
    args.fp16 = prec_cfg.get('fp16', False)
    args.activation_checkpoint = prec_cfg.get('activation_checkpoint', False)

    # Logging configuration
    log_cfg = cfg.get('logging', {})
    output_dir = log_cfg.get('output_dir', './output')
    
    # Resolve relative output_dir path
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(config_dir, output_dir))
    args.output_dir = output_dir
    
    args.log_freq = log_cfg.get('log_freq', 10)
    save_freq = log_cfg.get('save_freq', None)
    # If save_freq is 'none' or None, only save by epoch (not by steps)
    args.save_freq = None if save_freq in ['none', 'None', None] else save_freq
    args.save_best_only = log_cfg.get('save_best_only', False)

    # Validate required arguments
    required_args = {
        'dataset_type': args.dataset_type,
        'model_type': args.model_type,
        'model_name': args.model_name,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'output_dir': args.output_dir,
    }
    
    missing_args = [k for k, v in required_args.items() if v is None]
    if missing_args:
        raise ValueError(
            f"Missing required arguments: {missing_args}. Please specify in config file."
        )

    # Dataset-specific validation
    train = args.train
    if train is None:
        raise ValueError("train is required in dataset config")
    
    if args.dataset_type == 'RefseqPretrainDataset':
        if not isinstance(train, dict) or not train.get('ref_file'):
            raise ValueError("train.ref_file is required for RefseqPretrainDataset")
    elif args.dataset_type == 'DNAPretrainDataset':
        if not isinstance(train, dict) or not train.get('dna_file'):
            raise ValueError("train.dna_file is required for DNAPretrainDataset")
    elif args.dataset_type == 'Discrete1KGPretrainDataset':
        if not isinstance(train, dict) or not train.get('n1kg_file'):
            raise ValueError("train.n1kg_file is required for Discrete1KGPretrainDataset")
    elif args.dataset_type == 'USMAAPretrainDataset':
        if not isinstance(train, dict) or not any([
            train.get('ref_file'),
            train.get('dna_file'),
            train.get('n1kg_file')
        ]):
            raise ValueError(
                "At least one data file (ref_file/dna_file/n1kg_file) is required "
                "for USMAAPretrainDataset"
            )
    elif args.dataset_type in ['DenseDataset', 'DenseAADataset', 'NonContiguousDataset']:
        if not isinstance(train, str):
            raise ValueError(
                "train must be a file path string for "
                "DenseDataset/DenseAADataset/NonContiguousDataset"
            )

    return args


@torch.no_grad()
def evaluate(
    model_engine,
    dataloader: DataLoader,
    evaluator,
    device: torch.device,
    args: argparse.Namespace
) -> dict:
    """Evaluate model on validation or test set.
    
    Args:
        model_engine: DeepSpeed model engine.
        dataloader: DataLoader for evaluation data.
        evaluator: ClassificationEvaluator instance.
        device: Device to run evaluation on.
        args: Training arguments.
    
    Returns:
        Dictionary containing evaluation metrics (loss, accuracy, f1, mcc).
    """
    model_engine.eval()
    evaluator.reset()
    
    model_dtype = torch.bfloat16 if args.bf16 else (
        torch.float16 if args.fp16 else torch.float32
    )
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move data to device
        token_ids = inputs["token_ids"].to(device)
        chr_ids = inputs["chr_ids"].to(device)
        position_ids = inputs["position_ids"].to(device)
        strand_ids = inputs["strand_ids"].to(device)
        cigar_ids = inputs["cigar_ids"].to(device)
        seq_mask = inputs["seq_mask"].to(device, dtype=model_dtype)
        labels = targets["labels"].to(device)
        
        # Forward pass
        outputs = model_engine.module.forward(
            token_ids=token_ids,
            position_ids=position_ids,
            chr_ids=chr_ids,
            strand_ids=strand_ids,
            cigar_ids=cigar_ids,
            attention_mask=seq_mask
        )
        logits = outputs.logits
        
        # Accumulate predictions
        evaluator.add_batch(logits, labels, compute_loss=True)
    
    # Compute final metrics
    metrics = evaluator.compute_metrics()
    
    model_engine.train()
    return metrics


def main() -> None:
    """Main training entry point."""
    args = get_args()

    # Load YAML config for DeepSpeed settings
    cfg = {}
    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)

    # Initialize DeepSpeed distributed training
    deepspeed.init_distributed()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    device = torch.device(f"cuda:{local_rank}")

    # Create output directory
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print_rank_0(f"Model: {args.model_name}")
    print_rank_0(f"Dataset type: {args.dataset_type}")

    # Build tokenizer
    tokenizer = build_tokenizer(args.config)
    print_rank_0(f"Tokenizer vocab size: {len(tokenizer)}")

    # Build dataset based on type
    pretrain_types = [
        'RefseqPretrainDataset',
        'DNAPretrainDataset',
        'Discrete1KGPretrainDataset',
        'USMAAPretrainDataset'
    ]
    finetune_types = ['DenseDataset', 'DenseAADataset', 'NonContiguousDataset']
    
    if args.dataset_type in pretrain_types:
        dataset = build_dataset(args, tokenizer)
        val_dataset = None
    elif args.dataset_type in finetune_types:
        dataset = build_finetune_dataset(args, tokenizer, split="train")
        # For CV mode, always create val_dataset from train file
        if args.cv is not None:
            val_dataset = build_finetune_dataset(args, tokenizer, split="val")
        elif args.dev:
            val_dataset = build_finetune_dataset(args, tokenizer, split="val")
        else:
            val_dataset = None
    else:
        raise ValueError(
            f"Unknown dataset_type: {args.dataset_type}. "
            f"Available: {pretrain_types + finetune_types}"
        )

    # Create training DataLoader
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

    # Create validation DataLoader
    val_dataloader = None
    if val_dataset is not None:
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            sampler=val_sampler,
            num_workers=args.num_workers,
            collate_fn=val_dataset.collate_fn,
            pin_memory=True,
            prefetch_factor=2
        )

    # Build model
    model, config = build_model(
        vocab_size=len(tokenizer),
        model_type=args.model_type,
        model_name=args.model_name,
        num_classes=args.num_classes,
    )
    print_model_info(model, config)

    # Load pretrained weights if specified
    if args.checkpoint:
        print_rank_0(f"Loading pretrained weights from: {args.checkpoint}")
        load_pretrained_weights(model, args.checkpoint, strict=False)

    # Compute total training steps
    total_steps = len(dataloader) * args.num_epochs // args.gradient_accumulation_steps

    # Compute warmup steps
    if args.warmup_steps is None and args.warmup_ratio is not None:
        args.warmup_steps = int(total_steps * args.warmup_ratio)
    elif args.warmup_steps is None:
        args.warmup_steps = 0

    # Build DeepSpeed configuration
    ds_config_from_yaml = cfg.get('deepspeed', {})
    optimizer_cfg = cfg.get('optimizer', {})

    optimizer_type = optimizer_cfg.get('type')
    optimizer_betas = optimizer_cfg.get('betas')
    optimizer_eps = optimizer_cfg.get('eps')
    
    if optimizer_type is None:
        raise ValueError("optimizer.type must be specified in config")
    if optimizer_betas is None:
        raise ValueError("optimizer.betas must be specified in config")
    if optimizer_eps is None:
        raise ValueError("optimizer.eps must be specified in config")

    ds_config = {
        "train_batch_size": args.batch_size * world_size * args.gradient_accumulation_steps,
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "gradient_clipping": ds_config_from_yaml.get("gradient_clipping", args.grad_clip),
        "prescale_gradients": ds_config_from_yaml.get("prescale_gradients", False),
        "steps_per_print": ds_config_from_yaml.get("steps_per_print", args.log_freq),
        "wall_clock_breakdown": ds_config_from_yaml.get("wall_clock_breakdown", False),
        "optimizer": {
            "type": optimizer_type,
            "params": {
                "lr": args.lr,
                "betas": optimizer_betas,
                "eps": optimizer_eps,
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

    # Add optional DeepSpeed configurations
    if ds_config_from_yaml.get("tensorboard", {}).get("enabled", False):
        ds_config["tensorboard"] = ds_config_from_yaml["tensorboard"]
    if ds_config_from_yaml.get("data_sampling", {}).get("enabled", False):
        ds_config["data_sampling"] = ds_config_from_yaml["data_sampling"]
    if "zero_allow_untested_optimizer" in ds_config_from_yaml:
        ds_config["zero_allow_untested_optimizer"] = ds_config_from_yaml["zero_allow_untested_optimizer"]

    # Merge external DeepSpeed config if provided
    if args.deepspeed and os.path.exists(args.deepspeed):
        print_rank_0(f"Loading DeepSpeed config from: {args.deepspeed}")
        import json
        with open(args.deepspeed, 'r') as f:
            external_config = json.load(f)

        def merge_config(base: dict, override: dict) -> dict:
            """Recursively merge configs, skipping 'auto' values."""
            for key, value in override.items():
                if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                    merge_config(base[key], value)
                elif value != "auto":
                    base[key] = value
            return base

        ds_config = merge_config(ds_config, external_config)

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config,
        model_parameters=model.parameters()
    )

    # Build loss function
    criterion = build_loss(args.config, vocab_size=len(tokenizer))
    print_rank_0(f"Loss initialized: {type(criterion).__name__}")

    # Load checkpoint for resume or evaluation
    start_epoch = 0
    global_step = 0
    if args.resume or args.eval:
        output_path = Path(args.output_dir)
        loaded = False
        
        # For eval mode: first try to load from checkpoints/best_model.txt (safetensors format)
        if args.eval:
            ckpt_dir = output_path / "checkpoints"
            tag_file = ckpt_dir / "best_model.txt"
            if tag_file.exists():
                with open(tag_file, 'r') as f:
                    best_model_name = f.read().strip()
                best_ckpt_path = ckpt_dir / best_model_name
                if best_ckpt_path.exists():
                    print_rank_0(f"Loading best model from tag file: {best_ckpt_path}")
                    from safetensors.torch import load_file as load_safetensors
                    state_dict = load_safetensors(str(best_ckpt_path))
                    model_engine.module.load_state_dict(state_dict, strict=False)
                    loaded = True
                    print_rank_0(f"Successfully loaded checkpoint: {best_model_name}")
        
        # Fallback: try DeepSpeed checkpoint format
        if not loaded:
            checkpoints = []
            
            # Find best checkpoint
            best_ckpt = output_path / "checkpoint_best"
            if best_ckpt.exists():
                checkpoints.append(("best", best_ckpt))
            
            # Find epoch checkpoints
            for ckpt_dir in output_path.glob("checkpoint_epoch_*"):
                if ckpt_dir.is_dir():
                    try:
                        epoch_num = int(ckpt_dir.name.split("_")[-1])
                        checkpoints.append((epoch_num, ckpt_dir))
                    except ValueError:
                        continue
            
            if checkpoints:
                if args.eval:
                    # Use best checkpoint for evaluation
                    best_checkpoints = [(e, p) for e, p in checkpoints if e == "best"]
                    if best_checkpoints:
                        load_path = best_checkpoints[0][1]
                    else:
                        epoch_checkpoints = [(e, p) for e, p in checkpoints if isinstance(e, int)]
                        epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)
                        load_path = epoch_checkpoints[0][1]
                else:
                    # Use latest epoch checkpoint for resume
                    epoch_checkpoints = [(e, p) for e, p in checkpoints if isinstance(e, int)]
                    epoch_checkpoints.sort(key=lambda x: x[0], reverse=True)
                    load_path = epoch_checkpoints[0][1] if epoch_checkpoints else checkpoints[0][1]
                
                print_rank_0(f"Loading checkpoint from: {load_path}")
                try:
                    _, client_state = model_engine.load_checkpoint(
                        str(output_path), str(load_path.name)
                    )
                    if client_state:
                        start_epoch = client_state.get("epoch", 0)
                        global_step = client_state.get("global_step", 0)
                        print_rank_0(
                            f"Checkpoint loaded: epoch={start_epoch}, global_step={global_step}"
                        )
                except Exception as e:
                    print_rank_0(f"Failed to load checkpoint: {e}")
            else:
                print_rank_0(f"No checkpoint found in {output_path}")

    # Initialize TensorBoard
    writer = None
    if rank == 0 and not args.eval:
        writer = SummaryWriter(Path(args.output_dir) / "tensorboard")

    # Build evaluator if configured
    evaluator = None
    if cfg.get('evaluator'):
        evaluator = build_evaluator(args.config)
        print_rank_0(
            f"Evaluator initialized: {type(evaluator).__name__}, "
            f"best_metric={evaluator.best_metric_name}"
        )

    # ==================== Evaluation Only Mode ====================
    if args.eval:
        if evaluator is None:
            raise ValueError("Evaluator must be configured in YAML for --eval mode")
        
        print_rank_0("=" * 50)
        print_rank_0("Evaluation Only Mode")
        print_rank_0("=" * 50)
        
        # Evaluate on validation set
        if val_dataloader is not None:
            print_rank_0("Evaluating on validation set...")
            val_metrics = evaluate(model_engine, val_dataloader, evaluator, device, args)
            print_rank_0(
                f"Validation Results | Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | "
                f"MCC: {val_metrics['mcc']:.4f}"
            )
            if rank == 0:
                evaluator.save_results(val_metrics, args.output_dir, split="dev")
        
        # Evaluate on test set
        if args.test:
            print_rank_0("Evaluating on test set...")
            test_dataset = build_finetune_dataset(args, tokenizer, split="test")
            test_sampler = DistributedSampler(
                test_dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
            test_dataloader = DataLoader(
                test_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                sampler=test_sampler,
                num_workers=args.num_workers,
                collate_fn=test_dataset.collate_fn,
                pin_memory=True,
            )
            test_metrics = evaluate(model_engine, test_dataloader, evaluator, device, args)
            print_rank_0(
                f"Test Results | Loss: {test_metrics['loss']:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | "
                f"MCC: {test_metrics['mcc']:.4f}"
            )
            if rank == 0:
                evaluator.save_results(test_metrics, args.output_dir, split="test")
        
        print_rank_0("Evaluation completed!")
        return

    # ==================== Training Mode ====================
    zero_stage = ds_config.get("zero_optimization", {}).get("stage", 1)
    print_rank_0(
        f"Starting DeepSpeed training: {args.num_epochs} epochs, "
        f"{len(dataloader)} batches/epoch"
    )
    print_rank_0(
        f"ZeRO stage: {zero_stage}, BF16: {args.bf16}, "
        f"Activation Checkpoint: {args.activation_checkpoint}"
    )
    print_rank_0(f"Warmup steps: {args.warmup_steps}, Total steps: {total_steps}")

    # Track best model
    best_metric_value = -float('inf')
    best_epoch = -1
    best_models = []  # List of (metric_value, epoch, filepath)
    epochs_without_improvement = 0  # Early stopping counter

    # Create test DataLoader if test set is available
    test_dataloader = None
    if args.test and args.dataset_type in finetune_types:
        test_dataset = build_finetune_dataset(args, tokenizer, split="test")
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=args.batch_size * 2,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
            collate_fn=test_dataset.collate_fn,
            pin_memory=True,
        )
        print_rank_0(f"Test dataset loaded: {len(test_dataset)} samples")

    for epoch in range(start_epoch, args.num_epochs):
        sampler.set_epoch(epoch)

        # Run training epoch
        if args.dataset_type in finetune_types:
            avg_loss, avg_acc, global_step, epoch_time = train_epoch_finetune(
                model_engine, dataloader, criterion, device, epoch, args, writer, global_step
            )
        else:
            avg_loss, avg_acc, global_step, epoch_time = train_epoch(
                model_engine, dataloader, criterion, device, epoch, args, writer, global_step
            )

        print_rank_0(
            f"Epoch {epoch} completed | "
            f"Avg Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time / 60:.1f}min"
        )

        # Run validation
        if evaluator is not None and val_dataloader is not None:
            val_metrics = evaluate(model_engine, val_dataloader, evaluator, device, args)
            print_rank_0(
                f"Validation | Loss: {val_metrics['loss']:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | F1: {val_metrics['f1']:.4f} | "
                f"MCC: {val_metrics['mcc']:.4f}"
            )
            
            # Log to TensorBoard
            if writer is not None:
                writer.add_scalar("val/loss", val_metrics['loss'], epoch)
                writer.add_scalar("val/accuracy", val_metrics['accuracy'], epoch)
                writer.add_scalar("val/f1", val_metrics['f1'], epoch)
                writer.add_scalar("val/mcc", val_metrics['mcc'], epoch)
            
            # Save best model (keep top 2)
            current_metric = val_metrics.get(evaluator.best_metric_name, val_metrics['f1'])
            should_save = len(best_models) < 2 or current_metric > min(m[0] for m in best_models)
            
            if should_save and rank == 0:
                from safetensors.torch import save_file as save_safetensors
                ckpt_dir = Path(args.output_dir) / "checkpoints"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                
                # Save with epoch naming
                ckpt_path = ckpt_dir / f"model_epoch{epoch}.safetensors"
                save_safetensors(model_engine.module.state_dict(), str(ckpt_path))
                best_models.append((current_metric, epoch, ckpt_path))
                
                # Sort and keep top 2
                best_models.sort(key=lambda x: x[0], reverse=True)
                while len(best_models) > 2:
                    _, _, old_path = best_models.pop()
                    if old_path.exists():
                        old_path.unlink()
                
                # Update tag file
                tag_file = ckpt_dir / "best_model.txt"
                with open(tag_file, 'w') as f:
                    f.write(f"model_epoch{best_models[0][1]}.safetensors\n")
                
                if current_metric > best_metric_value:
                    best_metric_value = current_metric
                    best_epoch = epoch
                    epochs_without_improvement = 0  # Reset counter
                    print_rank_0(
                        f"New best model! {evaluator.best_metric_name}: {best_metric_value:.4f}"
                    )
                else:
                    epochs_without_improvement += 1
                    print_rank_0(f"Saved checkpoint: {ckpt_path}")
        
        # Run test evaluation after each epoch (if test set available)
        if test_dataloader is not None:
            test_metrics = evaluate(model_engine, test_dataloader, evaluator, device, args)
            print_rank_0(
                f"Test | Loss: {test_metrics['loss']:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | "
                f"MCC: {test_metrics['mcc']:.4f}"
            )
            
            # Log test metrics to TensorBoard
            if writer is not None:
                writer.add_scalar("test/loss", test_metrics['loss'], epoch)
                writer.add_scalar("test/accuracy", test_metrics['accuracy'], epoch)
                writer.add_scalar("test/f1", test_metrics['f1'], epoch)
                writer.add_scalar("test/mcc", test_metrics['mcc'], epoch)

        # Save epoch checkpoint (skip if save_best_only is enabled)
        if not args.save_best_only:
            save_checkpoint(model_engine, epoch + 1, global_step, args)
        
        # Early stopping check
        if args.early_stopping_patience is not None and epochs_without_improvement >= args.early_stopping_patience:
            print_rank_0(
                f"Early stopping triggered! No improvement for {args.early_stopping_patience} epochs. "
                f"Best {evaluator.best_metric_name}: {best_metric_value:.4f} at epoch {best_epoch}"
            )
            break

    # Final test evaluation with best model
    if evaluator is not None and test_dataloader is not None and best_epoch >= 0:
        print_rank_0("=" * 50)
        print_rank_0("Final Test Evaluation with Best Model")
        print_rank_0("=" * 50)
        
        # Load best checkpoint from tag file
        ckpt_dir = Path(args.output_dir) / "checkpoints"
        tag_file = ckpt_dir / "best_model.txt"
        if tag_file.exists():
            with open(tag_file, 'r') as f:
                best_model_name = f.read().strip()
            best_ckpt_path = ckpt_dir / best_model_name
            if best_ckpt_path.exists():
                print_rank_0(f"Loading best checkpoint from: {best_ckpt_path}")
                from safetensors.torch import load_file as load_safetensors
                state_dict = load_safetensors(str(best_ckpt_path))
                model_engine.module.load_state_dict(state_dict, strict=False)
        
        test_metrics = evaluate(model_engine, test_dataloader, evaluator, device, args)
        print_rank_0(
            f"Final Test Results (Best Model @ Epoch {best_epoch}) | "
            f"Loss: {test_metrics['loss']:.4f} | "
            f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | "
            f"MCC: {test_metrics['mcc']:.4f}"
        )
        
        # Save final test results
        if rank == 0:
            evaluator.save_results(
                test_metrics,
                args.output_dir,
                split="test",
                best_dev_metric=best_metric_value
            )

    if writer is not None:
        writer.close()

    print_rank_0("DeepSpeed training completed!")


if __name__ == "__main__":
    main()
