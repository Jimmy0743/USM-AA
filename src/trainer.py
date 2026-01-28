# Copyright (c) 2025, Tencent Inc. All rights reserved.
# USM-AA Pretraining Script - Standalone implementation without tgnn dependency
import argparse
import math
import os
import time
from datetime import datetime
from pathlib import Path

import yaml
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file as save_safetensors

try:
    from .usmaa import USMAAModel, USMAAConfig
    from .dataset import (
        USMAAPretrainDataset, RefseqPretrainDataset, DNAPretrainDataset, 
        Discrete1KGPretrainDataset, print_rank_0
    )
    from .build_model import (
        build_tokenizer, build_model, build_model_config,
        load_pretrained_weights, MODEL_CONFIGS, print_model_info
    )
except ImportError:
    from usmaa import USMAAModel, USMAAConfig
    from dataset import (
        USMAAPretrainDataset, RefseqPretrainDataset, DNAPretrainDataset,
        Discrete1KGPretrainDataset, print_rank_0
    )
    from build_model import (
        build_tokenizer, build_model, build_model_config,
        load_pretrained_weights, MODEL_CONFIGS, print_model_info
    )


def get_args():
    parser = argparse.ArgumentParser(description="USM-AA Pretraining")
    
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
    parser.add_argument("--sample_weights", type=str, default=None, 
                       help="Sampling weights as JSON string, e.g., '{\"ref\":1.0,\"dna\":0.5}'")
    
    # Model arguments
    parser.add_argument("--model_type", type=str, default=None, 
                       choices=["usm-aa-100m", "usm-aa-500m", "usm-aa-1b"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--pretrained", type=str, default=None, help="Load pretrained weights (safetensors/pt)")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size per GPU")
    parser.add_argument("--num_epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=None, help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=None, help="Warmup steps")
    parser.add_argument("--grad_clip", type=float, default=None, help="Gradient clipping")
    parser.add_argument("--mask_ratio", type=float, default=None, help="MLM mask ratio")
    parser.add_argument("--rc_ratio", type=float, default=None, help="Reverse complement ratio")
    
    # System arguments
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--log_freq", type=int, default=None, help="Log frequency")
    parser.add_argument("--save_freq", type=int, default=None, help="Save checkpoint frequency")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--fp16", action="store_true", help="Use float16")
    parser.add_argument("--activation_checkpoint", action="store_true", help="Use activation checkpointing")
    
    # Distributed arguments
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Parse sample_weights from JSON string if provided
    if args.sample_weights and isinstance(args.sample_weights, str):
        import json
        args.sample_weights = json.loads(args.sample_weights)
    
    # Load defaults from yaml config
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
        if args.sample_weights is None:
            args.sample_weights = data_cfg.get('sample_weights', {"ref": 1.0, "dna": 1.0, "n1kg": 1.0})
        
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
    
    # Set defaults if not specified
    if args.dataset_type is None:
        args.dataset_type = 'refseq'
    if args.seq_len is None:
        args.seq_len = 256
    if args.sample_weights is None:
        args.sample_weights = {"ref": 1.0, "dna": 1.0, "n1kg": 1.0}
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


def setup_distributed():
    """Initialize distributed training"""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def get_lr_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Warmup + Cosine learning rate scheduler"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    
    return LambdaLR(optimizer, lr_lambda)


class MLMLoss(torch.nn.Module):
    """Masked Language Model Loss"""
    
    def __init__(self, vocab_size: int, label_smoothing: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
        """
        Args:
            logits: [B, L, V] model output
            labels: [B, L] target labels
            mask: [B, L] mask positions (True = compute loss)
        
        Returns:
            loss: scalar loss
            accuracy: accuracy
        """
        # Only compute loss at masked positions
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)
        mask_flat = mask.view(-1)
        
        # Select masked positions
        masked_logits = logits_flat[mask_flat]
        masked_labels = labels_flat[mask_flat]
        
        if masked_logits.size(0) == 0:
            return torch.tensor(0.0, device=logits.device), torch.tensor(0.0, device=logits.device)
        
        loss = F.cross_entropy(masked_logits, masked_labels, label_smoothing=self.label_smoothing)
        
        # Compute accuracy
        with torch.no_grad():
            preds = masked_logits.argmax(dim=-1)
            accuracy = (preds == masked_labels).float().mean()
        
        return loss, accuracy


def train_epoch(
    model, dataloader, optimizer, scheduler, scaler, 
    criterion, device, epoch, args, writer, global_step
):
    """Train one epoch"""
    model.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    total_batches = len(dataloader)
    start_time = time.time()
    log_start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Move data to GPU
        token_ids = inputs["token_ids"].to(device)
        chr_ids = inputs["chr_ids"].to(device)
        position_ids = inputs["position_ids"].to(device)
        strand_ids = inputs["strand_ids"].to(device)
        cigar_ids = inputs["cigar_ids"].to(device)
        seq_mask = inputs["seq_mask"].to(device)
        
        labels = targets["labels"].to(device)
        mask = targets["mask"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        with autocast(enabled=args.bf16 or args.fp16, dtype=torch.bfloat16 if args.bf16 else torch.float16):
            # Use forward_mlm for pretraining (returns logits)
            if isinstance(model, DDP):
                outputs = model.module.forward_mlm(
                    input_ids=token_ids,
                    position_ids=position_ids,
                    chr_ids=chr_ids,
                    strand_ids=strand_ids,
                    cigar_ids=cigar_ids,
                    attention_mask=seq_mask.float()
                )
            else:
                outputs = model.forward_mlm(
                    input_ids=token_ids,
                    position_ids=position_ids,
                    chr_ids=chr_ids,
                    strand_ids=strand_ids,
                    cigar_ids=cigar_ids,
                    attention_mask=seq_mask.float()
                )
            logits = outputs["logits"]
            loss, acc = criterion(logits, labels, mask)
        
        # Backward pass
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
        
        scheduler.step()
        global_step += 1
        
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
        
        # Logging
        if global_step % args.log_freq == 0:
            elapsed = time.time() - log_start_time
            avg_loss = total_loss / num_batches
            lr = scheduler.get_last_lr()[0]
            current_time = datetime.now().strftime("%m-%d %H:%M:%S")
            
            # Format: [time] Epoch X | Step Y/Z | Loss | LR | elapsed
            print_rank_0(
                f"[{current_time}] Epoch {epoch} | Step {batch_idx + 1}/{total_batches} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.1f}s"
            )
            
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/learning_rate", lr, global_step)
            
            log_start_time = time.time()
        
        # Save checkpoint
        if global_step % args.save_freq == 0 and (not dist.is_initialized() or dist.get_rank() == 0):
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, args)
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc, global_step, epoch_time


def save_checkpoint(model, optimizer, scheduler, epoch, global_step, args):
    """Save checkpoint (safetensors only)"""
    save_dir = Path(args.output_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get model state
    if isinstance(model, DDP):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    
    # Save safetensors format model weights
    safetensors_path = save_dir / f"model_step{global_step}.safetensors"
    save_safetensors(model_state, str(safetensors_path))
    print_rank_0(f"Saved safetensors: {safetensors_path}")


def load_checkpoint(path, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    checkpoint = torch.load(path, map_location="cpu")
    
    if isinstance(model, DDP):
        model.module.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint["model"])
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    
    return checkpoint.get("epoch", 0), checkpoint.get("global_step", 0)


def main():
    args = get_args()
    
    # Distributed setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    
    # Create output directory
    if rank == 0:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build tokenizer
    tokenizer = build_tokenizer(kmer=6, dynamic=True)
    print_rank_0(f"Tokenizer vocab size: {len(tokenizer)}")
    
    # Build dataset
    dataset = build_dataset(args, tokenizer)
    
    # DataLoader
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    else:
        sampler = None
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
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
        pretrained_path=args.pretrained,
        device=device,
    )
    print_model_info(model, config)
    
    # Distributed model
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95)
    )
    
    # Learning rate scheduler
    total_steps = len(dataloader) * args.num_epochs
    scheduler = get_lr_scheduler(optimizer, args.warmup_steps, total_steps)
    
    # Mixed precision
    scaler = None
    if args.fp16:
        scaler = GradScaler()
    
    # Loss function
    criterion = MLMLoss(vocab_size=len(tokenizer))
    
    # Load checkpoint (pretrained weights already loaded in build_model)
    start_epoch = 0
    global_step = 0
    if args.checkpoint:
        start_epoch, global_step = load_checkpoint(args.checkpoint, model, optimizer, scheduler)
        print_rank_0(f"Resumed from epoch {start_epoch}, step {global_step}")
    
    # TensorBoard
    writer = None
    if rank == 0:
        writer = SummaryWriter(Path(args.output_dir) / "tensorboard")
    
    # Training loop
    print_rank_0(f"Starting training: {args.num_epochs} epochs, {len(dataloader)} batches/epoch")
    
    for epoch in range(start_epoch, args.num_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        avg_loss, avg_acc, global_step, epoch_time = train_epoch(
            model, dataloader, optimizer, scheduler, scaler,
            criterion, device, epoch, args, writer, global_step
        )
        
        print_rank_0(
            f"Epoch {epoch} completed | "
            f"Avg Loss: {avg_loss:.4f} | Avg Acc: {avg_acc:.4f} | "
            f"Time: {epoch_time/60:.1f}min"
        )
        
        # Save epoch checkpoint
        if rank == 0:
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, args)
    
    if writer is not None:
        writer.close()
    
    print_rank_0("Training completed!")


if __name__ == "__main__":
    main()
