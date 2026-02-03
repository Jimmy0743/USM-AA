# USM-AA Pretraining Script - DeepSpeed Version
import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import yaml
import torch
import torch.nn.functional as F
from safetensors.torch import save_file as save_safetensors

from src.model.modeling_usmaa import USMAAModel
from src.model.configuration_usmaa import USMAAConfig
from src.model.build import build_model, MODEL_CONFIGS, print_model_info
from src.tokenizer.build import build_tokenizer
from src.data.build import build_dataset


def print_rank_0(msg):
    """Print only on rank 0"""
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(msg)

def train_epoch(
    model_engine, dataloader, criterion, device, epoch, args, writer, global_step
):
    """Train one epoch (DeepSpeed version)"""
    model_engine.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    total_batches = len(dataloader)
    start_time = time.time()
    log_start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Determine model dtype
        model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        # Move data to GPU
        token_ids = inputs["token_ids"].to(device)
        chr_ids = inputs["chr_ids"].to(device)
        position_ids = inputs["position_ids"].to(device)
        strand_ids = inputs["strand_ids"].to(device)
        cigar_ids = inputs["cigar_ids"].to(device)
        seq_mask = inputs["seq_mask"].to(device, dtype=model_dtype)
        
        labels = targets["labels"].to(device)
        mask = targets["mask"].to(device)
        
        # Forward pass (use module to access original model methods)
        outputs = model_engine.module.forward_mlm(
            token_ids=token_ids,
            position_ids=position_ids,
            chr_ids=chr_ids,
            strand_ids=strand_ids,
            cigar_ids=cigar_ids,
            attention_mask=seq_mask
        )
        logits = outputs["logits"]
        loss, acc = criterion(logits, labels, mask)
        
        # Backward pass (DeepSpeed handles gradient scaling and accumulation automatically)
        model_engine.backward(loss)
        model_engine.step()
        
        global_step += 1
        
        total_loss += loss.item()
        total_acc += acc.item()
        num_batches += 1
        
        # Logging
        if global_step % args.log_freq == 0:
            elapsed = time.time() - log_start_time
            avg_loss = total_loss / num_batches
            lr = model_engine.get_lr()[0]
            current_time = datetime.now().strftime("%m-%d %H:%M:%S")
            
            print_rank_0(
                f"[{current_time}] Epoch {epoch} | Step {batch_idx + 1}/{total_batches} | "
                f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.1f}s"
            )
            
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/learning_rate", lr, global_step)
            
            log_start_time = time.time()
        
        # Save checkpoint by steps (if save_freq is set)
        if args.save_freq is not None and global_step % args.save_freq == 0:
            save_checkpoint(model_engine, epoch, global_step, args)
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc, global_step, epoch_time


def train_epoch_finetune(
    model_engine, dataloader, criterion, device, epoch, args, writer, global_step
):
    """Train one epoch for finetuning (classification/regression) - DeepSpeed version
    
    Unlike pretrain, finetune datasets return (inputs, label) without mask.
    The criterion can be any loss function that accepts (logits, labels).
    """
    model_engine.train()
    total_loss = 0
    total_acc = 0
    num_batches = 0
    
    total_batches = len(dataloader)
    start_time = time.time()
    log_start_time = time.time()
    
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # Determine model dtype
        model_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        
        # Move data to GPU
        token_ids = inputs["token_ids"].to(device)
        chr_ids = inputs["chr_ids"].to(device)
        position_ids = inputs["position_ids"].to(device)
        strand_ids = inputs["strand_ids"].to(device)
        cigar_ids = inputs["cigar_ids"].to(device)
        seq_mask = inputs["seq_mask"].to(device, dtype=model_dtype)
        
        # Labels for finetune - targets is a dict with "labels" key
        labels = targets["labels"].to(device)
        
        # Forward pass for classification (use module to access original model methods)
        outputs = model_engine.module.forward(
            token_ids=token_ids,
            position_ids=position_ids,
            chr_ids=chr_ids,
            strand_ids=strand_ids,
            cigar_ids=cigar_ids,
            attention_mask=seq_mask
        )
        logits = outputs.logits
        
        # Call criterion - returns dict with 'loss' and optionally 'accuracy'
        loss_dict = criterion(logits, labels)
        loss = loss_dict['loss']
        
        # Backward pass
        model_engine.backward(loss)
        model_engine.step()
        
        global_step += 1
        
        total_loss += loss.item()
        if 'accuracy' in loss_dict:
            total_acc += loss_dict['accuracy'].item()
        num_batches += 1
        
        # Logging
        if global_step % args.log_freq == 0:
            elapsed = time.time() - log_start_time
            avg_loss = total_loss / num_batches
            lr = model_engine.get_lr()[0]
            current_time = datetime.now().strftime("%m-%d %H:%M:%S")
            # Global batch size = per-GPU batch size * world_size
            world_size = int(os.environ.get("WORLD_SIZE", 1))
            global_batch_size = token_ids.shape[0] * world_size
            
            print_rank_0(
                f"[{current_time}] Epoch {epoch} | Step {batch_idx + 1}/{total_batches} | "
                f"Loss: {avg_loss:.4f} | BS: {global_batch_size} | LR: {lr:.2e} | {elapsed:.1f}s"
            )
            
            if writer is not None:
                writer.add_scalar("train/loss", avg_loss, global_step)
                writer.add_scalar("train/learning_rate", lr, global_step)
            
            log_start_time = time.time()
        
        # Save checkpoint by steps (if save_freq is set)
        if args.save_freq is not None and global_step % args.save_freq == 0:
            save_checkpoint(model_engine, epoch, global_step, args)
    
    epoch_time = time.time() - start_time
    avg_loss = total_loss / num_batches
    avg_acc = total_acc / num_batches
    
    return avg_loss, avg_acc, global_step, epoch_time


def save_checkpoint(model_engine, epoch, global_step, args, suffix=""):
    """Save checkpoint (safetensors only)
    
    Args:
        model_engine: DeepSpeed model engine
        epoch: Current epoch number
        global_step: Current global step
        args: Training arguments
        suffix: Optional suffix for checkpoint name (e.g., "_best")
    """
    save_dir = Path(args.output_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save safetensors format model weights (rank 0 only)
    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        # Get model state (remove module. prefix)
        model_state = model_engine.module.state_dict()
        if suffix:
            safetensors_path = save_dir / f"model{suffix}.safetensors"
        else:
            safetensors_path = save_dir / f"model_step{global_step}.safetensors"
        save_safetensors(model_state, str(safetensors_path))
        print_rank_0(f"Saved safetensors: {safetensors_path}")


def load_checkpoint(model_engine, args):
    """Load DeepSpeed checkpoint"""
    load_dir = Path(args.checkpoint)
    if load_dir.is_dir():
        # DeepSpeed checkpoint directory
        _, client_state = model_engine.load_checkpoint(str(load_dir.parent), str(load_dir.name))
        if client_state:
            return client_state.get("epoch", 0), client_state.get("global_step", 0)
    
    return 0, 0