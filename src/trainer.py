# USM-AA Trainer - Unified Trainer Class for Pretraining and Fine-tuning
import argparse
import csv
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import deepspeed
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from safetensors.torch import save_file as save_safetensors

try:
    from .usmaa import USMAAModel, USMAAConfig, USMAAForSequenceClassification
    from .dataset import (
        USMAAPretrainDataset, RefseqPretrainDataset, DNAPretrainDataset, 
        Discrete1KGPretrainDataset, DenseDataset, DenseAADataset, 
        NonContiguousDataset, print_rank_0
    )
    from .build_model import (
        build_tokenizer, build_model, build_model_config,
        load_pretrained_weights, MODEL_CONFIGS, print_model_info
    )
    from .evaluator import ClassificationEvaluator
    from .loss import build_loss
except ImportError:
    from usmaa import USMAAModel, USMAAConfig, USMAAForSequenceClassification
    from dataset import (
        USMAAPretrainDataset, RefseqPretrainDataset, DNAPretrainDataset,
        Discrete1KGPretrainDataset, DenseDataset, DenseAADataset,
        NonContiguousDataset, print_rank_0
    )
    from build_model import (
        build_tokenizer, build_model, build_model_config,
        load_pretrained_weights, MODEL_CONFIGS, print_model_info
    )
    from evaluator import ClassificationEvaluator
    from loss import build_loss


# ============ Training Arguments ============

@dataclass
class TrainingArguments:
    """Training configuration arguments."""
    # Mode
    mode: str = "pretrain"  # pretrain or finetune
    do_train: bool = True   # Whether to run training
    do_eval: bool = False   # Whether to run evaluation only
    
    # Data
    dataset_type: str = "refseq"
    ref_file: Optional[str] = None
    dna_file: Optional[str] = None
    n1kg_file: Optional[str] = None
    train_file: Optional[str] = None
    dev_file: Optional[str] = None
    test_file: Optional[str] = None
    seq_len: int = 256
    max_len: int = 512
    sparse_k: int = 4
    num_classes: int = 2
    sample_weights: Dict[str, float] = field(default_factory=lambda: {"ref": 1.0})
    
    # Model
    model_name: str = "USMAAModel"  # Model class name from usmaa.py
    model_type: str = "usm-aa-100m"
    checkpoint: Optional[str] = None
    freeze_encoder: bool = False
    
    # Loss
    loss_name: str = "mlm"  # mlm | classification | regression
    
    # Training
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    warmup_steps: Optional[int] = None
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    mask_ratio: float = 0.15
    rc_ratio: float = 0.3
    label_smoothing: float = 0.0
    gradient_accumulation_steps: int = 1
    
    # Evaluation
    eval_freq: Optional[int] = None  # None means eval per epoch only
    early_stopping_patience: int = 5
    best_metric_name: str = "mcc"  # metric for early stopping: mcc, f1, accuracy
    
    # Evaluator
    evaluator_type: str = "classification"  # classification | regression
    evaluator_threshold: Optional[float] = None  # None = argmax, float = binary threshold
    evaluator_batch_size: Optional[int] = None  # None means 2x training batch_size
    evaluator_num_workers: int = 4
    
    # System
    num_workers: int = 4
    output_dir: str = "./output"
    log_freq: int = 10
    save_freq: int = 1000
    save_best_only: bool = False
    bf16: bool = False
    fp16: bool = False
    activation_checkpoint: bool = False
    
    # DeepSpeed
    ds_config: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "TrainingArguments":
        """Load arguments from YAML file."""
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        
        args = cls()
        args.mode = cfg.get('mode', 'pretrain')
        
        # Data config
        data_cfg = cfg.get('data', {})
        args.dataset_type = data_cfg.get('dataset_type', 'refseq')
        args.ref_file = data_cfg.get('ref_file')
        args.dna_file = data_cfg.get('dna_file')
        args.n1kg_file = data_cfg.get('n1kg_file')
        args.train_file = data_cfg.get('train_file')
        args.dev_file = data_cfg.get('dev_file')
        args.test_file = data_cfg.get('test_file')
        args.seq_len = data_cfg.get('seq_len', 256)
        args.max_len = data_cfg.get('max_len', 512)
        args.sparse_k = data_cfg.get('sparse_k', 4)
        args.num_classes = data_cfg.get('num_classes', 2)
        args.sample_weights = data_cfg.get('sample_weights', {"ref": 1.0})
        
        # Model config
        model_cfg = cfg.get('model', {})
        args.model_name = model_cfg.get('name', 'USMAAModel')
        args.model_type = model_cfg.get('type', 'usm-aa-100m')
        args.checkpoint = model_cfg.get('checkpoint')
        args.freeze_encoder = model_cfg.get('freeze_encoder', False)
        
        # Loss config
        loss_cfg = cfg.get('loss', {})
        args.loss_name = loss_cfg.get('name', 'mlm')
        
        # Training config
        train_cfg = cfg.get('training', {})
        args.batch_size = train_cfg.get('batch_size', 32)
        args.num_epochs = train_cfg.get('num_epochs', 20)
        args.learning_rate = train_cfg.get('learning_rate', 1e-4)
        args.weight_decay = train_cfg.get('weight_decay', 0.1)
        args.warmup_steps = train_cfg.get('warmup_steps')
        args.warmup_ratio = train_cfg.get('warmup_ratio', 0.1)
        args.grad_clip = train_cfg.get('grad_clip', 1.0)
        args.mask_ratio = train_cfg.get('mask_ratio', 0.15)
        args.rc_ratio = train_cfg.get('rc_ratio', 0.3)
        args.label_smoothing = train_cfg.get('label_smoothing', 0.0)
        eval_freq = train_cfg.get('eval_freq', None)
        args.eval_freq = None if eval_freq is None or str(eval_freq).lower() == 'none' else int(eval_freq)
        args.early_stopping_patience = train_cfg.get('early_stopping_patience', 5)
        args.best_metric_name = train_cfg.get('best_metric', 'mcc')
        args.gradient_accumulation_steps = train_cfg.get('gradient_accumulation_steps', 1)
        
        # Evaluator config
        eval_cfg = cfg.get('evaluator', {})
        args.evaluator_type = eval_cfg.get('type', 'classification')
        args.evaluator_threshold = eval_cfg.get('threshold', None)
        args.evaluator_batch_size = eval_cfg.get('batch_size', None)
        args.evaluator_num_workers = eval_cfg.get('num_workers', 4)
        
        # DataLoader config
        loader_cfg = cfg.get('dataloader', {})
        args.num_workers = loader_cfg.get('num_workers', 4)
        
        # Precision config
        prec_cfg = cfg.get('precision', {})
        args.bf16 = prec_cfg.get('bf16', False)
        args.fp16 = prec_cfg.get('fp16', False)
        args.activation_checkpoint = prec_cfg.get('activation_checkpoint', False)
        
        # Logging config
        log_cfg = cfg.get('logging', {})
        args.output_dir = log_cfg.get('output_dir', './output')
        args.log_freq = log_cfg.get('log_freq', 10)
        args.save_freq = log_cfg.get('save_freq', 1000)
        args.save_best_only = log_cfg.get('save_best_only', False)
        
        # DeepSpeed config
        args.ds_config = cfg.get('deepspeed', {})
        
        return args


# ============ Unified Trainer ============

class USMAATrainer:
    """Unified trainer for pretraining and fine-tuning."""
    
    def __init__(self, args: TrainingArguments):
        self.args = args
        self.device = None
        self.rank = 0
        self.world_size = 1
        self.local_rank = 0
        self.model_engine = None
        self.tokenizer = None
        self.writer = None
        self.global_step = 0
        self.model_dtype = torch.float32
        
        # For fine-tuning
        self.criterion = None
        self.evaluator = None
        self.dev_loader = None
        self.test_loader = None
        self.best_metric = -float('inf')
        self.no_improve_count = 0
    
    def setup(self):
        """Initialize distributed training and device."""
        deepspeed.init_distributed()
        
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.rank = int(os.environ.get("RANK", 0))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        if self.args.bf16:
            self.model_dtype = torch.bfloat16
        elif self.args.fp16:
            self.model_dtype = torch.float16
        
        if self.rank == 0:
            Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(Path(self.args.output_dir) / "tensorboard")
        
        # Build tokenizer
        self.tokenizer = build_tokenizer(kmer=6, dynamic=True)
        print_rank_0(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def build_deepspeed_config(self, total_steps: int) -> Dict[str, Any]:
        """Build DeepSpeed configuration."""
        warmup_steps = self.args.warmup_steps or int(total_steps * self.args.warmup_ratio)
        ds_yaml = self.args.ds_config
        
        return {
            "train_batch_size": self.args.batch_size * self.world_size * self.args.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "gradient_clipping": ds_yaml.get("gradient_clipping", self.args.grad_clip),
            "prescale_gradients": ds_yaml.get("prescale_gradients", False),
            "steps_per_print": ds_yaml.get("steps_per_print", self.args.log_freq),
            "wall_clock_breakdown": ds_yaml.get("wall_clock_breakdown", False),
            "optimizer": {
                "type": "AdamW",
                "params": {
                    "lr": self.args.learning_rate,
                    "betas": [0.9, 0.95] if self.args.mode == 'pretrain' else [0.9, 0.999],
                    "eps": 1e-8,
                    "weight_decay": self.args.weight_decay
                }
            },
            "scheduler": {
                "type": "WarmupDecayLR",
                "params": {
                    "warmup_min_lr": 0,
                    "warmup_max_lr": self.args.learning_rate,
                    "warmup_num_steps": warmup_steps,
                    "total_num_steps": total_steps
                }
            },
            "zero_optimization": ds_yaml.get("zero_optimization", {
                "stage": 1,
                "allgather_partitions": True,
                "allgather_bucket_size": 5e8,
                "reduce_scatter": True,
                "reduce_bucket_size": 5e8,
                "contiguous_gradients": True
            }),
            "bfloat16": ds_yaml.get("bfloat16", {"enabled": self.args.bf16}),
            "fp16": ds_yaml.get("fp16", {"enabled": self.args.fp16}),
            "activation_checkpointing": {
                "partition_activations": self.args.activation_checkpoint,
                "cpu_checkpointing": False,
                "contiguous_memory_optimization": self.args.activation_checkpoint
            }
        }
    
    def build_model(self) -> nn.Module:
        """Build model from YAML config using model_name."""
        try:
            from . import usmaa as usmaa_module
        except ImportError:
            import usmaa as usmaa_module
        
        model_class_name = self.args.model_name
        if not hasattr(usmaa_module, model_class_name):
            raise ValueError(f"Model class '{model_class_name}' not found in usmaa.py. "
                           f"Available: USMAAModel, USMAAForSequenceClassification")
        model_class = getattr(usmaa_module, model_class_name)
        print_rank_0(f"Using model class: {model_class_name}")
        
        # Build model config
        model_config = build_model_config(
            vocab_size=len(self.tokenizer),
            model_type=self.args.model_type
        )
        model_config.num_classes = self.args.num_classes
        model_config.num_labels = self.args.num_classes
        
        # Build model
        model = model_class(model_config)
        
        # Load pretrained weights
        if self.args.checkpoint and os.path.exists(self.args.checkpoint):
            print_rank_0(f"Loading pretrained weights from: {self.args.checkpoint}")
            load_pretrained_weights(model, self.args.checkpoint, strict=False)
        
        # Freeze encoder if needed
        if self.args.freeze_encoder:
            for name, param in model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
            print_rank_0("Encoder parameters frozen")
        
        print_model_info(model, model_config)
        return model
    
    def build_loss(self) -> nn.Module:
        """Build loss function from YAML config."""
        loss_name = self.args.loss_name.lower()
        
        if loss_name in ['mlm', 'mlmloss']:
            return build_loss(
                loss_name="mlm",
                vocab_size=len(self.tokenizer),
                label_smoothing=self.args.label_smoothing
            )
        elif loss_name in ['classification', 'classificationloss', 'cross_entropy']:
            return build_loss(
                loss_name="classification",
                num_classes=self.args.num_classes,
                label_smoothing=self.args.label_smoothing
            )
        elif loss_name in ['regression', 'regressionloss', 'mse', 'mae']:
            return build_loss(loss_name=loss_name)
        else:
            raise ValueError(f"Unknown loss: {loss_name}")
    
    def build_datasets(self) -> tuple:
        """Build datasets based on mode."""
        args = self.args
        
        if args.mode == 'pretrain':
            return self._build_pretrain_datasets()
        else:
            return self._build_finetune_datasets()
    
    def _build_pretrain_datasets(self) -> tuple:
        """Build pretraining datasets."""
        args = self.args
        
        if args.dataset_type == 'refseq':
            print_rank_0("Using RefseqPretrainDataset")
            dataset = RefseqPretrainDataset(
                args.ref_file, self.tokenizer,
                mask_ratio=args.mask_ratio, rc_ratio=args.rc_ratio, seq_len=args.seq_len
            )
        elif args.dataset_type == 'dna':
            print_rank_0("Using DNAPretrainDataset")
            dataset = DNAPretrainDataset(
                args.dna_file, self.tokenizer,
                mask_ratio=args.mask_ratio, rc_ratio=args.rc_ratio, seq_len=args.seq_len
            )
        elif args.dataset_type == '1kg':
            print_rank_0("Using Discrete1KGPretrainDataset")
            dataset = Discrete1KGPretrainDataset(
                args.n1kg_file, self.tokenizer,
                mask_ratio=args.mask_ratio, seq_len=args.seq_len
            )
        else:
            print_rank_0("Using USMAAPretrainDataset (combined)")
            dataset = USMAAPretrainDataset(
                ref_file=args.ref_file, dna_file=args.dna_file, n1kg_file=args.n1kg_file,
                tokenizer=self.tokenizer, seq_len=args.seq_len,
                mask_ratio=args.mask_ratio, rc_ratio=args.rc_ratio,
                sample_weights=args.sample_weights
            )
        
        print_rank_0(f"Pretrain dataset size: {len(dataset)}")
        return dataset, None, None
    
    def _build_finetune_datasets(self) -> tuple:
        """Build fine-tuning datasets."""
        def _build_dataset(filename: str, split: str):
            if not filename or not os.path.exists(filename):
                print_rank_0(f"Warning: {split} file not found: {filename}")
                return None
            
            if self.args.dataset_type == 'dense':
                return DenseDataset(filename, self.tokenizer, split=split, max_len=self.args.max_len)
            elif self.args.dataset_type == 'dense_aa':
                return DenseAADataset(filename, self.tokenizer, split=split, max_len=self.args.max_len)
            else:
                return NonContiguousDataset(
                    filename, self.tokenizer, split=split,
                    max_len=self.args.max_len, sparse_k=self.args.sparse_k
                )
        
        train_dataset = _build_dataset(self.args.train_file, "train")
        dev_dataset = _build_dataset(self.args.dev_file, "dev")
        test_dataset = _build_dataset(self.args.test_file, "test")
        
        if train_dataset:
            print_rank_0(f"Train dataset: {len(train_dataset)} samples")
        if dev_dataset:
            print_rank_0(f"Dev dataset: {len(dev_dataset)} samples")
        if test_dataset:
            print_rank_0(f"Test dataset: {len(test_dataset)} samples")
        
        return train_dataset, dev_dataset, test_dataset
    
    def _prepare_batch(self, inputs: Dict, targets: Dict) -> tuple:
        """Move batch to device."""
        token_ids = inputs["token_ids"].to(self.device)
        chr_ids = inputs["chr_ids"].to(self.device)
        position_ids = inputs["position_ids"].to(self.device)
        strand_ids = inputs["strand_ids"].to(self.device)
        cigar_ids = inputs["cigar_ids"].to(self.device)
        seq_mask = inputs["seq_mask"].to(self.device, dtype=self.model_dtype)
        
        processed_targets = {}
        for k, v in targets.items():
            processed_targets[k] = v.to(self.device) if isinstance(v, torch.Tensor) else v
        
        return (token_ids, position_ids, chr_ids, strand_ids, cigar_ids, seq_mask), processed_targets
    
    def train_step(self, batch) -> Dict[str, float]:
        """Execute one training step."""
        inputs, targets = batch
        model_inputs, processed_targets = self._prepare_batch(inputs, targets)
        token_ids, position_ids, chr_ids, strand_ids, cigar_ids, seq_mask = model_inputs
        labels = processed_targets["labels"]
        
        if self.args.mode == 'pretrain':
            # MLM forward
            mask = processed_targets["mask"]
            outputs = self.model_engine.module.forward_mlm(
                token_ids=token_ids, position_ids=position_ids,
                chr_ids=chr_ids, strand_ids=strand_ids,
                cigar_ids=cigar_ids, attention_mask=seq_mask
            )
            loss_dict = self.criterion(outputs["logits"], labels, mask=mask)
            loss = loss_dict["loss"]
            metrics = {"loss": loss.item(), "accuracy": loss_dict.get("accuracy", torch.tensor(0.0)).item()}
        else:
            # Classification/Regression forward
            outputs = self.model_engine(
                token_ids=token_ids, position_ids=position_ids,
                chr_ids=chr_ids, strand_ids=strand_ids,
                cigar_ids=cigar_ids, attention_mask=seq_mask
            )
            loss_dict = self.criterion(outputs["logits"], labels)
            loss = loss_dict["loss"]
            metrics = {"loss": loss.item()}
        
        self.model_engine.backward(loss)
        self.model_engine.step()
        
        return metrics
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on validation/test set."""
        if self.evaluator is None:
            raise ValueError("Evaluator not initialized.")
        
        self.model_engine.eval()
        self.evaluator.reset()
        
        with torch.no_grad():
            for batch in dataloader:
                inputs, targets = batch
                model_inputs, processed_targets = self._prepare_batch(inputs, targets)
                token_ids, position_ids, chr_ids, strand_ids, cigar_ids, seq_mask = model_inputs
                labels = processed_targets["labels"]
                
                outputs = self.model_engine(
                    token_ids=token_ids, position_ids=position_ids,
                    chr_ids=chr_ids, strand_ids=strand_ids,
                    cigar_ids=cigar_ids, attention_mask=seq_mask
                )
                self.evaluator.add_batch(outputs["logits"], labels)
        
        metrics = self.evaluator.compute_metrics()
        self.model_engine.train()
        return metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        save_dir = Path(self.args.output_dir) / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        if self.rank == 0:
            model_state = self.model_engine.module.state_dict()
            model_state = {k: v.clone().contiguous() for k, v in model_state.items()}
            filename = "model_best.safetensors" if is_best else f"model_step{self.global_step}.safetensors"
            save_safetensors(model_state, str(save_dir / filename))
            print_rank_0(f"Saved checkpoint: {save_dir / filename}")
    
    def log_metrics(self, metrics: Dict[str, float], prefix: str = "train"):
        """Log metrics to tensorboard."""
        if self.writer is not None:
            for k, v in metrics.items():
                self.writer.add_scalar(f"{prefix}/{k}", v, self.global_step)
    
    def cleanup(self):
        """Cleanup resources."""
        if self.writer is not None:
            self.writer.close()
    
    def train(self):
        """Main training loop."""
        self.setup()
        
        # Build datasets and dataloaders
        train_dataset, dev_dataset, test_dataset = self.build_datasets()
        
        train_sampler = DistributedSampler(train_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=train_sampler,
            num_workers=self.args.num_workers, collate_fn=train_dataset.collate_fn,
            pin_memory=True, prefetch_factor=2
        )
        
        if dev_dataset:
            self.dev_loader = DataLoader(
                dev_dataset, batch_size=self.args.batch_size * 2, shuffle=False,
                num_workers=self.args.num_workers, collate_fn=dev_dataset.collate_fn, pin_memory=True
            )
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.args.batch_size * 2, shuffle=False,
                num_workers=self.args.num_workers, collate_fn=test_dataset.collate_fn, pin_memory=True
            )
        
        # Build model
        model = self.build_model()
        
        # DeepSpeed initialize
        total_steps = len(train_loader) * self.args.num_epochs // self.args.gradient_accumulation_steps
        ds_config = self.build_deepspeed_config(total_steps)
        
        self.model_engine, _, _, _ = deepspeed.initialize(
            model=model, config=ds_config, model_parameters=model.parameters()
        )
        
        # Build loss
        self.criterion = self.build_loss()
        print_rank_0(f"Loss: {self.args.loss_name}")
        
        # Initialize evaluator for finetune mode
        if self.args.mode == 'finetune':
            self.evaluator = ClassificationEvaluator(
                num_classes=self.args.num_classes,
                threshold=self.args.evaluator_threshold,
                best_metric_name=self.args.best_metric_name,
                label_smoothing=self.args.label_smoothing,
            )
            print_rank_0(f"Evaluator: threshold={'argmax' if self.evaluator.threshold is None else self.evaluator.threshold}")
        
        # Training loop
        mode_str = "pretraining" if self.args.mode == 'pretrain' else "fine-tuning"
        print_rank_0(f"Starting {mode_str}: {self.args.num_epochs} epochs, {len(train_loader)} batches/epoch")
        print_rank_0(f"ZeRO stage: {ds_config['zero_optimization'].get('stage', 1)}, BF16: {self.args.bf16}")
        
        for epoch in range(self.args.num_epochs):
            train_sampler.set_epoch(epoch)
            self.model_engine.train()
            
            total_loss, total_acc, num_batches = 0, 0, 0
            start_time = time.time()
            log_start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                metrics = self.train_step(batch)
                
                self.global_step += 1
                total_loss += metrics["loss"]
                if "accuracy" in metrics:
                    total_acc += metrics["accuracy"]
                num_batches += 1
                
                # Logging
                if self.global_step % self.args.log_freq == 0:
                    elapsed = time.time() - log_start_time
                    avg_loss = total_loss / num_batches
                    lr = self.model_engine.get_lr()[0]
                    current_time = datetime.now().strftime("%m-%d %H:%M:%S")
                    
                    if self.args.mode == 'pretrain':
                        print_rank_0(
                            f"[{current_time}] Epoch {epoch} | Step {batch_idx + 1}/{len(train_loader)} | "
                            f"Loss: {avg_loss:.4f} | Acc: {total_acc/num_batches:.4f} | LR: {lr:.2e} | {elapsed:.1f}s"
                        )
                        self.log_metrics({"loss": avg_loss, "accuracy": total_acc/num_batches, "learning_rate": lr})
                    else:
                        print_rank_0(
                            f"[{current_time}] Epoch {epoch} | Step {batch_idx + 1}/{len(train_loader)} | "
                            f"Loss: {avg_loss:.4f} | LR: {lr:.2e} | {elapsed:.1f}s"
                        )
                        self.log_metrics({"loss": avg_loss, "learning_rate": lr})
                    log_start_time = time.time()
                
                # Mid-epoch evaluation (finetune only)
                if self.args.mode == 'finetune' and self.args.eval_freq is not None:
                    if self.dev_loader is not None and self.global_step % self.args.eval_freq == 0:
                        self._run_evaluation(epoch)
                
                # Save checkpoint
                if not self.args.save_best_only and self.global_step % self.args.save_freq == 0:
                    self.save_checkpoint(epoch)
            
            # Epoch end
            epoch_time = time.time() - start_time
            if self.args.mode == 'pretrain':
                print_rank_0(
                    f"Epoch {epoch} completed | Avg Loss: {total_loss/num_batches:.4f} | "
                    f"Avg Acc: {total_acc/num_batches:.4f} | Time: {epoch_time/60:.1f}min"
                )
                self.save_checkpoint(epoch + 1)
            else:
                print_rank_0(f"Epoch {epoch} completed | Avg Loss: {total_loss/num_batches:.4f} | Time: {epoch_time/60:.1f}min")
                
                # Epoch-end evaluation
                if self.dev_loader is not None:
                    should_stop = self._run_evaluation(epoch, is_epoch_end=True)
                    if should_stop:
                        break
        
        # Final test evaluation (finetune only)
        if self.args.mode == 'finetune' and self.test_loader is not None:
            self._final_test_evaluation()
        
        self.cleanup()
        print_rank_0(f"{mode_str.capitalize()} completed!")
    
    def _run_evaluation(self, epoch: int, is_epoch_end: bool = False) -> bool:
        """Run evaluation on dev/test sets. Returns True if should early stop."""
        eval_metrics = self.evaluate(self.dev_loader)
        prefix = "dev_epoch" if is_epoch_end else "dev"
        print_rank_0(
            f"{'Epoch ' + str(epoch) + ' ' if is_epoch_end else '  '}Dev  | Loss: {eval_metrics['loss']:.4f} | "
            f"Acc: {eval_metrics['accuracy']:.4f} | F1: {eval_metrics['f1']:.4f} | MCC: {eval_metrics['mcc']:.4f}"
        )
        self.log_metrics(eval_metrics, prefix=prefix)
        
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            prefix = "test_epoch" if is_epoch_end else "test"
            print_rank_0(
                f"{'Epoch ' + str(epoch) + ' ' if is_epoch_end else '  '}Test | Loss: {test_metrics['loss']:.4f} | "
                f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}"
            )
            self.log_metrics(test_metrics, prefix=prefix)
        
        # Update best metric and early stopping
        metric_name = self.args.best_metric_name
        current_metric = eval_metrics[metric_name]
        if current_metric > self.best_metric:
            self.best_metric = current_metric
            self.no_improve_count = 0
            self.save_checkpoint(epoch, is_best=True)
            print_rank_0(f"  New best dev {metric_name}: {self.best_metric:.4f}")
        elif is_epoch_end:
            self.no_improve_count += 1
        
        if is_epoch_end and self.no_improve_count >= self.args.early_stopping_patience:
            print_rank_0(f"Early stopping at epoch {epoch} (no improvement for {self.no_improve_count} epochs)")
            return True
        return False
    
    def _final_test_evaluation(self):
        """Load best model and evaluate on test set."""
        print_rank_0("=" * 50)
        print_rank_0("Loading best model for final test evaluation...")
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        best_ckpt_path = Path(self.args.output_dir) / "checkpoints" / "model_best.safetensors"
        if best_ckpt_path.exists():
            from safetensors.torch import load_file as load_safetensors
            best_state = load_safetensors(str(best_ckpt_path), device="cpu")
            self.model_engine.module.load_state_dict(best_state, strict=False)
            print_rank_0(f"Loaded best model from {best_ckpt_path}")
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        test_metrics = self.evaluate(self.test_loader)
        print_rank_0(
            f"Final Test (Best Dev Model) | Loss: {test_metrics['loss']:.4f} | "
            f"Acc: {test_metrics['accuracy']:.4f} | F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}"
        )
        
        if self.rank == 0:
            results_file = self.evaluator.save_results(
                metrics=test_metrics,
                output_dir=self.args.output_dir,
                split="test",
                best_dev_metric=self.best_metric,
            )
            print_rank_0(f"Results saved to {results_file}")
    
    def eval_only(self):
        """Run evaluation only (no training)."""
        self.setup()
        
        # Build datasets
        _, dev_dataset, test_dataset = self.build_datasets()
        
        if dev_dataset is not None:
            dev_sampler = DistributedSampler(dev_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            self.dev_loader = DataLoader(
                dev_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=dev_sampler,
                num_workers=self.args.num_workers, collate_fn=dev_dataset.collate_fn, pin_memory=True
            )
        
        if test_dataset is not None:
            test_sampler = DistributedSampler(test_dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False)
            self.test_loader = DataLoader(
                test_dataset, batch_size=self.args.batch_size, shuffle=False, sampler=test_sampler,
                num_workers=self.args.num_workers, collate_fn=test_dataset.collate_fn, pin_memory=True
            )
        
        # Build model
        model = self.build_model()
        
        # Initialize DeepSpeed for inference
        ds_config = self.build_deepspeed_config(1)
        ds_config["train_batch_size"] = self.args.batch_size * self.world_size
        
        self.model_engine, _, _, _ = deepspeed.initialize(model=model, config=ds_config)
        
        # Load best model weights
        best_ckpt_path = Path(self.args.output_dir) / "checkpoints" / "model_best.safetensors"
        if best_ckpt_path.exists():
            from safetensors.torch import load_file as load_safetensors
            best_state = load_safetensors(str(best_ckpt_path), device="cpu")
            self.model_engine.module.load_state_dict(best_state, strict=False)
            print_rank_0(f"Loaded best model from {best_ckpt_path}")
        else:
            print_rank_0(f"Warning: Best model not found at {best_ckpt_path}, using current weights")
        
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        
        # Initialize evaluator
        self.evaluator = ClassificationEvaluator(
            num_classes=self.args.num_classes,
            threshold=self.args.evaluator_threshold,
            best_metric_name=self.args.best_metric_name,
            label_smoothing=self.args.label_smoothing,
        )
        
        print_rank_0("=" * 50)
        print_rank_0("Running evaluation only...")
        
        if self.dev_loader is not None:
            dev_metrics = self.evaluate(self.dev_loader)
            print_rank_0(
                f"Dev | Loss: {dev_metrics['loss']:.4f} | Acc: {dev_metrics['accuracy']:.4f} | "
                f"F1: {dev_metrics['f1']:.4f} | MCC: {dev_metrics['mcc']:.4f}"
            )
            if self.rank == 0:
                self.evaluator.save_results(metrics=dev_metrics, output_dir=self.args.output_dir, split="dev")
        
        if self.test_loader is not None:
            test_metrics = self.evaluate(self.test_loader)
            print_rank_0(
                f"Test | Loss: {test_metrics['loss']:.4f} | Acc: {test_metrics['accuracy']:.4f} | "
                f"F1: {test_metrics['f1']:.4f} | MCC: {test_metrics['mcc']:.4f}"
            )
            if self.rank == 0:
                results_file = self.evaluator.save_results(metrics=test_metrics, output_dir=self.args.output_dir, split="test")
                print_rank_0(f"Results saved to {results_file}")
        
        self.cleanup()
        print_rank_0("Evaluation completed!")


# ============ CLI Entry ============

def parse_args() -> TrainingArguments:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="USM-AA Training (Pretrain / Finetune)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                       help="Run mode: train or eval")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    cli_args = parser.parse_args()
    
    args = TrainingArguments.from_yaml(cli_args.config)
    args.do_train = (cli_args.mode == "train")
    args.do_eval = (cli_args.mode == "eval")
    
    return args


def main():
    args = parse_args()
    trainer = USMAATrainer(args)
    
    if args.do_train:
        trainer.train()
    else:
        trainer.eval_only()


if __name__ == "__main__":
    main()
