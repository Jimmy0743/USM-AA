from typing import Dict, Any, Optional
import yaml
import torch.nn as nn

from .mlm_loss import MLMLoss
from .cross_entropy import ClassificationLoss


# Loss registry
LOSS_REGISTRY = {
    # Pretraining losses
    "mlm": MLMLoss, 
    # Finetuning losses
    "classification": ClassificationLoss,
}


def build_loss(config_path: str, vocab_size: Optional[int] = None) -> nn.Module:
    """Build loss function from YAML config file.
    
    Args:
        config_path: Path to YAML config file
        vocab_size: Vocabulary size (required for MLM loss)
    
    Returns:
        Loss module instance
        
    Example config:
        loss:
          name: classification
          num_classes: 9
          label_smoothing: 0.1
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    loss_config = config.get('loss', {})
    if not loss_config:
        raise ValueError("Loss config is empty or missing 'loss' section")
    
    name = loss_config.get("name")
    if not name:
        raise ValueError("Loss name is required in config")
    
    name = name.lower()
    if name not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {name}. Available losses: {list(LOSS_REGISTRY.keys())}"
        )
    
    loss_cls = LOSS_REGISTRY[name]
    
    # Extract parameters
    num_classes = loss_config.get("num_classes")
    label_smoothing = loss_config.get("label_smoothing", 0.0)
    
    # Build loss based on type
    if name == "mlm":
        if vocab_size is None:
            raise ValueError("vocab_size is required for MLM loss")
        return loss_cls(vocab_size=vocab_size, label_smoothing=label_smoothing)
    
    elif name in ["classification"]:
        if num_classes is None:
            raise ValueError("num_classes is required for classification loss")
        return loss_cls(num_classes=num_classes, label_smoothing=label_smoothing)
    
    # For other losses, pass all extra parameters
    kwargs = {k: v for k, v in loss_config.items() 
              if k not in ["name", "num_classes", "label_smoothing"]}
    return loss_cls(**kwargs)


def print_loss_info(criterion: nn.Module):
    """Print loss function information."""
    print(f"Loss type: {criterion.__class__.__name__}")
    if hasattr(criterion, "num_classes"):
        print(f"  - num_classes: {criterion.num_classes}")
    if hasattr(criterion, "vocab_size"):
        print(f"  - vocab_size: {criterion.vocab_size}")
    if hasattr(criterion, "label_smoothing"):
        print(f"  - label_smoothing: {criterion.label_smoothing}")
    if hasattr(criterion, "gamma"):
        print(f"  - gamma: {criterion.gamma}")
