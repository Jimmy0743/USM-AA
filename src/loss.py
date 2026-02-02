"""
Loss functions for USM-AA training.
- MLMLoss: Masked Language Model loss for pretraining
- ClassificationLoss: Cross-entropy loss for classification
- RegressionLoss: MSE/MAE loss for regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any


class MLMLoss(nn.Module):
    """Masked Language Model loss for pretraining."""
    
    def __init__(
        self, 
        vocab_size: int,
        label_smoothing: float = 0.0,
        ignore_index: int = -100
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, L, V) prediction logits
            labels: (B, L) target labels, -100 for non-masked positions
        
        Returns:
            Dict with 'loss' and optionally 'accuracy'
        """
        # Reshape for cross entropy
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)
        
        loss = self.criterion(logits_flat, labels_flat)
        
        # Compute accuracy on masked tokens only
        with torch.no_grad():
            mask = labels_flat != self.ignore_index
            if mask.sum() > 0:
                preds = logits_flat[mask].argmax(dim=-1)
                targets = labels_flat[mask]
                accuracy = (preds == targets).float().mean()
            else:
                accuracy = torch.tensor(0.0, device=logits.device)
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }


class ClassificationLoss(nn.Module):
    """Cross-entropy loss for sequence classification."""
    
    def __init__(
        self,
        num_classes: int,
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, num_classes) prediction logits
            labels: (B,) target labels
        
        Returns:
            Dict with 'loss' and 'accuracy'
        """
        loss = self.criterion(logits, labels)
        
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()
        
        return {
            'loss': loss,
            'accuracy': accuracy
        }


class RegressionLoss(nn.Module):
    """MSE/MAE loss for regression tasks."""
    
    def __init__(self, loss_type: str = 'mse'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss()
        else:
            raise ValueError(f"Unknown regression loss type: {loss_type}")
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, 1) or (B,) predictions
            labels: (B,) target values
        
        Returns:
            Dict with 'loss'
        """
        logits = logits.squeeze(-1)
        loss = self.criterion(logits, labels.float())
        
        return {'loss': loss}


# Loss registry
LOSS_REGISTRY = {
    "mlm": MLMLoss,
    "classification": ClassificationLoss,
    "regression": RegressionLoss,
}


def build_loss(
    loss_name: str,
    vocab_size: Optional[int] = None,
    num_classes: Optional[int] = None,
    label_smoothing: float = 0.0,
    **kwargs
) -> nn.Module:
    """Build loss function by name.
    
    Args:
        loss_name: Name of the loss (mlm | classification | regression)
        vocab_size: Vocabulary size (for MLM loss)
        num_classes: Number of classes (for classification loss)
        label_smoothing: Label smoothing factor
        **kwargs: Additional arguments
    
    Returns:
        Loss module instance
    """
    loss_name_lower = loss_name.lower()
    
    if loss_name_lower not in LOSS_REGISTRY:
        raise ValueError(
            f"Unknown loss: {loss_name}. Available: {list(LOSS_REGISTRY.keys())}"
        )
    
    loss_class = LOSS_REGISTRY[loss_name_lower]
    
    if loss_class == MLMLoss:
        if vocab_size is None:
            raise ValueError("vocab_size required for MLM loss")
        return MLMLoss(vocab_size=vocab_size, label_smoothing=label_smoothing)
    
    elif loss_class == ClassificationLoss:
        if num_classes is None:
            raise ValueError("num_classes required for classification loss")
        return ClassificationLoss(
            num_classes=num_classes, 
            label_smoothing=label_smoothing,
            class_weights=kwargs.get('class_weights')
        )
    
    elif loss_class == RegressionLoss:
        return RegressionLoss(loss_type=kwargs.get('loss_type', 'mse'))
    
    else:
        raise ValueError(f"Unhandled loss class: {loss_class}")
