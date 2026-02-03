import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class ClassificationLoss(nn.Module):
    """Classification Loss for sequence classification tasks.
    
    Supports multi-class classification with optional label smoothing.
    """
    
    def __init__(
        self, 
        num_classes: int, 
        label_smoothing: float = 0.0,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Args:
            num_classes: Number of classes
            label_smoothing: Label smoothing factor (0.0 = no smoothing)
            class_weights: Optional class weights for imbalanced data
        """
        super().__init__()
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing
        )
        self._first_batch = True  # Flag to print first batch
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: [B, num_classes] model output logits
            labels: [B] target labels (long tensor)
        
        Returns:
            Dict with 'loss' and 'accuracy'
        """
        # Print first batch for debugging
        if self._first_batch:
            with torch.no_grad():
                preds = logits.argmax(dim=-1)
                print(f"[DEBUG] First batch:")
                print(f"  Labels: {labels.tolist()}")
                print(f"  Preds:  {preds.tolist()}")
                print(f"  Logits shape: {logits.shape}, Labels shape: {labels.shape}")
            self._first_batch = False
        
        loss = self.criterion(logits, labels)
        
        # Compute accuracy
        with torch.no_grad():
            preds = logits.argmax(dim=-1)
            accuracy = (preds == labels).float().mean()
        
        return {'loss': loss, 'accuracy': accuracy}
