
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLMLoss(nn.Module):
    """Masked Language Model Loss for pretraining.
    
    Computes cross-entropy loss only at masked positions.
    """
    
    def __init__(self, vocab_size: int, label_smoothing: float = 0.0):
        """
        Args:
            vocab_size: Vocabulary size
            label_smoothing: Label smoothing factor
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
    
    def forward(
        self, 
        logits: torch.Tensor, 
        labels: torch.Tensor, 
        mask: torch.Tensor
    ):
        """
        Args:
            logits: [B, L, V] model output logits
            labels: [B, L] target labels
            mask: [B, L] mask positions (True = compute loss)
        
        Returns:
            loss: scalar loss
            accuracy: accuracy at masked positions
        """
        # Only compute loss at masked positions
        logits_flat = logits.view(-1, self.vocab_size)
        labels_flat = labels.view(-1)
        mask_flat = mask.view(-1)
        
        # Select masked positions
        masked_logits = logits_flat[mask_flat]
        masked_labels = labels_flat[mask_flat]
        
        if masked_logits.size(0) == 0:
            return (
                torch.tensor(0.0, device=logits.device), 
                torch.tensor(0.0, device=logits.device)
            )
        
        loss = F.cross_entropy(
            masked_logits, masked_labels, 
            label_smoothing=self.label_smoothing
        )
        
        # Compute accuracy
        with torch.no_grad():
            preds = masked_logits.argmax(dim=-1)
            accuracy = (preds == masked_labels).float().mean()
        
        return loss, accuracy
