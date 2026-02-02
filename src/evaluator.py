# USM-AA Classification Evaluator
import csv
from pathlib import Path
from typing import Dict, List, Optional
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef


class ClassificationEvaluator:
    """Classification evaluator that accepts logits directly.
    
    Does NOT initialize or manage models - only computes metrics from logits.
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        threshold: Optional[float] = None,
        best_metric_name: str = "f1",
        label_smoothing: float = 0.0,
    ):
        """Initialize evaluator.
        
        Args:
            num_classes: Number of classes
            threshold: Classification threshold for binary classification.
                       None means use argmax (max prob). For binary, threshold
                       applies to positive class probability.
            best_metric_name: Metric name for best model selection
            label_smoothing: Label smoothing for loss computation
        """
        self.num_classes = num_classes
        self.threshold = threshold  # None = argmax (default)
        self.best_metric_name = best_metric_name
        self.label_smoothing = label_smoothing
        
        # Accumulated data for batch evaluation
        self.reset()
    
    @classmethod
    def from_config(cls, config: Dict) -> "ClassificationEvaluator":
        """Create evaluator from config dict."""
        data_cfg = config.get('data', {})
        training_cfg = config.get('training', {})
        eval_cfg = config.get('evaluator', {})
        
        return cls(
            num_classes=data_cfg.get('num_classes', 2),
            threshold=eval_cfg.get('threshold', None),
            best_metric_name=training_cfg.get('best_metric', 'f1'),
            label_smoothing=training_cfg.get('label_smoothing', 0.0),
        )
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "ClassificationEvaluator":
        """Create evaluator from YAML config file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls.from_config(config)
    
    def reset(self):
        """Reset accumulated predictions and labels."""
        self.all_logits: List[np.ndarray] = []
        self.all_labels: List[np.ndarray] = []
        self.total_loss = 0.0
        self.num_batches = 0
    
    def _logits_to_preds(self, logits: np.ndarray) -> np.ndarray:
        """Convert logits to predictions.
        
        Args:
            logits: [N, num_classes] logits array
        
        Returns:
            [N] predictions array
        """
        if self.threshold is None:
            # Default: argmax (max prob)
            return logits.argmax(axis=-1)
        else:
            # Binary classification with threshold
            if self.num_classes == 2:
                probs = self._softmax(logits)
                return (probs[:, 1] >= self.threshold).astype(np.int64)
            else:
                # For multiclass, threshold not applicable, use argmax
                return logits.argmax(axis=-1)
    
    @staticmethod
    def _softmax(logits: np.ndarray) -> np.ndarray:
        """Compute softmax."""
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        return exp_logits / exp_logits.sum(axis=-1, keepdims=True)
    
    def add_batch(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        compute_loss: bool = True,
    ):
        """Add a batch of predictions for evaluation.
        
        Args:
            logits: [B, num_classes] model output logits
            labels: [B] ground truth labels
            compute_loss: Whether to compute and accumulate loss
        """
        # Compute loss
        if compute_loss:
            loss = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
            self.total_loss += loss.item()
            self.num_batches += 1
        
        # Store for final metrics computation (convert bf16 to fp32 for numpy)
        self.all_logits.append(logits.detach().cpu().float().numpy())
        self.all_labels.append(labels.detach().cpu().numpy())
    
    def compute_metrics(self) -> Dict[str, float]:
        """Compute metrics from accumulated batches.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.all_logits:
            return {"loss": 0.0, "accuracy": 0.0, "f1": 0.0, "mcc": 0.0}
        
        # Concatenate all batches
        logits_all = np.concatenate(self.all_logits, axis=0)
        labels_all = np.concatenate(self.all_labels, axis=0)
        
        # Get predictions
        preds_all = self._logits_to_preds(logits_all)
        
        # Compute metrics
        metrics = {
            "loss": self.total_loss / max(self.num_batches, 1),
            "accuracy": accuracy_score(labels_all, preds_all),
            "f1": f1_score(labels_all, preds_all, average='binary' if self.num_classes == 2 else 'macro'),
            "mcc": matthews_corrcoef(labels_all, preds_all),
        }
        
        return metrics
    
    def evaluate_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """One-shot evaluation from logits and labels.
        
        Args:
            logits: [N, num_classes] all logits
            labels: [N] all labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        self.reset()
        self.add_batch(logits, labels, compute_loss=True)
        return self.compute_metrics()
    
    def save_results(
        self,
        metrics: Dict[str, float],
        output_dir: str,
        split: str = "test",
        best_dev_metric: Optional[float] = None,
    ):
        """Save evaluation results to CSV.
        
        Args:
            metrics: Evaluation metrics dict
            output_dir: Directory to save results
            split: Dataset split name (used in filename)
            best_dev_metric: Best dev metric value (optional)
        """
        results_dir = Path(output_dir) / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        results_file = results_dir / f"{split}_results.csv"
        
        file_exists = results_file.exists()
        
        with open(results_file, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                if best_dev_metric is not None:
                    writer.writerow([f'best_dev_{self.best_metric_name}', 'loss', 'accuracy', 'f1', 'mcc'])
                else:
                    writer.writerow(['loss', 'accuracy', 'f1', 'mcc'])
            
            if best_dev_metric is not None:
                writer.writerow([
                    f"{best_dev_metric:.4f}",
                    f"{metrics['loss']:.4f}",
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['mcc']:.4f}"
                ])
            else:
                writer.writerow([
                    f"{metrics['loss']:.4f}",
                    f"{metrics['accuracy']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['mcc']:.4f}"
                ])
        
        return results_file
