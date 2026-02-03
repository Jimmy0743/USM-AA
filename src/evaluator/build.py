# Evaluator Builder
from typing import Dict, Type

import yaml

from .evaluator import ClassificationEvaluator


# Evaluator registry: name -> class
EVALUATOR_REGISTRY: Dict[str, Type] = {
    "classification": ClassificationEvaluator,
}


def register_evaluator(name: str):
    """Decorator to register an evaluator class.
    
    Usage:
        @register_evaluator("my_evaluator")
        class MyEvaluator:
            ...
    """
    def decorator(cls):
        EVALUATOR_REGISTRY[name] = cls
        return cls
    return decorator


def build_evaluator(config_path: str):
    """Build evaluator from YAML config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Initialized evaluator instance
        
    Example config:
        evaluator:
          type: classification
          threshold: null
          batch_size: 64
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    eval_cfg = config.get('evaluator', {})
    evaluator_type = eval_cfg.get('type', 'classification')
    
    if evaluator_type not in EVALUATOR_REGISTRY:
        raise ValueError(
            f"Unknown evaluator type: '{evaluator_type}'. "
            f"Available: {list(EVALUATOR_REGISTRY.keys())}"
        )
    
    evaluator_cls = EVALUATOR_REGISTRY[evaluator_type]
    return evaluator_cls.from_config(config)
