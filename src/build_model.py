# Unified model and tokenizer builder for USM-AA
"""
Usage:
    from build_model import build_model_from_yaml, build_tokenizer, MODEL_CONFIGS
    
    # Build model from yaml config
    model, tokenizer, config = build_model_from_yaml("config.yaml")
    
    # Or build with custom model type
    model, tokenizer, config = build_model_from_yaml("config.yaml", model_type="usm-aa-1b")
    
    # Load pretrained weights
    model, tokenizer, config = build_model_from_yaml(
        "config.yaml", 
        pretrained_path="model.safetensors"
    )
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union

import yaml
import torch

try:
    from .usmaa import USMAAModel, USMAAConfig
    from .tokenizer import build_tokenizer as _build_tokenizer
except ImportError:
    from usmaa import USMAAModel, USMAAConfig
    from tokenizer import build_tokenizer as _build_tokenizer


# Model architecture configurations
MODEL_CONFIGS = {
    "usm-aa-100m": dict(
        num_layers=12, 
        num_heads=12, 
        embedding_dim=768, 
        ffn_dim=4096, 
        use_gate=True
    ),
    "usm-aa-500m": dict(
        num_layers=24, 
        num_heads=20, 
        embedding_dim=1280, 
        ffn_dim=6912, 
        use_gate=True
    ),
    "usm-aa-1b": dict(
        num_layers=36, 
        num_heads=32, 
        embedding_dim=1536, 
        ffn_dim=8192, 
        use_gate=True
    ),
}


def build_tokenizer(kmer: int = 6, dynamic: bool = True):
    """Build tokenizer.
    
    Args:
        kmer: K-mer size for tokenization
        dynamic: Whether to use dynamic k-mer tokenization
    
    Returns:
        tokenizer: Tokenizer instance
    """
    return _build_tokenizer(kmer=kmer, dynamic=dynamic)


def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
    
    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def build_model_config(
    vocab_size: int,
    model_type: str = "usm-aa-100m",
    dropout: float = 0.0,
    droppath: float = 0.0,
    norm_qk: bool = False,
    rotary_theta: float = 5_000_000,
    num_classes: Optional[int] = None,
    **kwargs
) -> USMAAConfig:
    """Build USMAAConfig from model type and parameters.
    
    Args:
        vocab_size: Vocabulary size from tokenizer
        model_type: One of "usm-aa-100m", "usm-aa-500m", "usm-aa-1b"
        dropout: Dropout rate
        droppath: Drop path rate
        norm_qk: Whether to normalize Q and K
        rotary_theta: Rotary position embedding theta
        num_classes: Number of classes for classification (optional)
        **kwargs: Additional config parameters
    
    Returns:
        config: USMAAConfig instance
    """
    if model_type not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_CONFIGS.keys())}")
    
    mcfg = MODEL_CONFIGS[model_type]
    
    config = USMAAConfig(
        vocab_size=vocab_size,
        embedding_dim=mcfg["embedding_dim"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        ffn_dim=mcfg["ffn_dim"],
        use_gate=mcfg["use_gate"],
        dropout=dropout,
        droppath=droppath,
        norm_qk=norm_qk,
        rotary_theta=rotary_theta,
        num_classes=num_classes,
        **kwargs
    )
    
    return config


def load_pretrained_weights(
    model: USMAAModel, 
    pretrained_path: str,
    strict: bool = False,
    verbose: bool = True
) -> int:
    """Load pretrained weights into model.
    
    Args:
        model: USMAAModel instance
        pretrained_path: Path to pretrained weights (.safetensors or .pt)
        strict: Whether to require exact match of state dict keys
        verbose: Whether to print loading info
    
    Returns:
        num_loaded: Number of loaded weight tensors
    """
    if pretrained_path.endswith('.safetensors'):
        from safetensors import safe_open
        state_dict = {}
        with safe_open(pretrained_path, framework='pt') as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    else:
        checkpoint = torch.load(pretrained_path, map_location="cpu")
        if 'module' in checkpoint:
            state_dict = checkpoint['module']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    
    # Remove "module." prefix if present (from DDP)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    
    if strict:
        model.load_state_dict(new_state_dict, strict=True)
        num_loaded = len(new_state_dict)
    else:
        # Load with partial matching
        model_state = model.state_dict()
        loaded_keys = []
        skipped_keys = []
        
        for key, value in new_state_dict.items():
            if key in model_state:
                if model_state[key].shape == value.shape:
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(f"{key} (shape mismatch: {value.shape} vs {model_state[key].shape})")
            else:
                skipped_keys.append(f"{key} (not in model)")
        
        model.load_state_dict(model_state)
        num_loaded = len(loaded_keys)
        
        if verbose and skipped_keys:
            print(f"Skipped {len(skipped_keys)} keys: {skipped_keys[:5]}...")
    
    if verbose:
        print(f"Loaded {num_loaded} weights from {pretrained_path}")
    
    return num_loaded


def build_model(
    vocab_size: int,
    model_type: str = "usm-aa-100m",
    pretrained_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    **config_kwargs
) -> Tuple[USMAAModel, USMAAConfig]:
    """Build USMAAModel with optional pretrained weights.
    
    Args:
        vocab_size: Vocabulary size from tokenizer
        model_type: One of "usm-aa-100m", "usm-aa-500m", "usm-aa-1b"
        pretrained_path: Path to pretrained weights (optional)
        device: Device to load model to (optional)
        dtype: Data type for model (optional)
        **config_kwargs: Additional config parameters
    
    Returns:
        model: USMAAModel instance
        config: USMAAConfig instance
    """
    config = build_model_config(vocab_size, model_type, **config_kwargs)
    model = USMAAModel(config)
    
    if pretrained_path:
        load_pretrained_weights(model, pretrained_path)
    
    if dtype is not None:
        model = model.to(dtype)
    
    if device is not None:
        model = model.to(device)
    
    return model, config


def build_model_from_yaml(
    config_path: str,
    model_type: Optional[str] = None,
    pretrained_path: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    dtype: Optional[torch.dtype] = None,
    kmer: int = 6,
    dynamic: bool = True,
) -> Tuple[USMAAModel, Any, USMAAConfig]:
    """Build model and tokenizer from YAML config file.
    
    Args:
        config_path: Path to YAML config file
        model_type: Override model type from config (optional)
        pretrained_path: Override pretrained path from config (optional)
        device: Device to load model to (optional)
        dtype: Data type for model (optional)
        kmer: K-mer size for tokenization
        dynamic: Whether to use dynamic k-mer tokenization
    
    Returns:
        model: USMAAModel instance
        tokenizer: Tokenizer instance
        config: USMAAConfig instance
    
    Example YAML config:
        model:
          type: usm-aa-100m
          pretrained: path/to/model.safetensors  # optional
        
        training:
          dropout: 0.0
          droppath: 0.0
    """
    # Load YAML config
    cfg = load_config_from_yaml(config_path)
    
    # Build tokenizer
    tokenizer = build_tokenizer(kmer=kmer, dynamic=dynamic)
    vocab_size = len(tokenizer)
    
    # Get model config from YAML
    model_cfg = cfg.get('model', {})
    _model_type = model_type or model_cfg.get('type', 'usm-aa-100m')
    _pretrained_path = pretrained_path or model_cfg.get('pretrained')
    
    # Get training config for dropout etc.
    train_cfg = cfg.get('training', {})
    dropout = train_cfg.get('dropout', 0.0)
    droppath = train_cfg.get('droppath', 0.0)
    
    # Get precision config
    prec_cfg = cfg.get('precision', {})
    
    # Build model
    model, config = build_model(
        vocab_size=vocab_size,
        model_type=_model_type,
        pretrained_path=_pretrained_path,
        device=device,
        dtype=dtype,
        dropout=dropout,
        droppath=droppath,
    )
    
    return model, tokenizer, config


def print_model_info(model: USMAAModel, config: USMAAConfig):
    """Print model information.
    
    Args:
        model: USMAAModel instance
        config: USMAAConfig instance
    """
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model type: {config.num_layers}L-{config.num_heads}H-{config.embedding_dim}D")
    print(f"Model parameters: {num_params:.1f}M")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Hidden dim: {config.embedding_dim}")
    print(f"FFN dim: {config.ffn_dim}")
    print(f"Num layers: {config.num_layers}")
    print(f"Num heads: {config.num_heads}")

