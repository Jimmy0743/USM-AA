import yaml

from .base_tokenizer import AbstractTokenizer
from src.utils import Registry

TOKENIZER_REGISTRY = Registry("tokenizer")
_GLOBAL_TOKENIZER = None


def create_tokenizer(cls_name: str, path: str, **kwargs) -> AbstractTokenizer:
    return TOKENIZER_REGISTRY[cls_name](path, **kwargs)


def build_tokenizer(config_path: str):
    """Build tokenizer from YAML config file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        KmerTokenizer instance
        
    Example config:
        tokenizer:
          kmer: 6
          dynamic: true
          special_tokens:
            - <pad>
            - <mask>
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    tokenizer_cfg = config.get('tokenizer', {})
    
    kmer = tokenizer_cfg.get('kmer', 6)
    dynamic = tokenizer_cfg.get('dynamic', True)
    special_tokens = tokenizer_cfg.get('special_tokens', ['<pad>', '<mask>'])
    
    from .kmer_tokenizer import KmerTokenizer
    return KmerTokenizer(kmer=kmer, dynamic=dynamic, special_tokens=special_tokens)
