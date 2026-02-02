from .base_tokenizer import AbstractTokenizer
from src.utils import Registry

TOKENIZER_REGISTRY = Registry("tokenizer")
_GLOBAL_TOKENIZER = None


def create_tokenizer(cls_name: str, path: str, **kwargs) -> AbstractTokenizer:
    return TOKENIZER_REGISTRY[cls_name](path, **kwargs)


def build_tokenizer(cfg=None, kmer=6, dynamic=True, special_tokens=None):
    """Build tokenizer from config or direct parameters.
    
    Args:
        cfg: Config object with tokenizer settings (optional)
        kmer: K-mer size (default: 6)
        dynamic: Whether to use dynamic k-mer (default: True)
        special_tokens: List of special tokens (optional)
    """
    if cfg is not None:
        # Build from config
        assert cfg.tokenizer.name, f"please specify tokenizer name, get {cfg.tokenizer.name}"
        tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.name](cfg)
    else:
        # Build directly with parameters
        from .kmer_tokenizer import KmerTokenizer
        special_tokens = special_tokens or ["<pad>", "<mask>"]
        tokenizer = KmerTokenizer(kmer=kmer, dynamic=dynamic, special_tokens=special_tokens)
    return tokenizer
