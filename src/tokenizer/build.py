from .base_tokenizer import AbstractTokenizer
from src.utils import Registry

TOKENIZER_REGISTRY = Registry("tokenizer")
_GLOBAL_TOKENIZER = None


def create_tokenizer(cls_name: str, path: str, **kwargs) -> AbstractTokenizer:
    return TOKENIZER_REGISTRY[cls_name](path, **kwargs)


def build_tokenizer(cfg):
    assert cfg.tokenizer.name, f"please specify tokenizer name, get {cfg.tokenizer.name}"
    tokenizer = TOKENIZER_REGISTRY[cfg.tokenizer.name](cfg)
    return tokenizer
