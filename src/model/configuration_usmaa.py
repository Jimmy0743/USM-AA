
from transformers import PretrainedConfig


class USMAAConfig(PretrainedConfig):
    """Configuration class for USM-AA model."""
    model_type = "usm-aa"

    def __init__(
            self,
            vocab_size: int = 512,
            num_cigar_ops: int = 10,
            num_chrs: int = 32,
            embedding_dim: int = 768,
            hidden_size: int = None,  # Alias for embedding_dim (HF compatibility)
            num_layers: int = 12,
            num_hidden_layers: int = None,  # Alias for num_layers (HF compatibility)
            num_heads: int = 12,
            num_attention_heads: int = None,  # Alias for num_heads (HF compatibility)
            ffn_dim: int = None,
            intermediate_size: int = None,  # Alias for ffn_dim (HF compatibility)
            sliding_window_size: int = None,
            norm_qk: bool = False,
            use_gate: bool = True,
            rotary_theta: float = 5_000_000,
            dropout: float = 0.0,
            hidden_dropout_prob: float = None,  # Alias for dropout (HF compatibility)
            droppath: float = 0.0,
            num_classes: int = None,
            num_labels: int = None,  # Alias for num_classes (HF compatibility)
            pooler: str = "attention",
            pad_token_id: int = 2,  # Default pad token id
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.num_cigar_ops = num_cigar_ops
        self.num_chrs = num_chrs
        # Handle aliases for HF compatibility
        self.embedding_dim = hidden_size or embedding_dim
        self.hidden_size = self.embedding_dim
        self.num_layers = num_hidden_layers or num_layers
        self.num_hidden_layers = self.num_layers
        self.num_heads = num_attention_heads or num_heads
        self.num_attention_heads = self.num_heads
        self.ffn_dim = intermediate_size or ffn_dim or self.embedding_dim * 4
        self.intermediate_size = self.ffn_dim
        self.sliding_window_size = sliding_window_size
        self.norm_qk = norm_qk
        self.use_gate = use_gate
        self.rotary_theta = rotary_theta
        self.dropout = hidden_dropout_prob if hidden_dropout_prob is not None else dropout
        self.hidden_dropout_prob = self.dropout
        self.droppath = droppath
        self.num_classes = num_labels or num_classes
        # Only set num_labels if num_classes is not None (avoid HF setter issue)
        if self.num_classes is not None:
            self.num_labels = self.num_classes
        self.pooler = pooler
