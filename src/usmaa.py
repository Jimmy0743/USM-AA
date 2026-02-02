# HuggingFace compatible USM-AA model
# Usage:
#   from transformers import AutoModel, AutoConfig
#   config = USMAAConfig.from_pretrained("path/to/model")
#   model = USMAAModel.from_pretrained("path/to/model", config=config, trust_remote_code=True)
#   outputs = model(input_ids)
#   hidden_states = outputs.last_hidden_state  # or outputs[0]

from dataclasses import dataclass
from typing import Optional, Tuple, Union, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import ModelOutput


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


@dataclass
class USMAAModelOutput(ModelOutput):
    """Output type for USMAAModel."""
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass 
class USMAALMOutput(ModelOutput):
    """Output type for USMAAModel with LM head."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class USMAASequenceClassifierOutput(ModelOutput):
    """Output type for USMAAForSequenceClassification."""
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


def precompute_freqs_cis(position_ids, theta, dim):
    """Compute rotary position embeddings."""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=position_ids.device).float() / dim))
    freqs = position_ids.unsqueeze(-1).float() * freqs.unsqueeze(0)
    freqs_cos = freqs.cos().unsqueeze(-2)
    freqs_sin = freqs.sin().unsqueeze(-2)
    return torch.stack([freqs_cos, freqs_sin], dim=-1)


def apply_discrete_rotary_emb(x, freqs_cis):
    """Apply rotary embeddings to input tensor."""
    x_reshape = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cos = freqs_cis[..., 0]
    freqs_sin = freqs_cis[..., 1]
    x_out = torch.stack([
        x_reshape[..., 0] * freqs_cos - x_reshape[..., 1] * freqs_sin,
        x_reshape[..., 1] * freqs_cos + x_reshape[..., 0] * freqs_sin,
    ], dim=-1)
    return x_out.flatten(-2).type_as(x)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        bias: bool = False,
        dropout: float = 0.0,
        norm_qk: bool = False,
        use_gate: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        self.use_gate = use_gate

        # QKV(G) projection
        attn_dim = 4 * dim if use_gate else 3 * dim
        self.c_attn = nn.Linear(dim, attn_dim, bias=bias)

        self.norm_qk = norm_qk
        if norm_qk:
            self.norm_q = RMSNorm(self.head_dim)
            self.norm_k = RMSNorm(self.head_dim)

        self.c_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        B, L, D = x.shape
        if self.use_gate:
            qkvg = self.c_attn(x).split([self.dim, self.dim, self.dim, self.dim], dim=-1)
            q, k, v, g = [t.view(B, L, self.num_heads, self.head_dim) for t in qkvg]
        else:
            qkv = self.c_attn(x).split([self.dim, self.dim, self.dim], dim=-1)
            q, k, v = [t.view(B, L, self.num_heads, self.head_dim) for t in qkv]
            g = None

        if self.norm_qk:
            q = self.norm_q(q)
            k = self.norm_k(k)

        if freqs_cis is not None:
            q = apply_discrete_rotary_emb(q, freqs_cis)
            k = apply_discrete_rotary_emb(k, freqs_cis)

        # [B, num_heads, L, head_dim]
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, L, D)
        if g is not None:
            y = y * g.view(B, L, D).sigmoid()
        y = F.dropout(self.c_proj(y), p=self.dropout, training=self.training)
        return y


class MLP(nn.Module):
    def __init__(self, dim: int, ffn_dim: int = None, dropout: float = 0.0):
        super().__init__()
        ffn_dim = ffn_dim or dim * 4
        # GLU structure: c_fc [dim -> ffn_dim], c_proj [ffn_dim//2 -> dim]
        self.c_fc = nn.Linear(dim, ffn_dim, bias=False)
        self.c_proj = nn.Linear(ffn_dim // 2, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # GLU: split c_fc output into two halves, one for activation, one for gate
        h = self.c_fc(x)
        h1, h2 = h.chunk(2, dim=-1)
        return self.dropout(self.c_proj(F.silu(h1) * h2))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_dim: int = None,
        norm_qk: bool = False,
        dropout: float = 0.0,
        droppath: float = 0.0,
        use_gate: bool = True,
    ):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, norm_qk=norm_qk, dropout=dropout, use_gate=use_gate)
        self.mlp = MLP(dim, ffn_dim, dropout=dropout)
        self.norm_attn = RMSNorm(dim)
        self.norm_mlp = RMSNorm(dim)
        self.drop_path = droppath

    def forward(self, x, freqs_cis=None, attn_mask=None):
        h = self.attn(self.norm_attn(x), freqs_cis=freqs_cis, attn_mask=attn_mask)
        if self.drop_path > 0 and self.training:
            h = F.dropout(h, p=self.drop_path, training=True)
        x = x + h
        x = x + self.mlp(self.norm_mlp(x))
        return x


class AttentionPooling(nn.Module):
    def __init__(self, dim: int, out_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.kv_proj = nn.Linear(dim, 2 * dim, bias=False)
        self.out_proj = nn.Linear(dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        q = self.query.expand(B, -1, -1).view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv_proj(x).view(B, L, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn_mask = None
        if mask is not None:
            attn_mask = mask[:, None, None, :].expand(-1, self.num_heads, 1, -1)
            attn_mask = attn_mask.masked_fill(~attn_mask.bool(), float('-inf'))

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        return self.out_proj(self.dropout(out)).squeeze(1)


class USMAAPreTrainedModel(PreTrainedModel):
    """Base class for USM-AA models."""
    config_class = USMAAConfig
    base_model_prefix = "usm_aa"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class USMAAModel(USMAAPreTrainedModel):
    """USM-AA base model for HuggingFace.
    
    Usage:
        config = USMAAConfig.from_pretrained("path/to/model")
        model = USMAAModel.from_pretrained("path/to/model", config=config, trust_remote_code=True)
        
        # Forward pass (HuggingFace style)
        outputs = model(input_ids)
        hidden_states = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        # or
        hidden_states = outputs[0]
        
        # Mean/Max pooling for embeddings
        embedding_mean = hidden_states.mean(dim=1)  # [batch, hidden_dim]
        embedding_max = hidden_states.max(dim=1)[0]  # [batch, hidden_dim]
    """

    def __init__(self, config: USMAAConfig):
        super().__init__(config)
        self.config = config
        self.dim = config.embedding_dim

        self.embedding = nn.Embedding(config.vocab_size, self.dim)
        self.chr_embedding = nn.Embedding(config.num_chrs, self.dim, padding_idx=0)
        self.strand_embedding = nn.Embedding(2, self.dim)
        self.cigar_embedding = nn.Embedding(config.num_cigar_ops, self.dim, padding_idx=9)

        self.layers = nn.ModuleList([
            TransformerBlock(
                self.dim,
                config.num_heads,
                config.ffn_dim,
                norm_qk=config.norm_qk,
                dropout=config.dropout,
                droppath=config.droppath,
                use_gate=config.use_gate
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(self.dim)
        self.lm_head = nn.Linear(self.dim, config.vocab_size, bias=False)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        chr_ids: torch.Tensor = None,
        strand_ids: torch.Tensor = None,
        cigar_ids: torch.Tensor = None,
        token_ids: torch.Tensor = None,  # Alias for input_ids (backward compatibility)
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, USMAAModelOutput]:
        """Forward pass.
        
        Args:
            input_ids: [batch_size, seq_len] token ids (HuggingFace standard)
            attention_mask: [batch_size, seq_len] attention mask (1=valid, 0=padding)
            position_ids: [batch_size, seq_len] position ids (auto-generated if None)
            chr_ids: [batch_size, seq_len] chromosome ids
            strand_ids: [batch_size, seq_len] strand ids
            cigar_ids: [batch_size, seq_len] cigar operation ids
            token_ids: Alias for input_ids (backward compatibility)
            output_hidden_states: Whether to return all hidden states
            output_attentions: Whether to return attention weights (not implemented)
            return_dict: Whether to return ModelOutput or tuple
        
        Returns:
            USMAAModelOutput or tuple containing last_hidden_state
        """
        # Handle input_ids alias
        if input_ids is None and token_ids is not None:
            input_ids = token_ids
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        batch_size, seq_len = input_ids.shape
        
        # Auto-generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(input_ids)
        if chr_ids is not None:
            x = x + self.chr_embedding(chr_ids)
        if strand_ids is not None:
            x = x + self.strand_embedding(strand_ids)
        if cigar_ids is not None:
            x = x + self.cigar_embedding(cigar_ids)

        freqs_cis = precompute_freqs_cis(
            position_ids,
            theta=self.config.rotary_theta,
            dim=self.dim // self.config.num_heads
        )

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].bool()

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            x = layer(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        x = self.final_norm(x)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        if not return_dict:
            return (x,) + (all_hidden_states,) if output_hidden_states else (x,)

        return USMAAModelOutput(
            last_hidden_state=x,
            hidden_states=all_hidden_states,
            attentions=None,
        )

    def forward_mlm(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        chr_ids: torch.Tensor = None,
        strand_ids: torch.Tensor = None,
        cigar_ids: torch.Tensor = None,
        token_ids: torch.Tensor = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        """Forward pass with LM head logits (for MLM pretraining).
        
        Returns:
            dict with "logits" key for backward compatibility with trainer
        """
        # Handle input_ids alias
        if input_ids is None and token_ids is not None:
            input_ids = token_ids
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(input_ids)
        if chr_ids is not None:
            x = x + self.chr_embedding(chr_ids)
        if strand_ids is not None:
            x = x + self.strand_embedding(strand_ids)
        if cigar_ids is not None:
            x = x + self.cigar_embedding(cigar_ids)

        freqs_cis = precompute_freqs_cis(
            position_ids,
            theta=self.config.rotary_theta,
            dim=self.dim // self.config.num_heads
        )

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].bool()

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            x = layer(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        if not return_dict:
            return (logits, all_hidden_states) if output_hidden_states else (logits,)

        return {"logits": logits, "hidden_states": all_hidden_states}

    @torch.no_grad()
    def get_embeddings(
        self,
        input_ids: torch.Tensor = None,
        token_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        chr_ids: torch.Tensor = None,
        strand_ids: torch.Tensor = None,
        cigar_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        pooling: str = "mean",
        layer: int = -1,
    ) -> torch.Tensor:
        """Extract embeddings from the model.
        
        Args:
            input_ids: [batch_size, seq_len] token ids
            token_ids: Alias for input_ids (backward compatibility)
            position_ids: [batch_size, seq_len] position ids
            chr_ids: [batch_size, seq_len] chromosome ids
            strand_ids: [batch_size, seq_len] strand ids
            cigar_ids: [batch_size, seq_len] cigar operation ids
            attention_mask: [batch_size, seq_len] attention mask (1 for valid, 0 for padding)
            pooling: Pooling strategy - "mean", "cls", "max", or "none"
            layer: Which layer to extract embeddings from (-1 for last layer)
        
        Returns:
            embeddings: [batch_size, dim] or [batch_size, seq_len, dim] if pooling="none"
        """
        # Handle input_ids alias
        if input_ids is None and token_ids is not None:
            input_ids = token_ids
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        batch_size, seq_len = input_ids.shape
        
        # Auto-generate position_ids if not provided
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(input_ids)
        if chr_ids is not None:
            x = x + self.chr_embedding(chr_ids)
        if strand_ids is not None:
            x = x + self.strand_embedding(strand_ids)
        if cigar_ids is not None:
            x = x + self.cigar_embedding(cigar_ids)

        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        freqs_cis = precompute_freqs_cis(
            position_ids,
            theta=self.config.rotary_theta,
            dim=self.dim // self.config.num_heads
        )

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].bool()

        # Forward through layers
        num_layers = len(self.layers)
        target_layer = layer if layer >= 0 else num_layers + layer

        for i, block in enumerate(self.layers):
            x = block(x, freqs_cis=freqs_cis, attn_mask=attn_mask)
            if i == target_layer:
                break

        x = self.final_norm(x)

        # Apply pooling
        if pooling == "none":
            return x
        elif pooling == "cls":
            return x[:, 0]
        elif pooling == "max":
            if attention_mask is not None:
                x = x.masked_fill(~attention_mask.unsqueeze(-1).bool(), float('-inf'))
            return x.max(dim=1)[0]
        else:  # mean
            if attention_mask is not None:
                x = (x * attention_mask.unsqueeze(-1)).sum(dim=1)
                x = x / attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                x = x.mean(dim=1)
            return x


class USMAAForSequenceClassification(USMAAPreTrainedModel):
    """USM-AA model for sequence classification."""

    def __init__(self, config: USMAAConfig):
        super().__init__(config)
        self.config = config
        self.num_labels = config.num_classes or config.num_labels
        self.dim = config.embedding_dim

        self.embedding = nn.Embedding(config.vocab_size, self.dim)
        self.chr_embedding = nn.Embedding(config.num_chrs, self.dim, padding_idx=0)
        self.strand_embedding = nn.Embedding(2, self.dim)
        self.cigar_embedding = nn.Embedding(config.num_cigar_ops, self.dim, padding_idx=9)

        self.layers = nn.ModuleList([
            TransformerBlock(
                self.dim,
                config.num_heads,
                config.ffn_dim,
                norm_qk=config.norm_qk,
                dropout=config.dropout,
                droppath=config.droppath,
                use_gate=config.use_gate
            )
            for _ in range(config.num_layers)
        ])
        self.final_norm = RMSNorm(self.dim)

        # Classification head
        self.classifier = nn.Linear(self.dim, self.num_labels)

        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        chr_ids: torch.Tensor = None,
        strand_ids: torch.Tensor = None,
        cigar_ids: torch.Tensor = None,
        token_ids: torch.Tensor = None,
        labels: torch.Tensor = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[Tuple, USMAASequenceClassifierOutput]:
        # Handle input_ids alias
        if input_ids is None and token_ids is not None:
            input_ids = token_ids
        if input_ids is None:
            raise ValueError("input_ids must be provided")
        
        batch_size, seq_len = input_ids.shape
        
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        x = self.embedding(input_ids)
        if chr_ids is not None:
            x = x + self.chr_embedding(chr_ids)
        if strand_ids is not None:
            x = x + self.strand_embedding(strand_ids)
        if cigar_ids is not None:
            x = x + self.cigar_embedding(cigar_ids)

        freqs_cis = precompute_freqs_cis(
            position_ids,
            theta=self.config.rotary_theta,
            dim=self.dim // self.config.num_heads
        )

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask[:, None, None, :].bool()

        all_hidden_states = () if output_hidden_states else None
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (x,)
            x = layer(x, freqs_cis=freqs_cis, attn_mask=attn_mask)

        x = self.final_norm(x)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (x,)

        # Mean pooling
        if attention_mask is not None:
            x = (x * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(1)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        if not return_dict:
            output = (logits,) + (all_hidden_states,) if output_hidden_states else (logits,)
            return ((loss,) + output) if loss is not None else output

        return USMAASequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=None,
        )


def convert_usmaa_to_hf(ckpt_path: str, output_dir: str, model_type: str = "usm-aa-100m"):
    """Convert original USM-AA checkpoint to HuggingFace format.
    
    Args:
        ckpt_path: Path to original checkpoint (.pt file)
        output_dir: Output directory for HuggingFace model
        model_type: One of "usm-aa-100m", "usm-aa-500m", "usm-aa-1b"
    """
    import os
    
    CONFIG_MAP = {
        "usm-aa-100m": dict(num_layers=12, num_heads=12, embedding_dim=768, ffn_dim=4096, use_gate=True),
        "usm-aa-1b": dict(num_layers=36, num_heads=32, embedding_dim=1536, ffn_dim=8192, use_gate=True),
    }
    
    mcfg = CONFIG_MAP[model_type]
    config = USMAAConfig(
        vocab_size=512,
        embedding_dim=mcfg["embedding_dim"],
        num_layers=mcfg["num_layers"],
        num_heads=mcfg["num_heads"],
        ffn_dim=mcfg["ffn_dim"],
        use_gate=mcfg["use_gate"],
    )
    
    model = USMAAModel(config)
    
    # Load original checkpoint
    state_dict = torch.load(ckpt_path, map_location="cpu")
    if "model" in state_dict:
        state_dict = state_dict["model"]
    
    # Convert state dict keys if needed
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove "module." prefix if present (from DDP)
        if k.startswith("module."):
            k = k[7:]
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=False)
    
    # Save to HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    config.save_pretrained(output_dir)
    
    print(f"Model saved to {output_dir}")
    return model

if __name__ == "__main__":
    main()