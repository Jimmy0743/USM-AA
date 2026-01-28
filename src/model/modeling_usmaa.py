# HuggingFace compatible USM-AA model
# Usage:
#   from transformers import AutoModel, AutoConfig
#   config = USMAAConfig.from_pretrained("path/to/model")
#   model = USMAAModel.from_pretrained("path/to/model", config=config, trust_remote_code=True)
#   outputs = model(input_ids)
#   hidden_states = outputs.last_hidden_state  # or outputs[0]
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput
from .module import RMSNorm, precompute_freqs_cis, MultiHeadAttention, DropPath
from .configuration_usmaa import USMAAConfig


@dataclass
class USMAAModelOutput(ModelOutput):
    """Output type for USMAAModel."""
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
            droppath: float = 0.0
    ):
        super().__init__()
        self.attn = MultiHeadAttention(dim, num_heads, norm_qk=norm_qk, dropout=dropout)
        self.mlp = MLP(dim, ffn_dim, dropout=dropout)
        self.norm_attn = RMSNorm(dim)
        self.norm_mlp = RMSNorm(dim)
        self.drop_path = DropPath(droppath)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        h = self.attn(self.norm_attn(x), freqs_cis=freqs_cis, attn_mask=attn_mask)
        x = x + self.drop_path(h)
        x = x + self.drop_path(self.mlp(self.norm_mlp(x)))
        return x


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
        if chr_ids is not None:
            chr_ids = torch.zeros_like(input_ids, dtype=torch.long)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)

        x = self.embedding(input_ids)
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
