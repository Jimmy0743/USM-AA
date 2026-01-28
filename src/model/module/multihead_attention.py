import torch.nn as nn
import torch.nn.functional as F
from .rms_norm import RMSNorm
from .rotary_embedding import apply_discrete_rotary_emb


class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            bias: bool = False,
            dropout: float = 0.0,
            norm_qk: bool = False
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout = dropout
        # QKV(G) projection
        self.c_attn = nn.Linear(dim, 4 * dim, bias=bias)
        self.norm_qk = norm_qk
        if norm_qk:
            self.norm_q = RMSNorm(self.head_dim)
            self.norm_k = RMSNorm(self.head_dim)

        self.c_proj = nn.Linear(dim, dim, bias=bias)

    def forward(self, x, freqs_cis=None, attn_mask=None):
        B, L, D = x.shape
        qkvg = self.c_attn(x).split([self.dim, self.dim, self.dim, self.dim], dim=-1)
        q, k, v, g = [t.view(B, L, self.num_heads, self.head_dim) for t in qkvg]
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
        y = y * g.view(B, L, D).sigmoid()
        y = F.dropout(self.c_proj(y), p=self.dropout, training=self.training)
        return y
