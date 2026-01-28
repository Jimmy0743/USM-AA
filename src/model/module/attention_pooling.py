import torch
import torch.nn as nn


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
