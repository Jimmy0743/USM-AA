import torch


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
