from typing import List, Optional
import torch


def collate_dense_tensors(tensors: List[torch.Tensor], pad_value=0) -> torch.Tensor:
    """Pad tensors of different lengths to the same length and stack."""
    max_len = max(t.size(0) for t in tensors)
    batch_size = len(tensors)

    if tensors[0].dim() == 1:
        result = torch.full((batch_size, max_len), pad_value, dtype=tensors[0].dtype)
    else:
        result = torch.full((batch_size, max_len, *tensors[0].shape[1:]), pad_value, dtype=tensors[0].dtype)

    for i, t in enumerate(tensors):
        result[i, :t.size(0)] = t

    return result


def get_random_mask(tokens: torch.Tensor, mask_ratio: float = 0.15,
                    filter_ids: Optional[List[int]] = None) -> torch.Tensor:
    """Generate random mask."""
    mask = torch.lt(torch.rand_like(tokens, dtype=torch.float32), mask_ratio)
    if filter_ids is not None:
        filter_ids = torch.tensor(list(set(filter_ids)), dtype=torch.long)
        fmask = (tokens.unsqueeze(-1) == filter_ids).any(dim=-1)
        mask = mask & (~fmask)
    return mask
