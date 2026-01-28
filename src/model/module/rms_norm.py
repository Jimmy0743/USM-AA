import numbers
from typing import Tuple, Optional, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size

_shape_t = Union[int, List[int], Size]


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization

    Ref：
        1. Root Mean Square Layer Normalization

    """
    __constants__ = ['normalized_shape', 'eps', 'elementwise_affine']
    normalized_shape: Tuple[int, ...]
    eps: Optional[float]

    def __init__(self,
                 normalized_shape: _shape_t,
                 eps: Optional[float] = 1e-6,
                 elementwise_affine: bool = True,
                 sequence_parallel: bool = False,
                 device=None,
                 dtype=None):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.eps = eps
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.empty(*normalized_shape, **factory_kwargs))
            setattr(self.weight, "sequence_parallel", sequence_parallel)
        else:
            self.register_parameter('weight', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            nn.init.ones_(self.weight)

    def _rms_norm(self, input):
        dtype = input.dtype
        dims = tuple(i for i in range(-1, -len(self.normalized_shape) - 1, -1))
        variance = input.to(torch.float32).pow(2).mean(dims, keepdim=True)
        input = input * torch.rsqrt(variance + self.eps)
        input = input.to(dtype)

        if self.weight is None:
            return input

        return self.weight * input

    def _forward_impl(self, x):
        if hasattr(F, "rms_norm"):
            return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

        output = self._rms_norm(x)
        return output

    def forward(self, input, mask=None, inplace=False):
        """
        Args:
            input: [*, dim]
            mask: [*,]

        Returns:
            [*, dim]
        """
        if mask is None:
            return self._forward_impl(input)

        output = input if inplace else input.clone()
        output[mask] = self._forward_impl(input[mask])
        return output

    def extra_repr(self):
        return f"{self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine}"
