import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Int
from torch import Tensor


class LayerNorm(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.w = nn.Parameter(torch.ones(in_dim))
        self.b = nn.Parameter(torch.zeros(in_dim))
    
    def forward(self, residual: Float[Tensor, " ... in_dim"]) -> Float[Tensor, " ... in_dim"]:
        # Normalize over the last dimension (embedding dimension) using flexible pattern
        mean = einops.reduce(residual, "... d_model -> ... 1", "mean")
        residual = residual - mean
        # Calculate the variance, square root it. Add in an epsilon to prevent divide by zero.
        scale = (einops.reduce(residual.pow(2), "... d_model -> ... 1", "mean") + 1e-5).sqrt()
        normalized = residual / scale
        normalized = normalized * self.w + self.b
        return normalized