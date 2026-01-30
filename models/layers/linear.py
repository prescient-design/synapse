import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Int
import einx
from torch import Tensor
import math

class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        """A linear layer initialized with truncated normal fan-in fan-out.

        Args:
            d_in: int
                The number of input features.
            d_out: int
                The number of output features.
        """

        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight: Float[Tensor, " d_out d_in"] = nn.Parameter(
            nn.init.trunc_normal_(torch.empty(d_out, d_in), std=std, a=-3 * std, b=3 * std), requires_grad=True
        )
        self.bias: Float[Tensor, " d_out"] = nn.Parameter(
            torch.zeros(d_out), requires_grad=True
        )

    def forward(self, x: Float[Tensor, " ... d_in"]) -> Float[Tensor, " ... d_out"]:
        output = einops.einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return output + self.bias