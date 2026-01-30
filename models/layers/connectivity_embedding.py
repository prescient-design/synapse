import torch
import torch.nn as nn
from jaxtyping import Float, Int
import einx
from torch import Tensor
from constants._connectivity import CONNECTIVITY_WEIGHTS


class ConnectivityEmbedding(nn.Module):
    def __init__(self, connectivity_format: str, emb_dim: int):
        super().__init__()
        self.connectivity_format = connectivity_format
        self.num_connectivity_types = len(CONNECTIVITY_WEIGHTS[connectivity_format])

        self.connectivity_embedding = nn.Parameter(torch.empty(self.num_connectivity_types, emb_dim), requires_grad=True)
        nn.init.normal_(self.connectivity_embedding, std=0.02)

    def forward(self, x: Int[Tensor, " ... "]) -> Float[Tensor, " ... emb_dim"]:
        return self.connectivity_embedding[x, :]