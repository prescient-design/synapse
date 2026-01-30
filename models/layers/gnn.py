import torch
import torch.nn as nn
from jaxtyping import Float, Int
from torch import Tensor
from models.layers.linear import Linear
from torch_geometric.nn import GINConv as PyGINConv

class GINConv(nn.Module):

    def __init__(self, in_dim: int, emb_dim: int, multiplier: int = 1):
        super().__init__()
        self.mlp = nn.Sequential(
            Linear(in_dim, multiplier * emb_dim),
            nn.LayerNorm(multiplier * emb_dim),
            nn.GELU(),
            Linear(multiplier * emb_dim, emb_dim),
        )
        self.layer = PyGINConv(nn=self.mlp, train_eps=True)

    def forward(self, x: Float[Tensor, " ... in_dim"], edge_index: Int[Tensor, " 2 num_edges"]) -> Float[Tensor, " ... emb_dim"]:
        """
        Args:
            x: [num_nodes, in_dim] - node features
            edge_index: [2, num_edges] - graph connectivity
        """
        return self.layer(x, edge_index)