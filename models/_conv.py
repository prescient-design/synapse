"""GNN and MLP models for graph regression."""

import math
import torch
import torch.nn as nn
import einops
from torch_geometric.nn import global_mean_pool

from constants._connectivity import CONNECTIVITY_WEIGHTS
from models.layers.linear import Linear
from models.layers.pos_enc import RotaryEmbedding
from models.layers.transformer import TransformerBlock
from models.layers.gnn import GINConv
from models.layers.connectivity_embedding import ConnectivityEmbedding


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.TokenEmbeddings = nn.Parameter(torch.empty(vocab_size, emb_dim))
        nn.init.normal_(self.TokenEmbeddings, std=0.02)

    def forward(self, x):
        return self.TokenEmbeddings[x, :]


class GINGNN(nn.Module):
    """GNN model with transformer encoder and GIN layers."""

    def __init__(self, num_layers, vocab_size, emb_dim, multiplier=1, bn=True,
                 sequence_aggregation='mean', use_connectivity_embedding=True,
                 connectivity_format='trispecific'):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_connectivity_embedding = use_connectivity_embedding
        self.connectivity_format = connectivity_format

        self.token_embedding = TokenEmbedding(vocab_size, emb_dim)

        if use_connectivity_embedding:
            self.connectivity_embedding = ConnectivityEmbedding(connectivity_format, 1)
        else:
            self.connectivity_embedding = None

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads=1, d_ff=emb_dim,
                           positional_encoder=RotaryEmbedding(context_length=297, dim=emb_dim))
            for _ in range(3)
        ])

        gin_input_dim = emb_dim + (1 if use_connectivity_embedding else 0)
        self.layers = nn.ModuleList([GINConv(gin_input_dim, emb_dim, multiplier)])
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(emb_dim, emb_dim, multiplier))

        self.post_mp = nn.Sequential(Linear(emb_dim, emb_dim), nn.GELU(), Linear(emb_dim, 1))


    def forward(self, x, edge_index, batch, edge_attr=None, connectivity_type=None):
        x = self.token_embedding(x.long())
        for block in self.transformer_blocks:
            x = block(x)
        x = einops.reduce(x, "n s e -> n e", "mean")

        if self.use_connectivity_embedding and connectivity_type is not None:
            conn_emb = self.connectivity_embedding(connectivity_type.long())
            x = torch.cat([x, conn_emb[batch]], dim=-1)

        for layer in self.layers:
            x = layer(x, edge_index)

        return self.post_mp(global_mean_pool(x, batch))


class MLP(nn.Module):

    def __init__(self, num_layers, vocab_size, emb_dim, multiplier=1, bn=True,
                 sequence_aggregation='mean', use_connectivity_embedding=False,
                 connectivity_format='trispecific'):
        super().__init__()
        self.emb_dim = emb_dim
        self.use_connectivity_embedding = use_connectivity_embedding
        self.connectivity_format = connectivity_format

        self.token_embedding = TokenEmbedding(vocab_size, emb_dim)

        if use_connectivity_embedding:
            self.connectivity_embedding = ConnectivityEmbedding(connectivity_format, 1)
        else:
            self.connectivity_embedding = None

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(emb_dim, num_heads=1, d_ff=emb_dim,
                           positional_encoder=RotaryEmbedding(context_length=297, dim=emb_dim))
            for _ in range(3)
        ])

        mlp_input_dim = emb_dim + (1 if use_connectivity_embedding else 0)
        self.layers = nn.ModuleList([
            nn.Sequential(Linear(mlp_input_dim, emb_dim), nn.LayerNorm(emb_dim), nn.GELU(), Linear(emb_dim, emb_dim))
        ])
        for _ in range(num_layers - 2):
            self.layers.append(nn.Sequential(
                Linear(emb_dim, emb_dim), nn.LayerNorm(emb_dim), nn.GELU(), Linear(emb_dim, emb_dim)
            ))

        self.post_mp = nn.Sequential(Linear(emb_dim, emb_dim), nn.GELU(), Linear(emb_dim, 1))

    def forward(self, x, edge_index, batch, edge_attr=None, connectivity_type=None):
        x = self.token_embedding(x.long())
        for block in self.transformer_blocks:
            x = block(x)
        x = einops.reduce(x, "n s e -> n e", "mean")

        if self.use_connectivity_embedding and connectivity_type is not None:
            conn_emb = self.connectivity_embedding(connectivity_type.long())
            x = torch.cat([x, conn_emb[batch]], dim=-1)

        for layer in self.layers:
            x = layer(x)

        return self.post_mp(global_mean_pool(x, batch))
