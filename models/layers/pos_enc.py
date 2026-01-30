import torch
import torch.nn as nn
import einops
from jaxtyping import Float, Int
from torch import Tensor
from models.layers.linear import Linear
import einx

class RotaryEmbedding(nn.Module):
    def __init__(self, context_length: int, dim: int, theta: float = 10000.0):
        super().__init__()
        self.register_buffer(
            "_freq_cis_cache", RotaryEmbedding._init_cache(context_length, dim, theta), persistent=False
        )

    @staticmethod
    def _init_cache(context_length: int, dim: int, theta: float) -> Float[Tensor, " 2 context_length half_dim"]:
        assert dim % 2 == 0

        d = torch.arange(0, dim, 2) / dim
        freqs = theta**-d
        t = torch.arange(context_length)

        freqs = einops.einsum(t, freqs, "t, f -> t f")

        cos, sin = torch.cos(freqs), torch.sin(freqs)
        return torch.stack((cos, sin))

    def forward(self, x: Float[Tensor, " ... seq d"], pos_ids: Int[Tensor, " ... seq"]) -> Float[Tensor, " ... seq d"]:
        x1, x2 = einops.rearrange(x, "... (half_d xy) -> xy ... half_d", xy=2)

        # Standard
        # cos, sin = self._freq_cis_cache[:, pos_ids, :]

        # einx
        cos, sin = einx.get_at("cos_sin [pos] half_dim, ... -> cos_sin ... half_dim", self._freq_cis_cache, pos_ids)

        # 2D rotation matrix applied to pairs in x
        x1_rot = cos * x1 - sin * x2
        x2_rot = sin * x1 + cos * x2
        result = einx.rearrange("... x_half, ... x_half -> ... (x_half (1 + 1))", x1_rot, x2_rot).contiguous()
        return result


class PosEmbed(nn.Module):
    def __init__(self, max_seq_len: int, emb_dim: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.emb_dim = emb_dim
        self.W_pos = nn.Parameter(torch.empty((max_seq_len, emb_dim)), requires_grad=True)
        nn.init.normal_(self.W_pos, std=0.02)
    
    def forward(self, tokens: Int[Tensor, " num_nodes sequence_length"]) -> Float[Tensor, " num_nodes sequence_length emb_dim"]:
        # tokens: [num_nodes, sequence_length] - but we only use it to get the shape
        seq_len = tokens.size(1)
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        pos_indices = torch.arange(seq_len, device=tokens.device)  # [sequence_length]
        # Get positional embeddings: [sequence_length, emb_dim]
        pos_embed = self.W_pos[pos_indices, :]  # [sequence_length, emb_dim]
        # Expand to match batch dimension: [num_nodes, sequence_length, emb_dim]
        pos_embed = einops.repeat(pos_embed, "seq_len emb_dim -> num_nodes seq_len emb_dim", num_nodes=tokens.size(0))
        return pos_embed