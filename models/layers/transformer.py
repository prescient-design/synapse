import torch
import torch.nn as nn
import einops
import math
from jaxtyping import Float, Int
import einx
from torch import Tensor
from models.layers.pos_enc import RotaryEmbedding, PosEmbed
from models.layers.norms import LayerNorm
from models.layers.linear import Linear



class Attention(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        n_heads = 4
        dim_head = in_dim // 4
        self.dim_head = dim_head
        self.W_Q = nn.Parameter(torch.empty((n_heads, in_dim, dim_head)))
        nn.init.normal_(self.W_Q, std=0.02)
        self.b_Q = nn.Parameter(torch.zeros((n_heads, dim_head)))
        self.W_K = nn.Parameter(torch.empty((n_heads, in_dim, dim_head)))
        nn.init.normal_(self.W_K, std=0.02)
        self.b_K = nn.Parameter(torch.zeros((n_heads, dim_head)))
        self.W_V = nn.Parameter(torch.empty((n_heads, in_dim, dim_head)))
        nn.init.normal_(self.W_V, std=0.02)
        self.b_V = nn.Parameter(torch.zeros((n_heads, dim_head)))
        self.W_O = nn.Parameter(torch.empty((n_heads, dim_head, in_dim)))
        nn.init.normal_(self.W_O, std=0.02)
        self.b_O = nn.Parameter(torch.zeros((in_dim)))
    
    def forward(self, normalized_resid_pre: Float[Tensor, " num_nodes query_pos d_model"]) -> Float[Tensor, " num_nodes query_pos d_model"]:
 
        
        q = einops.einsum(normalized_resid_pre, self.W_Q, "num_nodes query_pos d_model, n_heads d_model d_head -> num_nodes query_pos n_heads d_head") + self.b_Q
        k = einops.einsum(normalized_resid_pre, self.W_K, "num_nodes key_pos d_model, n_heads d_model d_head -> num_nodes key_pos n_heads d_head") + self.b_K
        
        attn_scores = einops.einsum(q, k, "num_nodes query_pos n_heads d_head, num_nodes key_pos n_heads d_head -> num_nodes n_heads query_pos key_pos")
        attn_scores = attn_scores / math.sqrt(self.dim_head)

        pattern = attn_scores.softmax(dim=-1) # [batch, n_head, query_pos, key_pos]

        v = einops.einsum(normalized_resid_pre, self.W_V, "num_nodes key_pos d_model, n_heads d_model d_head -> num_nodes key_pos n_heads d_head") + self.b_V

        z = einops.einsum(pattern, v, "num_nodes n_heads query_pos key_pos, num_nodes key_pos n_heads d_head -> num_nodes query_pos n_heads d_head")

        attn_out = einops.einsum(z, self.W_O, "num_nodes query_pos n_heads d_head, n_heads d_head d_model -> num_nodes query_pos d_model") + self.b_O
        return attn_out

class TransformerBlock(nn.Module):
    """A single Transformer layer.

    Args:
        d_model: int    
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        d_ff: int
            Dimensionality of the feed-forward inner layer (section 3.3).
        positional_encoder: RotaryEmbedding
            The RoPE module to use.

    Returns:
        FloatTensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        positional_encoder: RotaryEmbedding,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            positional_encoder=positional_encoder,
        )
        self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff)
        self.ln1 = nn.RMSNorm(d_model)
        self.ln2 = nn.RMSNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: FloatTensor of shape `(batch_size, sequence_length, d_model)`.
                The input to process with the Transformer block.

        Returns:
            FloatTensor of shape `(batch_size, sequence_length, d_model)`.
        """

        x_attn = self.attn(self.ln1(x))
        attn_sublayer_output = x + x_attn

        x_ffn = self.ffn(self.ln2(attn_sublayer_output))
        ffn_sublayer_output = attn_sublayer_output + x_ffn
        return ffn_sublayer_output

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x):
        return self.w2(nn.functional.silu(self.w1(x)) * self.w3(x))

class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention

    Args:
        d_model: int
            The dimensionality of the model embeddings and sublayer outputs.
        num_heads: int
            Number of heads to use in multi-headed attention. `d_model` must be
            evenly divisible by `num_heads`.
        positional_encoder: RotaryEmbedding
            The RoPE module to use.

    Returns:
        Tensor of shape `(batch_size, sequence_length, d_model)`.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        positional_encoder: RotaryEmbedding,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model // num_heads
        self.d_v = self.d_k

        self.q_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.k_proj = Linear(self.d_model, self.num_heads * self.d_k)
        self.v_proj = Linear(self.d_model, self.num_heads * self.d_v)

        self.output_proj = Linear(self.num_heads * self.d_v, self.d_model)

        self.positional_encoder = positional_encoder  # RoPE

    def forward(
        self, x: Float[Tensor, " ... seq d_k"], token_positions: Int[Tensor, " ... seq"] | None = None
    ) -> Float[Tensor, " ... seq d_v"]:
        """
        Args:
            x: The input to perform multi-headed self-attention on.
            positional_ids: The positional indices along the sequence dimension of the input embeddings.

        Returns:
            Self-attention outputs.
        """
        *b, sequence_length, d_model = x.size()
        assert d_model == self.d_model

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)

        # Take apart each head from the embedding dimension of Q, K, V to shape (..., num_heads, seq_len, d_k).
        Q, K, V = (
            einops.rearrange(X, "... seq (heads d) -> ... heads seq d", heads=self.num_heads)
            for X in (Q, K, V)
        )  # fmt: skip

        if token_positions is None:
            token_positions = einx.rearrange(
                "seq -> b... seq", torch.arange(sequence_length, device=x.device), b=[1] * len(b)
            )

        Q = self.positional_encoder(Q, token_positions)
        K = self.positional_encoder(K, token_positions)


        # Shape: (..., num_heads, sequence_length, d_k)
        attn_output = nn.functional.scaled_dot_product_attention(
            query=Q,
            key=K,
            value=V,
            is_causal=False,
            enable_gqa=False
        )

        # Concatenate the attention output from all heads.
        # (..., sequence_length, num_heads * d_v).
        attn_output = einops.rearrange(attn_output, "batch heads seq d_v -> batch seq (heads d_v)").contiguous()

        # Apply the output projection
        output = self.output_proj(attn_output)
        return output
