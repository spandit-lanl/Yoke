"""Variable Aggregation module.

nn.Modules to carry out *variable aggregation* after patch embedding. Variable
aggregation eliminates the variable dimension.

"""

import torch
import torch.nn as nn

from einops import rearrange


class ClimaX_AggVars(nn.Module):
    """ClimaX variable aggregation.

    Variable aggregation performed through a learnable Query vector and a
    re-mapping of the tensor dimensions.

    The goal is to get some quantity representative of cross-attention across
    all the variables. This eliminates the variable dimension which is
    important if the number of variables is large or possibly not fixed.

    Based on the paper, **ClimaX: A foundation model for weather and climate.**

    Args:
        embed_dim (int): Initial embedding dimension.
        num_heads (int): Number of heads in the MSA layers.

    """

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        """Initialization for ClimaX variable aggregation."""
        super().__init__()

        self.var_query = nn.Parameter(torch.zeros(1, 1, embed_dim), requires_grad=True)
        self.var_agg = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method for ClimaX variable aggregation."""
        # The input tensor is shape (B, NumVars, NumTokens, embed_dim)
        B, V, L, D = x.shape

        x = rearrange(x, "b v l d -> (b l) v d")  # BxL, V, D

        Q = self.var_query.repeat_interleave(x.shape[0], dim=0)  # BxL, 1, D

        # Attention is softmax(Q*K^T)*V. Q*K^T is shape (BxL, 1, V).
        # So Attention is shape (BxL, 1, D).
        #
        # Literally using Q=var_query, K=x, V=x ?!
        x, _ = self.var_agg(Q, x, x)

        x = x.squeeze()

        x = rearrange(x, "(b l) d -> b l d", b=B, l=L)

        return x


if __name__ == "__main__":
    """Usage Example.

    """

    # (B, V, token_number, E) = (3, 15, 128, 32)
    x = torch.rand(3, 15, 128, 32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = x.type(torch.FloatTensor).to(device)

    model = ClimaX_AggVars(embed_dim=32, num_heads=4).to(device)
    print("Input shape:", x.shape)
    print("Aggregate shape:", model(x).shape)
