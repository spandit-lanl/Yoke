"""The main encoder structure used in a SWIN transformer.

"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

import sys, os
sys.path.insert(0, os.getenv('YOKE_DIR'))
from models.vit.swin.windowed_msa import WindowMSA, ShiftedWindowMSA


class MLP(nn.Module):
    """A standard multi-layer perceptron structure using a GELU activtion, one
    hidden layer, and expanding the embedding size by 4x before contracting
    again.

    Args:
        emb_size (int): Embedding layer dimension from input layer.
    
    """
    
    def __init__(self, emb_size: int=64):
        super().__init__()
        self.ff = nn.Sequential(nn.Linear(emb_size, 4*emb_size),
                                nn.GELU(),
                                nn.Linear(4*emb_size, emb_size))
    
    def forward(self, x):
        return self.ff(x)


class SwinEncoder(nn.Module):
    """The main SWIN encoder alternates between a windowed-MSA and
    shifted-windowed-MSA block. MLP layers are used between and layer
    normalization is used prior to each layer. Residual connections are also
    included at each layer.

    Embedding size is the input dimension of the tokens. The embedding size
    must be evenly divisible by the number of heads. Moreover, the number of
    tokens, L, must satisfy L=patch_grid_size[0]*patch_grid_size[1]. The
    respective `window_size` dimensions must divide the `patch_grid_size`
    dimensions evenly.

    Args:
        emb_size (int): Incoming embedding dimension.
        num_heads (int): Number of heads to use in the MSA.
        patch_grid_size (int, int): Grid dimensions making up the token list.
        window_size (int, int): Dimensions of window to use on the patch grid.

    """
    
    def __init__(self,
                 emb_size: int=64,
                 num_heads: int=10,
                 patch_grid_size: (int, int)=(16, 32),
                 window_size: (int, int)=(8, 4)):
        super().__init__()
        # Check size compatibilities
        assert emb_size % num_heads == 0
        assert patch_grid_size[0] % window_size[0] == 0
        assert patch_grid_size[1] % window_size[1] == 0
        
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.patch_grid_size = patch_grid_size
        self.window_size = window_size

        self.WMSA = WindowMSA(emb_size=self.emb_size,
                              num_heads=self.num_heads,
                              patch_grid_size=self.patch_grid_size,
                              window_size=self.window_size)
        self.SWMSA = ShiftedWindowMSA(emb_size=self.emb_size,
                                      num_heads=self.num_heads,
                                      patch_grid_size=self.patch_grid_size,
                                      window_size=self.window_size)

        self.ln = nn.LayerNorm(self.emb_size)
        self.MLP = MLP(self.emb_size)
        
    def forward(self, x):
        # Window Attention
        x = x + self.WMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        # Shifted Window Attention
        x = x + self.SWMSA(self.ln(x))
        x = x + self.MLP(self.ln(x))
        
        return x


if __name__ == '__main__':
    """Usage Example.

    """

    # Assume original image is (1120, 800) and embedded with
    # patch-size (20, 20).
    #
    # (B, token_number, E) = (3, 1024, 64)
    x = torch.rand(3, 56*40, 64)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.type(torch.FloatTensor).to(device)

    num_heads = 8
    emb_size = 64
    window_size = (8, 10)  # Due to shift each dimension must be divisible by 2.
    patch_grid_size = (56, 40)
    model_swin_encoder = SwinEncoder(emb_size=emb_size,
                                     num_heads=num_heads,
                                     patch_grid_size=patch_grid_size,
                                     window_size=window_size).to(device)
        
    print('Input shape:', x.shape)
    print('SWIN encoder shape:', model_swin_encoder(x).shape)


