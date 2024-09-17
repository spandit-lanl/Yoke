"""Patch manipulation layers. Includes *merging*, *expanding*, *depatching*
layer definitions.

The goal of these layers is to reduce or increase the number of tokens of a
patch embedding while simultaneously modifying the embedding
dimension. Hypothetically this should allow a SWIN transformer to learn a
heirarchical feature representation for the image.

"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np
    

class PatchMerge(nn.Module):
    """Merge patches and increase embedding dimension.

    This layer is passed a (B, L, C) tensor, batches of L tokens, each of
    *emb_size* C. The *emb_size* must match the token embedding dimension from
    the previous layer. It is assumed L=H*W with H divisible by s1 and W
    divisible by s2.

    The tokens are first separated into (H', s1, W', s2) groups with H'=H/s1
    and W'=W/s2. The input is then remapped:

    *B (H' s1 W' s2) C -> B (H' W') (s1 s2 C)*

    The new token sets are then linear embedded to yield a new tensor of size

    (B, H'*W', emb_factor*C).

    Args:
        emb_size (int): Incoming embedding dimension
        emb_factor (int): Up-scaling factor for embedding dimension
        patch_grid_size (int, int): Incoming patch-grid dimensions within
                                    the token set
        s1 (int): Height reduction factor for the patch grid
        s2 (int): Width reduction factor for the patch grid

    """
    
    def __init__(self,
                 emb_size: int=64,
                 emb_factor: int=2,
                 patch_grid_size: (int, int)=(64, 64),
                 s1: int=2,
                 s2: int=2):
        super().__init__()
        # Check size compatibilities
        try:
            msg = 'Patch-grid height not divisible by height of patch-merge scale!!!'
            assert patch_grid_size[0] % s1 == 0, msg
        except AssertionError as e:
            msg_tuple = ('Patch-grid height:',
                         patch_grid_size[0],
                         'Patch-merge scale height:',
                         s1)
            e.args += msg_tuple
            raise

        try:
            msg = 'Patch-grid width not divisible by width of patch-merge scale!!!'
            assert patch_grid_size[1] % s2 == 0, msg
        except AssertionError as e:
            msg_tuple = ('Patch-grid width:',
                         patch_grid_size[1],
                         'Patch-merge scale width:',
                         s2)
            e.args += msg_tuple
            raise
        
        self.in_emb_size = emb_size
        
        # Patch grid parameters
        self.H = patch_grid_size[0]
        self.W = patch_grid_size[1]
        
        # Patch division factors
        self.s1 = s1
        self.s2 = s2

        # New patch grid
        self.out_patch_grid_size = (int(self.H/self.s1),
                                    int(self.W/self.s2))
        
        # Embedding dimension factor
        self.emb_factor = emb_factor
        self.out_emb_size = self.emb_factor*self.in_emb_size
        
        # Linear re-embedding
        self.linear = nn.Linear(self.s1*self.s2*self.in_emb_size,
                                self.out_emb_size)        
        
    def forward(self, x):
        # The input tensor is shape (B, num_tokens, embedding_dim)
        B, L, C = x.shape

        # NOTE: The number of tokens is assumed to be L=H*W
        assert L == self.H*self.W

        x = rearrange(x,
                      'b (h s1 w s2) c -> b (h w) (s1 s2 c)',
                      s1=self.s1,
                      s2=self.s2,
                      h=self.out_patch_grid_size[0],
                      w=self.out_patch_grid_size[1])
        x = self.linear(x)

        return x


class PatchExpand(nn.Module):
    """Expand patches and decrease embedding dimension.
    
    This layer is passed a (B, L, C) tensor, batches of L tokens, each of
    *emb_size* C. The *emb_size* must match the token embedding dimension from
    the previous layer. It is assumed L=H*W with (H, W) being defined by the
    previous layer's patch grid.

    Expansion follows:
    
        (B, H*W, C) ->[linear] (B, H*W, n*C)
            ->[rearrange] (B, H*s1*W*s2,  n*C/(s1*s2))
    
    The embedding dimension is first increased through a linear embedding by a
    factor `n`. The new embedding dimension is then divided by `s1*s2` and the
    number of patches is increased accordingly through *rearrange.

    NOTE: `n*C` must be divisible by `s1*s2`

    Args:
        emb_size (int): Incoming embedding dimension
        emb_factor (int): Up-scaling factor for embedding dimension
        patch_grid_size (int, int): Incoming patch-grid dimensions within
                                    the token set
        s1 (int): Height scaling factor for the patch grid
        s2 (int): Width scaling factor for the patch grid


    """
    
    def __init__(self,
                 emb_size: int=64,
                 emb_factor: int=2,
                 patch_grid_size: (int, int)=(64, 64),
                 s1: int=2,
                 s2: int=2):
        super().__init__()

        # Check size compatibilities
        try:
            msg = 'New embedding dimension not divisible by patch-expansion factors!!!'
            assert (emb_size*emb_factor) % (s1*s2) == 0, msg
        except AssertionError as e:
            msg_tuple = ('Input embedding size:', emb_size,
                         'Embedding factor:', emb_factor,
                         'Height expansion factor:', s1,
                         'Width expansion factor:', s2)
            e.args += msg_tuple
            raise

        # Input embedding size
        self.emb_size = emb_size
        
        # Patch grid parameters
        self.H = patch_grid_size[0]
        self.W = patch_grid_size[1]
        
        # Patch division factors
        self.s1 = s1
        self.s2 = s2

        # Embedding dimension factor
        self.emb_factor = emb_factor

        # New patch grid
        self.out_patch_grid_size = (int(self.H*self.s1),
                                    int(self.W*self.s2))

        # Add output embedding size for model building
        self.out_emb_size = int(self.emb_factor*self.emb_size/(self.s1*self.s2))

        # Linear re-embedding
        self.linear = nn.Linear(self.emb_size, self.emb_factor*self.emb_size)
        
    def forward(self, x):
        # The input tensor is shape (B, num_tokens, embedding_dim)
        B, L, C = x.shape

        # NOTE: The number of tokens is assumed to be L=H*W
        assert L == self.H*self.W

        # Linear embedding
        x = self.linear(x)
        
        # Rearrange
        x = rearrange(x,
                      'b (h w) (k s1 s2) -> b (h s1 w s2) k',
                      s1=self.s1,
                      s2=self.s2,
                      h=self.H,
                      w=self.W)

        return x

    
class Unpatchify(nn.Module):
    """Expansion from patches to variables and images used in the ClimaX model.

    (B, H*W,  V*p_h*p_w) ->[rearrange] (B, V, H*p_h, W*p_w)

    Based on the paper, **ClimaX: A foundation model for weather and
    climate.**

    Args:
        total_num_vars (int): Total number of variables to be output.
        patch_grid_size (int, int): Height and width grid size of patches
                                    making up the tokens.
        patch_size (int, int): Height and width of each patch.
    
    """
    
    def __init__(self,
                 total_num_vars: int=5,
                 patch_grid_size: (int, int)=(64, 64),
                 patch_size: (int, int)=(8, 8)):

        super().__init__()

        # Total number of variables
        self.V = total_num_vars
        
        # Patch grid parameters
        self.H = patch_grid_size[0]
        self.W = patch_grid_size[1]
        
        # Individual patch height and width
        self.p_h = patch_size[0]
        self.p_w = patch_size[1]
        
    def forward(self, x):
        # The input tensor is shape (B, num_tokens, embedding_dim)
        B, L, C = x.shape

        # Make sure shape requirements are met
        assert L == self.H*self.W

        assert C == self.V*self.p_h*self.p_w

        x = rearrange(x,
                      'b (h w) (v ph pw) -> b v (h ph) (w pw)',
                      h=self.H,
                      w=self.W,
                      v=self.V,
                      ph=self.p_h,
                      pw=self.p_w)
        
        return x


if __name__ == '__main__':
    """Usage Example.

    """

    # Original input before embedding: (B, V, H, W)
    img_size = (512, 128)
    num_vars = 5
    batch_size = 3
    emb_dim = 64
    patch_size = (16, 8)
    patch_grid_size = (int(img_size[0]/patch_size[0]),
                       int(img_size[1]/patch_size[1]))
    num_tokens = patch_grid_size[0]*patch_grid_size[1]

    # Input
    x = torch.rand(batch_size, num_tokens, emb_dim)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.type(torch.FloatTensor).to(device)

    s1 = 4
    s2 = 4
    emb_factor = 2
    merge_model = PatchMerge(emb_dim,
                             emb_factor=emb_factor,
                             patch_grid_size=patch_grid_size,
                             s1=s1,
                             s2=s2).to(device)

    print('Input shape:', x.shape)
    x = merge_model(x)
    print('Patch merge shape:', x.shape)

    # Grid size has been reduced through merge
    merged_patch_grid_size = (int(patch_grid_size[0]/s1),
                              int(patch_grid_size[1]/s2))
    expand_model = PatchExpand(x.shape[2],
                               emb_factor=int(s1*s2/emb_factor),
                               patch_grid_size=merged_patch_grid_size,
                               s1=s1,
                               s2=s2).to(device)
    x = expand_model(x)
    print('Patch expand shape:', x.shape)

    # Linear embed the last dimension into V*p_h*p_w
    linear = nn.Linear(emb_dim,
                       num_vars*patch_size[0]*patch_size[1]).to(device)
    x = linear(x)
    print('Embed to variable dimension shape:', x.shape)

    # Unpatch the variables and tokens
    unpatch_model = Unpatchify(total_num_vars=num_vars,
                               patch_grid_size=patch_grid_size,
                               patch_size=patch_size).to(device)

    x = unpatch_model(x)
    print('Unpatched image and variables shape:', x.shape)
