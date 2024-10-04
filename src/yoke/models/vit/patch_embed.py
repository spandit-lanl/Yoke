"""nn.Module classes defined here offer different methods to embed a
multi-channel image as a series of tokenized patches.

"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from einops import rearrange
from einops.layers.torch import Rearrange


def _get_conv2d_weights(in_channels,
                        out_channels,
                        kernel_size):

    weight = torch.empty(out_channels, in_channels, *kernel_size)

    return weight


def _get_conv2d_biases(out_channels):

    bias = torch.empty(out_channels)

    return bias


class ClimaX_ParallelVarPatchEmbed(nn.Module):
    """Variable to Patch Embedding with multiple variables in a single
    kernel. Key idea is to use Grouped Convolutions. This allows this layer to
    embed an arbitrary subset of a default list of variables.

    Based on the paper, **ClimaX: A foundation model for weather and
    climate.**

    NOTE: The img_size entries should be divisible by the corresponding
    patch_size entries.
    
    Args:
        max_vars (int): Maximum number of variables
        img_size ((int, int)): Image size
        patch_size ((int, int)): Patch size
        embed_dim (int): Embedding dimension
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.

    """

    def __init__(self,
                 max_vars: int = 5,
                 img_size: (int, int) = (128, 128),
                 patch_size: (int, int) = (16, 16),
                 embed_dim: int = 64,
                 norm_layer=None):

        super().__init__()
        # Check size compatibilities
        try:
            msg = 'Image height not divisible by patch height!!!'
            assert img_size[0] % patch_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = ('Image height:',
                         img_size[0],
                         'Patch height:',
                         patch_size[0])
            e.args += msg_tuple
            raise

        try:
            msg = 'Image width not divisible by patch widht!!!'
            assert img_size[1] % patch_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = ('Image width:',
                         img_size[1],
                         'Patch width:',
                         patch_size[1])
            e.args += msg_tuple
            raise

        self.max_vars = max_vars
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // self.patch_size[0],
                          img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.embed_dim = embed_dim

        # Conv patch embedding weights and biases
        grouped_weights = torch.stack(
            [_get_conv2d_weights(1, self.embed_dim, self.patch_size) for _ in range(max_vars)],
            dim=0)

        self.proj_weights = nn.Parameter(grouped_weights)

        grouped_biases = torch.stack(
            [_get_conv2d_biases(self.embed_dim) for _ in range(max_vars)],
            dim=0)
        self.proj_biases = nn.Parameter(grouped_biases)

        # Layer normalization
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

        # Initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_vars):
            nn.init.kaiming_uniform_(self.proj_weights[idx], a=math.sqrt(5))
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.proj_weights[idx])

            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.proj_biases[idx], -bound, bound)

    def forward(self, x, vars=None):
        B, C, H, W = x.shape

        if vars is None:
            vars = range(self.max_vars)

        weights = self.proj_weights[vars].flatten(0, 1)
        biases = self.proj_biases[vars].flatten(0, 1)

        # Each variable's image is embedded separately. Each variable's image
        # gets mapped to an a set of patches of embeddings. The variable's
        # embeddings are concatenated.
        #
        # For embedding dimension E and patch size p=(p1, p2),
        # (B, V, H, W) -> (B, VxE, H', W') with H'=H/p1, W'=W/p2
        groups = len(vars)
        proj = F.conv2d(x, weights, biases, groups=groups, stride=self.patch_size)

        # Flatten the patch arrays and separate the variables and embeddings.
        proj = rearrange(proj,
                         'b (v e) h1 h2 -> b v (h1 h2) e',
                         v=groups,
                         e=self.embed_dim)

        # Normalize the layer output
        proj = self.norm(proj)

        return proj


class SwinEmbedding(nn.Module):
    """This SWIN embedding layer takes a *channels-first* image, breaks it into
    linear patch embeddings, then rearranges those embedded patches into sets
    of tokens.

    For an input tensor of size (B, C, H, W) the linear embedding, with
    embedding dimension E and patch-size P=(ph, pw), returns a (B, E, H', W')
    filtered image. Here, H' and W' are the resulting heights and widths of the
    (H, W) after going through a convolutional kernel of size (ph, pw) and
    stride (ph, pw). The embedded image is then rearranged to a batch of tokens
    each of size E. Resulting in a (B, H'*W', E) tensor.

    NOTE: The img_size entries should be divisible by the corresponding
    patch_size entries.
    
    Args:
        max_vars (int): Maximum number of variables
        img_size ((int, int)): Image size
        patch_size ((int, int)): Patch size
        embed_dim (int): Embedding dimension
        norm_layer (nn.Module, optional): Normalization layer. Defaults to None.

    """

    def __init__(self,
                 num_vars: int = 5,
                 img_size: (int, int) = (128, 128),
                 patch_size: (int, int) = (16, 16),
                 embed_dim: int = 64,
                 norm_layer=None):
        super().__init__()
        # Check size compatibilities
        try:
            msg = 'Image height not divisible by patch height!!!'
            assert img_size[0] % patch_size[0] == 0, msg
        except AssertionError as e:
            msg_tuple = ('Image height:',
                         img_size[0],
                         'Patch height:',
                         patch_size[0])
            e.args += msg_tuple
            raise

        try:
            msg = 'Image width not divisible by patch widht!!!'
            assert img_size[1] % patch_size[1] == 0, msg
        except AssertionError as e:
            msg_tuple = ('Image width:',
                         img_size[1],
                         'Patch width:',
                         patch_size[1])
            e.args += msg_tuple
            raise

        self.num_vars = num_vars
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = (self.img_size[0] // self.patch_size[0],
                          self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.linear_embedding = nn.Conv2d(self.num_vars,
                                          self.embed_dim,
                                          kernel_size=self.patch_size,
                                          stride=self.patch_size)
        self.rearrange = Rearrange('b c h w -> b (h w) c')

        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.rearrange(x)

        x = self.norm(x)

        return x


if __name__ == '__main__':
    """Usage Example.

    """

    # (B, C, H, W) = (3, 5, 128, 128)
    x = torch.rand(3, 4, 128, 128)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.type(torch.FloatTensor).to(device)

    PPembed_model = ClimaX_ParallelVarPatchEmbed(max_vars=5,
                                                 img_size=(128, 128),
                                                 patch_size=(16, 16),
                                                 embed_dim=72,
                                                 norm_layer=nn.LayerNorm).to(device)

    print('Input shape:', x.shape)
    print('Parallel-patch embed shape:', PPembed_model(x, vars=[0, 1, 3, 4]).shape)

    swin_embed_model = SwinEmbedding(num_vars=4,
                                     img_size=(128, 128),
                                     patch_size=(16, 16),
                                     embed_dim=72,
                                     norm_layer=nn.LayerNorm).to(device)

    print('Input shape:', x.shape)
    print('SWIN embed shape:', swin_embed_model(x).shape)
