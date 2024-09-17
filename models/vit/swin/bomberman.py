"""nn.Module allowing processing of variable channel image input through a
SWIN-V2 U-Net architecture then re-embedded as a variable channel image.

This network architecture will serve as the foundation for a hydro-code
emulator.

"""

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import numpy as np

import sys, os
sys.path.insert(0, os.getenv('YOKE_DIR'))
from models.vit.swin.encoder import SwinEncoder2
from models.vit.swin.unet import SwinUnetBackbone
from models.vit.patch_embed import ClimaX_ParallelVarPatchEmbed
from models.vit.patch_manipulation import PatchMerge, PatchExpand

from models.vit.aggregate_variables import ClimaX_AggVars
from models.vit.embedding_encoders import ClimaX_VarEmbed, ClimaX_PosEmbed, ClimaX_TimeEmbed


class LodeRunner(nn.Module):
    """Parallel-patch embedding with SWIN U-Net backbone and
    unpatchification. This module will take in a variable-channel image format
    and output an equivalent variable-channel image formate. This will serves
    as a prototype foundational architecture for multi-material, multi-physics,
    surrogate models.
    
    Args:
        emb_size (int): Initial embedding dimension.
        emb_factor (int): Scale of embedding in each patch merge/exand.
        patch_grid_size (int, int): Initial patch-grid height and width for input.
        block_structure (int, int, int, int): Tuple specifying the number of SWIN
                                              encoders in each block structure
                                              separated by the patch-merge layers.
        num_heads (int): Number of heads in the MSA layers.
        window_sizes (list(4*(int, int))): Window sizes within each SWIN encoder/decoder.
        patch_merge_scales (list(3*(int, int))): Height and width scales used in
                                                 each patch-merge layer.
        verbose (bool): When TRUE, windowing and merging dimensions are printed
                        during initialization.

    """
    
    def __init__(self,
                 default_vars,
                 image_size: (int, int)=(1120, 800),
                 patch_size: (int, int)=(10, 10),
                 embed_dim: int=128,
                 emb_factor: int=2,
                 num_heads: int=8,
                 block_structure: (int, int, int, int)=(1, 1, 3, 1),
                 window_sizes: [(int, int), (int, int), (int, int), (int, int)]=[(8, 8), (8, 8), (4, 4), (2, 2)],
                 patch_merge_scales: [(int, int), (int, int), (int, int)]=[(2, 2), (2, 2), (2, 2)],
                 verbose: bool=False):
        super().__init__()

        self.default_vars = default_vars
        self.max_vars = len(self.default_vars)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.emb_factor =  emb_factor
        self.num_heads = num_heads
        self.block_structure = block_structure
        self.window_sizes = window_sizes
        self.patch_merge_scales = patch_merge_scales

        
        self.parallel_embed = ClimaX_ParallelVarPatchEmbed(max_vars=self.max_vars,
                                                           img_size=self.image_size,
                                                           patch_size=self.patch_size,
                                                           embed_dim=self.embed_dim,
                                                           norm_layer=None)

        self.var_embed_layer = ClimaX_VarEmbed(self.default_vars, self.embed_dim)
    
        self.agg_vars = ClimaX_AggVars(self.embed_dim, self.num_heads)

        self.pos_embed = ClimaX_PosEmbed(self.embed_dim,
                                         self.patch_size,
                                         self.image_size,
                                         self.parallel_embed.num_patches)

        self.temporal_encoding = ClimaX_TimeEmbed(self.embed_dim)
    
        self.unet = SwinUnetBackbone(emb_size=self.embed_dim,
                                     emb_factor=self.emb_factor,
                                     patch_grid_size=self.parallel_embed.grid_size,
                                     block_structure=self.block_structure,
                                     num_heads=self.num_heads,
                                     window_sizes=self.window_sizes,
                                     patch_merge_scales=self.patch_merge_scales,
                                     verbose=verbose)

    def forward(self, x, vars, lead_times: torch.Tensor, device):
        print('Input shape:', x.shape)
        
        # First embed input
        varIDXs = self.var_embed_layer.get_var_ids(tuple(vars), device)
        x = self.parallel_embed(x, varIDXs)
        print('Shape after parallel patch-embed:', x.shape)
        
        # Encode variables
        x = self.var_embed_layer(x, vars)
        print('Shape after variable encoding:', x.shape)

        # Aggregate variables
        x = self.agg_vars(x)
        print('Shape after variable aggregation:', x.shape)

        # Encode patch positions, spatial information
        x = self.pos_embed(x)
        print('Shape after position encoding:', x.shape)

        # Encode temporal information
        x = self.temporal_encoding(x, lead_times)
        print('Shape after temporal encoding:', x.shape)

        # Pass through SWIN-V2 U-Net encoder
        x = self.unet(x)
        print('Shape after U-Net:', x.shape)
        
        return x


if __name__ == '__main__':
    import sys, os
    sys.path.insert(0, os.getenv('YOKE_DIR'))
    from torch_training_utils import count_torch_params

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    default_vars = ['cu_pressure',
                    'cu_density',
                    'cu_temperature',
                    'al_pressure',
                    'al_density',
                    'al_temperature',
                    'ss_pressure',
                    'ss_density',
                    'ss_temperature',
                    'ply_pressure',
                    'ply_density',
                    'ply_temperature',
                    'air_pressure',
                    'air_density',
                    'air_temperature',
                    'hmx_pressure',
                    'hmx_density',
                    'hmx_temperature',
                    'r_vel',
                    'z_vel']

    # (B, C, H, W)
    x = torch.rand(5, 4, 1120, 800)
    x = x.type(torch.FloatTensor).to(device)
    
    lead_times = torch.rand(5)  # Lead time for each entry in batch
    x_vars=['cu_density',
            'ss_density',
            'ply_density',
            'air_density']
    x_varIDX = [1, 7, 10, 13]
    embed_dim = 128
    emb_factor = 2
    patch_size = (10, 10)
    image_size = (1120, 800)
    block_structure = (1, 1, 3, 1)
    num_heads = 8
    window_sizes = [(8, 8), (8, 8), (4, 4), (2, 2)]
    patch_merge_scales = [(2, 2), (2, 2), (2, 2)]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = x.type(torch.FloatTensor).to(device)

    # Test LodeRunner architecture.
    lode_runner = LodeRunner(default_vars=default_vars,
                             image_size=image_size,
                             patch_size=patch_size,
                             embed_dim=embed_dim,
                             emb_factor=emb_factor,
                             num_heads=num_heads,
                             block_structure=block_structure,
                             window_sizes=window_sizes,
                             patch_merge_scales=patch_merge_scales,
                             verbose=False).to(device)
    print('Lode Runner output shape:', lode_runner(x, x_vars, lead_times, device).shape)
    print('Lode Runner parameters:', count_torch_params(lode_runner, trainable=True))
