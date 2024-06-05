"""Module containing torch nn.Module classes forming CNNs mapping vectors of
scalar inputs to images.

"""

import sys
import os
import math
from typing import List
import torch
import torch.nn as nn

sys.path.insert(0, os.getenv('YOKE_DIR'))
from models.cnn_utils import count_parameters


class jekelCNNsurrogate(nn.Module):
    def __init__(self, 
                 input_size: int=29,
                 linear_features: tuple[int, int]=(4, 4),
                 kernel: tuple[int, int]=(3, 3),
                 nfeature_list: List[int]=[512, 512, 512, 512, 256, 128, 64, 32],
                 output_image_size: tuple[int, int]=(1120, 800),
                 act_layer=nn.GELU):
        """Convolutional Neural Network Module that creates a scalar-to-image 
        surrogate using a sequence of ConvTranspose2D, Batch Normalization, and
        Activation layers.

        This architecture is meant to reproduce the architecture described in
        Jekel et. al. 2022 *Using conservation laws to infer deep learning
        model accuracy of Richtmyer-Meshkov instabilities.*

        Args:
            input_size (int): Size of input
            linear_features (tuple[int, int]): Window size scalar parameters are 
                                               originally mapped into
            kernel (tuple[int, int]): Size of transpose-convolutional kernel
            nfeature_list (List[int]): List of number of features in each 
                                       T-convolutional layer
            output_image_size (tuple[int, int]): Image size to output, (H, W). 
                                                 Channels are automatically inherited.
            act_layer(nn.modules.activation): torch neural network layer class 
                                              to use as activation

        """

        super().__init__()

        self.input_size = input_size
        self.output_image_size = output_image_size
        self.linear_features = linear_features
        self.nfeature_list = nfeature_list
        self.kernel = kernel
        self.nConvT = len(self.nfeature_list)

        # First linear remap
        out_features = self.linear_features[0]*self.linear_features[1]*self.nfeature_list[0]
        self.dense_expand = nn.Linear(in_features=self.input_size,
                                      out_features=out_features,
                                      bias=False)

        normLayer = nn.BatchNorm2d(self.nfeature_list[0])
        nn.init.constant_(normLayer.weight, 1)
        normLayer.weight.requires_grad = False
            
        self.inNorm = normLayer
        self.inActivation = act_layer()

        # Module list to hold transpose convolutions
        self.TConvList = nn.ModuleList()
        self.BnormList = nn.ModuleList()
        self.ActList = nn.ModuleList()

        # Create transpose convolutional layer for each entry in feature list.
        for i in range(self.nConvT-1):
            tconv = nn.ConvTranspose2d(in_channels=self.nfeature_list[i],
                                       out_channels=self.nfeature_list[i+1], 
                                       kernel_size=self.kernel, 
                                       stride=2, 
                                       padding=1,
                                       output_padding=1,
                                       bias=False)
            
            self.TConvList.append(tconv)

            normLayer = nn.BatchNorm2d(self.nfeature_list[i+1])
            nn.init.constant_(normLayer.weight, 1)
            normLayer.weight.requires_grad = False

            self.BnormList.append(normLayer)
            self.ActList.append(act_layer())

        # Final Transpose Conv layer followed by hyperbolic tanh activation
        self.final_tconv = nn.ConvTranspose2d(in_channels=self.nfeature_list[-1],
                                              out_channels=1, 
                                              kernel_size=self.kernel, 
                                              stride=2, 
                                              padding=1,
                                              output_padding=1,
                                              bias=True)

        # If normalizing to [-1, 1]
        #self.final_act = nn.Tanh()

        # Else...
        self.final_act = nn.Identity()
        
        # NOTE: Upsample layer is alternative to resizing image using
        # nn.functional.interpolate. However, pytorch claims the interpolate
        # method is better for general resizing.
        #
        # Upsample layer to get image size correct
        #self.upsample = nn.Upsample(size=self.output_image_size,
        #                            mode='bilinear')

    def forward(self, x):
        ## Input Layers
        x = self.dense_expand(x)
        # Reshape to a 2D block with channels
        # NOTE: -1 infers batch size
        x = x.view(-1,
                   self.nfeature_list[0],
                   self.linear_features[0],
                   self.linear_features[1])

        x = self.inNorm(x)
        x = self.inActivation(x)
        #print('After dense-map shape:', x.shape)
        
        ## ConvT layers
        for i in range(self.nConvT-1):
            x = self.TConvList[i](x)
            x = self.BnormList[i](x)
            x = self.ActList[i](x)
            #print(f'After convT{i:d} shape:', x.shape)

        # Final ConvT
        x = self.final_tconv(x)
        x = self.final_act(x)
        #print('After final convT shape:', x.shape)

        # Reshape to output image size.
        #print('Pre-Upsample shape:', x.shape)
        #x = self.upsample(x)

        # Alternate resize
        x = nn.functional.interpolate(x,
                                      size=self.output_image_size,
                                      mode='bilinear',
                                      antialias=True)
        #print('Post-Upsample shape:', x.shape)
        
        return x


if __name__ == '__main__':
    """For testing and debugging.

    """

    # Excercise model setup
    # NOTE: Model takes (BatchSize, ScalarDims) tensor.
    scalar_input = torch.rand(4, 29)
    jCNN = jekelCNNsurrogate(input_size=29,
                             linear_features=(4, 4),
                             kernel=(3, 3),
                             nfeature_list=[512, 512, 512, 512, 256, 128, 64, 32],
                             output_image_size=(1120, 800),
                             act_layer=nn.GELU)

    jCNN.eval()
    jCNN_pred = jCNN(scalar_input)

    print('Input shape:', scalar_input.shape)
    print('Output shape:', jCNN_pred.shape)

    N_jCNN_param = count_parameters(jCNN)
