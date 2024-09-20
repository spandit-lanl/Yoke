"""Functions and classes to calculate normalization values for the *lsc240420*
dataset. For example, finding the average density for each *av_density* image.

These quantities are saved in an NPZ file for use during training in a Torch
dataloader.

When run as a script an NPZ of calculated parameters is generated.

"""

####################################
## Packages
####################################
import os, sys, argparse
import glob
import random
import typing
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NoneStr = typing.Union[None, str]

from yoke.datasets.lsc_dataset import LSCnpz2key, LSCcsv2bspline_pts
from yoke.datasets.lsc_dataset import LSCread_npz, LSC_cntr2rho_DataSet


###################################################################
# Define command line argument parser
descr_str = ('Generate lsc240420 normalization.')
parser = argparse.ArgumentParser(prog='Generate Normalization',
                                 description=descr_str,
                                 fromfile_prefix_chars='@')

parser.add_argument('--design_file',
                    action='store',
                    type=str,
                    default='design_lsc240420_MASTER.csv',
                    help='.csv file that contains the truth values for data files')

parser.add_argument('--eval_filelist',
                    action='store',
                    type=str,
                    default='lsc240420_train_80pct.txt',
                    help='Path to list of files to evaluate normalizations for.')


# Main script
args = parser.parse_args()

# YOKE env variables
YOKE_DIR = os.getenv('YOKE_DIR')
LSC_NPZ_DIR = os.getenv('LSC_NPZ_DIR')
LSC_DESIGN_DIR = os.getenv('LSC_DESIGN_DIR')

## Data Paths
design_file = os.path.abspath(LSC_DESIGN_DIR+args.design_file)
eval_files = YOKE_DIR + 'filelists/' + args.eval_filelist

## Create filelist
with open(eval_files, 'r') as f:
    eval_filelist = [line.rstrip() for line in f]

Nsamp = len(eval_filelist)
print('Number of samples:', Nsamp)

avg_time_dict = {}
nsamp_time_dict = {}

# Build arrays to normalize with
for k, filepath in enumerate(eval_filelist):
    ## Get the input image
    npz = np.load(LSC_NPZ_DIR+filepath)

    true_image = LSCread_npz(npz, 'av_density')
    true_image = np.concatenate((np.fliplr(true_image), true_image), axis=1)
    nY, nX = true_image.shape
    #print('True image shape:', nY, nX)

    ## Get the contours and sim_time
    sim_key = LSCnpz2key(LSC_NPZ_DIR+filepath)
    Bspline_nodes = LSCcsv2bspline_pts(design_file, sim_key)
    #print('Shape of Bspline node array:', Bspline_nodes.shape)
    
    sim_time = npz['sim_time']
    #print('Sim. Time:', sim_time)
    round_sim_time = str(round(4.0*sim_time)/4.0)
    #print('Nearest 0.25us Sim. Time:', round_sim_time)
    
    npz.close()

    if round_sim_time in avg_time_dict.keys():
        avg_time_dict[round_sim_time] += true_image
        nsamp_time_dict[round_sim_time] += 1
    else:
        avg_time_dict[round_sim_time] = true_image
        nsamp_time_dict[round_sim_time] = 1

    # Calculate normalization quantities
    if k == 0:
        image_avg = true_image
        image_min = np.min(true_image)
        image_max = np.max(true_image)
        Bspline_avg = Bspline_nodes
        Bspline_min = Bspline_nodes
        Bspline_max = Bspline_nodes
        #print('Image_min:', image_min)
        #print('Image_max:', image_max)
    else:
        image_avg += true_image
        image_min = min(image_min, np.min(true_image))
        image_max = max(image_max, np.max(true_image))
        Bspline_avg += Bspline_nodes
        Bspline_min = np.minimum(Bspline_nodes, Bspline_min)
        Bspline_max = np.maximum(Bspline_nodes, Bspline_max)
        #print('Bspline min array:', Bspline_min)
        #print('Bspline max array:', Bspline_max)
        #print('Image_min:', image_min)
        #print('Image_max:', image_max)
        
    #print('============')
    
# Calculate averages
image_avg = image_avg/Nsamp
Bspline_avg = Bspline_avg/Nsamp

for k, v in avg_time_dict.items():
    avg_time_dict[k] = v/nsamp_time_dict[k]

# Save normalization information
np.savez('./lsc240420_norm.npz',
         image_avg=image_avg,
         image_min=image_min,
         image_max=image_max,
         Bspline_avg=Bspline_avg,
         Bspline_min=Bspline_min,
         Bspline_max=Bspline_max,
         **avg_time_dict)



