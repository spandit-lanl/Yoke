"""This dataloader processes only PVI .npz files and returns a single specified
hydrodynamic field.

Contains functions that process the .npz files for the nested cylinder
dataset

Note that some .npz file names contain the "pvi" flag and contain hydrodynamic
fields and some .npz file names contain the "pdv" and contain the photon
doppler velocimetry traces

"""

####################################
## Packages
####################################
import os
import sys
import glob
import random
import typing
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NoneStr = typing.Union[None, str]


################################################
## Functions for processing nested cylinder data
################################################
def npz2key(npz_file: str):
    """Function to extract study information from the name of an .npz file

        Args:
            npz_file (str): file path from working directory to .npz file
        
        Returns: 
            key (str): The study information for the simulation that
                       generated the .npz file; of the form "nc231213_Sn_id####"

    """

    key = npz_file.split('/')[-1].split('_')
    key = '_'.join(key[0:3])

    return key


def csv2scalar(csv_file: str, key:str, scalar:str):
    """Function to extract the scalar value from the design .csv file given the
    study key.
        
    Args:
        csv_file (str): file path from working directory to the .csv design file
        key (str): The study information for a given simulation; of the 
                   form "nc231213_Sn_id####"
        scalar (str): column name of scalar to extract from the design file
        
    Returns:
        value (float): the value of the scalar for the specified key

    """
    
    design_df = pd.read_csv(csv_file,
                            sep=',',
                            header=0,
                            index_col=0,
                            engine='python')
    
    #removed spaces from headers
    for col in design_df.columns:
        design_df.rename(columns={col: col.strip()}, inplace=True)

    assert scalar in design_df.columns, "csv2scalar: selected scalar is not in the design file"

    value = design_df.at[key, scalar]

    return value


def npz_pvi2field(npz: np.lib.npyio.NpzFile, field: str):
    """Function to extract a field "picture" array from an .npz file

    Args:
        npz (np.lib.npyio.NpzFile): a loaded .npz file
        field (str): name of field to extract

    Returns:
        pic (np.ndarray[(1700, 500), float]): field 

    """
    
    pic = npz[field]
    pic = pic[800:, :250]
    pic = np.concatenate((np.fliplr(pic), pic), axis=1)

    return pic


####################################
## DataSet Class
####################################
class PVI_SingleField_DataSet(Dataset):
    def __init__(self,
                 NC_NPZ_DIR: str,
                 filelist: str,
                 input_field: str='rho',
                 predicted: str='ptw_scale',
                 design_file: str='/data2/design_nc231213_Sn_MASTER.csv'):

        """The definition of a dataset object for the simple nested cylinder 
        problem: Nested Cylinder MOI density -> PTW scale value

        Args:
            NC_NPZ_DIR (str): Location of NC NPZ files. A yoke environment variable.
            filelist (str): Text file listing file names to read
            input_field (str): The radiographic/hydrodynamic field the model 
                               is trained on
            predicted (str): The scalar value that a model predicts
            design_file (str): .csv file with master design study parameters

        """

        ## Model Arguments
        self.NC_NPZ_DIR = NC_NPZ_DIR
        self.input_field = input_field
        self.predicted = predicted
        self.filelist = filelist
        self.design_file = design_file

        ## Create filelist
        with open(filelist, 'r') as f:
            self.filelist = [line.rstrip() for line in f]
            
        self.Nsamples = len(self.filelist)

    def __len__(self):
        """Return number of samples in dataset.

        """

        return self.Nsamples

    def __getitem__(self, index):
        """Return a tuple of a batch's input and output data for training at a given index.

        """

        ## Get the input image
        filepath = self.filelist[index]
        npz = np.load(self.NC_NPZ_DIR+filepath)
        img_input = npz_pvi2field(npz, self.input_field)
        in_y, in_x = img_input.shape
        img_input = img_input.reshape((1, in_y, in_x))
        img_input = torch.tensor(img_input).to(torch.float32)

        ## Get the ground truth.
        # NOTE: This will not work with Dcj being predicted.
        key = npz2key(filepath)
        truth = csv2scalar(self.design_file, key, self.predicted)

        return img_input, truth

    # def __getitems__(self, indices):
    #     """A mysterious method that PyTorch may have support for and documentation
    #     implies may produce a speed up that returns a batch as a list of tensors
    #     instead of a single tensor.

    #     """


# if __name__ == '__main__':
#     """For testing and debugging.

#     """

#     # Imports for plotting
#     # To view possible matplotlib backends use
#     # >>> import matplotlib
#     # >>> bklist = matplotlib.rcsetup.interactive_bk
#     # >>> print(bklist)
#     import matplotlib
#     #matplotlib.use('MacOSX')
#     matplotlib.use('TkAgg')
#     # Get rid of type 3 fonts in figures
#     matplotlib.rcParams['pdf.fonttype'] = 42
#     matplotlib.rcParams['ps.fonttype'] = 42
#     import matplotlib.pyplot as plt
#     # Ensure LaTeX font
#     font = {'family': 'serif'}
#     plt.rc('font', **font)
#     plt.rcParams['figure.figsize'] = (6, 6)
#     from mpl_toolkits.axes_grid1 import make_axes_locatable

#     # Get yoke environment variables
#     YOKE_DIR = os.getenv('YOKE_DIR')
#     NC_NPZ_DIR = os.getenv('NC_NPZ_DIR')
#     NC_DESIGN_DIR = os.getenv('NC_DESIGN_DIR')

#     pvi_test_ds = PVI_SingleField_DataSet(NC_NPZ_DIR,
#                                           YOKE_DIR+'filelists/nc231213_test_10pct.txt',
#                                           input_field='rho',
#                                           predicted='ptw_scale',
#                                           design_file=NC_DESIGN_DIR+'design_nc231213_Sn_MASTER.csv')

#     sampIDX = 123
#     rho_samp, ptw_scale_samp = pvi_test_ds.__getitem__(sampIDX)

#     print('Shape of density field tensor: ', rho_samp.shape)
#     rho_samp = np.squeeze(rho_samp.numpy())
    
#     # Plot normalized radiograph and density field for diagnostics.
#     fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
#     img1 = ax1.imshow(rho_samp,
#                       aspect='equal',
#                       origin='lower',
#                       cmap='jet')
#     ax1.set_ylabel("Z-axis", fontsize=16)                 
#     ax1.set_xlabel("R-axis", fontsize=16)
#     ax1.set_title('c-PTW={:.3f}'.format(float(ptw_scale_samp)), fontsize=18)

#     divider1 = make_axes_locatable(ax1)
#     cax1 = divider1.append_axes('right', size='10%', pad=0.1)
#     fig1.colorbar(img1,
#                   cax=cax1).set_label('Density',
#                                       fontsize=14)

#     plt.show()
    
