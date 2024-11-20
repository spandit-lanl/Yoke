"""Functions and classes for torch DataSets which sample the Layered Shaped
Charge data, *lsc240420*.

"""

####################################
# Packages
####################################
import os
import sys
from pathlib import Path
import typing
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

NoneStr = typing.Union[None, str]


################################################
# Functions for returning the run *key* from
# the npz-file name
################################################
def LSCnpz2key(npz_file: str):
    """Function to extract simulation *key* from the name of an .npz file.

    A study key looks like **lsc240420_id00001** and a NPZ filename is like
    **lsc240420_id00001_pvi_idx00000.npz**

    Args:
        npz_file (str): file path from working directory to .npz file

    Returns:
        key (str): The correspond simulation key for the NPZ file.

    """
    key = npz_file.split("/")[-1].split("_pvi_")[0]

    return key


def LSCcsv2bspline_pts(design_file: str, key: str):
    """Function to extract the B-spline nodes from the design .csv file given the
    study key.

    Args:
        csv_file (str): file path from working directory to the .csv design file
        key (str): The study information for a given simulation; of the
                   form *lsc240420_id?????*

    Returns:
        bspline_pts (numpy array): The B-spline nodes defining the geometry of
                                   the Layered Shaped Charge

    """
    design_df = pd.read_csv(design_file, sep=",", header=0, index_col=0, engine="python")

    # removed spaces from headers
    for col in design_df.columns:
        design_df.rename(columns={col: col.strip()}, inplace=True)

    bspline_pts = design_df.loc[key, "sa1":"ct7"].values

    return bspline_pts.astype(float)


def LSCread_npz(npz: np.lib.npyio.NpzFile, field: str):
    """Function to extract a value corresponding to an NPZ key.

    Args:
        npz (np.lib.npyio.NpzFile): a loaded .npz file
        field (str): name of field to extract

    """
    return npz[field]


####################################
# DataSet Classes
####################################
class LSC_cntr2rho_DataSet(Dataset):
    def __init__(self, LSC_NPZ_DIR: str, filelist: str, design_file: str):
        """The definition of a dataset object for the *Layered Shaped Charge* data
        which produces pairs of B-spline contour-node vectors and simulation times
        together with an average density field.

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files. A YOKE env variable.
            filelist (str): Text file listing file names to read
            design_file (str): .csv file with master design study parameters

        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.filelist = filelist
        self.design_file = design_file

        # Create filelist
        with open(filelist) as f:
            self.filelist = [line.rstrip() for line in f]

        self.Nsamples = len(self.filelist)

    def __len__(self):
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index):
        """Return a tuple of a batch's input and output data for training at a given
        index.

        """
        # Get the input image
        filepath = self.filelist[index]
        npz = np.load(self.LSC_NPZ_DIR + filepath)

        true_image = LSCread_npz(npz, "av_density")
        true_image = np.concatenate((np.fliplr(true_image), true_image), axis=1)
        nY, nX = true_image.shape
        true_image = true_image.reshape((1, nY, nX))
        true_image = torch.tensor(true_image).to(torch.float32)

        # Get the contours and sim_time
        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + filepath)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)
        sim_time = npz["sim_time"]
        npz.close()

        sim_params = np.append(Bspline_nodes, sim_time)
        sim_params = torch.from_numpy(sim_params).to(torch.float32)

        return sim_params, true_image


class LSCnorm_cntr2rho_DataSet(Dataset):
    def __init__(
        self, LSC_NPZ_DIR: str, filelist: str, design_file: str, normalization_file: str
    ):
        """The definition of a dataset object for the *Layered Shaped Charge* data
        which produces pairs of B-spline contour-node vectors and simulation times
        together with an average density field.

        This class uses pre-calculated normalization constants to normalize the
        geometry parameters and the density field. The time values are also
        normalized to [0, 1] at regular intervals of 0.25 us.

        Args:
            LSC_NPZ_DIR (str): Location of LSC NPZ files. A YOKE env variable.
            filelist (str): Text file listing file names to read
            design_file (str): .csv file with master design study parameters
            normalization_file: Full-path to the NPZ file containing the pre-calculated
                                normalization quantities.

        """
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.filelist = filelist
        self.design_file = design_file
        self.normalization_file = normalization_file

        # Open normalization file
        # np.savez('./lsc240420_norm.npz',
        #  image_avg=image_avg,
        #  image_min=image_min,
        #  image_max=image_max,
        #  Bspline_avg=Bspline_avg,
        #  Bspline_min=Bspline_min,
        #  Bspline_max=Bspline_max,
        #  **avg_time_dict)

        norm_npz = np.load(self.normalization_file)
        time_keys = [k for k in norm_npz.keys()]
        time_keys.remove("image_avg")
        time_keys.remove("image_min")
        time_keys.remove("image_max")
        time_keys.remove("Bspline_avg")
        time_keys.remove("Bspline_min")
        time_keys.remove("Bspline_max")
        self.time_keys = sorted([float(k) for k in time_keys])

        self.avg_rho_by_time = dict()
        for tk in self.time_keys:
            self.avg_rho_by_time[str(tk)] = norm_npz[str(tk)]

        self.Bspline_min = norm_npz["Bspline_min"]
        self.Bspline_max = norm_npz["Bspline_max"]

        # Create filelist
        with open(filelist) as f:
            self.filelist = [line.rstrip() for line in f]

        self.Nsamples = len(self.filelist)

    def __len__(self):
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index):
        """Return a tuple of a batch's input and output data for training at a given
        index.

        """
        # Get the input image
        filepath = self.filelist[index]
        npz = np.load(self.LSC_NPZ_DIR + filepath)

        # Get time associated with image
        sim_time = npz["sim_time"]
        round_sim_time = round(4.0 * sim_time) / 4.0

        # Load image
        true_image = LSCread_npz(npz, "av_density")
        true_image = np.concatenate((np.fliplr(true_image), true_image), axis=1)
        # unbias image
        unbias_true_image = true_image - self.avg_rho_by_time[str(round_sim_time)]
        # unbias_true_image = self.avg_rho_by_time[str(round_sim_time)]
        nY, nX = unbias_true_image.shape
        unbias_true_image = unbias_true_image.reshape((1, nY, nX))
        unbias_true_image = torch.tensor(unbias_true_image).to(torch.float32)

        # Get the contours and sim_time
        sim_key = LSCnpz2key(self.LSC_NPZ_DIR + filepath)
        Bspline_nodes = LSCcsv2bspline_pts(self.design_file, sim_key)

        npz.close()

        # Scale round_sim_time
        norm_time = round_sim_time / 25.0

        # Normalize Bspline nodes
        norm_Bspline_nodes = (Bspline_nodes - self.Bspline_min) / (
            self.Bspline_max - self.Bspline_min
        )
        norm_sim_params = np.append(norm_Bspline_nodes, norm_time)
        norm_sim_params = torch.from_numpy(norm_sim_params).to(torch.float32)

        return norm_sim_params, unbias_true_image


class LSC_rho2rho_temporal_DataSet(Dataset):
    """This dataset returns multi-channel images at two different times
    from the *Layered Shaped Charge* simulation. The *maximum time-offset*
    can be specified. The channels in the images returned are the densities
    for each material at a given time as well as the (R, Z)-velocity
    fields. The time-offset between the two images is also returned.

    NOTE: The way time indices are chosen necessitates *max_timeIDX_offset*
    being less than or equal to 3 in the lsc240420 data.

    Args:
        LSC_NPZ_DIR (str): Location of LSC NPZ files.
        file_prefix_list (str): Text file listing unique prefixes corresponding
                                to unique simulations.
        max_timeIDX_offset (int): Maximum timesteps-ahead to attempt
                                  prediction for. A prediction image will be chosen
                                  within this timeframe at random.
        max_file_checks (int): This dataset generates two random time indices and 
                               checks if the corresponding files exist. This 
                               argument controls the maximum number of times indices 
                               are generated before throwing an error.

    """
    def __init__(
            self,
            LSC_NPZ_DIR: str,
            file_prefix_list: str,
            max_timeIDX_offset: int,
            max_file_checks: int,
    ) -> None:        
        # Model Arguments
        self.LSC_NPZ_DIR = LSC_NPZ_DIR
        self.max_timeIDX_offset = max_timeIDX_offset
        self.max_file_checks = max_file_checks
        
        # Create filelist
        with open(file_prefix_list) as f:
            self.file_prefix_list = [line.rstrip() for line in f]

        # Shuffle the list of prefixes in-place
        random.shuffle(self.file_prefix_list)
        
        self.Nsamples = len(self.file_prefix_list)

        # Lists of fields to return images for 
        self.hydro_fields = ['density_case',
                             'density_cushion',
                             'density_maincharge',
                             'density_outside_air',
                             'density_striker',
                             'density_throw',
                             'Uvelocity',
                             'Wvelocity']
        
        # Initialize random number generator for time index selection
        self.rng = np.random.default_rng()
        
    def __len__(self):
        """Return number of samples in dataset."""
        return self.Nsamples

    def __getitem__(self, index):
        """Return a tuple of a batch's input and output data for training at a given
        index.

        """
        # Rotate index if necessary
        index = index % self.Nsamples
        
        # Get the input image. Try several indices if necessary.
        prefix_attempt = 0
        prefix_loop_break = False
        while prefix_attempt < 5:
            file_prefix = self.file_prefix_list[index]

            # Use `while` loop to search until a pair of files which exists is
            # found.
            attempt = 0
            while attempt < self.max_file_checks:
                # Files have name format
                # *lsc240420_id01001_pvi_idx00000.npz*. Choose random starting
                # index 0-96 so the end index will be less than or equal to 99.
                startIDX = self.rng.integers(0, 97)
                endIDX = self.rng.integers(1, self.max_timeIDX_offset + 1) + startIDX

                # Construct file names
                start_file = file_prefix + f'_pvi_idx{startIDX:05d}.npz'
                end_file = file_prefix + f'_pvi_idx{endIDX:05d}.npz'

                # Check if both files exist
                start_file_path = Path(self.LSC_NPZ_DIR + start_file)
                end_file_path = Path(self.LSC_NPZ_DIR + end_file)

                if start_file_path.is_file() and end_file_path.is_file():
                    prefix_loop_break = True
                    break

                attempt += 1

            if attempt == self.max_file_checks:
                fnf_msg = ("In LSC_rho2rho_temporal_DataSet, "
                           "max_file_checks "
                           f"reached for prefix: {file_prefix}")
                print(fnf_msg, file=sys.stderr)

            # Break outer loop if time-pairs were found.
            if prefix_loop_break:
                break

            # Try different prefix if no time-pairs are found.
            print(f"Prefix attempt {prefix_attempt + 1} failed. Trying next prefix.", 
                  file=sys.stderr)
            prefix_attempt += 1
            index = (index + 1) % self.Nsamples  # Rotate index if necessary
                    
        # Load NPZ files. Raise exceptions if file is not able to be loaded.
        try:
            start_npz = np.load(self.LSC_NPZ_DIR + start_file)
        except Exception as e:
            print(f"Error loading start file: {self.LSC_NPZ_DIR + start_file}",
                  file=sys.stderr)
            raise e

        try:
            end_npz = np.load(self.LSC_NPZ_DIR + end_file)
        except Exception as e:
            print(f"Error loading end file: {self.LSC_NPZ_DIR + end_file}",
                  file=sys.stderr)
            start_npz.close()
            raise e

        start_img_list = []
        end_img_list = []
        for hfield in self.hydro_fields:
            tmp_img = LSCread_npz(start_npz, hfield)
            # Remember to replace all NaNs with 0.0
            tmp_img = np.nan_to_num(tmp_img, nan=0.0)
            tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            start_img_list.append(tmp_img)

            tmp_img = LSCread_npz(end_npz, hfield)
            # Remember to replace all NaNs with 0.0
            tmp_img = np.nan_to_num(tmp_img, nan=0.0)
            tmp_img = np.concatenate((np.fliplr(tmp_img), tmp_img), axis=1)
            end_img_list.append(tmp_img)

        # Concatenate images channel first.
        start_img = torch.tensor(np.stack(start_img_list, axis=0)).to(torch.float32)
        end_img = torch.tensor(np.stack(end_img_list, axis=0)).to(torch.float32)

        # Get the time offset
        Dt = 0.25*(endIDX - startIDX)

        # Close the npzs
        start_npz.close()
        end_npz.close()

        return start_img, end_img, Dt

