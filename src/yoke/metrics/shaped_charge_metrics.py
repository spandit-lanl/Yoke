################################################################
# This file defines a class for computing shaped-charge metrics.
#
# Created by Derek Armstrong, XCP-8, June 2024
################################################################

import numpy as np
import scipy as sp
import os

# import matplotlib stuff and set as desired
import matplotlib

# Get rid of type 3 fonts in figures
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)
from mpl_toolkits.axes_grid1 import make_axes_locatable


# function for reading the npz files
def singlePVIarray(npzfile="./lsc_nonconvex_pvi_idx00115.npz", FIELD="rho"):
    """Function to grab single array from NPZ.

    Args:
       indir (str): File name for NPZ.
       FIELD (str): Field to return array for.

    Returns:
       field (np.array): Array of hydro-dynamic field for plotting

    """
    NPZ = np.load(npzfile)
    arrays_dict = dict()
    for key in NPZ.keys():
        arrays_dict[key] = NPZ[key]

    NPZ.close()

    return arrays_dict[FIELD]


class SCmetrics:
    # initialization class, primarily called by SCmetrics(filename),
    # where filename is the name of the npz file.
    # variable liner is for the name of the shaped-charge liner
    # in the Pagosa simulations (more specifically, as named in
    # the npz files).
    def __init__(self, filename, liner="throw"):
        self.filename = filename

        # initialize density and vf of liner
        rhofield = singlePVIarray(npzfile=filename, FIELD="density_" + liner)
        self.density = self.get_field("density_" + liner)

        # initialize volume fraction for liner andW-velocity
        self.vofm = self.get_field("vofm_" + liner)
        self.Wvelocity = self.get_field("Wvelocity")

        # get mesh coordinates
        self.Rcoord = singlePVIarray(npzfile=filename, FIELD="Rcoord")
        self.Zcoord = singlePVIarray(npzfile=filename, FIELD="Zcoord")
        # extend coordinate vectors to contain end points of mesh
        self.Rcoord = np.append(self.Rcoord, 2 * self.Rcoord[-1] - self.Rcoord[-2])
        self.Zcoord = np.append(self.Zcoord, 2 * self.Zcoord[-1] - self.Zcoord[-2])

        # initialize other fields such as volume,
        # volume is the cell volume for 2D cylindrical meshes
        self.volume = self.compute_volume()

        # create regions field using density of liner
        # this returns density field of connected components
        # that lie on the vertical axis (axis of symmetry)
        self.regions = self.compute_regions(mask=True)

        # initialize some vars to None and only compute them once
        self.jet_mass = None
        self.jet_kinetic_energy = None
        self.HE_mass = None

        # hard-coding names of some materials
        self.HE_field_name = "density_maincharge"
        self.HE_vofm_field_name = "vofm_maincharge"

    # function for getting a field from a npz file
    # function checks for nans, as they can occur, for example,
    # in the density fields
    def get_field(self, field_name):
        field = singlePVIarray(npzfile=self.filename, FIELD=field_name)
        field_map = np.zeros(field.shape)
        Dind = np.where(np.isfinite(field))
        field_map[Dind] = field[Dind]
        return field_map

    # function to compute and return HE mass
    def get_HE_mass(self):
        if self.HE_mass == None:
            HEdensity = self.get_field(self.HE_field_name)
            HEvofm = self.get_field(self.HE_vofm_field_name)
            return np.sum(self.volume * HEdensity * HEvofm)
        else:
            return self.HE_mass

    ###############################################################
    # Function to compute jet width statistics.
    # Function returns avg width, std dev of width, and max width.
    #
    # Variable vel_thres sets the velocity threshold and the
    # "jet" is only considered when its velocity exceeds the
    # threshold.
    ###############################################################
    def get_jet_width_stats(self, vel_thres=0.0):
        if vel_thres > 0.0:
            # get jet locations above threshold
            Vind = np.where((self.regions) & (self.Wvelocity > vel_thres))
            skeleton = np.zeros(self.regions.shape)
            skeleton[Vind] = self.regions[Vind]
        else:
            skeleton = self.regions

        # get jet width as a function of z (vertical axis)
        Rcoord_map = np.repeat(
            np.reshape(self.Rcoord[1:], (1, -1)), skeleton.shape[0], axis=0
        )
        Rcoord_mask = skeleton * Rcoord_map
        width = np.max(Rcoord_mask, axis=0)

        # compute stats
        # multiplying by 2 to consider a "true" width instead of a "radius",
        # since we're looking at 2D cylindrical simulations
        avg_width = 2.0 * np.mean(width)
        std_width = 2.0 * np.std(width)
        max_width = 2.0 * np.max(width)

        return avg_width, std_width, max_width

    ###############################################################
    # Function to compute cumulative value of jet density times
    # velocity squared over a 2D cross section of jet.
    # This is primarly intended for 2D axi-symmetric calculations.
    #
    # Variable vel_thres allows for parts of jet below the
    # threshold to be ignored.
    ###############################################################
    def get_jet_rho_velsq_2D(self, vel_thres=0.1):
        Vind = np.where((self.Wvelocity >= vel_thres) & (self.regions))
        return np.sum(self.density[Vind] * np.square(self.Wvelocity[Vind]))

    ###############################################################
    # Function to compute cumulative value of jet sqrt(density)
    # times velocity over a 2D cross section of jet.
    # This is primarly intended for 2D axi-symmetric calculations.
    #
    # Variable vel_thres allows for parts of jet below the
    # threshold to be ignored.
    ###############################################################
    def get_jet_sqrt_rho_vel_2D(self, vel_thres=0.1):
        Vind = np.where((self.Wvelocity > vel_thres) & (self.regions))
        return np.sum(np.sqrt(self.density[Vind]) * self.Wvelocity[Vind])

    ###############################################################
    # Function to compute jet kinetic energy of effective jet.
    #
    # This differs from function get_jet_rho_velsq_2D in that
    # it computes the kinetic energy of the actual 3D jet object.
    ###############################################################
    def get_jet_kinetic_energy(self, vel_thres=0.1):
        eff_jet_mass_map = self.get_eff_jet_mass_map(vel_thres=vel_thres)
        return 0.5 * np.sum(eff_jet_mass_map * np.square(self.Wvelocity))

    ###############################################################
    # Function to compute spatially integrated quantity of
    # sqrt(0.5 * mass) * velocity.
    #
    # This differs from function get_jet_sqrt_rho_velsq_2D in that
    # it computes the quantity for the actual 3D jet object.
    ###############################################################
    def get_jet_sqrt_kinetic_energy(self, vel_thres=0.1):
        eff_jet_mass_map = np.sqrt(self.get_eff_jet_mass_map(vel_thres=vel_thres))
        return 0.70710678 * np.sum(np.sqrt(eff_jet_mass_map) * self.Wvelocity)

    ###############################################################
    # Function to compute volume for 2D axis-symmetric grid cells.
    ###############################################################
    def compute_volume(self):
        surf_area = np.pi * (np.square(self.Rcoord[1:]) - np.square(self.Rcoord[0:-1]))
        height = self.Zcoord[1:] - self.Zcoord[0:-1]
        volume = np.matmul(
            np.reshape(height, (len(height), 1)),
            np.reshape(surf_area, (1, len(surf_area))),
        )
        return volume

    ###############################################################
    # Compute and return jet mass
    ###############################################################
    def get_jet_mass(self):
        if self.jet_mass == None:
            return np.sum(self.volume * self.density * self.vofm)
        else:
            return self.jet_mass

    ###############################################################
    # Returns the connected components that touch the
    # central axis (axis of symmetry for 2D runs).
    # Initial field is taken as the density-liner field.
    #
    # If mask is True, then only return zero/one with
    # one representing an on-axis jet component.
    # Otherwise, each different connected component will be
    # labeled with an "ID" (just a number) for the component.
    ###############################################################
    def compute_regions(self, mask=False):
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
        field_regions, n_regions = sp.ndimage.label(self.density, structure)

        # get region labels for regions that are on-axis
        axis_regions = np.unique(field_regions[:, 0])

        # removing connected components/regions that are not on-axis
        count = 1  # label for first connected component, needs to
        # be greater than zero.
        # A value of zero represents locations that are
        # not part of any connected component.
        for ilabel in range(1, np.max(field_regions) + 1, 1):
            Aind = np.where(field_regions == ilabel)
            if ilabel not in axis_regions:  # region is not on-axis
                field_regions[Aind] = 0
            else:  # region is on-axis
                field_regions[Aind] = count
                if not mask:  # increment label for next region
                    count = count + 1

        return field_regions

    ##############################################################
    # Function to compute maximum field value over contiguous jet.
    # To be included in the contiguous jet, a location must
    # be in a connected component that is touching the "central"
    # axis (which is the 2D axis of symmetry).
    #
    # Input:
    #   field - field to take max value from
    ##############################################################
    def max_regions(self, field):
        Vind = np.where(self.regions > 0)  # get on-axis jet regions
        maxv = np.max(field[Vind])
        return maxv

    ##############################################################
    # Function to compute average field value, where average
    # is taken over connected jet regions that are on-axis.
    ##############################################################
    def avg_regions(self, field, thresh=0.0):
        Vind = np.where((self.regions) > 0 and (field >= thresh))
        avg = np.mean(field[Vind])
        return avg

    ##############################################################
    # Function to return maximum vertical velocity
    ##############################################################
    def max_Wvelocity(self):
        return self.max_regions(self.Wvelocity)

    ##############################################################
    # Function to return average vertical velocity
    ##############################################################
    def avg_Wvelocity(self, Wthresh=0.0):
        return self.avg_regions(self.Wvelocity, Wthresh=Wthresh)

    ##############################################################
    # Function to perform PCA --- fit ellipse to the on-axis
    # jet regions. This function is a work in progress.
    ##############################################################
    def pca(self):
        Vind = np.where(self.regions > 0)
        Rvalues = self.Rcoord[Vind[1]]
        Zvalues = self.Zcoord[Vind[0]]
        meanZ = np.mean(Zvalues)
        meanR = np.mean(Rvalues)
        Zvalues = Zvalues - meanZ
        Rvalues = Rvalues - meanR
        nd = len(Rvalues)
        RZ_matrix = np.zeros((nd, 2))
        RZ_matrix[:, 0] = Rvalues
        RZ_matrix[:, 1] = Zvalues
        Rvar = np.var(Rvalues)
        Zvar = np.var(Zvalues)
        print("Rvar", Rvar)
        print("Zvar", Zvar)
        # Zvar = np.var(Zvalues)
        # RZcovar = np.dot(Rvalues,Zvalues)
        U, S, Vh = np.linalg.svd(RZ_matrix)
        print("D")
        print(S)
        # rows of Vh are the eigenvectors
        print(Vh[0, :])
        print(Vh[1, :])
        print("variances", S[0] * S[0] / nd, S[1] * S[1] / nd)

    ##############################################################
    # Function to compute effective jet mass.
    # Effective jet mass is mass of jet with Wvelocity above
    # a threshold and for a connected component that lies on
    # the vertical axis.
    ##############################################################
    def get_eff_jet_mass(self, vel_thres=0.1, asPercent=False):
        eff_jet_mass = np.sum(self.get_eff_jet_mass_map())
        if asPercent:
            return eff_jet_mass / self.get_jet_mass()
        else:
            return eff_jet_mass

    ##############################################################
    # Function to compute effective jet mass map.
    # Return effective jet mass for each cell/zone in simulation.
    ##############################################################
    def get_eff_jet_mass_map(self, vel_thres=0.1):
        Vind = np.where((self.Wvelocity >= vel_thres) & (self.regions))
        eff_jet_mass_map = self.volume[Vind] * self.density[Vind] * self.vofm[Vind]
        return eff_jet_mass_map

    ##############################################################
    # Function for plotting the density-liner field.
    # Modify later to plot for user selected variable field.
    ##############################################################
    def plotField(self):
        # Plot normalized radiograph and density field for diagnostics.
        field = np.concatenate((np.fliplr(self.density), self.density), axis=1)
        fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
        img1 = ax1.imshow(
            field,
            aspect="equal",
            extent=[
                -self.Rcoord.max(),
                self.Rcoord.max(),
                self.Zcoord.min(),
                self.Zcoord.max(),
            ],
            origin="lower",
            cmap="jet",
        )
        # cmap='cividis' if FIELD=='pRad' else 'jet')
        ax1.set_ylabel("Z-axis (um)", fontsize=16)
        ax1.set_xlabel("R-axis (um)", fontsize=16)
        # ax1.set_title('T={:.2f}us'.format(float(simtime)), fontsize=18)

        divider1 = make_axes_locatable(ax1)
        cax1 = divider1.append_axes("right", size="10%", pad=0.1)
        fig1.colorbar(img1, cax=cax1).set_label("density", fontsize=14)
        # fig1.colorbar(img1,
        #               cax=cax1).set_label(f'{FIELD}',
        #                                   fontsize=14)

        fig1.savefig(
            os.path.join("png", self.filename + ".density.png"), bbox_inches="tight"
        )
