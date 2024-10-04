"""Function to return single array from complete NPZ file generated from the
NestedCylinder PVI scipts. As a script this can plot a single image of a single
hydro-dynamic field. The directory of NPZ files must be specified and the
filenames are assumed to be of the form `{runID}_pvi_idx{pviIDX:05d}.npz`

"""

import os
import argparse
import numpy as np

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib

# matplotlib.use('MacOSX')
matplotlib.use("pdf")
# Get rid of type 3 fonts in figures
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)
from mpl_toolkits.axes_grid1 import make_axes_locatable


###################################################################
# Define command line argument parser
descr_str = "Read in single NPZ file and plot a hydro-dynamic field."
parser = argparse.ArgumentParser(prog="Plot field from NPZ", description=descr_str)

# indir
parser.add_argument(
    "--indir",
    "-D",
    action="store",
    type=str,
    default="/data2/nc231213_npzs",
    help="Directory to find NPZ files.",
)

# outdir
parser.add_argument(
    "--outdir",
    "-O",
    action="store",
    type=str,
    default="./",
    help="Directory to output images to.",
)

# run index
parser.add_argument(
    "--runID",
    "-R",
    action="store",
    type=str,
    default="nc231213_Sn_id0415",
    help="Run identifier. Is prefix to _pvi_idx?????.npz",
)

# PVI index
parser.add_argument(
    "--pviIDX", "-I", action="store", type=int, default=125, help="PVI index to plot."
)

# Hydro-dynamic field from PVI
parser.add_argument(
    "--field",
    "-F",
    action="store",
    type=str,
    default="volhr_sum",
    help=(
        "Hydro-dynamic field to plot: rho, hr_outerWall, "
        "hr_bottomWall, hr_mcSide, hr_mcBottom, hr_innerCylSide, "
        "hr_innerCylBottom, hr_MOI, volhr_outerWall, "
        "volhr_bottomWall, volhr_mcSide, volhr_mcBottom, "
        "volhr_innerCylSide, volhr_innerCylBottom, volhr_MOI, "
        "volhr_sum, pressure, temperature, melt_state, porosity, "
        "eqps, eqps_rate, eff_stress, bulk_mod, sound_speed, rVel, zVel."
    ),
)

parser.add_argument("--save", "-S", action="store_true", help="Flag to save image.")


def singlePVIarray(indir="./", runID="Sn00", pviIDX=41, FIELD="rho"):
    """Function to grab single array from NPZ.

    Args:
       indir (str): Directory where all NPZ files live.
       runIDX (int): Index of run
       pviIDX (int): Index of PVI output
       FIELD (str): Field to return array for, i.e. rho, pressure, temperature,
                    melt_state, porosity, eqps, eqps_rate, eff_stress, bulk_mod,
                    sound_speed, rVel, zVel

    Returns:
       field (np.array): Array of hydro-dynamic field for plotting
       rcoord (np.array): Array of radial coordinates
       zcoord (np.array): Array of z-axis coordinates
       simtime (float): Simulation time of array

    """
    # Get field array
    filename = os.path.join(indir, f"{runID}_pvi_idx{pviIDX:05d}.npz")

    data = np.load(filename)
    if FIELD == "volhr_sum":
        # Look at sum of vol-avg density for parts
        volhr_outerWall = data["volhr_outerWall"]
        volhr_bottomWall = data["volhr_bottomWall"]
        volhr_mcSide = data["volhr_mcSide"]
        volhr_mcBottom = data["volhr_mcBottom"]
        volhr_MOI = data["volhr_MOI"]
        volhr_innerCylSide = data["volhr_innerCylSide"]
        volhr_innerCylBottom = data["volhr_innerCylBottom"]
        field = (
            volhr_MOI
            + volhr_innerCylSide
            + volhr_innerCylBottom
            + volhr_outerWall
            + volhr_bottomWall
            + volhr_mcSide
            + volhr_mcBottom
        )
    else:
        field = data[FIELD]

    # # Look at difference between material density methods
    # volhr_MOI = data['volhr_MOI']
    # hr_MOI = data['hr_MOI']
    # field = volhr_MOI - hr_MOI

    # # Look at sum of material density for parts
    # hr_MOI = data['hr_MOI']
    # hr_innerCylSide = data['hr_innerCylSide']
    # hr_innerCylBottom = data['hr_innerCylBottom']
    # field = hr_MOI + hr_innerCylSide + hr_innerCylBottom

    # Get simulation time
    simtime = data["sim_time"]

    # Get coordinate arrays
    Rcoord = data["Rcoord"]
    Zcoord = data["Zcoord"]

    return field, Rcoord, Zcoord, simtime


if __name__ == "__main__":
    # Parse commandline arguments
    args_ns = parser.parse_args()

    # Assign command-line arguments
    indir = args_ns.indir
    outdir = args_ns.outdir
    runID = args_ns.runID
    pviIDX = args_ns.pviIDX
    FIELD = args_ns.field
    SAVEFIG = args_ns.save

    # Get the field
    Hfield, Rcoord, Zcoord, simtime = singlePVIarray(
        indir=indir, runID=runID, pviIDX=pviIDX, FIELD=FIELD
    )

    Hfield = np.concatenate((np.fliplr(Hfield), Hfield), axis=1)
    print("Shape of Hfield: ", Hfield.shape)
    # Hfield = Hfield[0:300, 100:1100]
    print(Hfield.shape)

    # Plot normalized radiograph and density field for diagnostics.
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
    img1 = ax1.imshow(
        Hfield,
        aspect="equal",
        extent=[-Rcoord.max(), Rcoord.max(), Zcoord.min(), Zcoord.max()],
        origin="lower",
        cmap="cividis" if FIELD == "pRad" else "jet",
    )
    ax1.set_ylabel("Z-axis (um)", fontsize=16)
    ax1.set_xlabel("R-axis (um)", fontsize=16)
    ax1.set_title(f"T={float(simtime):.2f}us", fontsize=18)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="10%", pad=0.1)
    fig1.colorbar(img1, cax=cax1).set_label(f"{FIELD}", fontsize=14)

    if SAVEFIG:
        fig1.savefig(
            os.path.join(outdir, f"{runID}_idx{pviIDX:05d}_{FIELD}.png"),
            bbox_inches="tight",
        )
        # fig1.savefig(os.path.join(outdir,
        #                           f'{runID}_idx{pviIDX:05d}_hr_sum.png'),
        #              bbox_inches='tight')
    else:
        plt.show()
