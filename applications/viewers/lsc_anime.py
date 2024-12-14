"""Script to produce animation of LSC simulation.

This script allows production of an animation of a single hydrodynamic field
within one lsc240420 simulation set of NPZ files.

"""

import os
import glob
import argparse
import numpy as np

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib

# matplotlib.use('MacOSX')
# matplotlib.use('pdf')
# matplotlib.use('QtAgg')
# Get rid of type 3 fonts in figures
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

# Ensure LaTeX font
font = {"family": "serif"}
plt.rc("font", **font)
plt.rcParams["figure.figsize"] = (6, 6)


###################################################################
# Define command line argument parser
descr_str = "Create animation of single hydro-field for lsc240420 simulation IDX."
parser = argparse.ArgumentParser(prog="Animation from NPZ", description=descr_str)

# indir
parser.add_argument(
    "--indir",
    "-D",
    action="store",
    type=str,
    default="/data2/lsc240420",
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
# Example: lsc240420_id00405_pvi_idx00100.npz
parser.add_argument(
    "--runID",
    "-R",
    action="store",
    type=int,
    default=405,
    help="Run identifier index.",
)

# Hydro-dynamic field from PVI
parser.add_argument(
    "--field",
    "-F",
    action="store",
    type=str,
    default="av_density",
    help="Depends on keys stored in file. Use -K option to print keys.",
)

parser.add_argument(
    "--keys", "-K", action="store_true", help="Flag to print keys of NPZ file."
)

parser.add_argument(
    "--verbose",
    "-V",
    action="store_true",
    help="Flag to turn on debugging output."
)


def print_NPZ_keys(npzfile: str="./lsc240420_id00405_pvi_idx00100.npz") -> None:
    """Print keys of NPZ file."""
    NPZ = np.load(npzfile)
    print("NPZ file keys:")
    for key in NPZ.keys():
        print(key)

    NPZ.close()

    return


def singlePVIarray(
    npzfile: str="./lsc240420_id00405_pvi_idx00100.npz", FIELD: str="rho"
) -> np.array:
    """Function to grab single array from NPZ.

    Args:
       npzfile (str): File name for NPZ.
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


if __name__ == "__main__":
    # Parse commandline arguments
    args_ns = parser.parse_args()

    # Assign command-line arguments
    indir = args_ns.indir
    outdir = args_ns.outdir
    runID = args_ns.runID
    FIELD = args_ns.field
    KEYS = args_ns.keys
    VERBOSE = args_ns.verbose

    # Assemble filename
    # Example: lsc240420_id00405_pvi_idx00100.npz
    npz_glob = os.path.join(indir, f"lsc240420_id{runID:05d}_pvi_idx?????.npz")
    npz_list = sorted(glob.glob(npz_glob))

    if VERBOSE:
        print("NPZ files:", npz_list)

    if KEYS:
        print_NPZ_keys(npzfile=npz_list[0])

    else:
        for npzfile in npz_list:
            # Get index
            pviIDX = npzfile.split('idx')[1]
            pviIDX = int(pviIDX.split('.')[0])
            
            # Get the fields
            Hfield = singlePVIarray(npzfile=npzfile, FIELD=FIELD)
            simtime = singlePVIarray(npzfile=npzfile, FIELD="sim_time")
            Rcoord = singlePVIarray(npzfile=npzfile, FIELD="Rcoord")
            Zcoord = singlePVIarray(npzfile=npzfile, FIELD="Zcoord")

            # Plot normalized radiograph and density field for diagnostics.
            fig1, ax1 = plt.subplots(1, 1, figsize=(12, 12))
            img1 = ax1.imshow(
                Hfield,
                aspect="equal",
                extent=[0.0, Rcoord.max(), Zcoord.min(), Zcoord.max()],
                origin="lower",
                cmap="jet",
            )
            ax1.set_ylabel("Z-axis (cm)", fontsize=16)
            ax1.set_xlabel("R-axis (cm)", fontsize=16)
            ax1.set_title(f"T={float(simtime):.2f}us", fontsize=18)

            divider1 = make_axes_locatable(ax1)
            cax1 = divider1.append_axes("right", size="10%", pad=0.1)
            fig1.colorbar(img1, cax=cax1).set_label(f"{FIELD}", fontsize=14)

            fig1.savefig(
                os.path.join(outdir,
                             f"lsc240420_id{runID:05d}_{FIELD}_idx{pviIDX:05d}.png"),
                bbox_inches="tight",
            )
            plt.close()
