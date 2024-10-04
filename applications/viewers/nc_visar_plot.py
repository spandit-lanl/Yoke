"""Read visar file produced from Nested-Cylinder simulation and plot velocity
vs. time signal for all PDV positions.

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
matplotlib.use('Agg')
# Get rid of type 3 fonts in figures
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
# Ensure LaTeX font
font = {'family': 'serif'}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (6, 6)


class Visar:
    """Visar data file class

    Methods of this class allow quick reading, manipulating, and plotting from
    a 'Visar' file. The file is read line-by-line so multi-channel PDV is
    supported.

    The data dictionary created for Visar/PDV has integer channel indices as
    keys. Each channel key corresponds to a dictionary with variable names as
    keys. A variable name key combined with a channel index corresponds to a
    numpy vector of that variable on that channel.

    The variables for each Visar/PDV channel are: 

    - **origin**
    - **direction**
    - **time**
    - **distance**
    - **velocity**

    """

    def __init__(self, visarfilename):
        # Define data dictionary
        self.data = {}

        # Open the file
        self.visarfilename = visarfilename

        with open(self.visarfilename) as pdvfile:
            for line in pdvfile:
                # Strip whitespace
                line = line.strip()
                if '#' in line:
                    # Parse the header for a channel. Record channel index,
                    # origin and direction.
                    if 'Channel' in line:
                        channelIDX = int(line.split()[-1])
                        self.data[channelIDX] = {}
                        self.data[channelIDX]['time'] = []
                        self.data[channelIDX]['distance'] = []
                        self.data[channelIDX]['velocity'] = []
                    elif 'Origin' in line:
                        coords = line.split(':')[-1].split()
                        origin = np.array([float(coords[0].strip()),
                                           float(coords[1].strip()),
                                           float(coords[2].strip())])
                        self.data[channelIDX]['origin'] = origin
                    elif 'Direction' in line:
                        coords = line.split(':')[-1].split()
                        direction = np.array([float(coords[0].strip()),
                                              float(coords[1].strip()),
                                              float(coords[2].strip())])
                        self.data[channelIDX]['direction'] = direction
                    else:
                        pass
                elif len(line) == 0:
                    # Test if line is blank
                    pass
                else:
                    # Parse time, distance, and velocity
                    coords = line.split()
                    self.data[channelIDX]['time'].append(float(coords[0].strip()))
                    self.data[channelIDX]['distance'].append(float(coords[1].strip()))
                    self.data[channelIDX]['velocity'].append(float(coords[2].strip()))

        # Change the time, distance, and velocity lists to numpy arrays
        for idx in self.data.keys():
            self.data[idx]['time'] = np.array(self.data[idx]['time'])
            self.data[idx]['distance'] = np.array(self.data[idx]['distance'])
            self.data[idx]['velocity'] = np.array(self.data[idx]['velocity'])

    def channels(self):
        """Get list of channel indices.

        """
        return list(self.data.keys())

    def variables(self, channelIDX=1):
        """Get list of variable names on a channel.

        Args:
            channelIDX (int): Integer index for a PDV channel

        """
        return list(self.data[channelIDX].keys())

    def getarray(self, channelIDX, varname):
        """Return array associated with channelIDX and variable, `varname`.

        Args:
            channelIDX (int): Integer index for a PDV channel
            varname (str): *origin*, *direction*, *time*, *distance*, or *velocity*

        """
        return self.data[channelIDX][varname]

    def plotVelTH(self, channels):
        """Plot Visar-velocity time history for multiple channels.

        To use this function a Matplotlib figure object and axes must already
        be instantiated. To view or save the figure `plt.show()` or
        `plt.savefig()` should be called after this method.

        Args:
            channels (list of ints): List of integer channel indices to plot 
                                     velocity time-history for

        """
        start_times = []
        stop_times = []
        for idx in channels:
            time = self.getarray(idx, 'time')
            velocity = self.getarray(idx, 'velocity')

            start_times.append(time[0])
            stop_times.append(time[-1])
            plt.plot(time,
                     velocity,
                     linewidth=1.7,
                     label=f'Velocity, channel {idx:03d}')

        plt.xlim([min(start_times), max(stop_times)])
        plt.xlabel('Time', fontsize=18)
        plt.ylabel("Velocity", fontsize=18)
        plt.tick_params(axis='both', labelsize=18)

        titlestr = 'Velocity Time History'
        plt.title(titlestr, fontsize=20)
        plt.legend(fontsize=16)

    def plotDistTH(self, channels):
        """Plot Visar-distance time history.

        Plot time history of Visar origin distance from surface. To use this
        function a Matplotlib figure object and axes must already be
        instantiated. To view or save the figure `plt.show()` or
        `plt.savefig()` should be called after this method.

        Args:
            channels (list of ints): List of integer channel indices to plot 
                                     distance time-history for

        """
        start_times = []
        stop_times = []
        for idx in channels:
            time = self.getarray(idx, 'time')
            distance = self.getarray(idx, 'distance')

            start_times.append(time[0])
            stop_times.append(time[-1])
            plt.plot(time,
                     distance,
                     linewidth=1.7,
                     label=f'Distance, channel {idx:03d}')

        plt.xlim([min(start_times), max(stop_times)])
        plt.xlabel('Time', fontsize=18)
        plt.ylabel("Distance", fontsize=18)
        plt.tick_params(axis='both', labelsize=18)

        titlestr = 'Distance Time History'
        plt.title(titlestr, fontsize=20)
        plt.legend(fontsize=16)


###################################################################
# Define command line argument parser
descr_str = 'Read in PDV file and plot velocity vs. time.'
parser = argparse.ArgumentParser(prog='Plot PDV output',
                                 description=descr_str)

# run index
parser.add_argument('--runID', '-R',
                    action='store',
                    type=str,
                    default='runID',
                    help='Run identifier.')

# indir
parser.add_argument('--indir', '-D',
                    action='store',
                    type=str,
                    default='./',
                    help='Directory to find NPZ files.')

# outdir
parser.add_argument('--outdir', '-O',
                    action='store',
                    type=str,
                    default='./',
                    help='Directory to output images to.')

parser.add_argument('--save', '-S',
                    action='store_true',
                    help='Flag to save image.')


if __name__ == '__main__':
    # Parse commandline arguments
    args_ns = parser.parse_args()

    # Assign command-line arguments
    indir = args_ns.indir
    outdir = args_ns.outdir
    runID = args_ns.runID
    SAVEFIG = args_ns.save

    # Read in PDV file, e.g. nc231213_Sn_id0001_pdv.npz
    filename = os.path.join(indir,
                            f'{runID}_pdv.npz')

    pdv_data = np.load(filename)

    time = pdv_data['time']
    origin = pdv_data['origin']
    direction = pdv_data['direction']
    distance = pdv_data['distance']
    velocity = pdv_data['velocity']

    print('PDV time shape:', time.shape)
    print('PDV origin shape:', origin.shape)
    print('PDV direction shape:', direction.shape)
    print('PDV dist shape:', distance.shape)
    print('PDV vel shape:', velocity.shape)

    fig1, ax1 = plt.subplots(1, 1, figsize=(16, 16))
    for channel in range(time.shape[0]):
        plt.plot(time[channel],
                 velocity[channel],
                 # '-k',
                 linewidth=1.7,
                 label=f'r={origin[channel][0]}, z={origin[channel][2]}')

    plt.xlim([time[0, 0], time[0, -1]])
    plt.xlabel(r'Time ($\mu$s)', fontsize=18)
    plt.ylabel(r'Velocity ($\frac{cm}{\mu s}$)', fontsize=18)
    plt.tick_params(axis='both', labelsize=18)
    plt.legend()

    if SAVEFIG:
        # Save figure
        fig1.savefig(f'{runID}_PDV.png',
                     bbox_inches='tight')
    else:
        plt.show()
