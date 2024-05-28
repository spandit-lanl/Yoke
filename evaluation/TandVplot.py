"""This script reads the training and validation record files and plots the
network learning curves. The data files are CSV files which are read with
Pandas and only contain metric evaluation information.

NOTE: For each network evaluation this script should be submitted using a
job-submission script.

"""

import os, sys, argparse
import glob
import numpy as np
import pandas as pd

# Imports for plotting
# To view possible matplotlib backends use
# >>> import matplotlib
# >>> bklist = matplotlib.rcsetup.interactive_bk
# >>> print(bklist)
import matplotlib
#matplotlib.use('MacOSX')
#matplotlib.use('pdf')
# Get rid of type 3 fonts in figures
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
# Ensure LaTeX font
font = {'family': 'serif'}
plt.rc('font', **font)
plt.rcParams['figure.figsize'] = (6, 6)
from mpl_toolkits.axes_grid1 import make_axes_locatable


###################################################################
# Define command line argument parser
descr_str = ('Plot learning curves.')
parser = argparse.ArgumentParser(prog='Training and validation curves.',
                                 description=descr_str,
                                 fromfile_prefix_chars='@')

parser.add_argument('--basedir',
                    action='store',
                    type=str,
                    default='./study_directory',
                    help='Directory to look for studies.')

# parser.add_argument('--runtype', '-T',
#                     action='store',
#                     type=str,
#                     default='pRad',
#                     help='Runtype to observe normalization for.')

parser.add_argument('--IDX', '-I',
                    action='store',
                    type=int,
                    default=0,
                    help='Index of study to plot curves for.')

parser.add_argument('--savedir',
                    action='store',
                    type=str,
                    default='./',
                    help='Directory for saving images.')

parser.add_argument('--savefig', '-S',
                    action='store_true',
                    help='Flag to save figures.')

args_ns = parser.parse_args()

basedir = args_ns.basedir
#runtype = args_ns.runtype
IDX = args_ns.IDX
savedir = args_ns.savedir
SAVEFIG = args_ns.savefig

# Number of batches to include in each plot point
Nbatch_per_pt = 2000

# Read the data using Pandas
# Of the form: training_study009_epoch0100.csv
trn_glob = f'{basedir}/study_{IDX:03d}/training_study{IDX:03d}_epoch*.csv'
trn_csv_list = sorted(glob.glob(trn_glob))
trn_file_epochs = []
for Tcsv in trn_csv_list:
    epoch = Tcsv.split('epoch')[1]
    epoch = int(epoch.split('.')[0])
    trn_file_epochs.append(epoch)

# Of the form: validation_study009_epoch0010.csv
val_glob = f'{basedir}/study_{IDX:03d}/validation_study{IDX:03d}_epoch*.csv'
val_csv_list = sorted(glob.glob(val_glob))
val_file_epochs = []
for Vcsv in val_csv_list:
    epoch = Vcsv.split('epoch')[1]
    epoch = int(epoch.split('.')[0])
    val_file_epochs.append(epoch)

# val_DF = pd.read_csv(val_csvfile,
#                      sep=', ',
#                      header=None,
#                      names=['Epoch',
#                             'Step',
#                             'Loss'],
#                      engine='python')

# # Retrieve indexes
# trn_idxlist = trn_DF.index.values
# val_idxlist = val_DF.index.values

# Plot loss for training over all steps and epochs
fig1 = plt.figure(num=1, figsize=(6, 6))
ax = plt.gca()
# Nepochs = min(trn_DF['Epoch'].max(), val_DF['Epoch'].max())
# startIDX = 0
vIDX = 0
startIDX = 0
for tIDX, Tcsv in enumerate(trn_csv_list):
    ptIDX_list = []
    trn_DF = pd.read_csv(Tcsv,
                         sep=', ',
                         header=None,
                         names=['Epoch',
                                'Batch',
                                'Loss'],
                         engine='python')
    #print('Train IDX:', trn_file_epochs[tIDX])

    trn_epoch_loss = trn_DF.loc[:, 'Loss'].values.reshape((Nbatch_per_pt, -1))
    #print('Loss array shape:', trn_epoch_loss.shape)

    trn_positions = np.arange(trn_epoch_loss.shape[1])
    plt.boxplot(trn_epoch_loss,
                positions=startIDX+trn_positions,
                showfliers=False)

    startIDX += trn_positions[-1]
    
    # if trn_file_epochs[tIDX] == val_file_epochs[vIDX]:
    #     val_DF = pd.read_csv(val_csv_list[vIDX],
    #                          sep=', ',
    #                          header=None,
    #                          names=['Epoch',
    #                                 'Batch',
    #                                 'Loss'],
    #                          engine='python')
    #     print('Validation IDX:', val_file_epochs[vIDX])
        
    #     vIDX += 1
        
    # Training
#    trn_epoch = trn_DF[trn_DF['Epoch'] == k+1]
#     trn_epoch_loss = trn_epoch.loc[:, 'Loss']
#     trn_epoch_steps = trn_epoch.loc[:, 'Step'].values

#     if k == 0:
#         plt.plot(startIDX+trn_epoch_steps, trn_epoch_loss, '-b', label='Training')
#     else:
#         plt.plot(startIDX+trn_epoch_steps, trn_epoch_loss, '-b')

#     startIDX += trn_epoch_steps.max()

#     # Validation
#     val_epoch = val_DF[val_DF['Epoch'] == k+1]
#     val_epoch_mse = val_epoch.loc[:, 'MSE']
#     val_epoch_steps = val_epoch.loc[:, 'Step'].values

#     if k == 0:
#         plt.plot(startIDX+val_epoch_steps, val_epoch_mse, '-r', label='Validation')
#     else:
#         plt.plot(startIDX+val_epoch_steps, val_epoch_mse, '-r')

#     startIDX += val_epoch_steps.max()

# # Make legend
# plt.legend(fontsize=16)

# No xlim
ax.set_ylim(0.0, 0.03)

# Set axis labels
ax.set_ylabel('Loss', fontsize=16)
ax.set_xlabel('Evaluation Index', fontsize=16)

# Save or plot images
if SAVEFIG:
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    plt.figure(fig1.number)
    filenameA = f'{savedir}/study{IDX:03d}_TandV_curve.png'
    plt.savefig(filenameA, bbox_inches='tight')
else:
    plt.show()
