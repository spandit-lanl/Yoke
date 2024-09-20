#!/bin/bash

# NOTE: Number of CPUs per GPU seems to be one more than ntasks-per-gpu

#SBATCH --job-name=NC_CNN_study<studyIDX>
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-gpu=8
#SBATCH --gpu-bind=closest
#SBATCH --gres-flags=enforce-binding
#SBATCH --mem-bind=local
#SBATCH --time=10:00:00
#SBATCH --output=study<studyIDX>_epoch0001.out
#SBATCH --error=study<studyIDX>_epoch0001.err
#SBATCH -vvv

# Check available GPUs
sinfo  -o "%P %.24G %N"
srun /usr/bin/echo $CUDA_AVAILABLE_DEVICES

# Load correct conda environment
source $YOKE_CONDA/bin/activate $YOKE_TORCH

# Get start time
export date00=`date`

# Start the Code
python ../train_nc_density_CNN.py @study<studyIDX>_START.input

# Get end time and print to stdout
export date01=`date`

echo "===================TIME STARTED==================="
echo $date00
echo "===================TIME FINISHED==================="
echo $date01
