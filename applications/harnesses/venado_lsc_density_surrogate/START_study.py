"""Script to start training a study."""

####################################
# Packages
####################################
import os
import argparse
import pandas as pd

from src.yoke.helpers import strings


####################################
# Process Hyperparameters
####################################
# .csv argparse argument
descr_str = "Starts execution of Nested Cylinder CNN training"
parser = argparse.ArgumentParser(prog="NC-CNN START", description=descr_str)
parser.add_argument(
    "--csv",
    action="store",
    type=str,
    default="./hyperparameters.csv",
    help="CSV file containing study hyperparameters",
)
args = parser.parse_args()

training_input_tmpl = "./training_input.tmpl"
training_slurm_tmpl = "./training_slurm.tmpl"
training_START_input = "./training_START.input"
training_START_slurm = "./training_START.slurm"

# Process Hyperparmaeters File
studyDF = pd.read_csv(
    args.csv, sep=",", header=0, index_col=0, comment="#", engine="python"
)
varnames = studyDF.columns.values
idxlist = studyDF.index.values

# Save Hyperparameters to list of dictionaries
studylist = []
for i in idxlist:
    studydict = {}
    studydict["studyIDX"] = int(i)

    for var in varnames:
        studydict[var] = studyDF.loc[i, var]

    studylist.append(studydict)

####################################
# Run Studies
####################################
# Iterate Through Dictionary List to Run Studies
for k, study in enumerate(studylist):
    # Make Study Directory
    studydirname = "study_{:03d}".format(study["studyIDX"])

    if not os.path.exists(studydirname):
        os.makedirs(studydirname)

    # Make new training_input.tmpl file
    with open(training_input_tmpl) as f:
        training_input_data = f.read()

    training_input_data = strings.replace_keys(study, training_input_data)
    training_input_filepath = os.path.join(studydirname, "training_input.tmpl")

    with open(training_input_filepath, "w") as f:
        f.write(training_input_data)

    # Make new training_slurm.tmpl file
    with open(training_slurm_tmpl) as f:
        training_slurm_data = f.read()

    training_slurm_data = strings.replace_keys(study, training_slurm_data)
    training_slurm_filepath = os.path.join(studydirname, "training_slurm.tmpl")

    with open(training_slurm_filepath, "w") as f:
        f.write(training_slurm_data)

    # Make new training_START.input file
    with open(training_START_input) as f:
        START_input_data = f.read()

    START_input_data = strings.replace_keys(study, START_input_data)
    START_input_name = "study{:03d}_START.input".format(study["studyIDX"])
    START_input_filepath = os.path.join(studydirname, START_input_name)

    with open(START_input_filepath, "w") as f:
        f.write(START_input_data)

    # Make a new training_START.slurm file
    with open(training_START_slurm) as f:
        START_slurm_data = f.read()

    START_slurm_data = strings.replace_keys(study, START_slurm_data)
    START_slurm_name = "study{:03d}_START.slurm".format(study["studyIDX"])
    START_slurm_filepath = os.path.join(studydirname, START_slurm_name)

    with open(START_slurm_filepath, "w") as f:
        f.write(START_slurm_data)

    # Submit Job
    os.system(f"cd {studydirname}; sbatch {START_slurm_name}; cd ..")
