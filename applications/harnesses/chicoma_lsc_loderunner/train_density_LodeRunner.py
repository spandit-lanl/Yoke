"""Initial training setup for LodeRunner on LSC material densities.

This version of training uses only the lsc240420 data with only per-material
density along with the velocity field. A single timestep is input, a single
timestep is predicted. The number of input variables is fixed throughout
training.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import torch
import torch.nn as nn

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
import yoke.torch_training_utils as tr

#############################################
# Inputs
#############################################
descr_str = (
    "Trains LodeRunner architecture on single-timstep input and output of the "
    "lsc240420 per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="Initial LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)

#############################################
# Learning Problem
#############################################
parser.add_argument(
    "--studyIDX",
    action="store",
    type=int,
    default=1,
    help="Study ID number to match hyperparameters",
)

#############################################
# File Paths
#############################################
parser.add_argument(
    "--FILELIST_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../filelists/"),
    help="Directory where filelists are located.",
)

parser.add_argument(
    "--LSC_NPZ_DIR",
    action="store",
    type=str,
    default=os.path.join(os.path.dirname(__file__), "../../../data_examples/lsc240420/"),
    help="Directory in which LSC *.npz files live.",
)

parser.add_argument(
    "--train_filelist",
    action="store",
    type=str,
    default="lsc240420_prefixes_train_80pct.txt",
    help="Path to list of files to train on.",
)

parser.add_argument(
    "--validation_filelist",
    action="store",
    type=str,
    default="lsc240420_prefixes_validation_10pct.txt",
    help="Path to list of files to validate on.",
)

parser.add_argument(
    "--test_filelist",
    action="store",
    type=str,
    default="lsc240420_prefixes_test_10pct.txt",
    help="Path to list of files to test on.",
)

#############################################
# Model Parameters
#############################################

#############################################
# Training Parameters
#############################################
parser.add_argument(
    "--init_learnrate",
    action="store",
    type=float,
    default=1e-3,
    help="Initial learning rate",
)

parser.add_argument(
    "--LRepoch_per_step",
    action="store",
    type=float,
    default=10,
    help="Number of epochs per LR reduction.",
)

parser.add_argument(
    "--LRdecay", action="store", type=float, default=0.5, help="LR decay factor."
)

parser.add_argument(
    "--batch_size", action="store", type=int, default=64, help="Batch size"
)

parser.add_argument(
    "--num_workers",
    action="store",
    type=int,
    default=4,
    help=("Number of processes simultaneously loading batches of data. "
          "NOTE: If set too big workers will swamp memory!!")
)

#############################################
# Epoch Parameters
#############################################
parser.add_argument(
    "--total_epochs", action="store", type=int, default=10, help="Total training epochs"
)

parser.add_argument(
    "--cycle_epochs",
    action="store",
    type=int,
    default=5,
    help=(
        "Number of epochs between saving the model and re-queueing "
        "training process; must be able to be completed in the "
        "set wall time"
    ),
)

parser.add_argument(
    "--train_batches",
    action="store",
    type=int,
    default=250,
    help="Number of batches to train on in a given epoch",
)

parser.add_argument(
    "--val_batches",
    action="store",
    type=int,
    default=25,
    help="Number of batches to validate on in a given epoch",
)

parser.add_argument(
    "--TRAIN_PER_VAL",
    action="store",
    type=int,
    default=10,
    help="Number of training epochs between each validation epoch",
)

parser.add_argument(
    "--trn_rcrd_filename",
    action="store",
    type=str,
    default="./default_training.csv",
    help="Filename for text file of training loss and metrics on each batch",
)

parser.add_argument(
    "--val_rcrd_filename",
    action="store",
    type=str,
    default="./default_validation.csv",
    help="Filename for text file of validation loss and metrics on each batch",
)

parser.add_argument(
    "--continuation",
    action="store_true",
    help="Indicates if training is being continued or restarted",
)

parser.add_argument(
    "--checkpoint",
    action="store",
    type=str,
    default="None",
    help="Path to checkpoint to continue training from",
)

#############################################
#############################################
if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Study ID
    studyIDX = args.studyIDX

    # Data Paths
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    test_filelist = args.FILELIST_DIR + args.test_filelist

    # Model Parameters

    # Training Parameters
    initial_learningrate = args.init_learnrate
    LRepoch_per_step = args.LRepoch_per_step
    LRdecay = args.LRdecay
    batch_size = args.batch_size

    # Number of workers controls how batches of data are prefetched and,
    # possibly, pre-loaded onto GPUs. If the number of workers is large they
    # will swamp memory and jobs will fail.
    #
    #num_workers = int(os.environ["SLURM_JOB_CPUS_PER_NODE"])
    num_workers = args.num_workers
    train_per_val = args.TRAIN_PER_VAL

    # Epoch Parameters
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    START = not CONTINUATION
    checkpoint = args.checkpoint

    #############################################
    # Check Devices
    #############################################
    print("\n")
    print("Slurm & Device Information")
    print("=========================================")
    print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    print("Pytorch Cuda Available:", torch.cuda.is_available())
    print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    print("Number of System CPUs:", os.cpu_count())
    print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])

    print("\n")
    print("Model Training Information")
    print("=========================================")

    #############################################
    # Initialize Model
    #############################################
    model = LodeRunner(
        default_vars=['density_case',
                      'density_cushion',
                      'density_maincharge',
                      'density_outside_air',
                      'density_striker',
                      'density_throw',
                      'Uvelocity',
                      'Wvelocity'],
        image_size=(1120, 800),
        patch_size=(10, 10),
        embed_dim=128,
        emb_factor=2,
        num_heads=8,
        block_structure=(1, 1, 3, 1),  #  This should vary as in the SWIN models.
        window_sizes=[
            (8, 8),
            (8, 8),
            (4, 4),
            (2, 2),
        ],
        patch_merge_scales=[
            (2, 2),
            (2, 2),
            (2, 2),
        ],
    )

    print("Lode Runner parameters:", tr.count_torch_params(model, trainable=True))
    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.

    #############################################
    # Initialize Optimizer
    #############################################
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=initial_learningrate,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    #############################################
    # Initialize Loss
    #############################################
    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    print("Model initialized.")

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
        print("Model state loaded for continuation.")
    else:
        starting_epoch = 0

    #############################################
    # Move model and optimizer state to GPU
    #############################################
    model.to(device)

    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)

    #############################################
    # Setup LR scheduler
    #############################################
    stepLRsched = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=LRepoch_per_step,
        gamma=LRdecay,
        last_epoch=starting_epoch - 1,
    )

    # # Sanity check for architecture and inputs
    # # Scramble to see if model takes input expected
    # img_input = torch.rand(batch_size, 8, 1120, 800)
    # img_input = img_input.to(device, non_blocking=True)
    # Dt = 0.25 * torch.ones(batch_size, dtype=torch.float32)
    # Dt = Dt.to(device, non_blocking=True)
    # print('train_density_LodeRunner.py, Dt.shape:', Dt.shape)
    # # Both in_vars and out_vars correspond to indices for every variable in
    # # this training setup...
    # #
    # # in_vars = ['density_case',
    # #            'density_cushion',
    # #            'density_maincharge',
    # #            'density_outside_air',
    # #            'density_striker',
    # #            'density_throw',
    # #            'Uvelocity',
    # #            'Wvelocity']            
    # in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device)
    # out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7]).to(device)

    # pred_img = model(img_input, in_vars, out_vars, Dt)
    # print('In train_density_LodeRunner.py, pred_img.shape:', pred_img.shape)
    
    #############################################
    # Script and compile model on device
    #############################################
    # The LodeRunner model has flow-control statements in the forward method
    # and, thus, cannot be scripted.
    #scripted_model = torch.jit.script(model)

    # Model compilation has some interesting parameters to play with.
    #
    # NOTE: Compiled model is not able to be loaded from checkpoint for some
    # reason.
    # compiled_model = torch.compile(
    #     model,
    #     fullgraph=True,  #  If TRUE, throw error if
    #                      #  whole graph is not
    #                      #  compileable.
    #     mode="reduce-overhead",
    # )
    
    #############################################
    # Initialize Data
    #############################################
    train_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=2,  # This could be a variable.
        max_file_checks=10,
    )
    val_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=2,  # This could be a variable.
        max_file_checks=10,
    )
    test_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=test_filelist,
        max_timeIDX_offset=2,  # This could be a variable.
        max_file_checks=10,
    )

    print("Datasets initialized...")

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    # Setup Dataloaders
    train_dataloader = tr.make_dataloader(
        train_dataset, batch_size, train_batches, num_workers=num_workers
    )
    val_dataloader = tr.make_dataloader(
        val_dataset, batch_size, val_batches, num_workers=num_workers
    )
    print("DataLoaders initialized...")
    
    for epochIDX in range(starting_epoch, ending_epoch):
        # Time each epoch and print to stdout
        startTime = time.time()

        # Train an Epoch
        tr.train_simple_loderunner_epoch(
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename,
            device=device,
        )

        # Increment LR scheduler
        stepLRsched.step()

        endTime = time.time()
        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        print("Completed epoch " + str(epochIDX) + "...")
        print("Epoch time (minutes):", epoch_time)

    # Save Model Checkpoint
    print("Saving model checkpoint at end of epoch " + str(epochIDX) + ". . .")

    # Move the model back to CPU prior to saving to increase portability
    model.to("cpu")
    # Move optimizer state back to CPU
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to("cpu")

    # Save model and optimizer state in hdf5
    h5_name_str = "study{0:03d}_modelState_epoch{1:04d}.hdf5"
    new_h5_path = os.path.join("./", h5_name_str.format(studyIDX, epochIDX))
    tr.save_model_and_optimizer_hdf5(
        model, optimizer, epochIDX, new_h5_path, compiled=False
    )

    #############################################
    # Continue if Necessary
    #############################################
    FINISHED_TRAINING = epochIDX + 1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(
            new_h5_path, studyIDX, last_epoch=epochIDX
        )
        os.system(f"sbatch {new_slurm_file}")

    ###########################################################################
    # For array prediction, especially large array prediction, the network is
    # not evaluated on the test set after training. This is performed using
    # the *evaluation* module as a separate post-analysis step.
    ###########################################################################
