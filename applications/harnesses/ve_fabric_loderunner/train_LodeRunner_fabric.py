"""Use Fabric to do DDP training with LodeRunner.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import torch
import torch.nn as nn
from lightning.fabric import Fabric
from lightning.pytorch.plugins.environments import SLURMEnvironment

from yoke.models.vit.swin.bomberman import LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
import yoke.torch_training_utils as tr
from yoke.parallel_utils import LodeRunner_DataParallel
from yoke.lr_schedulers import CosineWithWarmupScheduler


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
# Data Parallelism
#############################################
parser.add_argument(
    '--Ngpus',
    action="store",
    type=int,
    default=1,
    help='Number of GPUs per node.'
)

parser.add_argument(
    '--Knodes',
    action="store",
    type=int,
    default=1,
    help='Number of nodes.'
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
parser.add_argument(
    "--block_structure",
    action="store",
    type=int,
    nargs="+",
    default=[1, 1, 3, 1],
    help="List of number of SW-MSA layers in each SWIN block.",
)

parser.add_argument(
    "--embed_dim",
    action="store",
    type=int,
    default=128,
    help="Initial embedding dimension for SWIN-Unet.",
)

#############################################
# Training Parameters
#############################################
#---------------------
# Learning Rate Params
#---------------------
parser.add_argument(
    "--anchor_lr",
    action="store",
    type=float,
    default=1e-4,
    help="Learning rate at the peak of cosine scheduler.",
)

parser.add_argument(
    "--num_cycles",
    action="store",
    type=float,
    default=0.5,
    help="Learning rate at the peak of cosine scheduler.",
)

parser.add_argument(
    "--min_fraction",
    action="store",
    type=float,
    default=0.5,
    help="Fraction of anchor-LR at cosine trough.",
)

parser.add_argument(
    "--terminal_steps",
    action="store",
    type=int,
    default=1000,
    help="Number of steps for cosine to go through specified cycles.",
)

parser.add_argument(
    "--warmup_steps",
    action="store",
    type=int,
    default=500,
    help="Number of steps for scheduler to reach anchor-LR."
)

#---------------------
#---------------------

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
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Study ID
    studyIDX = args.studyIDX

    # Resources
    Ngpus = args.Ngpus
    Knodes = args.Knodes

    # Data Paths
    train_filelist = args.FILELIST_DIR + args.train_filelist
    validation_filelist = args.FILELIST_DIR + args.validation_filelist
    test_filelist = args.FILELIST_DIR + args.test_filelist

    # Model Parameters
    embed_dim = args.embed_dim
    block_structure = tuple(args.block_structure)
    
    # Training Parameters
    anchor_lr = args.anchor_lr
    num_cycles = args.num_cycles
    min_fraction = args.min_fraction
    terminal_steps = args.terminal_steps
    warmup_steps = args.warmup_steps

    # Number of workers controls how batches of data are prefetched and,
    # possibly, pre-loaded onto GPUs. If the number of workers is large they
    # will swamp memory and jobs will fail.
    num_workers = args.num_workers

    # Epoch Parameters
    batch_size = args.batch_size
    total_epochs = args.total_epochs
    cycle_epochs = args.cycle_epochs
    train_batches = args.train_batches
    val_batches = args.val_batches
    train_per_val = args.TRAIN_PER_VAL
    trn_rcrd_filename = args.trn_rcrd_filename
    val_rcrd_filename = args.val_rcrd_filename
    CONTINUATION = args.continuation
    START = not CONTINUATION
    checkpoint = args.checkpoint

    # Examine the SLURM environment a bit...
    # env = SLURMEnvironment()

    # print("SLURM detected:", env.detect())
    # print("Job name:", env.job_name())
    
    # Setup fabric
    torch.set_float32_matmul_precision('medium')  # or `high`
    fabric = Fabric(
        accelerator="gpu",
        devices=Ngpus,
        num_nodes=Knodes,
        strategy="ddp"
    )

    fabric.launch()

    #############################################
    # Check Devices
    #############################################
    fabric.print("\n")
    fabric.print("Slurm & Device Information")
    fabric.print("=========================================")
    fabric.print("Slurm Job ID:", os.environ["SLURM_JOB_ID"])
    fabric.print("Pytorch Cuda Available:", torch.cuda.is_available())
    fabric.print("GPU ID:", os.environ["SLURM_JOB_GPUS"])
    fabric.print("Number of System CPUs:", os.cpu_count())
    fabric.print("Number of CPUs per GPU:", os.environ["SLURM_JOB_CPUS_PER_NODE"])

    fabric.print("\n")
    fabric.print("Model Training Information")
    fabric.print("=========================================")

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
        image_size=(1120, 400),
        patch_size=(10, 5),  # Since using half-image, halve patch size.
        embed_dim=embed_dim,
        emb_factor=2,
        num_heads=8,
        block_structure=block_structure,
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

    fabric.print("Lode Runner parameters:", tr.count_torch_params(model, trainable=True))
    # Wait to move model to GPU until after the checkpoint load. Then
    # explicitly move model and optimizer state to GPU.

    #############################################
    # Initialize Optimizer
    #############################################
    # Using LR scheduler so optimizer LR is fixed and small.
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-6,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=0.01,
    )

    #############################################
    # Initialize Loss
    #############################################
    # Use `reduction='none'` so loss on each sample in batch can be recorded.
    loss_fn = nn.MSELoss(reduction="none")

    fabric.print("Model initialized.")

    #############################################
    # Load Model for Continuation
    #############################################
    if CONTINUATION and fabric.global_rank == 0:
        starting_epoch = tr.load_model_and_optimizer_hdf5(model, optimizer, checkpoint)
        fabric.print("Model state loaded for continuation.")
    else:
        starting_epoch = 0

    #############################################
    # LR scheduler
    #############################################
    # We will take a scheduler step every back-prop step so the number of steps
    # is the number of previous batches.
    if starting_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = train_batches * (starting_epoch - 1)
    LRsched = CosineWithWarmupScheduler(
        optimizer,
        anchor_lr=anchor_lr,
        terminal_steps=terminal_steps,
        warmup_steps=warmup_steps,
        num_cycles=num_cycles,
        min_fraction=min_fraction,
        last_epoch=last_epoch,
    )

    #############################################
    # Setup Fabric
    #############################################
    model, optimizer = fabric.setup(model, optimizer)
    
    #############################################
    # Initialize Data
    #############################################
    train_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=train_filelist,
        max_timeIDX_offset=2,  # This could be a variable.
        max_file_checks=10,
        half_image=True,
    )
    val_dataset = LSC_rho2rho_temporal_DataSet(
        args.LSC_NPZ_DIR,
        file_prefix_list=validation_filelist,
        max_timeIDX_offset=2,  # This could be a variable.
        max_file_checks=10,
        half_image=True,
    )

    fabric.print("Datasets initialized...")

    #############################################
    # Training Loop
    #############################################
    # Train Model
    fabric.print("Training Model . . .")
    starting_epoch += 1
    ending_epoch = min(starting_epoch + cycle_epochs, total_epochs + 1)

    # Setup Dataloaders
    train_dataloader = tr.make_dataloader(
        train_dataset,
        batch_size,
        train_batches,
        num_workers=num_workers,
        prefetch_factor=2
    )
    train_dataloader = fabric.setup_dataloaders(train_dataloader)
    val_dataloader = tr.make_dataloader(
        val_dataset,
        batch_size,
        val_batches,
        num_workers=num_workers,
        prefetch_factor=2
    )
    val_dataloader = fabric.setup_dataloaders(val_dataloader)
    fabric.print("DataLoaders initialized...")

    TIME_EPOCH = True
    for epochIDX in range(starting_epoch, ending_epoch):
        # For timing epochs
        if TIME_EPOCH:
            # Synchronize before starting the timer
            fabric.barrier()  # Ensure that all nodes sync
            torch.cuda.synchronize()  # Ensure GPUs on each node sync
            # Time each epoch and print to stdout
            startTime = time.time()

        # Train an Epoch
        tr.train_fabric_loderunner_epoch(
            fabric,
            training_data=train_dataloader,
            validation_data=val_dataloader,
            model=model,
            optimizer=optimizer,
            loss_fn=loss_fn,
            LRsched=LRsched,
            epochIDX=epochIDX,
            train_per_val=train_per_val,
            train_rcrd_filename=trn_rcrd_filename,
            val_rcrd_filename=val_rcrd_filename
        )

        if TIME_EPOCH:
            # Synchronize before starting the timer
            torch.cuda.synchronize()  # Ensure GPUs on each node sync
            fabric.barrier()  # Ensure that all nodes sync
            # Time each epoch and print to stdout
            endTime = time.time()

        epoch_time = (endTime - startTime) / 60

        # Print Summary Results
        fabric.print(f"Completed epoch {epochIDX}...")
        fabric.print(f"Epoch time (minutes): {epoch_time:.2f}")

        # Clear GPU memory after each epoch
        torch.cuda.empty_cache()

    # Save Model Checkpoint
    # Move the model back to CPU prior to saving to increase portability
    fabric.print(f"Saving model checkpoint at end of epoch {epochIDX}...")
    if fabric.global_rank == 0:
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
            model.module, optimizer, epochIDX, new_h5_path, compiled=False
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
