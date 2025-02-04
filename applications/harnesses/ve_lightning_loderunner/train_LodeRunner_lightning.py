"""Training for Lightning-wrapped LodeRunner on LSC material densities.

This version of training uses only the lsc240420 data with only per-material
density along with the velocity field. A single timestep is input, a single
timestep is predicted. The number of input variables is fixed throughout
training.

`lightning` is used to train a LightningModule wrapper for LodeRunner to allow
multi-node, multi-GPU, distributed data-parallel training.

"""

#############################################
# Packages
#############################################
import os
import time
import argparse
import torch
import torch.nn as nn
import lightning.pytorch as L

from yoke.models.vit.swin.bomberman import LodeRunner, Lightning_LodeRunner
from yoke.datasets.lsc_dataset import LSC_rho2rho_temporal_DataSet
import yoke.torch_training_utils as tr
from yoke.parallel_utils import LodeRunner_DataParallel
from yoke.lr_schedulers import CosineWithWarmupScheduler
from yoke.helpers import cli


#############################################
# Inputs
#############################################
descr_str = (
    "Trains lightning-wrapped LodeRunner on single-timstep input and output of the "
    "lsc240420 per-material density fields."
)
parser = argparse.ArgumentParser(
    prog="Initial LodeRunner Training", description=descr_str, fromfile_prefix_chars="@"
)
parser = cli.add_default_args(parser=parser)
parser = cli.add_filepath_args(parser=parser)
parser = cli.add_computing_args(parser=parser)
parser = cli.add_model_args(parser=parser)
parser = cli.add_training_args(parser=parser)
parser = cli.add_cosine_lr_scheduler_args(parser=parser)

# LSC experiment paths and files:
parser.add_argument(
    "--LSC_NPZ_DIR",
    action="store",
    type=str,
    default=os.path.join(
        os.path.dirname(__file__), "../../../data_examples/lsc240420/"
    ),
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
#############################################
if __name__ == "__main__":
    #############################################
    # Process Inputs
    #############################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    prefetch_factor = args.prefetch_factor

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
        default_vars=[
            "density_case",
            "density_cushion",
            "density_maincharge",
            "density_outside_air",
            "density_striker",
            "density_throw",
            "Uvelocity",
            "Wvelocity",
        ],
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

    print("Lode Runner parameters:", tr.count_torch_params(model, trainable=True))
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

    print("Model initialized...")

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

    print("Datasets initialized...")

    #############################################
    # Training Loop
    #############################################
    # Train Model
    print("Training Model . . .")

    # Setup Dataloaders
    train_dataloader = tr.make_dataloader(
        train_dataset,
        batch_size,
        train_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    val_dataloader = tr.make_dataloader(
        val_dataset,
        batch_size,
        val_batches,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )
    print("DataLoaders initialized...")

    #############################################
    # Lightning wrap
    #############################################
    # Get start_epoch from checkpoint filename
    # Format: study{studyIDX:03d}_modelState_epoch{final_epoch:04d}.hdf5
    if CONTINUATION:
        starting_epoch = checkpoint.split("epoch")[1]
        starting_epoch = int(start_epoch.split(".")[0])
        last_epoch = train_batches * (starting_epoch - 1)
    else:
        last_epoch = -1
        starting_epoch = 0

    in_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    out_vars = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7])
    L_loderunner = Lightning_LodeRunner(
        model,
        in_vars=in_vars,
        out_vars=out_vars,
        lrscheduler=CosineWithWarmupScheduler,
        scheduler_params={
            "warmup_steps": warmup_steps,
            "anchor_lr": anchor_lr,
            "terminal_steps": terminal_steps,
            "num_cycles": num_cycles,
            "min_fraction": min_fraction,
            "last_epoch": last_epoch,
        },
    )

    # Have to initialize the optimizer so custom load checkpoint works.
    optimizers = L_loderunner.configure_optimizers()
    if isinstance(optimizers, dict):
        optimizer = optimizers["optimizer"]
    else:
        optimizer = optimizers

    L_loderunner._optimizers = [optimizer]

    # Use lightning Trainer, Logger, and fit.
    if CONTINUATION:
        L_loderunner.load_h5_chkpt = checkpoint
        L_loderunner.on_load_checkpoint()
        starting_epoch = L_loderunner.current_epoch_override
    else:
        starting_epoch = 0

    logger = L.loggers.CSVLogger(
        save_dir="./",
        prefix=f"{starting_epoch:03d}_",
        flush_logs_every_n_steps=100,
    )

    cycle_epochs = min(cycle_epochs, total_epochs - starting_epoch + 1)
    final_epoch = starting_epoch + cycle_epochs - 1
    save_h5_path = f"./study{studyIDX:03d}_modelState_epoch{final_epoch:04d}.hdf5"
    L_loderunner.save_h5_chkpt = save_h5_path

    trainer = L.Trainer(
        max_epochs=final_epoch + 1,
        limit_train_batches=train_batches,
        check_val_every_n_epoch=train_per_val,
        limit_val_batches=val_batches,
        accelerator="gpu",
        devices=Ngpus,  # Number of GPUs per node
        num_nodes=Knodes,
        strategy="ddp",
        enable_progress_bar=False,
        logger=logger,
    )

    trainer.fit(
        L_loderunner,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    #############################################
    # Continue if Necessary
    #############################################
    FINISHED_TRAINING = final_epoch + 1 > total_epochs
    if not FINISHED_TRAINING:
        new_slurm_file = tr.continuation_setup(
            save_h5_path, studyIDX, last_epoch=final_epoch
        )
        os.system(f"sbatch {new_slurm_file}")

    ###########################################################################
    # For array prediction, especially large array prediction, the network is
    # not evaluated on the test set after training. This is performed using
    # the *evaluation* module as a separate post-analysis step.
    ###########################################################################
