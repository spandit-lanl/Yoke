YOKE: Yielding Optimal Knowledge Enhancement
============================================

![Get YOKEd!](./YOKE_DALLE_512x512.png)

> [!WARNING]
> We're in the process of updating Yoke to be installable
> using Flit.  To install locally (in your personal conda
> environment)...
> 
> ```
> >> flit install --user --symlink
> ```

About:
------

A general prototyping, training, and testing harness for pytorch used
for models developed under the **ArtIMis: Multi-physics/Multi-material
Applications** and **ASC-PEM-EADA(Enabling Agile Design and Assessment)**
projects.

Module is divided into submodules:

- datasets
- models
- torch_training_utils.py
- evaluation
- harnesses
- viewers

Data for training is not housed within YOKE and the python environment
is not controlled or specified, currently, through YOKE. To aid
portability a bash script, `yoke_env_setup.sh`, has been included to
help the user set the necessary environment variables.

> [!NOTE]
> **First step is to edit `yoke_env_setup.sh` to work with your current
>  environment and then source that script.**

Testing:
--------

To run the tests use...

```
>> pytest
>> pytest --cov
>> pytest --cov --cov-report term-missing
```

**yoke_env_setup.sh**
---------------------

File exists as a way to setup a local environment without turning
`Yoke` into an installable module. Due to limited storage for per-user
python environments this is neccessary on LANL HPC/IC systems.

The script sets environment variables:

- `YOKE_DIR`: Absolute path to top level directory where this
  README.md is. Mostly used to access `filelists`.

- `YOKE_CONDA`: Absolute path to conda install. Necessary in *Slurm*
  scripts to load conda environment.

- `YOKE_TORCH`: Name of the conda environment with torch
  installed. Used during runs with a *harness*

- `LSC_DESIGN_DIR`: Absolute path to *design.txt* file for the
  **lsc240420** dataset.

- `LSC_NPZ_DIR`: Absolute path to the directory with NPZ files for the
  **lsc240420** dataset.

- `NC_DESIGN_DIR`: Absolute path to *design.txt* file for the
  **nc231213** dataset.

- `NC_NPZ_DIR`: Absolute path to the directory with NPZ files for the
  **nc231213** dataset.
