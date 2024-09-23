YOKE: Yielding Optimal Knowledge Enhancement
============================================

![Get YOKEd!](./YOKE_DALLE_512x512.png)


About:
------

A general prototyping, training, and testing harness for pytorch used
for models developed under the **ArtIMis: Multi-physics/Multi-material
Applications** and **ASC-PEM-EADA(Enabling Agile Design and Assessment)**
projects.

Module is divided into submodules, installed in a python environment:

- torch_training_utils.py
- datasets
- models
- metrics

Helper utilities and examples:

- harnesses
- filelists
- evaluation
- viewers

Data for training is not housed within YOKE, data locations are
specified through command-line arguments passed to the programs in
`harnesses`, `evaluation`, and `viewers`.


Installation:
-------------

The python environment is specified through an Anaconda
`environment.yml` file included in this repo. **Make sure to edit the
`environment.yml` file appropriately. You can then build a copy of a
working environment and activate it using...

```
>> conda env create -f environment.yml
>> conda activate yoke_<operating_system>_<YYMMDD>
```

For **developers**, you can install a **development version** of your
`yoke` checkout using...

```
>> flit install --user --symlink
```

For **non-developers**, you can install `yoke` using...

```
>> flit install
```

Testing:
--------

To run the tests use...

```
>> pytest
>> pytest --cov
>> pytest --cov --cov-report term-missing
```

[DEPRECATED] **yoke_env_setup.sh**
--------------------------------

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
