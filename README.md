YOKE: Yielding Optimal Knowledge Enhancement
============================================


[![pipeline status](https://gitlab.lanl.gov/multiphysmultimatapps/yoke/badges/main/pipeline.svg)](https://gitlab.lanl.gov/multiphysmultimatapps/yoke/-/commits/main) 
 [![coverage report](https://gitlab.lanl.gov/multiphysmultimatapps/yoke/badges/main/coverage.svg)](https://gitlab.lanl.gov/multiphysmultimatapps/yoke/-/commits/main) 
  [![Latest Release](https://gitlab.lanl.gov/multiphysmultimatapps/yoke/-/badges/release.svg)](https://gitlab.lanl.gov/multiphysmultimatapps/yoke/-/releases) 

![Get YOKEd!](./YOKE_DALLE_512x512.png)


About:
------

A general prototyping, training, and testing harness for pytorch used
for models developed under the **ArtIMis: Multi-physics/Multi-material
Applications** and **ASC-PEM-EADA(Enabling Agile Design and Assessment)**
projects.

The YOKE module is divided into submodules, installed in a python environment:

- datasets/
- models/
- metrics/
- torch_training_utils.py
- lr_schedulers.py
- parallel_utils.py

Helper utilities and examples are under `applications`:

- harnesses
- makefilelists.py
- filelists
- normalization
- evaluation
- viewers

NOTE: Data for training is not housed within YOKE. The data locations are
specified through command-line arguments passed to the programs in
`harnesses`, `evaluation`, and `viewers`.


Installation:
-------------

The python environment is specified through the `pyproject.toml`
file. YOKE is meant to be installed using `flit` in a minimal python
environment.

Setup your base environment and activate it (we use conda):

```
>> conda create -n <yoke_env_name> python=3.9 flit
>> conda activate <yoke_env_name>
```

> **WARNING!!**
>
> Now, at least for some environments `flit`, the install manager for
> YOKE will not default to installing in the conda environment. To
> remedy this first checkout your `USER_BASE` and `USER_SITE` variables
> using
>
> ```
> >> python -m site
> ```
>
> If `USER_BASE` and `USER_SITE` don't appear to be associated with
> `<yoke_env_name>` then set the `PYTHONUSERBASE` environment variable
> prior to installing YOKE:
>
> ```
> >> export PYTHONUSERBASE=$CONDA_PREFIX
> ```
>
> Rerun `python -m site` to ensure `USER_BASE` and `USER_SITE` have
> changed.

For **developers**, you can install a **development version** of
`YOKE` checkout using...

```
>> flit install --user --symlink --deps=all
```

For **non-developers**, you can install `YOKE` using...

```
>> flit install --deps=all
```

> **WARNING**
> 
> This install process does not guarantee that PyTorch is installed to
> utilize your GPUs. If you want to ensure that PyTorch is installed to
> make optimal use of your hardware we suggest manually installing
> `torch` prior to installing `YOKE` with `flit`.

Testing:
--------

To run the tests use...

```
>> pytest
>> pytest --cov
>> pytest --cov --cov-report term-missing
```

Linting:
--------

The `ruff` linter is used in `YOKE` to enforce coding and formatting
standards. To run the linter do

```
>> ruff check
>> ruff check --preview
```

You can make `ruff` fix automatic standards using

```
>> ruff check --fix
>> ruff check --preview --fix
```

Use `ruff` to then check your code formatting and show you what would
be adjusted, then fix formatting

```
>> ruff format --check --diff
>> ruff format
```
