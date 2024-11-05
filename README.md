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

Module is divided into submodules, installed in a python environment:

- torch_training_utils.py
- datasets
- models
- metrics

Helper utilities and examples:

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

The python environment is specified through a Anaconda *environment
files*. There are two in this repo...

- `sample_environment.yml`
- `osx_environment.yml`

On OSX you should be able to edit the `osx_environment.yml` to replace
the `<YYMMDD>` token with the relevant date. Then...

```
>> conda env create -f osx_environment.yml
>> conda activate yoke_OSX_<YYMMDD>
```

For **developers**, you can install a **development version** of your
`yoke` checkout using...

```
>> flit install --user --symlink
```

For **non-developers**, you can install `yoke` in your own environment
using...

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

Linting:
--------

The `ruff` linter is used in `Yoke` to enforce coding and formatting
standards. To run the linter do

```
>> ruff check
>> ruff check --preview
```

You can make `ruff` fix automatic standards using

```
>> ruff check fix
```

Use `ruff` to then check your code formatting and show you what would
be adjusted, then fix formatting

```
>> ruff format --check --diff
>> ruff format
```


Create Environment Specification:
---------------------------------

Conda environment file was created using

```
>> conda env export > environment.yml
```

and then removing the final hashes from each package specification.