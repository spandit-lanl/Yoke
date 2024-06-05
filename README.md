YOKE: Yielding Optimal Knowledge Enhancement
============================================

![Get YOKEd!](./YOKE_DALLE_512x512.png)

A general prototyping, training, and testing harness for pytorch used
for models developed under the ASC-PEM-EADA project.

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

**First step is to edit `yoke_env_setup.sh` to work with your current
  environment and then source that script.**
