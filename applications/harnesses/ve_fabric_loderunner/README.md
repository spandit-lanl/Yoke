LodeRunner Training - `lightning.fabric - Venado
================================================

An example setup training LodeRunner using `lightning.fabric`.

The training system works currently but is unstable on Venado beyond 3
Nodes. Currently, it seems using PyTorch's built-in
`DistributedDataParallel` is preferrable.

It seems that beyond 3 Nodes communication conflicts arise.

```
RuntimeError: CUDA error: uncorrectable ECC error encountered

CUDA kernel errors might be asynchronously reported at some other API
call, so the stacktrace below might be incorrect.

For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

