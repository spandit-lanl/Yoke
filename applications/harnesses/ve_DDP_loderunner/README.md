LodeRunner Training - DDP - Venado
==================================

An example setup training LodeRunner using PyTorch
`DistributedDataParallel`.

The training system works currently within limitations but seems more
stable than `lightning.fabric`.

1. It seems that beyond 4 Nodes communication conflicts arise
intermittently.

```
RuntimeError: CUDA error: uncorrectable ECC error encountered

CUDA kernel errors might be asynchronously reported at some other API
call, so the stacktrace below might be incorrect.

For debugging consider passing CUDA_LAUNCH_BLOCKING=1.

Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
```

2. With 8 `lsc240420` fields each GPU can only fit 5 samples at a
time.

