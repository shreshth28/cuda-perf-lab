# cuda-perf-lab

Small CUDA programs I wrote while learning how to make GPU code fast. Each one has a few versions of the same kernel, from the obvious one to something better, plus the cuBLAS equivalent so there's a real baseline to compare against.

| program | kernels | cuBLAS baseline |
|---|---|---|
| vecadd | naive, stride, vec4 | `cublasSaxpy` |
| reduce | atomic, tree, shfl, multi | `cublasSasum` |
| transpose | naive, tiled, tiled_pad | `cublasSgeam` |
| matmul | naive, tiled, reg1d, reg2d, vec4 | `cublasSgemm` |
| matmul_cpu | naive, ikj, omp | - |

## Build

```
make            # ARCH=sm_86 by default (3070 Ti), change it for your card
make ARCH=sm_80
```

Needs the CUDA toolkit (nvcc + cuBLAS) and a compiler with OpenMP for `matmul_cpu`.

## Run

Every program takes the same args:

```
./bin/matmul --kernel reg2d --n 2048 --iters 20 --warmup 3
kernel=reg2d n=2048 time_ms=... gflops=... ok=1
```

- `time_ms` is just the kernel, timed with cuda events and averaged over `--iters` after `--warmup` runs. Setup, copies and the correctness check aren't in it.
- `ok=1` means the output matched a CPU reference (or cuBLAS, for big matmuls where the CPU takes forever). `--no-check` skips it.
- vecadd/reduce/transpose report `gbps`, matmul reports `gflops`.
- reg1d needs n to be a multiple of 64, reg2d and vec4 a multiple of 128.

## Benchmarking

`bench/` has a spec per program for [ansibench](https://github.com/shreshth28/ansibench), which builds everything on the GPU box over ssh, runs every kernel at a few sizes with repeats and warmups, and gives back a CSV and a summary table:

```
ansibench run bench/matmul.yml -i inventory.ini
```

The specs use `timing: from: output`, so the numbers are the `time_ms` each program prints and not the process wall time (which is mostly CUDA starting up). Point `hosts:` at your own machine, or pass `-l`.
