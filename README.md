# GPU Napkin Math

Use a handful of measured ceilings to decide whether a GPU workload is limited
by compute, memory, transfers, communication, launch overhead, or capacity. The
equations on this page apply to every GPU; the numbers do not.

## Pick a GPU profile

| Profile | Architecture | Modal request | Status |
| --- | --- | --- | --- |
| [H100](H100.md) | Hopper | `H100!` | Calibrated on Modal |
| [B200](B200.md) | Blackwell | `B200` | Vendor-ceiling planning profile |
| [B300](B300.md) | Blackwell Ultra | `B300` | Vendor-ceiling planning profile |

Modal also offers `H200`. It is a Hopper GPU with more and faster memory than
H100, not the predecessor of B300. There is no `H300` request on Modal. For
benchmarking H100 specifically, use `H100!`: an unpinned `H100` request may be
upgraded to H200. `B200+` similarly opts into either B200 or B300.

## The metrics that matter on any GPU

| Metric | Symbol | Why it matters |
| --- | ---: | --- |
| Usable device-memory capacity | `C` | Whether the workload fits at all |
| Achievable compute throughput at the actual dtype | `F` | Compute-time floor |
| Achievable HBM bandwidth | `M` | Device-memory-time floor |
| Host-to-device and device-to-host bandwidth | `P_h2d`, `P_d2h` | Input and output transfer floors |
| GPU-to-GPU bandwidth | `P_g2g` | Tensor/pipeline parallel communication floor |
| Kernel-launch latency | `L` | Cost of many small kernels |

Keep the qualifiers attached to every number: exact SKU and form factor,
precision, dense versus sparse, theoretical versus achieved, direction,
topology, and software stack. No throughput number is universal across dtypes,
layouts, shapes, kernels, or systems.

## The reusable arithmetic

For a workload with `W` FLOPs and `Q` bytes of HBM traffic:

```text
arithmetic intensity = W / Q                         FLOP/byte
ridge point          = F / M                         FLOP/byte
compute time         = W / F
memory time          = Q / M
device time          = max(compute time, memory time)
```

The unit shortcut is:

```text
ridge point = compute TFLOP/s × 1,000 / memory GB/s
```

If arithmetic intensity is below the ridge point, the roofline model predicts
a memory-bound kernel. If it is above the ridge point, it predicts a
compute-bound kernel.

Add data movement and launch overhead when they are on the critical path:

```text
host copy time   = host bytes / host bandwidth
H2D time         = H2D bytes / P_h2d
D2H time         = D2H bytes / P_d2h
GPU-to-GPU time  = communicated bytes / P_g2g
launch time      = kernel launches × L

serial estimate  = host copy + H2D + device + GPU-to-GPU + D2H + launch
```

The serial sum is conservative. When stages genuinely overlap, model the
overlapped section with `max(...)`, not a sum. Count algorithmic traffic, not
just allocation size: a copy reads and writes two buffer-sized byte streams,
while `C = A + B` reads two and writes one.

## Capacity is a separate constraint

Speed does not matter when the working set does not fit:

```text
working set = weights + activations + KV cache + temporary workspace
fits        = working set <= usable device memory
```

For replicated weights, a quick lower bound is:

```text
weight bytes = parameter count × bits per parameter / 8
```

Leave headroom for runtime state, allocator fragmentation, communication
buffers, and workspaces. Multi-GPU capacity scales only when the software
actually shards the relevant tensors.

## Estimation discipline

1. Choose the exact GPU and precision.
2. Measure or obtain ceilings with matching dense/sparse conventions.
3. Count FLOPs and bytes for the workload.
4. Compute the roofline lower bound.
5. Add transfers, communication, launches, and non-overlapped CPU work.
6. Report one or two significant digits and name the predicted bottleneck.
7. Check the estimate against a representative end-to-end run.

A napkin estimate is a lower-bound model, not a latency promise. Cache effects,
occupancy, synchronization, collectives, framework overhead, power limits, and
contention can make real execution slower.

## Calculator

The CLI implements the same arithmetic using a measured JSON profile. It
currently defaults to the checked-in H100 calibration:

```sh
python3 gpu_napkin.py \
  --dtype bf16 \
  --flops 2e14 \
  --bytes 1e12 \
  --h2d-bytes 2e9 \
  --launches 20
```

It can also include `--cpu-bytes`, `--d2h-bytes`, and
`--gpu-to-gpu-bytes`. Extra printed digits do not make the estimate more
certain.

## Website

The interactive estimator is an Astro static site deployed with Cloudflare
Workers Static Assets. Website builds run locally; Modal remains reserved for
GPU benchmarks and tests.

```sh
npm install
npm run dev
npm run build
npm run deploy
```

PostHog Product Analytics, Session Replay, and Web Analytics are initialized
when `PUBLIC_POSTHOG_KEY` and `PUBLIC_POSTHOG_HOST` are present at build time.
Copy `.env.example` to `.env` for local development and keep production values
in the deployment environment.

The interface takes visual cues from [turbopuffer](https://turbopuffer.com/):
monospace type, ruled technical panels, restrained color, and dense operational
information.

## Evidence

The [evidence guide](BENCHMARKS.md) defines the source-of-truth policy and maps
each metric to NVIDIA datasheets, NVBandwidth, NCCL Tests, Nsight Compute,
cuBLAS/cuBLASLt, or MLPerf. The checked-in [H100 report](results/h100-sxm.md)
and [JSON](results/h100-sxm.json) are prototype calibration artifacts.

Modal availability and request semantics were checked against the
[Modal GPU guide](https://modal.com/docs/guide/gpu) and with live `nvidia-smi`
allocations on 2026-08-09. Hardware ceilings come from NVIDIA's
[H100 specifications](https://www.nvidia.com/en-us/data-center/h100/) and
[HGX B200/B300 specifications](https://www.nvidia.com/en-us/data-center/hgx/).

This project is inspired by Simon Eskildsen's
[Napkin Math](https://github.com/sirupsen/napkin-math) and is MIT licensed.

## Resources

Sources behind the checked-in numbers:

- [NVIDIA H100 specifications](https://www.nvidia.com/en-us/data-center/h100/)
  and [HGX B200/B300 specifications](https://www.nvidia.com/en-us/data-center/hgx/).
  Datasheet ceilings for capacity, HBM, FLOP/s, and NVLink.
- [Modal GPU guide](https://modal.com/docs/guide/gpu). GPU request names and
  upgrade semantics such as `H100!`.
- [NVBandwidth](https://github.com/NVIDIA/nvbandwidth). Host↔device and
  device↔device bandwidth and latency across the real topology.
- [NCCL Tests](https://github.com/NVIDIA/nccl-tests). AllReduce, AllGather, and
  ReduceScatter message-size curves.
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/).
  Speed of Light, memory workload, and roofline sections for real kernels.
- [cuBLAS/cuBLASLt](https://docs.nvidia.com/cuda/cublas/). Achievable GEMM
  ceilings by dtype and shape.
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/).
  End-to-end results to check napkin predictions against.

### Napkin math for GPUs

- [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html).
  The clearest explanation of compute-, memory-, and overhead-bound work, which
  is exactly the split this repository's equations make.
- [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). A whole
  book of roofline-first napkin math for training and inference, with a
  dedicated [GPU chapter](https://jax-ml.github.io/scaling-book/gpus/) covering
  SMs, HBM, NVLink, and collectives.
- [Roofline: An Insightful Visual Performance Model](https://dl.acm.org/doi/10.1145/1498765.1498785).
  The original paper behind arithmetic intensity and ridge points.
- [Transformer Math 101](https://blog.eleuther.ai/transformer-math/). The
  `6 × parameters × tokens` training-FLOPs rule and memory accounting for
  weights, optimizer state, and activations.
- [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/).
  Back-of-the-envelope KV-cache sizing, memory- versus compute-bound decoding,
  and batch-size crossover points.
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102).
  Latency, throughput, and partitioning trade-offs for large-model inference,
  derived from first-principles cost models.
- [Efficient Large-Scale Language Model Training on GPU Clusters](https://arxiv.org/abs/2104.04473).
  Megatron-LM's tensor/pipeline/data-parallel cost model and its
  model-FLOPs-utilization arithmetic.
- [PaLM](https://arxiv.org/abs/2204.02311). The standard definition
  of model FLOPs utilization (MFU) versus hardware FLOPs utilization.
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361).
  Where the compute-per-token approximation used everywhere comes from.
- [FlashAttention](https://arxiv.org/abs/2205.14135). A worked example of
  counting HBM traffic instead of FLOPs and winning by moving fewer bytes.
- [PagedAttention (vLLM)](https://arxiv.org/abs/2309.06180). Why KV-cache
  capacity, not compute, often caps serving throughput.
- [Accelerating Generative AI with PyTorch II: GPT, Fast](https://pytorch.org/blog/accelerating-generative-ai-2/).
  Decoding as a memory-bandwidth problem: tokens per second ≈ HBM bandwidth /
  weight bytes, and how quantization and speculative decoding move that bound.

### GPU architecture and vendor guides

- [Modal GPU Glossary](https://modal.com/gpu-glossary). Short, linked
  definitions of SMs, warps, tensor cores, memory hierarchy, and the CUDA
  software stack.
- [NVIDIA GPU Performance Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)
  and [Matrix Multiplication Background](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html).
  NVIDIA's own arithmetic-intensity walkthrough plus tile and wave
  quantization effects that make GEMM shapes matter.
- [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
  and the [Blackwell Architecture Technical Overview](https://resources.nvidia.com/en-us-blackwell-architecture).
  Where the per-generation ceilings and new features come from.
- [Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/) and
  [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/).
  What changed per generation for shared memory, L2, and occupancy.
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
  and [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/).
  The reference for the execution and memory model, coalescing, and transfer
  overlap.
- [Nvidia's H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/p/nvidias-h100-funny-l2-and-tons-of-bandwidth).
  Independent microbenchmarks of H100 cache and memory latency and bandwidth.
- [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499)
  and [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826).
  Measured latencies and throughputs for memory levels and instructions, the
  GPU counterpart to Agner's instruction tables.
- [Which GPU(s) to Get for Deep Learning](https://timdettmers.com/2023/01/30/which-gpu-for-deep-learning/).
  Practical comparison of GPUs through memory bandwidth, tensor cores, and
  cache sizes.

### Kernels, profiling, and benchmarking

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/).
  Timeline profiling to find launch gaps, host stalls, and non-overlapped
  transfers before reaching for kernel-level tools.
- [How to Implement Performance Metrics in CUDA C/C++](https://developer.nvidia.com/blog/how-implement-performance-metrics-cuda-cc/).
  Timing with CUDA events and computing effective bandwidth correctly.
- [PyTorch Benchmark recipe](https://pytorch.org/tutorials/recipes/recipes/benchmark.html).
  `torch.utils.benchmark` handles warmup, synchronization, and threading so
  timings are not just asynchronous launch times.
- [Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data!](https://www.thonking.ai/p/strangely-matrix-multiplications).
  Power and clock throttling mean the input values change GEMM throughput;
  benchmark with realistic data.
- [CUDA Graphs](https://developer.nvidia.com/blog/cuda-graphs/). The standard
  fix when the `launches × L` term dominates.
- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM)
  and [Outperforming cuBLAS on H100](https://cudaforfun.substack.com/p/outperforming-cublas-on-h100-a-worklog).
  Step-by-step worklogs that show how far a kernel is from the roofline at
  each step.
- [CUTLASS](https://github.com/NVIDIA/cutlass) and [Triton](https://triton-lang.org/).
  The usual ways to reach library-level GEMM and fused-kernel performance
  without hand-written PTX.
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/) and
  [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
  (`cuobjdump`, `nvdisasm`). For inspecting what the compiler actually emitted.
- [Godbolt's compiler explorer](https://gcc.godbolt.org/#). Also compiles CUDA
  with NVCC and shows PTX and SASS next to the source.
- [GPU MODE lectures](https://github.com/gpu-mode/lectures) and
  [resource stream](https://github.com/gpu-mode/resource-stream). A large,
  curated collection of GPU programming talks, papers, and posts.

### Communication, I/O, and multi-GPU

- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/).
  Environment knobs, topology detection, and collective semantics.
- [NCCL Tests performance notes](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md).
  How `algbw` and `busbw` are defined and how to convert between them for each
  collective.
- [Machine Learning Engineering Open Book](https://github.com/stas00/ml-engineering).
  Field notes on accelerators, network, storage, debugging, and training at
  scale, with many benchmark scripts.
- [GPUDirect Storage](https://docs.nvidia.com/gpudirect-storage/). The
  storage-to-HBM path that skips the host bounce buffer.

### Shared with Napkin Math

Host-side code, data loading, and launches still run on CPUs, so much of the
original [Napkin Math](https://github.com/sirupsen/napkin-math) resource list
still applies:

- ["How to get consistent results when benchmarking on Linux?"](https://easyperf.net/blog/2019/08/02/Perf-measurement-environment-on-Linux).
  CPU affinity, turbo boost, and other knobs for reliable host-side timing.
  Pair it with locked GPU clocks for device-side timing.
- [LLVM benchmarking tips](https://www.llvm.org/docs/Benchmarking.html).
  Dedicating CPUs and disabling address space randomization.
- [Top-Down performance analysis methodology](https://easyperf.net/blog/2019/02/09/Top-Down-performance-analysis-methodology).
  Finding host-side bottlenecks, such as data loaders or Python overhead, that
  starve the GPU.
- [Agner's Instruction Tables](https://www.agner.org/optimize/instruction_tables.pdf).
  Expected throughput for CPU instructions in preprocessing and launch paths.
- [Systems Performance (book)](https://www.brendangregg.com/systems-performance-2nd-edition-book.html).
  The USE method and OS-level analysis that apply unchanged to GPU hosts.
- [io_uring](https://lwn.net/Articles/776703/). Fast asynchronous I/O for
  feeding datasets and checkpoints to the GPU.
- [How Long Does It Take To Make a Context Switch?](https://blog.tsunanet.net/2010/11/how-long-does-it-take-to-make-context.html).
  Host scheduling costs that show up between kernel launches.
- [Files are hard](https://danluu.com/file-consistency/). Required reading
  before trusting checkpoint writes.
