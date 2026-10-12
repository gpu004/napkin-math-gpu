# GPU Napkin Math

The goal of this project is to collect numbers and techniques to quickly
estimate GPU workload performance from first principles — how long does it
take to read 1 GB from HBM on an H100? By composing them you should be able to
answer questions like: is this kernel compute- or memory-bound, how long does
the upload take, and how many GPUs does the model need? It follows Simon
Eskildsen's [Napkin Math](https://github.com/sirupsen/napkin-math).

## Numbers

Below are numbers rounded for memorization, not faux precision.
Measured on Modal (`H100!:2`, 2026-08-08, driver 580.95.05, CUDA 12.8, cuBLAS via the
[cudarc](https://github.com/chelsea0x3b/cudarc) bindings).

**Note 1:** Compute figures are dense, not sparse. NVIDIA's headline Tensor
Core numbers assume structured sparsity; dense is half.

**Note 2:** Take the numbers with a grain of salt. GB means 10^9 bytes. For
transfers, [NVBandwidth](https://github.com/NVIDIA/nvbandwidth) is the state of
the art; for collectives, [NCCL Tests](https://github.com/NVIDIA/nccl-tests).

<!-- napkin table --gpu h100 -->
| Operation | Latency | Throughput | 1 MB | 1 GB |
| --- | ---: | ---: | ---: | ---: |
| Kernel launch (back-to-back) | 5 μs |  |  |  |
| Launch + sync round trip |  |  |  |  |
| Pinned host RAM copy |  | 70 GB/s | 10 μs | 10 ms |
| Host → device, pinned |  | 60 GB/s | 20 μs | 20 ms |
| Device → host, pinned |  | 60 GB/s | 20 μs | 20 ms |
| HBM random read (pointer chase) |  |  |  |  |
| HBM copy (read + write) |  | 3 TB/s | 300 ns | 300 μs |
| Elementwise add |  | 3 TB/s | 300 ns | 300 μs |
| GPU ↔ GPU, NVLink one-way |  | 400 GB/s | 3 μs | 3 ms |

| Dense GEMM | Throughput | 1 GFLOP | 1 TFLOP | Ridge point |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 700 TFLOP/s | 1 μs | 1 ms | 230 FLOP/B |
| bf16 | 700 TFLOP/s | 1 μs | 1 ms | 240 FLOP/B |
| tf32 | 400 TFLOP/s | 3 μs | 3 ms | 130 FLOP/B |
| fp32 | 50 TFLOP/s | 20 μs | 20 ms | 17 FLOP/B |
<!-- /napkin -->

## Other GPUs

<!-- napkin compare -->
| GPU | Memory | HBM | Dense BF16 | BF16 ridge | NVLink | Host link | Launch | Basis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A100 SXM 80GB | 80 GB | 2 TB/s | 300 TFLOP/s | 170 FLOP/B | 600 GB/s | 70 GB/s | 5 μs | measured 2026-08-08 |
| H100 SXM 80GB | 80 GB | 3 TB/s | 700 TFLOP/s | 240 FLOP/B | 900 GB/s | 70 GB/s | 5 μs | measured 2026-08-08 |
| B200 SXM 180GB | 180 GB | 8 TB/s | 2 PFLOP/s | 280 FLOP/B | 2 TB/s | — | — | datasheet |
| B300 SXM 288GB | 288 GB | 8 TB/s | 2 PFLOP/s | 280 FLOP/B | 2 TB/s | — | — | datasheet |

| GPU | HBM measured / ceiling | BF16 measured / ceiling |
| --- | ---: | ---: |
| A100 SXM 80GB | 1750 / 2039 GB/s (86%) | 289 / 312 TFLOP/s (93%) |
| H100 SXM 80GB | 3000 / 3350 GB/s (90%) | 717 / 990 TFLOP/s (73%) |
<!-- /napkin -->

Modal request semantics matter: `H100!` pins an H100 — a plain `H100` request
may be upgraded to H200 — and `B200+` explicitly allows either B200 or B300.
Rows labelled "datasheet" are vendor ceilings, not achieved throughput.

## Arithmetic

For a workload with `W` FLOPs and `Q` bytes of HBM traffic:

```text
arithmetic intensity = W / Q                          FLOP/byte
ridge point          = F / M = TFLOP/s × 1,000 / GB/s FLOP/byte
compute time         = W / F
memory time          = Q / M
device time          = max(compute time, memory time)
```

Below the ridge point the kernel is memory-bound; above it, compute-bound.
Add the stages that sit on the critical path:

```text
H2D time         = H2D bytes / P_h2d
D2H time         = D2H bytes / P_d2h
GPU-to-GPU time  = communicated bytes / P_g2g
launch time      = kernel launches × L

serial estimate     = host + H2D + device + GPU-to-GPU + D2H + launch
overlapped estimate = max(those stages)
```

Count algorithmic traffic, not allocation size: a copy reads and writes, while
`C = A + B` reads two streams and writes one. Capacity is separate from speed:
`weights = params × bits / 8`, and the working set must fit in usable memory
with headroom for runtime state, buffers, and fragmentation.

## Examples

A BF16 job with 200 TFLOPs, 1 TB of HBM traffic, a 2 GB upload, and 20 kernel
launches on an H100:

```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 200T --bytes 1TB --h2d 2GB --launches 20
H100 SXM 80GB · bf16 · measured 2026-08-08
intensity   200 FLOP/B             ridge 240 FLOP/B             memory-bound
compute     280 ms
memory      330 ms
device      330 ms
h2d         36 ms
launch      93 μs
total       370 ms                memory-bound
```

A 70B bf16 decode step must read the full 140 GB of weights once per token.
Does it fit, and how fast can one GPU's bandwidth feed it?

```text
$ napkin fit --gpu h100 --params 70B --bits 16
H100 SXM 80GB · 70B params · 16-bit
weights     140 GB
working set 140 GB
usable      72 GB per GPU (10% headroom)
fits        no — needs at least 2 GPUs (weights sharded)

$ napkin estimate --gpu h100 --dtype bf16 --flops 140G --bytes 140GB
H100 SXM 80GB · bf16 · measured 2026-08-08
intensity   1 FLOP/B               ridge 240 FLOP/B             memory-bound
compute     200 μs
memory      47 ms
device      47 ms
total       47 ms                 memory-bound
```

Memory-bound: about 50 ms/token on one GPU's bandwidth, and it splits roughly
linearly across tensor-parallel GPUs (plus communication).

## Usage

```sh
cargo install --path crates/napkin

napkin gpus                                       # profiles and their basis
napkin table --gpu h100                           # the tables above
napkin compare                                    # every GPU side by side
napkin estimate --gpu h100 --dtype bf16 --flops 200T --bytes 1TB [--h2d 2GB]
                [--d2h ..] [--host ..] [--g2g ..] [--launches 20]
                [--overlap] [--ceiling]
napkin fit --gpu h100 --params 70B --bits 16 [--extra 20GB] [--headroom 0.1]
napkin readme --write README.md                   # regenerate the marker blocks
```

## Benchmarks

The numbers come from `napkin-bench` (Rust + cudarc), run on Modal:

```sh
modal run modal/bench.py --gpu h100     # writes profiles/h100.measured.json
napkin readme --write README.md         # refreshes the tables
```

- `launch_us`: back-to-back empty-kernel launches, CUDA-event timed.
- `sync_us`: launch plus stream-synchronize round trip, wall clock.
- `host_copy_gbps`: pinned host memcpy on min(8, cpus) threads; read + write
  bytes counted.
- `h2d_gbps` / `d2h_gbps`: 512 MiB pinned async copies, event timed; the bytes
  that cross the link.
- `h2d_latency_us` / `d2h_latency_us`: 4-byte synchronous copies, wall clock.
- `hbm_gbps`: 512 MiB device-to-device memcpy; read + write counted (2x).
- `elementwise_gbps`: NVRTC `c[i] = a[i] + b[i]` fp32; two reads + one write (3x).
- `hbm_latency_ns`: single-thread pointer chase over a random cycle in a
  1 GiB buffer, ns per dependent load.
- `g2g_gbps`: 512 MiB peer copy, slower of the two directions; `None` with
  fewer than 2 GPUs or no peer access.
- `compute_tflops`: cuBLAS GemmEx 8192³, `2·N³` FLOPs counted, validated
  against a CPU float64 reference per dtype.

## Techniques

- **Count algorithmic bytes, not allocations.** A copy is 2x its buffer; an
  elementwise add is 3x. KV cache and weight reads are traffic too.
- **Dense is not sparse.** NVIDIA's headline FLOPs assume structured sparsity;
  halve them or measure.
- **Keep the qualifiers attached.** dtype, direction, topology, dense/sparse,
  measured/theoretical, software stack — no throughput number is universal.
- **Sum serial stages; max overlapped ones.** Overlap is a property of the
  workload, not of the GPU.
- **One or two significant digits.** You want order-of-magnitude answers;
  extra digits are not extra certainty.
- **Validate against one real end-to-end run.** A napkin estimate is a lower
  bound; check it once before trusting it.

## Resources

* [Making Deep Learning Go Brrrr From First Principles](https://horace.io/brrr_intro.html). The best short introduction to compute-, memory-, and overhead-bound GPU work. Start here.
* [How to Scale Your Model](https://jax-ml.github.io/scaling-book/). Book-length napkin math for accelerators: rooflines, sharding, communication, and inference arithmetic.
* [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://dl.acm.org/doi/10.1145/1498765.1498785). Williams, Waterman, and Patterson's original roofline paper.
* [Transformer Inference Arithmetic](https://kipp.ly/transformer-inference-arithmetic/). FLOP and byte counting for LLM inference, with KV-cache and latency estimates.
* [Transformer Math 101](https://blog.eleuther.ai/transformer-math/). Training compute and memory rules of thumb from EleutherAI.
* [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102). Pope et al. on the memory/compute/communication trade-offs of large-model inference.
* [Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473). Tensor, pipeline, and data parallelism with communication accounting.
* [FlashAttention](https://arxiv.org/abs/2205.14135). A worked example of winning by counting HBM bytes instead of FLOPs.
* [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance](https://siboehm.com/articles/22/CUDA-MMM). Step-by-step from naive to near-cuBLAS, with the arithmetic for each step.
* [NVIDIA GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html). NVIDIA's own arithmetic-intensity and roofline primer.
* [NVIDIA Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). Tile and wave quantization, and why GEMM shapes matter.
* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). Memory coalescing, transfers, occupancy, and measurement.
* [Modal GPU Glossary](https://modal.com/gpu-glossary). Fast definitions for SMs, warps, HBM, Tensor Cores, and the rest of the vocabulary.
* [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826). The template for measuring latencies and cache hierarchies yourself.
* [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499). Microbenchmarked Hopper latencies, Tensor Cores, and memory.
* [NVBandwidth](https://github.com/NVIDIA/nvbandwidth). NVIDIA's tool for host↔device and device↔device bandwidth; the state of the art for transfer numbers.
* [NCCL Tests](https://github.com/NVIDIA/nccl-tests). Collective benchmarks; read [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) for algorithm versus bus bandwidth.
* [Nsight Compute](https://developer.nvidia.com/nsight-compute). Per-kernel roofline, memory, and occupancy analysis.
* [cudarc](https://github.com/chelsea0x3b/cudarc). The Rust CUDA bindings `napkin-bench` is built on.
* [Napkin Math](https://github.com/sirupsen/napkin-math). Simon Eskildsen's original project, which this one follows.
