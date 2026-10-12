# GPU Napkin Math

The goal of this project is to collect numbers and techniques to quickly
estimate GPU workload performance from first principles — how long does it
take to read 1 GB from HBM on an H100? By composing them you should be able to
answer questions like: is this kernel compute- or memory-bound, how long does
the upload take, and how many GPUs does the model need? It follows Simon
Eskildsen's [Napkin Math](https://github.com/sirupsen/napkin-math).

## Numbers

Below are numbers rounded for memorization, not faux precision.
All four GPUs were measured on Modal on 2026-10-12 by `napkin-bench` (Rust +
cudarc, CUDA-event timing, CUDA 12.9, driver 580.95.05 on H100 and
610.57.04 on A100/B200/B300): `H100!:2`, `A100-80GB:2`, `B200:2`, and `B300:2`.
Each request returned two GPUs; the GPU↔GPU row is a `cuMemcpyPeerAsync` copy
between them with peer access enabled — NVLink where the hardware has it.
Host copies ran on A100 at PCIe Gen4 x16, and H100 and B200 at Gen5 x16. B300's
GPU link trains at Gen6 x16, but `nvidia-smi` reports `Host Max: 5`: each HGX
B300 GPU connects at Gen6 to the PCIe switch inside its ConnectX-8 NIC, then
Gen5 x16 to the CPU. B300 host copies are therefore Gen5-bound, like B200 and
H100.

**Note 1:** Compute figures are dense, not sparse. NVIDIA's headline Tensor
Core numbers assume structured sparsity; dense is half.

**Note 2:** Take the numbers with a grain of salt. GB means 10^9 bytes. For
transfers, [NVBandwidth](https://github.com/NVIDIA/nvbandwidth) is the state of
the art; for collectives, [NCCL Tests](https://github.com/NVIDIA/nccl-tests).

<!-- napkin table --gpu h100 -->
| Operation | Latency | Throughput | 1 MB | 1 GB |
| --- | ---: | ---: | ---: | ---: |
| Kernel launch (back-to-back) | 2 μs |  |  |  |
| Launch + sync round trip | 7 μs |  |  |  |
| Pinned host RAM copy |  | 100 GB/s † | 10 μs | 10 ms |
| Host → device, pinned | 9 μs | 50 GB/s † | 20 μs | 20 ms |
| Device → host, pinned | 9 μs | 60 GB/s † | 20 μs | 20 ms |
| HBM random read (pointer chase) | 300 ns |  |  |  |
| HBM copy (read + write) |  | 3 TB/s | 300 ns | 300 μs |
| Elementwise add |  | 3 TB/s | 300 ns | 300 μs |
| GPU ↔ GPU, NVLink one-way | 10 μs | 400 GB/s | 3 μs | 3 ms |

| Dense GEMM, 8192³ | Throughput | 1 GFLOP | 1 TFLOP | Ridge point |
| --- | ---: | ---: | ---: | ---: |
| fp16 | 700 TFLOP/s † | 1 μs | 1 ms | 220 FLOP/B |
| bf16 | 700 TFLOP/s † | 1 μs | 1 ms | 230 FLOP/B |
| tf32 | 400 TFLOP/s † | 3 μs | 3 ms | 120 FLOP/B |
| fp32 | 50 TFLOP/s | 20 μs | 20 ms | 17 FLOP/B |

Dense GEMM rows are the cuBLAS `GemmEx` 8192³ rate, a large-GEMM library ceiling, not the rate of every GEMM; measured shapes follow below.

† Rounds disagree by more than 5%: host RAM copy 13%, host → device 24%, device → host 6.4%, fp16 GEMM 10%, bf16 GEMM 14%, tf32 GEMM 13%, square fp16 GEMM 11%, square bf16 GEMM 11%, prefill fp16 GEMM 16%, prefill bf16 GEMM 13%. Spread is the slowest measured round over the fastest, minus one: the observed range, not a confidence interval. The cell shows the median round.

| cuBLASLt GEMM shape | Dimensions (M×N×K) | dtype | Throughput | of 8192³ peak | 1 TFLOP |
| --- | ---: | --- | ---: | ---: | ---: |
| square | 8192×8192×8192 | fp16 | 700 TFLOP/s † | 100% | 1 ms |
| square | 8192×8192×8192 | bf16 | 700 TFLOP/s † | 100% | 1 ms |
| prefill | 4096×28672×8192 | fp16 | 700 TFLOP/s † | 100% | 1 ms |
| prefill | 4096×28672×8192 | bf16 | 700 TFLOP/s † | 100% | 1 ms |
| decode-m8 | 8×8192×8192 | fp16 | 20 TFLOP/s | 3.5% | 50 ms |
| decode-m8 | 8×8192×8192 | bf16 | 20 TFLOP/s | 3.3% | 50 ms |
| decode-m64 | 64×8192×8192 | fp16 | 200 TFLOP/s | 27% | 5 ms |
| decode-m64 | 64×8192×8192 | bf16 | 200 TFLOP/s | 26% | 5 ms |
| square-k8191 | 8192×8192×8191 | fp16 | 100 TFLOP/s | 21% | 10 ms |
| square-k8191 | 8192×8192×8191 | bf16 | 100 TFLOP/s | 20% | 10 ms |

Shape suite: cuBLASLt 120901 (cublasLtGetVersion) captured 2026-10-12T23:51:29Z from H100!:2 on NVIDIA H100 80GB HBM3 (driver 580.95.05, CUDA 12.9, PCIe Gen5 x16); a different run from the generic peak captured 2026-10-12T22:23:05Z; % of peak uses the preserved profile's generic 8192³ `GemmEx` rate. TN column-major, fp32 accumulation and CUDA_R_32F scaling, 32 MiB heuristic workspace; query once before warm-up and reuse the selected algorithm.

| GEMM rounds | SM clock | Memory clock | Power | Temperature | Clock limits |
| --- | ---: | ---: | ---: | ---: | --- |
| fp16 | 1140–1350 MHz | 2619 MHz | 622 W | 51–57 °C | sw_power_cap 100% |
| bf16 | 1170–1425 MHz | 2619 MHz | 710 W | 56–62 °C | sw_power_cap 100% |
| tf32 | 1200–1470 MHz | 2619 MHz | 719 W | 57–62 °C | sw_power_cap 100% |
| fp32 | 1905–1980 MHz | 2619 MHz | 710 W | 64–66 °C | sw_power_cap 86% |

Read through NVML 13.580.95.05 via nvml-wrapper 0.13 every 5 ms on the GPU that ran the GEMMs, during the timed rounds only. Board limits: 700 W, SM clock 1980 MHz, memory clock 2619 MHz. Power is the median round's energy-counter change between its first and last counter update; the counter updates only a few times per round, so a single round in `gpu_state` can read above the board limit. Clock limits give the share of polls that reported each NVML clock-limit reason.
<!-- /napkin -->

## Other GPUs

<!-- napkin compare -->
| GPU | Memory | HBM | Dense BF16 | BF16 ridge | NVLink | Host link | Launch | Basis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A100 SXM 80GB | 80 GB | 2 TB/s | 300 TFLOP/s | 150 FLOP/B | 600 GB/s | 10 GB/s Gen4 x16 | 3 μs | measured 2026-10-12 |
| H100 SXM 80GB | 80 GB | 3 TB/s | 700 TFLOP/s | 230 FLOP/B | 900 GB/s | 50 GB/s Gen5 x16 | 2 μs | measured 2026-10-12 |
| B200 SXM 180GB | 180 GB | 6 TB/s | 1 PFLOP/s | 210 FLOP/B | 2 TB/s | 60 GB/s Gen5 x16 | 3 μs | measured 2026-10-12 |
| B300 SXM 288GB | 288 GB | 6 TB/s | 1 PFLOP/s | 220 FLOP/B | 2 TB/s | 60 GB/s Gen5 x16 (GPU Gen6) | 4 μs | measured 2026-10-12 |

| GPU | HBM measured / ceiling | BF16 measured / ceiling |
| --- | ---: | ---: |
| A100 SXM 80GB | 1740 / 2039 GB/s (85%) | 264 / 312 TFLOP/s (85%) |
| H100 SXM 80GB | 3002 / 3350 GB/s (90%) | 691 / 990 TFLOP/s (70%) |
| B200 SXM 180GB | 6414 / 8000 GB/s (80%) | 1325 / 2250 TFLOP/s (59%) |
| B300 SXM 288GB | 6494 / 8000 GB/s (81%) | 1423 / 2250 TFLOP/s (63%) |
<!-- /napkin -->

Modal request semantics matter: `H100!` pins an H100 — a plain `H100` request
may be upgraded to H200 — and `B200+` explicitly allows either B200 or B300.
Rows labelled "datasheet" are vendor ceilings, not achieved throughput.

## Arithmetic

For a workload with `W` FLOPs and `Q` bytes of HBM traffic:

```text
arithmetic  = W / Q                          FLOP/byte
ridge point          = F / M = TFLOP/s × 1,000 / GB/s FLOP/byte
compute time         = W / F
memory time          = Q / M
device time          = max(compute time, memory time)
```

Below the ridge point the kernel is memory-bound; above it, compute-bound.
Add the stages that sit on the critical path:

```text
copy time        = waits × L_copy + bytes / P      (per kind: H2D, D2H, GPU-to-GPU)
waits            = copies (sync) or 1 (async)
launch time      = kernel launches × L_launch

serial estimate     = host + H2D + device + GPU-to-GPU + D2H + launch
overlapped estimate = max(those stages)
```

`L_copy` is the measured 4-byte synchronous copy: issue, copy, and the stream
synchronize that waits for it, on the wall clock. `P` is the bulk 512 MiB
bandwidth. A sync copy (`cudaMemcpy`, or an async copy followed by
`cudaStreamSynchronize`) pays `L_copy` every time; `--copy-mode async` queues
all copies of one kind and waits once, so it pays `L_copy` once and is a lower
bound, because the cost of queueing each extra copy is not measured. `L_copy`
already contains a synchronize, so neither `L_launch` nor the launch + sync
round trip is added to copies, and copies are not counted as launches. Without
a measured `L_copy` (for example `--ceiling`), napkin uses `bytes / P` alone,
prints `≥`, and says why.

### Why a 4-byte copy is not in the bandwidth row

The H100 bandwidth row moves 512 MiB in about 9.7 ms, timed with CUDA events
on the GPU. The ~10 μs fixed cost of a copy is 0.1% of that, so it vanishes
into the throughput. Divide 4 bytes by the same 55 GB/s and you get 0.07 ns;
the measured 4-byte synchronous copy takes 9.8 μs, about 140,000 times longer.
Small copies are all fixed cost: driver call, DMA setup, and the wait for
completion, none of which scale with bytes. The size where the two terms are
equal is `L_copy × P = 9.8 μs × 55 GB/s ≈ 500 KB`; below it a copy is
latency-bound, above it bandwidth-bound. So moving 1 GB as 100,000 small
copies is about 50 times slower than one large copy:

```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 4B
h2d         9.8 μs   1 sync copy: 1 × 9.8 μs + 4 B / 55 GB/s

$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB
h2d         18 ms   1 sync copy: 1 × 9.8 μs + 1 GB / 55 GB/s

$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB --h2d-copies 100000
h2d         1 s   100000 sync copies: 100000 × 9.8 μs + 1 GB / 55 GB/s

$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB --h2d-copies 100000 --copy-mode async
h2d         ≥ 18 ms   100000 async copies: 1 × 9.8 μs + 1 GB / 55 GB/s; lower bound: per-copy cost of queued async copies is unmeasured
```

This latency-plus-bandwidth line is a straight-line fit through two measured
points. Real transfers between the two sizes, pageable host memory, and
copies that overlap with each other can land off it.

Count algorithmic traffic, not allocation size: a copy reads and writes, while
`C = A + B` reads two streams and writes one. Capacity is separate from speed:
`weights = params × bits / 8`, and the working set must fit in usable memory
with headroom for runtime state, buffers, and fragmentation.

For a decoder-only model, KV-cache capacity is
`KV bytes = 2 × layers × KV heads × head dim × live tokens × bytes`, where
live tokens are summed across requests. `--extra` is shardable with weights;
`--per-gpu` memory is replicated and does not shrink with GPU count.

## Examples

A BF16 job with 200 TFLOPs, 1 TB of HBM traffic, a 2 GB upload, a 1 GB
peer transfer, and 20 kernel launches on an H100:

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 200T --bytes 1TB --h2d 2GB --launches 20 -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 200T --bytes 1TB --h2d 2GB --launches 20
H100 SXM 80GB · bf16 · measured 2026-10-12
intensity   200 FLOP/B   ridge 230 FLOP/B   memory-bound
compute     290 ms
memory      330 ms
device      330 ms
h2d         37 ms   1 sync copy: 1 × 8.8 μs + 2 GB / 54 GB/s
launch      34 μs
total       370 ms   memory-bound
```
<!-- /napkin -->

### LLM inference

Use a Llama-3-70B-shaped dense decoder-only model. These are lower-bound
Fermi counts: assume 2 FLOPs per parameter per token, all parameters are
matmul weights, causal attention skips the masked half, activations hit a
fused-kernel read/write floor, decode is costed at its last step, and there is
no communication.

<!-- napkin example llm --gpu h100 --layers 80 --hidden 8192 --heads 64 --kv-heads 8 --head-dim 128 --params 70B --batch 8 --prompt 2048 --generated 512 -->
```text
$ napkin llm --gpu h100 --layers 80 --hidden 8192 --heads 64 --kv-heads 8 --head-dim 128 --params 70B --batch 8 --prompt 2048 --generated 512
H100 SXM 80GB · 70B params · 80 layers · 64 heads (8 KV heads, GQA 8:1) × 128 head dim · hidden 8192
weights 16-bit · KV 16-bit · activations bf16 · batch 8 × (2048 prompt + 512 generated)

memory: what must stay resident
weights      70B × 16 / 8 = 140 GB
KV/token     2 × 80 layers × 8 KV heads × 128 × 2 B = 328 KB
live tokens  8 × (2048 + 512) = 20480
KV cache     328 KB × 20480 = 6.71 GB
$ napkin fit --gpu h100 --params 70B --bits 16 --extra 6.71GB

prefill: one forward over 8 × 2048 = 16384 prompt tokens
linear       2 × 70B × 16384 = 2.29P FLOP
attention    2 × 80 × 64 × 128 × 8 × 2048² (causal) = 44T FLOP
weights      read once = 140 GB
KV write     328 KB × 16384 = 5.37 GB
activations  2 × 80 × 16384 × 8192 × 2 B (floor) = 42.9 GB
$ napkin estimate --gpu h100 --dtype bf16 --flops 2.34P --bytes 188GB

decode step: 8 new tokens at context 2560 (the last, largest step)
linear       2 × 70B × 8 = 1.12T FLOP
attention    4 × 80 × 64 × 128 × 8 × 2560 = 53.7G FLOP
weights      read once, shared by 8 requests = 140 GB
KV read      328 KB × 8 × 2560 = 6.71 GB
activations  2 × 80 × 8 × 8192 × 2 B (floor) = 21 MB
per token    147 GB / 8 = 18.3 GB
$ napkin estimate --gpu h100 --dtype bf16 --flops 1.17T --bytes 147GB

hypothetical: the 147 GB working set exceeds 72 GB usable on one H100 SXM 80GB,
so both estimates assume one GPU's bandwidth and compute. Sharding needs at least 3 GPUs;
it divides weights and KV, not per-GPU memory, and a tensor-parallel speedup needs
a communication term (napkin collective) before you claim one.
```
<!-- /napkin -->

Prefill is compute-bound: 16,384 prompt tokens share one 140 GB weight read, so
intensity is about 12,000 FLOP/B, far above the H100's 240 FLOP/B ridge. Decode
is memory-bound at 8 FLOP/B: every step reads all the weights to produce one
token per request. Batching amortizes that read: 147 GB per step serves 8
tokens, 18 GB each, against about 141 GB for a single request's token. KV reads
do not amortize, because they grow with batch × context. At batch 8 and a
32,768-token context the same formula gives an 86 GB KV read per step,
comparable to the 140 GB of weights.

Capacity and traffic are different questions. The 6.71 GB KV cache is what must
stay resident at the end of generation. Per-step KV traffic is what one decode
step reads, and it grows from 5.4 GB at the first step to 6.7 GB at the last.
Precision is set separately: `--weight-bits 4` with 16-bit KV shrinks the
weights 4× and leaves KV bytes unchanged. GQA (8 KV heads for 64 query heads) is
why KV/token is 328 KB instead of 2.6 MB, while attention FLOPs still count all
64 query heads.

These formulas are architecture-specific. A mixture-of-experts model multiplies
only its active experts per token, but a large batch may still read most experts
each step; sliding-window attention caps the context term; latent-attention
(MLA) caches change KV/token. Change the inputs, or the formula, rather than
reaching for a universal tokens-per-second rule.

The working set does not fit one H100, so both estimates are hypothetical
single-GPU numbers. By capacity alone, `fit` needs at least 3 GPUs. That divides
the weights and KV, not the per-GPU runtime memory, and dividing the 49 ms step
by 3 is not a speedup until the communication is added;
[Tensor parallelism](#tensor-parallelism-device-work-plus-communication) below
does that with measured NCCL curves.

Feed the printed calls to the existing commands:

<!-- napkin example fit --gpu h100 --params 70B --bits 16 --extra 6.71GB -->
```text
$ napkin fit --gpu h100 --params 70B --bits 16 --extra 6.71GB
H100 SXM 80GB · 70B params · 16-bit
weights     140 GB
extra       6.71 GB
working set 147 GB
usable      72 GB per GPU (10% headroom)
fits        no — needs at least 3 GPUs (weights + extra sharded, per-GPU replicated)
```
<!-- /napkin -->

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 2.34P --bytes 188GB -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 2.34P --bytes 188GB
H100 SXM 80GB · bf16 · measured 2026-10-12
intensity   12000 FLOP/B   ridge 230 FLOP/B   compute-bound
compute     3.4 s
memory      63 ms
device      3.4 s
total       3.4 s   compute-bound
```
<!-- /napkin -->

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 1.17T --bytes 147GB -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 1.17T --bytes 147GB
H100 SXM 80GB · bf16 · measured 2026-10-12
intensity   8 FLOP/B   ridge 230 FLOP/B   memory-bound
compute     1.7 ms
memory      49 ms
device      49 ms
total       49 ms   memory-bound
```
<!-- /napkin -->

### Tensor parallelism: device work plus communication

Eight H100s hold the 147 GB working set. With Megatron-style tensor
parallelism each GPU reads an eighth of the weights and KV per decode step,
140 GB / 8 + 6.71 GB / 8 = 18.3 GB, and does an eighth of the FLOPs:

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 146G --bytes 18.3GB -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 146G --bytes 18.3GB
H100 SXM 80GB · bf16 · measured 2026-10-12
intensity   8 FLOP/B   ridge 230 FLOP/B   memory-bound
compute     210 μs
memory      6.1 ms
device      6.1 ms
total       6.1 ms   memory-bound
```
<!-- /napkin -->

Sharding adds communication one GPU never did. Each of the 80 layers ends its
attention block and its MLP block with an AllReduce of the activations:
batch 8 × hidden 8192 × 2 bytes = 128 KiB per GPU, 160 AllReduces per step.
Their cost comes from an NCCL Tests curve measured on exactly this setup, eight
H100s on NVLink:

<!-- napkin example collective --gpu h100 --count 8 --topology nvlink --op all-reduce --bytes 128KiB -->
```text
$ napkin collective --gpu h100 --count 8 --topology nvlink --op all-reduce --bytes 128KiB
H100 SXM 80GB × 8 · nvlink · AllReduce · bf16 sum · 128 KiB (per-rank buffer)
regime      latency (fixed per-call cost is at least half the time)
time        38.9 μs
collective  algbw 3.37 GB/s · busbw 5.89 GB/s (busbw = algbw × 2(n-1)/n)
curve       128 KiB 38.9 μs · floor 36.8 μs · peak busbw 479 GB/s at 4 GiB
peer copy   395 GB/s measured (napkin-bench cuMemcpyPeer, one GPU pair; not used here)
nvlink      900 GB/s per-GPU datasheet aggregate, both directions (not used here)
evidence    NCCL 2.27.3 · nccl-tests v2.21.1 (afd59ab) · H100!:8 · 2026-10-12 · profiles/nccl-tests/h100x8
```
<!-- /napkin -->

```text
device         6.1 ms    18.3 GB per GPU at the measured HBM rate
communication  6.2 ms    160 AllReduces × 38.9 μs, latency regime
total          12 ms     serial: each AllReduce waits for its layer's output
```

Eight GPUs give about 4x over the hypothetical one-GPU 49 ms, not 8x, and at
this batch communication is half the step. The 128 KiB AllReduce sits in the
latency regime of the curve: dividing it by the 479 GB/s bus bandwidth NCCL
reaches at 4 GiB would claim 0.48 μs, 80x too fast. The 395 GB/s peer-copy
rate and the 900 GB/s datasheet NVLink figure answer different questions and
are not used. The same message costs 11.1 μs on two H100s, but two GPUs cannot
hold this working set, and `napkin collective` refuses counts and topologies
it has no curve for (`--count 4` is an error) rather than scaling a curve.

## Cost

The fastest GPU is not always the cheapest way to finish a job. You pay for
billed wall time on every GPU you rent, not for FLOPs:

```text
running   = busy / utilization
wall      = startup + running + idle
billed    = wall, rounded up to the billing increment
cost      = GPUs × billed hours × price per GPU-hour
useful    = busy / billed
work / $  = work / cost
```

`busy` is the time the job keeps each GPU working at the rate you modelled,
such as `napkin estimate`'s total. `utilization` is the fraction of its on-GPU
time the job actually sustains that rate: stalls, gaps between requests, and
kernels too small to fill the GPU all lower it. Startup, idle time, and
rounding up to the billing increment are paid for without doing work.
`napkin cost` takes each of these as a flag and has no built-in price: pass
`--price` per GPU-hour, or `--offer` one of the dated records below.

A 10-second job on two GPUs at $3/GPU-hour:

```text
$ napkin cost --gpus 2 --price 3 --time 10s
2 GPUs · $3/GPU-hour (--price)
busy        10 s per GPU (--time)
utilization 100% (default; pass --utilization)
running     10 s = busy / utilization
startup     0 s (default; pass --startup)
idle        0 s (default; pass --idle)
wall        10 s = startup + running + idle
billed      10 s per GPU (exact; pass --granularity)
GPU-hours   0.00556 = 2 GPUs × billed
cost        $0.0167 per job
useful      100% of billed GPU time is busy
work        none (pass --work or --throughput: a workload rate, never peak FLOPs)
```

The same job with a 30-second cold start, a minute kept warm afterwards, and
per-minute billing costs 12x as much:

```text
$ napkin cost --gpus 2 --price 3 --time 10s --startup 30s --idle 60s --granularity 60s
2 GPUs · $3/GPU-hour (--price)
busy        10 s per GPU (--time)
utilization 100% (default; pass --utilization)
running     10 s = busy / utilization
startup     30 s (--startup)
idle        60 s (--idle)
wall        100 s = startup + running + idle
billed      120 s per GPU (rounded up to 60 s)
GPU-hours   0.0667 = 2 GPUs × billed
cost        $0.2 per job
useful      8.3% of billed GPU time is busy
work        none (pass --work or --throughput: a workload rate, never peak FLOPs)
```

Tokens per dollar needs a workload rate: `--work 25K --unit tokens` for what
the job completes, or `--throughput 2.5K --unit tokens` for its measured rate
while busy. `napkin cost` never derives work from peak FLOPs. The GPU count is
also yours to choose: `napkin fit`'s "at least 2 GPUs" is a capacity floor,
not a runtime. Two GPUs halve `--time` only if your estimate of the split,
communication included, says so.

### Price records

Dated list prices live in `prices/*.json`, apart from the hardware profiles.
Each record names the provider, SKU, currency, date, source URL, and
on-demand versus spot terms. `napkin prices` prints each record's age and
flags any older than 90 days as stale; re-check the source before relying on
one.

<!-- napkin prices -->
| Record | Provider | SKU | GPU | Price | Per GPU-hour | Terms | Dated | Source |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `modal-a100` | Modal | Nvidia A100, 80 GB | a100 | $0.000694/s | $2.5 | on-demand, preemptible | 2026-10-12 | [modal.com/pricing](https://modal.com/pricing) |
| `modal-h100` | Modal | Nvidia H100 SXM5 | h100 | $0.001097/s | $3.95 | on-demand, preemptible | 2026-10-12 | [modal.com/pricing](https://modal.com/pricing) |
| `modal-b200` | Modal | Nvidia B200 | b200 | $0.001736/s | $6.25 | on-demand, preemptible | 2026-10-12 | [modal.com/pricing](https://modal.com/pricing) |
| `modal-b300` | Modal | Nvidia B300 | b300 | $0.001972/s | $7.1 | on-demand, preemptible | 2026-10-12 | [modal.com/pricing](https://modal.com/pricing) |

- `modal-a100`, `modal-h100`, `modal-b200`, `modal-b300`: Serverless base price, billed per second. Region selection adds 1.15-1.75x (https://modal.com/docs/guide/region-selection); GPU functions can be preempted (https://modal.com/docs/guide/preemption).
<!-- /napkin -->

### Utilization can reverse a price-performance ranking

Ranked by datasheet dense BF16 per dollar at the Modal prices above, a B200
beats an H100: 2,250 TFLOP/s at $6.25/hour is 360 TFLOP/s per dollar-hour,
against 990 TFLOP/s at $3.95/hour for 250. Take a compute-bound 1 EFLOP BF16
job. `napkin estimate --dtype bf16 --flops 1E --bytes 1TB` gives 1400 s on
the H100 and 750 s on the B200 at their measured BF16 rates. Modal bills per
second.

Now suppose the job's steps are too small to fill a B200's 148 SMs, and
profiling shows it sustains 50% of the modelled rate on the H100 but only 30%
on the B200. These utilizations are assumptions for illustration, not
measurements:

| GPU | Price | Busy | Cost at 100% | Utilization | Cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| H100 | $3.95/h | 1400 s | $1.54 | 50% | $3.07 |
| B200 | $6.25/h | 750 s | $1.30 | 30% | $4.34 |

```sh
napkin cost --gpus 1 --price 3.95 --time 1400s --utilization 0.5 --granularity 1s
napkin cost --gpus 1 --price 6.25 --time 750s --utilization 0.3 --granularity 1s
```

At full utilization the B200 is cheaper, as the ranking predicts; at these
utilizations the H100 is. With measured rates, the B200 wins only while its
utilization stays above `(691 / 3.95) / (1325 / 6.25) ≈ 0.83` of the H100's;
here it is `0.3 / 0.5 = 0.6`.

## Practice problems

[PROBLEMS.md](PROBLEMS.md) has seven problems with worked solutions hidden
until you open them: HBM traffic for a copy versus an add, tiny-copy latency,
a roofline crossover, KV-cache capacity, kernel fusion, decode batching, and
cost per token. Try each on paper first.

## Usage

```sh
cargo install --path crates/napkin

napkin gpus                                       # profiles and their basis
napkin table --gpu h100                           # the tables above
napkin compare                                    # every GPU side by side
napkin estimate --gpu h100 --dtype bf16 --flops 200T --bytes 1TB [--h2d 2GB]
napkin estimate --gpu h100 --dtype bf16 --gemm 4096x4096x4096
napkin estimate --gpu h100 --dtype bf16 --tflops 300 --flops 1e12 --bytes 1e9
                [--h2d-copies 1] [--d2h ..] [--d2h-copies ..] [--host ..]
                [--g2g ..] [--g2g-copies ..] [--copy-mode sync|async]
                [--launches 20] [--overlap] [--ceiling]
napkin fit --gpu h100 --params 70B --bits 16 [--extra 20GB] [--per-gpu 2GB] [--headroom 0.1]
napkin llm --gpu h100 --layers 80 --hidden 8192 --heads 64 --kv-heads 8
           --head-dim 128 --params 70B --batch 8 --prompt 2048 --generated 512
           [--weight-bits 4] [--kv-bits 8] [--per-gpu 2GB] [--headroom 0.1]
napkin cost --gpus 2 --time 10s --price 3       # or --offer modal-h100
            [--utilization 0.5] [--startup 30s] [--idle 60s] [--granularity 1s]
            [--work 25K | --throughput 2.5K] [--unit tokens]
napkin prices                                     # dated price records and their age
napkin collective --gpu h100 --count 2 --topology nvlink --op all-reduce
                   --bytes 16KiB [--dtype bf16]       # NCCL Tests curves only
napkin readme --write README.md PROBLEMS.md       # regenerate the marker blocks
napkin validate [--write VALIDATION.md | --check VALIDATION.md]
```

`--gemm MxNxK` (or `M×N×K`) derives dense FLOPs and minimum A/B/C traffic.
It uses a measured cuBLASLt rate only for an exact shape and dtype match;
otherwise it labels the estimate as unmatched and uses the separate 8192³
cuBLAS `GemmEx` peak. Napkin does not interpolate between shapes.
`--tflops` supplies an explicit positive, finite decimal TFLOP/s rate and
cannot be combined with `--gemm`. `--gemm` cannot be combined with `--ceiling`.

## Benchmarks

Compute, HBM, launch, and latency measurements come from `napkin-bench` (Rust +
cudarc), run on Modal:

```sh
modal run modal/bench.py --gpu h100     # writes profiles/h100.measured.json
napkin readme --write README.md PROBLEMS.md  # refreshes the tables
```

- `launch_us`: back-to-back empty-kernel launches, CUDA-event timed.
- `sync_us`: launch plus stream-synchronize round trip, wall clock.
- `host_copy_gbps`: pinned host memcpy on min(8, cpus) threads; read + write
  bytes counted.
- `h2d_gbps` / `d2h_gbps`: napkin-bench's 512 MiB pinned async-copy cross-check;
  the published transfer rates come from NVBandwidth below.
- `h2d_latency_us` / `d2h_latency_us`: 4-byte synchronous copies, wall clock;
  each sample includes the stream synchronize, so it is a per-copy fixed cost,
  not a bandwidth.
- `hbm_gbps`: 512 MiB device-to-device memcpy; read + write counted (2x).
- `elementwise_gbps`: NVRTC `c[i] = a[i] + b[i]` fp32; two reads + one write (3x).
- `hbm_latency_ns`: single-thread pointer chase over a random cycle in a
  1 GiB buffer, ns per dependent load.
- `g2g_gbps`: napkin-bench's 512 MiB peer-copy cross-check, slower of the two
  directions; `None` with fewer than 2 GPUs or no peer access.
- `g2g_latency_us`: 4-byte peer copy plus stream synchronize, wall clock.
- `compute_tflops`: the generic cuBLAS `GemmEx` 8192³ peak, `2·N³` FLOPs counted, validated
  against a CPU float64 reference per dtype. Each dtype first runs untimed
  GEMMs for 1 s, then times 7 rounds sized from the last warm-up batch to last about
  200 ms each (each round's measured length is stored as `wall_ms`). Tensor Core
  GEMMs draw enough power that the GPU starts at its maximum SM clock and drops
  it once the software power cap engages: on `H100!:2`, 10 warm-up calls and 10-call
  rounds (about 15 ms each) caught the boost in the first rounds and the cap
  in the later ones, a 15–26% spread on H100 and B300
  ([#5](https://github.com/gpu004/napkin-math-gpu/issues/5)). The longer
  warm-up reports the sustained rate under the power cap, which is the rate a
  long training or prefill GEMM gets.
- `gemm_shapes`: separate cuBLASLt measurements for fp16 and bf16 `square`
  (8192×8192×8192), `prefill` (4096×28672×8192), `decode-m8`
  (8×8192×8192), `decode-m64` (64×8192×8192), and `square-k8191`
  (8192×8192×8191). Each records TN column-major layout, fp32 accumulation,
  the cuBLASLt version and heuristic-selected algorithm, the 32 MiB heuristic
  workspace limit, achieved rate, and a deterministic 256-output check against
  a float64 reference. The heuristic is queried once before warm-up; the
  selected algorithm is reused by all timed calls. Shape-specific rates coexist
  with, and never replace, the 8192³ peak.
- `gpu_state`: NVML (`libnvidia-ml`, loaded at run time) polls the GEMM GPU
  every 5 ms during the timed rounds and stores, per round, its SM and memory
  clocks, temperature, average power from the energy counter, and how often
  each clock-limit reason (e.g. `sw_power_cap`) was active. `napkin table`
  prints a row per dtype and shape.
- Round spread: every metric keeps its per-round samples in `samples_ms`, and
  the tables show the median round. When a metric's slowest round is more than
  5% slower than its fastest, `napkin table` marks the cell with † and names
  the spread underneath. Seven rounds' min and max are an observed range, not
  a confidence interval.

Collective curves come from NVIDIA NCCL Tests, also run on Modal:

```sh
modal run modal/nccl.py --gpu h100 --count 2   # writes profiles/nccl-tests/h100x2/
```

See [BENCHMARKS.md](BENCHMARKS.md#nccl-tests) for what is recorded and how
`napkin collective` reads the curve.

### NVBandwidth

Transfer throughput is ingested from NVIDIA NVBandwidth v0.10.0, pinned to
commit `82fc4e8c6afa0babb8687793678f615b3b8d793e`. Run the Modal launcher with
an exact GPU request such as:

```sh
modal run modal/nvbandwidth.py --gpu h100
```

It runs these NVBandwidth arguments with verification enabled:

```sh
nvbandwidth --format json --disableAffinity --bufferSize 512 --testSamples 7 \
  --testcase host_to_device_memcpy_ce device_to_host_memcpy_ce \
  host_to_device_memcpy_sm device_to_host_memcpy_sm \
  device_to_device_memcpy_read_ce device_to_device_memcpy_write_ce \
  device_to_device_memcpy_read_sm device_to_device_memcpy_write_sm \
  host_device_latency_sm device_to_device_latency_sm
```

Do not add `--skipVerification` (`-s`): every testcase must pass. NVBandwidth
reports decimal GB/s. The `512` buffer size is 512 MiB (536,870,912 bytes);
each value is the median of 7 samples. Bandwidth counts the one-way payload
once, not both the read and write traffic. CE uses the copy engine and SM uses
GPU streaming multiprocessors. The peer READ matrix's value at row GPU *r*,
column GPU *c* measures data flowing
*c* → *r*, issued in *r*'s context; WRITE measures *r* → *c*, also issued in
*r*'s context.

The published host rates are the minimum CE value across the GPUs. The
published GPU↔GPU rate is the minimum CE value across both directions and both
READ/WRITE access patterns. The table retains the SM values and directional
entries for comparison. On all four checked-in GPUs, the mean absolute
deviation of the CE/napkin-bench ratio from 1 is below 1%; the table shows
each individual ratio.

`--disableAffinity` is needed because Modal rejects NVML's GPU-affinity
configuration call; it disables that affinity optimization but leaves
verification enabled. On Modal, `nvidia-smi topo -m` is unavailable; its failed
probe and the successful `nvidia-smi nvlink --status` output are retained with
the run metadata.

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
  bound; check it once before trusting it. See the [committed H100 validation
  report](VALIDATION.md) and the evidence in `validation/<case>/`.

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
* [NVIDIA GPU Performance Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html). NVIDIA's own arithmetic- and roofline primer.
* [NVIDIA Matrix Multiplication Background User's Guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html). Tile and wave quantization, and why GEMM shapes matter.
* [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/). Memory coalescing, transfers, occupancy, and measurement.
* [Modal pricing](https://modal.com/pricing). Per-second GPU list prices behind the dated records in `prices/modal.json`.
* [Modal GPU Glossary](https://modal.com/gpu-glossary). Fast definitions for SMs, warps, HBM, Tensor Cores, and the rest of the vocabulary.
* [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826). The template for measuring latencies and cache hierarchies yourself.
* [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499). Microbenchmarked Hopper latencies, Tensor Cores, and memory.
* [NVBandwidth](https://github.com/NVIDIA/nvbandwidth). NVIDIA's tool for host↔device and device↔device bandwidth; the state of the art for transfer numbers.
* [NCCL Tests](https://github.com/NVIDIA/nccl-tests). Collective benchmarks; read [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) for algorithm versus bus bandwidth.
* [Nsight Compute](https://developer.nvidia.com/nsight-compute). Per-kernel roofline, memory, and occupancy analysis.
* [cudarc](https://github.com/chelsea0x3b/cudarc). The Rust CUDA bindings `napkin-bench` is built on.
* [Napkin Math](https://github.com/sirupsen/napkin-math). Simon Eskildsen's original project, which this one follows.
