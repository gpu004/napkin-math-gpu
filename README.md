# GPU Napkin Math

The goal of this project is to collect numbers and techniques to quickly
estimate GPU workload performance from first principles — how long does it
take to read 1 GB from HBM on an H100? By composing them you should be able to
answer questions like: is this kernel compute- or memory-bound, how long does
the upload take, and how many GPUs does the model need? It follows Simon
Eskildsen's [Napkin Math](https://github.com/sirupsen/napkin-math).

## Numbers

Below are numbers rounded for memorization, not faux precision. Learn this
table first; the detailed tables after it give each number's basis, its
spread between rounds, and its time for 1 MB and 1 GB.

<!-- napkin summary --gpu h100 -->
| Operation | Latency | Throughput |
| --- | ---: | ---: |
| Kernel launch | 2 μs ‡ |  |
| Launch + sync round trip | 7 μs ‡ |  |
| HBM random read | 300 ns |  |
| HBM copy, read + write counted |  | 3 TB/s |
| Host ↔ device, pinned | 9 μs ‡ | 60 GB/s ‡ |
| GPU ↔ GPU, NVLink one-way | 10 μs ‡ | 400 GB/s |
| AllReduce, 8 GPUs on NVLink, bus bandwidth | 40 μs | 500 GB/s |
| Dense BF16 GEMM, 8192³ |  | 700 TFLOP/s |
| BF16 GEMM, 8×8192×8192 (decode, 8 tokens) |  | 20 TFLOP/s |

H100 SXM 80GB, one significant digit. Host ↔ device is the slower direction. Host and NVLink throughput come from NVBandwidth 2026-10-12, the rest from napkin-bench's median round. The AllReduce row is the fastest call and the highest bus bandwidth on the NCCL Tests curve (H100!:8, 2026-10-12).

‡ napkin-bench's number for this cell differs by more than 15% between runs on different machines; `napkin variation` has the range.
<!-- /napkin -->

Units are decimal: GB is 10⁹ bytes and TB/s is 10¹² bytes per second.
Simon's table is binary (GiB, TiB), so a number carried between the two is off
by 7% at GiB (2³⁰ / 10⁹ = 1.074) and 10% at TiB (2⁴⁰ / 10¹² = 1.10). His
7 TiB/s B200 memory row is 7.7 TB/s, NVIDIA's 8 TB/s datasheet figure;
napkin-bench's HBM copy measured 6.4 TB/s.

Every number comes from one canonical run per GPU model, on one Modal machine.
Repeats on other machines ([Between machines](#between-machines)) show which
numbers belong to the machine rather than to the GPU model; ‡ marks them.

All four GPUs were measured on Modal on 2026-10-12 by `napkin-bench` (Rust +
cudarc, CUDA-event timing, CUDA 12.9, driver 580.95.05 on A100/H100 and
610.57.04 on B200/B300): `H100!:2`, `A100-80GB:2`, `B200:2`, and `B300:2`.
Each request returned two GPUs; the GPU↔GPU row is a `cuMemcpyPeerAsync` copy
between them with peer access enabled — NVLink where the hardware has it.
Host copies ran on A100 at PCIe Gen4 x16, and H100 and B200 at Gen5 x16. B300's
GPU link trains at Gen6 x16, but `nvidia-smi` reports `Host Max: 5`: each HGX
B300 GPU connects at Gen6 to the PCIe switch inside its ConnectX-8 NIC, then
Gen5 x16 to the CPU. B300 host copies are therefore Gen5-bound, like B200 and
H100.

**Note 1:** Compute figures are dense, not sparse. NVIDIA's headline Tensor
Core numbers assume structured sparsity; dense is half.

**Note 2:** Take the numbers with a grain of salt. For
transfers, [NVBandwidth](https://github.com/NVIDIA/nvbandwidth) is the state of
the art; for collectives, [NCCL Tests](https://github.com/NVIDIA/nccl-tests).

<!-- napkin table --gpu h100 -->
| Operation | Latency | Throughput | 1 MB | 1 GB | Basis |
| --- | ---: | ---: | ---: | ---: | --- |
| Kernel launch (back-to-back) | 2 μs |  |  |  | napkin-bench |
| Launch + sync round trip | 7 μs |  |  |  | napkin-bench |
| Pinned host RAM copy |  | 100 GB/s † | 10 μs | 10 ms | napkin-bench |
| Host → device, pinned | 9 μs | 60 GB/s | 20 μs | 20 ms | NVBandwidth CE; latency napkin-bench |
| Device → host, pinned | 9 μs | 60 GB/s | 20 μs | 20 ms | NVBandwidth CE; latency napkin-bench |
| HBM random read (pointer chase) | 300 ns |  |  |  | napkin-bench |
| HBM copy (read + write) |  | 3 TB/s | 300 ns | 300 μs | napkin-bench |
| Elementwise add |  | 3 TB/s | 300 ns | 300 μs | napkin-bench |
| GPU ↔ GPU, NVLink one-way, slower direction | 10 μs | 400 GB/s | 3 μs | 3 ms | NVBandwidth CE; latency napkin-bench |
| GPU ↔ GPU across machines, RDMA one-way (400 Gbit/s NIC per GPU) | 5 μs | 50 GB/s | 20 μs | 20 ms | datasheet line rate |

The RDMA row is not measured: no NCCL Tests curve here spans two machines. It is the DGX H100, ConnectX-7 InfiniBand line rate, 400 Gbit/s / 8 = 50 GB/s per GPU in each direction, with 8 GPUs per machine and one NIC each. Source: https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html (checked 2026-11-05) — eight H100s and eight single-port ConnectX-7 cards, InfiniBand up to 400 Gbps each. NVIDIA publishes no end-to-end latency; 5 μs is the GPU ↔ GPU RDMA row of https://github.com/sirupsen/napkin-math (checked 2026-11-05), which also rounds the rate to 40 GiB/s.

| Dense GEMM, 8192³ | Throughput | 1 GFLOP | 1 TFLOP | Ridge point | Basis |
| --- | ---: | ---: | ---: | ---: | --- |
| fp16 | 700 TFLOP/s † | 1 μs | 1 ms | 220 FLOP/B | napkin-bench |
| bf16 | 700 TFLOP/s † | 1 μs | 1 ms | 230 FLOP/B | napkin-bench |
| tf32 | 400 TFLOP/s † | 3 μs | 3 ms | 120 FLOP/B | napkin-bench |
| fp32 | 50 TFLOP/s | 20 μs | 20 ms | 17 FLOP/B | napkin-bench |

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

### Between machines

The † marks above are spread between rounds within one run on one machine.
`modal run modal/bench.py --gpu h100 --repeat 3` repeats the full run in
fresh containers on whatever machines Modal assigns, and records each
machine's provider, region, and GPU UUIDs. Two B200 runs below predate the
one-run-per-container rule and share a machine. Simon averages three machine types
and keeps a latency only if it holds within 15% on every machine he measured;
this table applies the same rule to every GPU here.

<!-- napkin variation -->
| napkin-bench number | A100 (4 runs) | H100 (4 runs) | B200 (4 runs) | B300 (4 runs) |
| --- | ---: | ---: | ---: | ---: |
| Kernel launch, μs | 2.21–3.22 (46%) ‡ | 1.69–2.06 (22%) ‡ | 2.66–3.74 (41%) ‡ | 1.5–3.81 (150%) ‡ |
| Launch + sync round trip, μs | 8.97–10.6 (18%) ‡ | 6.9–7.94 (15%) ‡ | 9.23–9.95 (7.8%) | 7.01–11.5 (63%) ‡ |
| HBM random read, ns | 384–386 (0.48%) | 346–350 (1.1%) | 347–353 (1.8%) | 346–358 (3.4%) |
| HBM copy, GB/s | 1740–1740 (0.083%) | 2980–3000 (0.64%) | 6410–6420 (0.12%) | 6400–6490 (1.5%) |
| Pinned host RAM copy, GB/s | 21.7–95.4 (340%) ‡ | 99.6–117 (18%) ‡ | 67.6–79.1 (17%) ‡ | 42.8–243 (470%) ‡ |
| Host → device latency, μs | 13.3–14.5 (8.5%) | 8.8–11.5 (31%) ‡ | 10.6–10.9 (3.3%) | 9.39–13.3 (42%) ‡ |
| Host → device, GB/s | 24.7–26.8 (8.5%) | 53.7–55.5 (3.5%) | 55.5–55.6 (0.11%) | 55.7–57.7 (3.7%) |
| Device → host latency, μs | 11.4–13.2 (15%) ‡ | 9.02–11.5 (28%) ‡ | 13.3–13.7 (3.3%) | 11.9–14.9 (25%) ‡ |
| Device → host, GB/s | 26–26.3 (1.3%) | 47.6–55.5 (16%) ‡ | 57.1–57.1 (0.047%) | 57–57.4 (0.62%) |
| GPU ↔ GPU latency, μs | 13–15.6 (21%) ‡ | 9.61–12.4 (29%) ‡ | 12.7–13.1 (3.4%) | 11–18.3 (66%) ‡ |
| GPU ↔ GPU, slower direction, GB/s | 274–276 (0.38%) | 395–395 (0.058%) | 765–766 (0.089%) | 765–769 (0.5%) |
| Dense BF16 GEMM, 8192³, TFLOP/s | 262–284 (8.6%) | 679–697 (2.8%) | 1330–1380 (3.8%) | 1400–1460 (4%) |

Each cell is the lowest–highest median round over the runs, then the spread: highest over lowest, minus one. ‡ marks more than 15%, the most sirupsen/napkin-math lets a number move between machines. Transfers here are napkin-bench's own copies, not NVBandwidth.

| GPU | Run captured (UTC) | Driver | Provider | Region | GPU UUIDs |
| --- | --- | --- | --- | --- | --- |
| A100 | 2026-10-12T23:01:23Z (canonical) | 580.95.05 | not recorded | not recorded | not recorded |
| A100 | 2026-11-05T23:50:17Z | 610.57.04 | AZURE | eu-south | d5d6f0af, fed028f8 |
| A100 | 2026-11-05T23:50:18Z | 610.57.04 | UNSPECIFIED | eu | aae01d9b, ac95cbed |
| A100 | 2026-11-05T23:50:21Z | 610.57.04 | OCI | ap | 5484281b, b2a9b951 |
| H100 | 2026-10-12T22:23:05Z (canonical) | 580.95.05 | not recorded | not recorded | not recorded |
| H100 | 2026-11-05T23:42:53Z | 610.57.04 | OCI | us-central | 77ac2056, b623b800 |
| H100 | 2026-11-05T23:42:57Z | 610.57.04 | OCI | us-central | 6f73cd39, ad83bdbc |
| H100 | 2026-11-05T23:43:24Z | 610.57.04 | UNSPECIFIED | ca | 02e51b08, b1081b44 |
| B200 | 2026-10-12T22:33:57Z (canonical) | 610.57.04 | not recorded | not recorded | not recorded |
| B200 | 2026-11-05T23:43:07Z | 610.57.04 | AWS | us-west | a699ba9c, b01b92dc |
| B200 | 2026-11-05T23:46:00Z | 610.57.04 | AWS | us-west | a699ba9c, b01b92dc |
| B200 | 2026-11-05T23:46:55Z | 610.57.04 | AWS | eu-north | 15ff6185, 33a5ab6d |
| B300 | 2026-10-12T22:50:21Z (canonical) | 610.57.04 | not recorded | not recorded | not recorded |
| B300 | 2026-11-05T23:44:00Z | 610.57.04 | AWS | us-east | 0c9b9f4a, ad6f25ac |
| B300 | 2026-11-05T23:46:50Z | 610.57.04 | UNSPECIFIED | us-central | 44504982, 9001452d |
| B300 | 2026-11-05T23:46:52Z | 610.57.04 | UNSPECIFIED | us-central | 60d37169, 67008a4f |

GPU UUIDs are unique per physical GPU, shown by their first 8 hex digits, so runs with different UUIDs used different GPUs. Modal shows no machine ID inside a container: two 2-GPU runs in the same provider and region may still share one 8-GPU node.
<!-- /napkin -->

The repeats split the numbers in two. HBM bandwidth and latency, NVLink
bandwidth, and dense BF16 GEMM stay within 10% on every machine: they belong
to the GPU model. Kernel launch and the pinned host RAM copy move more than
15% on every GPU, and the 4-byte copy latencies do on most. They run through
the host CPU, its memory, and the driver, so they belong to the machine: read
them as orders of magnitude. The repeats ran on 2026-11-05, three weeks after
the canonical runs, and the A100 and H100 machines had moved from driver
580.95.05 to 610.57.04 in between, so this spread mixes machine and software.

## Other GPUs

<!-- napkin compare -->
| GPU | Memory | HBM | Dense BF16 | BF16 ridge | NVLink | Host link | Launch | Basis | Transfer basis |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A100 SXM 80GB | 80 GB | 2 TB/s | 300 TFLOP/s | 150 FLOP/B | 600 GB/s | 20 GB/s Gen4 x16 | 3 μs | napkin-bench 2026-10-12 | NVBandwidth 2026-10-12 |
| H100 SXM 80GB | 80 GB | 3 TB/s | 700 TFLOP/s | 230 FLOP/B | 900 GB/s | 60 GB/s Gen5 x16 | 2 μs | napkin-bench 2026-10-12 | NVBandwidth 2026-10-12 |
| B200 SXM 180GB | 180 GB | 6 TB/s | 1 PFLOP/s | 210 FLOP/B | 2 TB/s | 60 GB/s Gen5 x16 | 3 μs | napkin-bench 2026-10-12 | NVBandwidth 2026-10-12 |
| B300 SXM 288GB | 288 GB | 6 TB/s | 1 PFLOP/s | 220 FLOP/B | 2 TB/s | 60 GB/s Gen5 x16 (GPU Gen6) | 4 μs | napkin-bench 2026-10-12 | NVBandwidth 2026-10-12 |

| GPU | HBM measured / ceiling | BF16 measured / ceiling |
| --- | ---: | ---: |
| A100 SXM 80GB | 1737 / 2039 GB/s (85%) | 267 / 312 TFLOP/s (86%) |
| H100 SXM 80GB | 3002 / 3350 GB/s (90%) | 691 / 990 TFLOP/s (70%) |
| B200 SXM 180GB | 6414 / 8000 GB/s (80%) | 1325 / 2250 TFLOP/s (59%) |
| B300 SXM 288GB | 6494 / 8000 GB/s (81%) | 1423 / 2250 TFLOP/s (63%) |

| GPU | Transfer | NVBandwidth CE | NVBandwidth SM | napkin-bench | CE / napkin-bench | PCIe (NVBandwidth / napkin-bench) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| A100 SXM 80GB | H2D | 24.9 GB/s | 26.5 GB/s | 24.7 GB/s | 1.009 | Gen4 x16 / Gen4 x16 |
| A100 SXM 80GB | D2H | 26.3 GB/s | 26.3 GB/s | 26.3 GB/s | 1.000 | Gen4 x16 / Gen4 x16 |
| A100 SXM 80GB | GPU0→GPU1 | 279 GB/s | 230 GB/s | 275 GB/s | 1.014 | n/a (NVLink) |
| A100 SXM 80GB | GPU1→GPU0 | 277 GB/s | 235 GB/s | 279 GB/s | 0.992 | n/a (NVLink) |
| H100 SXM 80GB | H2D | 55.5 GB/s | 51 GB/s | 53.7 GB/s | 1.035 | Gen5 x16 / Gen5 x16 |
| H100 SXM 80GB | D2H | 55.5 GB/s | 40.9 GB/s | 55.3 GB/s | 1.004 | Gen5 x16 / Gen5 x16 |
| H100 SXM 80GB | GPU0→GPU1 | 395 GB/s | 368 GB/s | 395 GB/s | 0.999 | n/a (NVLink) |
| H100 SXM 80GB | GPU1→GPU0 | 395 GB/s | 369 GB/s | 395 GB/s | 0.999 | n/a (NVLink) |
| B200 SXM 180GB | H2D | 55.6 GB/s | 51.3 GB/s | 55.6 GB/s | 1.000 | Gen5 x16 / Gen5 x16 |
| B200 SXM 180GB | D2H | 56.9 GB/s | 52.6 GB/s | 57.1 GB/s | 0.996 | Gen5 x16 / Gen5 x16 |
| B200 SXM 180GB | GPU0→GPU1 | 766 GB/s | 717 GB/s | 767 GB/s | 0.998 | n/a (NVLink) |
| B200 SXM 180GB | GPU1→GPU0 | 765 GB/s | 717 GB/s | 766 GB/s | 0.999 | n/a (NVLink) |
| B300 SXM 288GB | H2D | 55.7 GB/s | 51.6 GB/s | 55.7 GB/s | 1.001 | Gen6 x16 / Gen6 x16 |
| B300 SXM 288GB | D2H | 57.1 GB/s | 52.5 GB/s | 57.4 GB/s | 0.994 | Gen6 x16 / Gen6 x16 |
| B300 SXM 288GB | GPU0→GPU1 | 764 GB/s | 717 GB/s | 765 GB/s | 0.998 | n/a (NVLink) |
| B300 SXM 288GB | GPU1→GPU0 | 764 GB/s | 717 GB/s | 766 GB/s | 0.998 | n/a (NVLink) |

| GPU | Copy | Latency `L` (4 B) | Bandwidth `P` | Latency size `0.05 × L × P` | Break-even `L × P` |
| --- | --- | ---: | ---: | ---: | ---: |
| A100 SXM 80GB | H2D | 14 μs | 25 GB/s | 18 KB | 360 KB |
| A100 SXM 80GB | D2H | 13 μs | 26 GB/s | 17 KB | 350 KB |
| A100 SXM 80GB | GPU ↔ GPU | 15 μs | 280 GB/s | 210 KB | 4.3 MB |
| H100 SXM 80GB | H2D | 8.8 μs | 56 GB/s | 24 KB | 490 KB |
| H100 SXM 80GB | D2H | 9.4 μs | 55 GB/s | 26 KB | 520 KB |
| H100 SXM 80GB | GPU ↔ GPU | 10 μs | 400 GB/s | 200 KB | 4.1 MB |
| B200 SXM 180GB | H2D | 11 μs | 56 GB/s | 29 KB | 590 KB |
| B200 SXM 180GB | D2H | 14 μs | 57 GB/s | 39 KB | 780 KB |
| B200 SXM 180GB | GPU ↔ GPU | 13 μs | 760 GB/s | 480 KB | 9.7 MB |
| B300 SXM 288GB | H2D | 13 μs | 56 GB/s | 37 KB | 740 KB |
| B300 SXM 288GB | D2H | 15 μs | 57 GB/s | 42 KB | 850 KB |
| B300 SXM 288GB | GPU ↔ GPU | 18 μs | 760 GB/s | 700 KB | 14 MB |

A latency is one copy at a time, each waiting for the last, at the largest size that takes as long as the smallest. napkin-bench times only the smallest, a 4-byte copy plus the stream synchronize that waits for it, so the last two columns come from the line `time = L + bytes / P`, not from a sweep of sizes. At the latency size, a copy of `0.05 × L × P` bytes takes 5% longer than `L`, the spread at which rounds are flagged as disagreeing. At break-even the bytes cost as much as the fixed cost and the copy takes `2 × L`: smaller copies are latency-bound, larger ones bandwidth-bound. `P` is the rate `napkin estimate` charges, with the basis in the first table.
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

### Copy latency and the break-even size

Following Simon's [Napkin Math](https://github.com/sirupsen/napkin-math), a
latency is one operation at a time, each waiting for the last, at the largest
size that takes as long as the smallest. The smallest copy is 4 bytes, and
napkin-bench times exactly that: issue, copy, and the stream synchronize that
waits for it, on the wall clock. That is `L_copy`.

The bandwidth row is a different measurement: 512 MiB timed with CUDA events
on the GPU, where a fixed cost of microseconds is about 0.1% of the time and
vanishes into the throughput. 4 bytes at 55 GB/s would take under a tenth of a
nanosecond, yet the measured copy takes microseconds. Small copies are all fixed
cost: driver call, DMA setup, and the wait for completion, none of which scale
with bytes.

napkin-bench does not sweep sizes, so napkin draws the straight line
`time = L_copy + bytes / P` between the two measured points. Two sizes fall out
of it, and the last table under [Other GPUs](#other-gpus) gives both for every
GPU and copy kind:

```text
latency size = 0.05 × L_copy × P   a copy this big takes 5% longer than 4 bytes
break-even   = L_copy × P          the bytes cost as much as the fixed cost: 2 × L_copy
```

Up to the latency size, a copy takes as long as the smallest one, so by
Simon's rule it costs `L_copy`. Below break-even a copy is latency-bound, above
it bandwidth-bound. On an H100 the two sizes are about 20 KB and 500 KB for
host → device. So moving 1 GB as 100,000 copies of 10 KB, each under the
break-even size, pays the fixed cost 100,000 times; queueing them
asynchronously and waiting once pays it once, as a lower bound:

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
intensity   0 FLOP/B   ridge 230 FLOP/B   h2d-bound
compute     0 s
memory      0 s
device      0 s
h2d         18 ms   1 sync copy: 1 × 8.8 μs + 1 GB / 56 GB/s   NVBandwidth CE
total       18 ms   h2d-bound
```
<!-- /napkin -->

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB --h2d-copies 100000 -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB --h2d-copies 100000
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
intensity   0 FLOP/B   ridge 230 FLOP/B   h2d-bound
compute     0 s
memory      0 s
device      0 s
h2d         900 ms   100000 sync copies: 100000 × 8.8 μs + 1 GB / 56 GB/s   NVBandwidth CE
total       900 ms   h2d-bound
```
<!-- /napkin -->

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB --h2d-copies 100000 --copy-mode async -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 1GB --h2d-copies 100000 --copy-mode async
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
intensity   0 FLOP/B   ridge 230 FLOP/B   h2d-bound
compute     0 s
memory      0 s
device      0 s
h2d         ≥ 18 ms   100000 async copies: 1 × 8.8 μs + 1 GB / 56 GB/s; lower bound: per-copy cost of queued async copies is unmeasured   NVBandwidth CE
total       ≥ 18 ms   h2d-bound
```
<!-- /napkin -->

The line is fitted through two measured points. Real transfers between the
two sizes, pageable host memory, and copies that overlap with each other can
land off it, and a size sweep would replace the latency-size column with a
measurement.

Count algorithmic traffic, not allocation size: a copy reads and writes, while
`C = A + B` reads two streams and writes one. Capacity is separate from speed:
`weights = params × bits / 8`, and the working set must fit in usable memory
with headroom for runtime state, buffers, and fragmentation.

For a decoder-only model, KV-cache capacity is
`KV bytes = 2 × layers × KV heads × head dim × live tokens × bytes`, where
live tokens are summed across requests. `--extra` is shardable with weights;
`--per-gpu` memory is replicated and does not shrink with GPU count.

## Examples

A BF16 job with 200 TFLOPs, 1 TB of HBM traffic, a 2 GB upload, and 20 kernel
launches on an H100:

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 200T --bytes 1TB --h2d 2GB --launches 20 -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 200T --bytes 1TB --h2d 2GB --launches 20
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
intensity   200 FLOP/B   ridge 230 FLOP/B   memory-bound
compute     290 ms
memory      330 ms
device      330 ms
h2d         36 ms   1 sync copy: 1 × 8.8 μs + 2 GB / 56 GB/s   NVBandwidth CE
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
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
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
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
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
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
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
network     50 GB/s per-GPU RDMA datasheet line rate, one direction, across machines (not used here)
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

### Across machines: a datasheet RDMA link

Every NCCL Tests curve above was measured on one machine. Modal's
[multi-node clusters](https://modal.com/docs/guide/multi-node-clusters)
(checked 2026-11-05) advertise 3,200 Gbps of RDMA per H100 machine, the same
8 × 400 Gbit/s, and `modal run modal/nccl.py --gpu h100 --count 16` is ready
to measure it, but no run here has used one yet. Until then the link between
machines is a datasheet row, the last row of the H100 table:

```text
line rate   400 Gbit/s / 8 bits = 50 GB/s per GPU, each direction
per machine 8 GPUs × one ConnectX-7 each = 8 × 50 GB/s = 400 GB/s out
latency     5 μs one-way
```

The 400 Gbit/s NIC per GPU comes from NVIDIA's [DGX H100 user
guide](https://docs.nvidia.com/dgx/dgxh100-user-guide/introduction-to-dgxh100.html).
It is a line rate, a ceiling like the 900 GB/s NVLink figure, not what NCCL
achieves. NVIDIA publishes no end-to-end latency for it, so 5 μs is the order
of magnitude from Simon's [GPU ↔ GPU RDMA
row](https://github.com/sirupsen/napkin-math). That row rounds the same link
to 40 GiB/s (43 GB/s), below the 50 GB/s (46.6 GiB/s) line rate.

`napkin collective` still refuses 16 GPUs, because no curve covers them. You can
estimate them by hand by composing measured curves with the link. Take a
data-parallel AllReduce of 4 GiB of bf16 gradients per GPU over two machines
of eight H100s. Assume the three-stage schedule hierarchical AllReduce uses:

1. ReduceScatter inside each machine over NVLink. Each GPU then holds a
   4 GiB / 8 = 512 MiB shard of its machine's partial sums.
2. Each GPU runs a two-rank AllReduce of its shard with the matching GPU in the
   other machine, over its own NIC. Two ranks send `2(n-1)/n` = 1× the shard,
   512 MiB = 537 MB, so 537 MB / 50 GB/s = 10.7 ms, plus two 5 μs steps.
3. AllGather inside each machine to hand every GPU the full result.

The two NVLink stages come from the measured eight-GPU curves:

```sh
napkin collective --gpu h100 --count 8 --topology nvlink --op reduce-scatter --bytes 4GiB
napkin collective --gpu h100 --count 8 --topology nvlink --op all-gather --bytes 4GiB
```

```text
reduce-scatter 10.6 ms    measured, 8 H100s on NVLink
network        10.7 ms    datasheet: 537 MB / 50 GB/s + 2 × 5 μs
all-gather     10.6 ms    measured, 8 H100s on NVLink
serial         32 ms      the stages one after another
pipelined      ≥ 11 ms    the slowest stage, if NCCL overlaps the chunks
```

One machine does the same 4 GiB AllReduce in a measured 15.7 ms. Two machines
land somewhere between 11 and 32 ms. The network stage is only as slow as the
NVLink stages because each GPU has its own NIC: if all eight GPUs shared one
400 Gbit/s NIC, the same stage would take 8 × 10.7 = 86 ms. Several things can
move the real number off this sketch. NCCL may pick a flat ring or a tree
instead of this schedule, in-network reduction (SHARP) changes the traffic,
and achieved RDMA bandwidth sits below the line rate: 43 GB/s would make the
network stage 12.5 ms.

Small messages behave differently. A tensor-parallel 128 KiB AllReduce that
spans the two machines pays the same three stages: 39.9 μs ReduceScatter,
2 × 5 μs on the network, and 42.4 μs AllGather, for about 90 μs. That is more
than twice the 38.9 μs one-machine call, and 160 of them per decode step
cost 15 ms instead of 6.2 ms. This is why tensor parallelism usually stays
inside one machine and only data or pipeline parallelism crosses the network.
Measure a two-machine NCCL Tests curve before trusting either figure.

### Loading the weights: storage comes before PCIe

The host → device row starts from pinned host RAM. Before that, a real job
has to get its weights or dataset into RAM, from blob storage (S3, GCS, Azure
Blob) or a local disk. `napkin-bench` does not measure either step, so this
example borrows two rows from Simon Eskildsen's
[Napkin Math](https://github.com/sirupsen/napkin-math#numbers). He measured
them on 48 vCPU CPU VMs on GCP and AWS, not on GPU hosts:

| Stage | Throughput | Basis |
| --- | ---: | --- |
| Blob storage read | 3 GiB/s = 3.2 GB/s | Simon's "Blob Storage, Zonal Read" row: 32–128 connections in parallel, capped by the VM's NIC |
| Local SSD sequential read | 5 GiB/s = 5.4 GB/s | Simon's "SSD Sequential Read" row: every local SSD on the VM striped together |
| Host → device, pinned | 56 GB/s | NVBandwidth CE, from the H100 table above |

Load the 140 GB of 16-bit Llama-3-70B weights from the LLM example each way:

```text
blob storage   140 GB / 3.2 GB/s ≈ 44 s
local SSD      140 GB / 5.4 GB/s ≈ 26 s
PCIe           140 GB / 56 GB/s  ≈ 2.5 s
```

The PCIe line is the measured row:

<!-- napkin example estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 140GB -->
```text
$ napkin estimate --gpu h100 --dtype bf16 --flops 0 --bytes 0 --h2d 140GB
H100 SXM 80GB · bf16 · napkin-bench 2026-10-12
intensity   0 FLOP/B   ridge 230 FLOP/B   h2d-bound
compute     0 s
memory      0 s
device      0 s
h2d         2.5 s   1 sync copy: 1 × 8.8 μs + 140 GB / 56 GB/s   NVBandwidth CE
total       2.5 s   h2d-bound
```
<!-- /napkin -->

The disk or network step is 10 to 20 times slower than the PCIe step the
tables measure, so storage, not PCIe, sets how long a cold start waits before
the first kernel. The stages follow the same rule as the arithmetic above:
sum serial stages, take the max of overlapped ones. Downloading the file to
local SSD and then loading it pays `44 + 26 + 2.5 ≈ 72 s`. A loader that
streams chunks from blob storage into pinned buffers, copying one chunk to the
GPU while the next downloads, pays about the slowest stage, 44 s.

Sharding does not shrink the storage step. The model needs at least three
H100s (`napkin fit` above), and the tensor-parallel example uses eight. Each
GPU then receives 17.5 GB over its own PCIe link, about 0.3 s if all eight
links run at full rate at once; the host copies above were measured one GPU
at a time, so that concurrent rate is an assumption. All 140 GB still
crosses the host's one NIC or one disk array, so the storage step stays at
44 or 26 s.

These assumptions move the answer the most:

- **Connections.** Simon's notes measured about 100 MiB/s on one connection.
  A loader that issues one GET at a time takes `140 GB / 0.105 GB/s ≈ 1300 s`,
  about 22 minutes. The 3 GiB/s needs dozens of range reads in flight.
- **Provider.** With 32–128 connections, S3 peaked at 2.0 GiB/s and GCS at
  4.3–4.9 GiB/s on those VMs: 65 s and 27–30 s for 140 GB.
- **The host.** A GPU host has a different NIC and different local disks
  than a 48 vCPU CPU VM, so its numbers can land on either side of these.
  Measuring blob → GPU and SSD → GPU on Modal would replace the borrowed
  rows.
- **What is not counted.** Container start, imports, CUDA context creation,
  and converting or deserializing the weights all add to a cold start, so
  these times are lower bounds.

Simon's [benchmarking notes](https://github.com/sirupsen/napkin-math/blob/master/BENCHMARKING.md#blob-storage-rows-march-2026)
describe both measurements: the per-connection and per-provider blob rates,
and the [striped local SSDs](https://github.com/sirupsen/napkin-math/blob/master/BENCHMARKING.md#ssd-network-disk-and-hdd-rows).

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

The same job with an assumed 30-second cold start, a minute kept warm
afterwards, and per-minute billing costs 12x as much:

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

A cold start can be derived instead of assumed. Eight H100s serving the 140 GB
model wait at least 44 s for its weights to stream from blob storage
([Loading the weights](#loading-the-weights-storage-comes-before-pcie)), so
the same 10 seconds of work bills 54 s on each GPU:

```text
$ napkin cost --gpus 8 --price 3.95 --time 10s --startup 44s
8 GPUs · $3.95/GPU-hour (--price)
busy        10 s per GPU (--time)
utilization 100% (default; pass --utilization)
running     10 s = busy / utilization
startup     44 s (--startup)
idle        0 s (default; pass --idle)
wall        54 s = startup + running + idle
billed      54 s per GPU (exact; pass --granularity)
GPU-hours   0.12 = 8 GPUs × billed
cost        $0.474 per job
useful      19% of billed GPU time is busy
work        none (pass --work or --throughput: a workload rate, never peak FLOPs)
```

Loading from local SSD instead (`--startup 26s`) bills 36 s and $0.316, 28%
useful. The PCIe step is at most 2.5 s of either startup, so storage is the
term to shrink: keep weights on fast local disk, or keep a GPU warm.

Tokens per dollar needs a workload rate: `--work 25K --unit tokens` for what
the job completes, or `--throughput 2.5K --unit tokens` for its measured rate
while busy. `napkin cost` never derives work from peak FLOPs. The GPU count is
also yours to choose: `napkin fit`'s "at least 2 GPUs" is a capacity floor,
not a runtime. Two GPUs halve `--time` only if your estimate of the split,
communication included, says so.

### Price records

Dated list prices live in `prices/*.json`, apart from the hardware profiles.
Each record names the provider, SKU, currency, date, source URL, the number
of GPUs the listed price buys, and its terms: on-demand, spot, or a 1- or
3-year commitment. `napkin prices` prints each record's age and flags any
older than 90 days as stale; re-check the source before relying on one.

Two things keep these per-GPU-hour prices from being like for like. The
clouds rent 8-GPU instances whose price includes host CPUs, memory, local
SSD, and networking, divided here by 8; Modal's price is for the GPU alone,
with CPU and memory billed on top. And a commitment bills every hour of its
term, busy or not: its per-GPU-hour price holds only if you keep the GPUs
busy all term, so divide it by the fraction of the term you use.
`napkin cost --offer` prints both caveats when they apply.

<!-- napkin prices -->
| Record | Provider | SKU | GPU | Price | Per GPU-hour | Terms | Dated | Source |
| --- | --- | --- | --- | ---: | ---: | --- | --- | --- |
| `aws-h100` | AWS | p5.48xlarge, 8x H100 SXM, us-east-1 | h100 | $55.04/h for 8 GPUs | $6.88 | on-demand | 2026-11-05 | [b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json) |
| `aws-h100-1y` | AWS | p5.48xlarge, 8x H100 SXM, us-east-1 | h100 | $43.344/h for 8 GPUs | $5.42 | 1-year commitment | 2026-11-05 | [pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json](https://pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json) |
| `aws-h100-3y` | AWS | p5.48xlarge, 8x H100 SXM, us-east-1 | h100 | $40.42138/h for 8 GPUs | $5.05 | 3-year commitment | 2026-11-05 | [pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json](https://pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json) |
| `aws-b200` | AWS | p6-b200.48xlarge, 8x B200, us-east-1 | b200 | $113.9328/h for 8 GPUs | $14.2 | on-demand | 2026-11-05 | [b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json](https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/US%20East%20%28N.%20Virginia%29/Linux/index.json) |
| `aws-b200-1y` | AWS | p6-b200.48xlarge, 8x B200, us-east-1 | b200 | $89.72208/h for 8 GPUs | $11.2 | 1-year commitment | 2026-11-05 | [pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json](https://pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json) |
| `aws-b200-3y` | AWS | p6-b200.48xlarge, 8x B200, us-east-1 | b200 | $83.67225/h for 8 GPUs | $10.5 | 3-year commitment | 2026-11-05 | [pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json](https://pricing.us-east-1.amazonaws.com/savingsPlan/v1.0/aws/AWSComputeSavingsPlan/20261104210253/us-east-1/index.json) |
| `azure-h100` | Azure | Standard_ND96isr_H100_v5, 8x H100 SXM, eastus | h100 | $98.32/h for 8 GPUs | $12.3 | on-demand | 2026-11-05 | [prices.azure.com/api/retail/prices](https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27eastus%27%20and%20armSkuName%20eq%20%27Standard_ND96isr_H100_v5%27) |
| `azure-h100-spot` | Azure | Standard_ND96isr_H100_v5, 8x H100 SXM, eastus | h100 | $18.169536/h for 8 GPUs | $2.27 | spot, preemptible | 2026-11-05 | [prices.azure.com/api/retail/prices](https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27eastus%27%20and%20armSkuName%20eq%20%27Standard_ND96isr_H100_v5%27) |
| `azure-h100-1y` | Azure | Standard_ND96isr_H100_v5, 8x H100 SXM, eastus | h100 | $551221/year for 8 GPUs | $7.87 | 1-year commitment | 2026-11-05 | [prices.azure.com/api/retail/prices](https://prices.azure.com/api/retail/prices?$filter=armRegionName%20eq%20%27eastus%27%20and%20armSkuName%20eq%20%27Standard_ND96isr_H100_v5%27) |
| `gcp-h100` | GCP | a3-highgpu-8g, 8x H100 SXM, us-central1 | h100 | $88.490000119/h for 8 GPUs | $11.1 | on-demand | 2026-11-05 | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| `gcp-h100-spot` | GCP | a3-highgpu-8g, 8x H100 SXM, us-central1 | h100 | $52.700460712/h for 8 GPUs | $6.59 | spot, preemptible | 2026-11-05 | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| `gcp-h100-1y` | GCP | a3-highgpu-8g, 8x H100 SXM, us-central1 | h100 | $61.383674231/h for 8 GPUs | $7.67 | 1-year commitment | 2026-11-05 | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| `gcp-h100-3y` | GCP | a3-highgpu-8g, 8x H100 SXM, us-central1 | h100 | $38.864383195/h for 8 GPUs | $4.86 | 3-year commitment | 2026-11-05 | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| `gcp-b200-spot` | GCP | a4-highgpu-8g, 8x B200, us-central1 | b200 | $39.6336/h for 8 GPUs | $4.95 | spot, preemptible | 2026-11-05 | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| `gcp-b200-1y` | GCP | a4-highgpu-8g, 8x B200, us-central1 | b200 | $88.9272/h for 8 GPUs | $11.1 | 1-year commitment | 2026-11-05 | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| `gcp-b200-3y` | GCP | a4-highgpu-8g, 8x B200, us-central1 | b200 | $56.7072/h for 8 GPUs | $7.09 | 3-year commitment | 2026-11-05 | [cloud.google.com/products/compute/pricing/accelerator-optimized](https://cloud.google.com/products/compute/pricing/accelerator-optimized) |
| `modal-a100` | Modal | Nvidia A100, 80 GB | a100 | $0.000694/s | $2.5 | on-demand, preemptible | 2026-11-05 | [modal.com/pricing](https://modal.com/pricing) |
| `modal-h100` | Modal | Nvidia H100 SXM5 | h100 | $0.001097/s | $3.95 | on-demand, preemptible | 2026-11-05 | [modal.com/pricing](https://modal.com/pricing) |
| `modal-b200` | Modal | Nvidia B200 | b200 | $0.001736/s | $6.25 | on-demand, preemptible | 2026-11-05 | [modal.com/pricing](https://modal.com/pricing) |
| `modal-b300` | Modal | Nvidia B300 | b300 | $0.001972/s | $7.1 | on-demand, preemptible | 2026-11-05 | [modal.com/pricing](https://modal.com/pricing) |

- `aws-h100`, `aws-b200`: Linux on-demand instance price from the JSON behind https://aws.amazon.com/ec2/pricing/on-demand/. AWS rents whole instances, so the hourly price includes the host CPUs, memory, NVMe, and EFA networking.
- `aws-h100-1y`, `aws-h100-3y`, `aws-b200-1y`, `aws-b200-3y`: Compute Savings Plan, No Upfront: you commit to the hourly spend for every hour of the term. EC2 Instance Savings Plans are cheaper (p5.48xlarge 1-year $34.6752/h, 3-year $23.77728/h) but lock in the instance family and list no 1-year plan for p6-b200.
- `azure-h100`: Linux pay-as-you-go VM price from the Azure Retail Prices API. The VM price includes the host CPUs, memory, and InfiniBand. Azure lists no B200 VM; ND GB200 v6 uses Grace-Blackwell GB200, not the B200 profile.
- `azure-h100-spot`: Spot price from the Azure Retail Prices API; Azure moves spot prices with spare capacity and can evict spot VMs at any time.
- `azure-h100-1y`: Reserved VM instance: the API lists one total for the whole 1-year term, billed whether or not the VM runs.
- `gcp-h100`: Machine-type price in the page's default region, Iowa (us-central1); it includes the host vCPUs, memory, and local SSD. GCP lists no on-demand price for the B200 a4-highgpu-8g, only DWS, spot, and committed-use prices.
- `gcp-h100-spot`, `gcp-b200-spot`: "Current Spot pricing" column in the page's default region, Iowa (us-central1). GCP can change spot prices up to once a day (https://cloud.google.com/compute/docs/instances/spot) and can preempt spot VMs at any time.
- `gcp-h100-1y`, `gcp-h100-3y`, `gcp-b200-1y`, `gcp-b200-3y`: "Compute Resource CUDs" column in the page's default region, Iowa (us-central1): a committed-use discount billed for every hour of the term.
- `modal-a100`, `modal-h100`, `modal-b200`, `modal-b300`: Serverless base GPU price, billed per second; CPU cores and memory are billed separately on the same page. Region selection adds 1.15-1.75x (https://modal.com/docs/guide/region-selection); GPU functions can be preempted (https://modal.com/docs/guide/preemption).
<!-- /napkin -->

### Cost of moving 1 TB or computing 1 EFLOP

Like Simon's cost-per-TiB-read table, this one turns measured throughput into
dollars: the seconds 1 TB (or 1 EFLOP) takes at the measured rate, times the
price per GPU-second above.

<!-- napkin unit-costs -->
| Work | H100 `aws-h100` | H100 `aws-h100-1y` | H100 `aws-h100-3y` | B200 `aws-b200` | B200 `aws-b200-1y` | B200 `aws-b200-3y` | H100 `azure-h100` | H100 `azure-h100-spot` | H100 `azure-h100-1y` | H100 `gcp-h100` | H100 `gcp-h100-spot` | H100 `gcp-h100-1y` | H100 `gcp-h100-3y` | B200 `gcp-b200-spot` | B200 `gcp-b200-1y` | B200 `gcp-b200-3y` | A100 `modal-a100` | H100 `modal-h100` | B200 `modal-b200` | B300 `modal-b300` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Price per GPU-hour | $6.88 | $5.42 | $5.05 | $14.2 | $11.2 | $10.5 | $12.3 | $2.27 | $7.87 | $11.1 | $6.59 | $7.67 | $4.86 | $4.95 | $11.1 | $7.09 | $2.5 | $3.95 | $6.25 | $7.1 |
| 1 TB of HBM traffic, read + write counted | $0.0006 | $0.0005 | $0.0005 | $0.0006 | $0.0005 | $0.0005 | $0.001 | $0.0002 | $0.0007 | $0.001 | $0.0006 | $0.0007 | $0.0004 | $0.0002 | $0.0005 | $0.0003 | $0.0004 | $0.0004 | $0.0003 | $0.0003 |
| 1 TB host ↔ device, pinned, slower direction | $0.03 | $0.03 | $0.03 | $0.07 | $0.06 | $0.05 | $0.06 | $0.01 | $0.04 | $0.06 | $0.03 | $0.04 | $0.02 | $0.02 | $0.06 | $0.04 | $0.03 | $0.02 | $0.03 | $0.04 |
| 1 TB GPU → GPU over NVLink, both GPUs billed | $0.01 | $0.008 | $0.007 | $0.01 | $0.008 | $0.008 | $0.02 | $0.003 | $0.01 | $0.02 | $0.009 | $0.01 | $0.007 | $0.004 | $0.008 | $0.005 | $0.005 | $0.006 | $0.005 | $0.005 |
| 1 EFLOP of dense BF16 GEMM at the 8192³ rate | $3 | $2 | $2 | $3 | $2 | $2 | $5 | $0.9 | $3 | $4 | $3 | $3 | $2 | $1 | $2 | $1 | $3 | $2 | $1 | $1 |

Each cell is the time one GPU takes at the rates in `napkin table` and `napkin compare`, times the GPUs billed, times the price record, at one significant digit. It assumes billing covers only that work at full speed: no startup, idle time, or rounding up to the billing increment, which `napkin cost` adds.
<!-- /napkin -->

Bytes that stay in HBM are almost free; bytes that cross PCIe cost about 50
times as much per terabyte, because the host link is about 50 times slower and
the GPU is billed while it waits.

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

That break-even ratio needs only each GPU's rate and price, so it can be
recomputed for every provider that prices both GPUs on the same terms.
`napkin breakeven` pairs the records above and divides the H100's work per
dollar by the B200's:

<!-- napkin breakeven --gpu h100 --vs b200 --dtype bf16 -->
| Provider | Terms | H100 per GPU-hour | B200 per GPU-hour | H100 bf16 per dollar | B200 bf16 per dollar | B200 cheaper per job above |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| AWS | on-demand | $6.88 | $14.2 | 360 PFLOP | 330 PFLOP | 1.1× H100's utilization |
| AWS | 1-year commitment | $5.42 | $11.2 | 460 PFLOP | 430 PFLOP | 1.1× H100's utilization |
| AWS | 3-year commitment | $5.05 | $10.5 | 490 PFLOP | 460 PFLOP | 1.1× H100's utilization |
| GCP | spot, preemptible | $6.59 | $4.95 | 380 PFLOP | 960 PFLOP | 0.39× H100's utilization |
| GCP | 1-year commitment | $7.67 | $11.1 | 320 PFLOP | 430 PFLOP | 0.76× H100's utilization |
| GCP | 3-year commitment | $4.86 | $7.09 | 510 PFLOP | 670 PFLOP | 0.76× H100's utilization |
| Modal | on-demand, preemptible | $3.95 | $6.25 | 630 PFLOP | 760 PFLOP | 0.83× H100's utilization |

Rates: H100 691 TFLOP/s bf16 (napkin-bench), B200 1.33 PFLOP/s bf16 (napkin-bench). Work per dollar is rate × 3600 s ÷ price per GPU-hour.
<!-- /napkin -->

The ranking does not survive every price list. On-demand, AWS charges $14.2
per B200 GPU-hour against $6.88 for an H100, 2.07 times as much, but the B200
does only 1.92 times the H100's measured BF16 work. At AWS prices the B200
costs more per job even at equal utilization, and the job above costs $5.35
on the H100 against $9.89 on the B200. GCP's spot prices point the other way:
a B200 GPU-hour costs less than an H100's, so the B200 stays cheaper per job
down to 0.39 of the H100's utilization. GCP lists no on-demand B200, and
Azure lists no B200 at all, so neither appears in the on-demand comparison.

As a cross-check, Simon Eskildsen's [napkin-math cost
table](https://github.com/sirupsen/napkin-math#cost-numbers) rounds AWS B200
prices to $10,000 per month on-demand and $8,000 with a 1-year plan. At 730
hours per month, `aws-b200` and `aws-b200-1y` give $10,400 and $8,190.

## Practice problems

[PROBLEMS.md](PROBLEMS.md) has seven problems with worked solutions hidden
until you open them: HBM traffic for a copy versus an add, tiny-copy latency,
a roofline crossover, KV-cache capacity, kernel fusion, decode batching, and
cost per token. Try each on paper first.

## Usage

```sh
cargo install --path crates/napkin

napkin gpus                                       # profiles and their basis
napkin summary --gpu h100                         # the memorization table
napkin table --gpu h100                           # the detailed tables
napkin unit-costs                                 # cost of 1 TB or 1 EFLOP per GPU
napkin variation                                  # spread across repeated runs
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
napkin breakeven --gpu h100 --vs b200 --dtype bf16  # work per dollar at paired prices
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

The numbers come from `napkin-bench` (Rust + cudarc), run on Modal:

```sh
modal run modal/bench.py --gpu h100     # writes profiles/h100.measured.json
modal run modal/bench.py --gpu h100 --repeat 3  # adds profiles/repeats/h100/*.json
napkin readme --write README.md PROBLEMS.md  # refreshes the tables
```

- `launch_us`: back-to-back empty-kernel launches, CUDA-event timed.
- `sync_us`: launch plus stream-synchronize round trip, wall clock.
- `host_copy_gbps`: pinned host memcpy on min(8, cpus) threads; read + write
  bytes counted.
- `h2d_gbps` / `d2h_gbps`: 512 MiB pinned async copies, event timed; the bytes
  that cross the link.
- `h2d_latency_us` / `d2h_latency_us`: 4-byte synchronous copies, wall clock;
  each sample includes the stream synchronize, so it is a per-copy fixed cost,
  not a bandwidth. The 2026-10-12 profiles predate
  [#32](https://github.com/gpu004/napkin-math-gpu/issues/32): their H2D copies
  also allocated and freed the 4-byte destination, which on `H100!:2` read
  8.9 μs against 6.0 μs for the copy alone.
- `hbm_gbps`: 512 MiB device-to-device memcpy; read + write counted (2x).
- `elementwise_gbps`: NVRTC `c[i] = a[i] + b[i]` fp32; two reads + one write (3x).
- `hbm_latency_ns`: single-thread pointer chase over a random cycle in a
  1 GiB buffer, ns per dependent load.
- `g2g_gbps`: 512 MiB peer copy, slower of the two directions; `None` with
  fewer than 2 GPUs or no peer access.
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
modal run modal/nccl.py --gpu h100 --count 2    # writes profiles/nccl-tests/h100x2/
modal run modal/nccl.py --gpu h100 --count 16   # two machines over RDMA: profiles/nccl-tests/h100x16/
```

See [BENCHMARKS.md](BENCHMARKS.md#nccl-tests) for what is recorded and how
`napkin collective` reads the curve.

## Techniques

- **Count algorithmic bytes, not allocations.** A copy is 2x its buffer; an
  elementwise add is 3x. KV cache and weight reads are traffic too.
- **Dense is not sparse.** NVIDIA's headline FLOPs assume structured sparsity;
  halve them or measure.
- **Keep the qualifiers attached.** dtype, direction, topology, dense/sparse,
  measured/theoretical, software stack — no throughput number is universal.
- **Sum serial stages; max overlapped ones.** Overlap is a property of the
  workload, not of the GPU.
- **Weights barely compress, so compression won't speed up loading them.**
  As Simon notes, floats that use their full mantissa barely compress, and
  BF16 weights do. Only the 8-bit exponent is predictable:
  [DFloat11](https://arxiv.org/abs/2504.11651) entropy-codes it losslessly and
  saves about 30%. Loading Llama-3-70B's 140 GB over a 56 GB/s host link takes
  2.5 s; 30% fewer bytes saves under a second and adds a decompression kernel.
  Fewer bits per weight (`napkin llm --weight-bits 4`) cut the bytes 4x, but
  they change the model.
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
* [AWS price list files](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/using-ppslong.html), [Azure Retail Prices API](https://learn.microsoft.com/en-us/rest/api/cost-management/retail-prices/azure-retail-prices), and [GCP accelerator-optimized pricing](https://cloud.google.com/products/compute/pricing/accelerator-optimized). Machine-readable sources behind `prices/aws.json`, `prices/azure.json`, and `prices/gcp.json`.
* [Modal GPU Glossary](https://modal.com/gpu-glossary). Fast definitions for SMs, warps, HBM, Tensor Cores, and the rest of the vocabulary.
* [Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking](https://arxiv.org/abs/1804.06826). The template for measuring latencies and cache hierarchies yourself.
* [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/abs/2402.13499). Microbenchmarked Hopper latencies, Tensor Cores, and memory.
* [NVBandwidth](https://github.com/NVIDIA/nvbandwidth). NVIDIA's tool for host↔device and device↔device bandwidth; the state of the art for transfer numbers.
* [NCCL Tests](https://github.com/NVIDIA/nccl-tests). Collective benchmarks; read [PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) for algorithm versus bus bandwidth.
* [Nsight Compute](https://developer.nvidia.com/nsight-compute). Per-kernel roofline, memory, and occupancy analysis.
* [cudarc](https://github.com/chelsea0x3b/cudarc). The Rust CUDA bindings `napkin-bench` is built on.
* [Napkin Math](https://github.com/sirupsen/napkin-math). Simon Eskildsen's original project, which this one follows.
