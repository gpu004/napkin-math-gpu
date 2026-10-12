# Behind the GPU Napkin Math numbers

The repository should ingest, normalize, and cross-check authoritative evidence.
It should not reimplement mature NVIDIA benchmark and profiling tools.

The user-facing product remains the rounded numbers and arithmetic in the
[README](README.md). Modal is an execution environment for the evidence tools,
not itself a measurement methodology.

## Source-of-truth policy

| Napkin input | Primary authority | What the repository keeps |
| --- | --- | --- |
| Peak FLOPs by dtype | NVIDIA product datasheet | Dense/sparse convention, precision, form factor, source URL, published value |
| HBM bandwidth and capacity | NVIDIA product datasheet | Theoretical bandwidth and capacity |
| NVLink generation and peak bandwidth | NVIDIA product or system datasheet | Per-GPU theoretical aggregate and topology assumptions |
| CPU↔GPU and GPU↔GPU bandwidth | NVIDIA NVBandwidth | Version, command, topology, raw JSON, normalized directional values |
| P2P latency and connectivity | CUDA Samples `p2pBandwidthLatencyTest` | Pinned CUDA Samples revision, command, raw matrix |
| AllReduce | NVIDIA NCCL Tests `all_reduce_perf` | Algorithm bandwidth, bus bandwidth, message-size curve, topology |
| AllGather | NVIDIA NCCL Tests `all_gather_perf` | Algorithm bandwidth, bus bandwidth, message-size curve, topology |
| ReduceScatter | NVIDIA NCCL Tests `reduce_scatter_perf` | Algorithm bandwidth, bus bandwidth, message-size curve, topology |
| L1/L2/HBM behavior and roofline | NVIDIA Nsight Compute | `.ncu-rep` export, selected sections/metrics, kernel identity |
| Achievable GEMM | cuBLAS/cuBLASLt | Library/CUDA version, dtype, shape, layout, algorithm, measured throughput |
| End-to-end model validity | Published or reproduced MLPerf results | Matched system/workload, prediction, observed result, error ratio |

If an authoritative source exists, a homegrown throughput loop must not define a
published napkin number. Small bespoke code is still appropriate for parsing,
normalization, correctness checks, prediction, and gaps such as a framework-level
tiny-kernel floor.

## Important CUDA tooling update

CUDA Samples removed `bandwidthTest` in CUDA 12.9 because NVIDIA found it could
produce inaccurate results. NVIDIA now directs bandwidth measurement to
[NVBandwidth](https://github.com/NVIDIA/nvbandwidth), which covers host↔device,
device↔device, copy-engine and SM paths, directional and bidirectional tests,
latency, verification, and JSON output.

Use CUDA Samples'
[`p2pBandwidthLatencyTest`](https://github.com/NVIDIA/cuda-samples/tree/v13.3/cpp/5_Domain_Specific/p2pBandwidthLatencyTest)
for its P2P latency/connectivity matrix, not as the sole bandwidth authority.
CUDA Samples 13.4 removed it as obsolete, so pin `v13.3` or earlier; the link
above points at that tag. Always pin the CUDA Samples revision so a profile can
be reproduced after samples move or change.

## Evidence flow

```text
vendor datasheets         NVIDIA/MLCommons tools
        │                        │
        └── pinned raw evidence ──┘
                     │
             typed normalization
                     │
             consistency checks
                     │
              versioned profile
                     │
              rounded README
                     │
             MLPerf prediction check
```

Every normalized metric should carry:

- authority and tool name;
- source URL or exact command;
- tool, CUDA, driver, and library versions where applicable;
- GPU model, form factor, count, and topology;
- theoretical versus achieved classification;
- precision and dense/sparse convention;
- raw artifact path;
- units and byte/FLOP counting convention.

Theoretical and achieved values must remain separate. For example, a datasheet
BF16 peak is a hardware ceiling; a cuBLASLt result is an achievable library
ceiling. Both can be useful, but they answer different questions.

## Tool responsibilities

### Vendor datasheets

Datasheets own stable hardware facts: memory capacity, theoretical HBM bandwidth,
peak arithmetic throughput, PCIe generation, and peak NVLink bandwidth. Record
the exact SKU and form factor. Do not silently mix H100 SXM with H100 NVL, or A100
PCIe with A100 SXM.

Start with NVIDIA's [H100 product
specifications](https://www.nvidia.com/en-us/data-center/h100/) and [A100 80 GB
datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a100/pdf/nvidia-a100-datasheet-nvidia-us-2188504-web.pdf),
then pin the exact source used by each profile.

Datasheet tensor figures must record whether structured sparsity is enabled. The
napkin table should normally use dense numbers unless the workload explicitly
depends on supported sparsity.

### NVBandwidth and CUDA Samples

NVBandwidth owns achieved memcpy bandwidth across the actual system topology.
At minimum collect:

- `host_to_device_memcpy_ce`;
- `device_to_host_memcpy_ce`;
- directional device-to-device copy-engine bandwidth;
- directional device-to-device SM bandwidth;
- latency for the relevant host/device and device/device paths;
- JSON output with verification enabled.

CUDA Samples' `p2pBandwidthLatencyTest` supplies an independently maintained P2P
connectivity and latency matrix. Store both the sample revision and its raw
stdout.

### NCCL Tests

Use NVIDIA's [NCCL Tests](https://github.com/NVIDIA/nccl-tests). Collect full
message-size curves for `all_reduce_perf`, `all_gather_perf`, and
`reduce_scatter_perf`, not one headline number. Retain both `algbw` and `busbw`:
the former estimates application-visible time; the latter helps compare link
utilization across collective algorithms.

Correctness checks stay enabled. Record NCCL version, GPU count, rank layout,
topology, datatype, reduction operator, warmups, iterations, and whether CUDA
graphs or buffer registration were enabled.

`modal/nccl.py` builds NCCL Tests v2.21.1 (`afd59ab`, `MPI=0`) against the
NCCL 2.27.3 packaged in `nvidia/cuda:12.9.1-devel-ubuntu24.04` and writes one
directory per GPU and count, `profiles/nccl-tests/<id>x<count>/`:

- `run.json`: Modal request, GPU count, rank layout (one process, one thread,
  `-g <count>` GPUs), topology label, dtype (`bfloat16`), reduction (`sum`),
  NCCL Tests revision and build line, CUDA image, NCCL packages, the
  `nvidia-smi` GPU/driver/PCIe query, the environment and every exact command.
- `<collective>_perf.txt` and `.json`: raw stdout and `-J` JSON of the sweep
  `-b 8 -e 4G -f 2 -w 5 -n 20 -c 1` (8 B to 4 GiB doubling, 5 warmups, 20
  timed iterations, correctness checks on, no CUDA graphs, no buffer
  registration). Both out-of-place and in-place rows are kept.
- `topo.txt`: `nvidia-smi topo -m`, `topo -p2p n` and `nvlink -s`. Modal's
  sandbox fails `topo -m` ("Failed to run topology matrix"), so the failure is
  kept as-is and the other two stand in for it.
- `nccl_topology.txt`: NCCL's own `INIT,GRAPH` log for one tiny AllReduce.
  The `nvlink` topology label means NCCL built every ring over NVLink (graph
  type `NVL`) and every channel is `via P2P`; the launcher refuses to label a
  run `nvlink` otherwise.

`crates/napkin/build.rs` parses these files at build time and rejects any run
whose correctness checks failed or were off, whose `algbw` does not equal
`size / time` or whose `busbw` does not equal `algbw` times the
[PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md)
factor (`2(n-1)/n` for AllReduce, `(n-1)/n` for AllGather and ReduceScatter).
`size` keeps the NCCL Tests meaning: the per-rank buffer for AllReduce, the
total gathered output for AllGather, the total per-rank input for
ReduceScatter. The estimator uses the out-of-place curve.

`napkin collective` reads a time off the curve for the exact GPU, count,
topology and dtype that was measured, and refuses anything else. It never
scales a curve to another GPU count or topology, and it never extrapolates past
the largest measured size. Below the smallest size it returns the smallest
size's time; between two sizes it interpolates time linearly in bytes. It
reports the latency regime when the time is less than twice the curve's
fastest call, meaning the fixed per-call cost is at least half the answer.
The collective bandwidths stay separate from the `napkin-bench` peer-copy
rate (one GPU pair, `cuMemcpyPeer`) and the datasheet NVLink figure.

### Nsight Compute

Use [Nsight Compute](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)'s
Speed of Light, Memory Workload Analysis, and roofline sections on
representative kernels. The profiler owns achieved L1/L2/HBM behavior,
arithmetic intensity, and hierarchical roofline placement.

Keep the `.ncu-rep` artifact and exported metrics. Do not convert a profiled
kernel's achieved bandwidth into a universal hardware specification; compare it
with the datasheet ceiling and label it as workload-specific.

### cuBLAS and cuBLASLt

Use [cuBLAS/cuBLASLt](https://docs.nvidia.com/cuda/cublas/) for GEMM rather than
a handwritten matrix-multiply kernel. The evidence record must include shape,
dtype, accumulation type, layouts, epilogue, workspace, and selected algorithm.
Query cuBLASLt heuristics once and reuse the selected algorithm during timing.

The existing `compute_tflops` field remains the cuBLAS `GemmEx` 8192³ peak; it
is not a universal rate for arbitrary GEMM dimensions. `gemm_shapes` stores
separate cuBLASLt measurements for the `square`, `prefill`, `decode-m8`,
`decode-m64`, and `square-k8191` cases in fp16 and bf16. They use TN
column-major layouts, fp32 accumulation and scale, default epilogue, and a
32 MiB heuristic workspace limit. Each heuristic is queried once before the
existing warm-up and sustained-round timing, then that algorithm is reused for
all timed calls. Per-round samples and matching NVML state are retained; a
deterministic 256-output sample is checked against a float64 reference built
from the dtype-rounded inputs. The estimate command selects a shape rate only
for an exact dtype and M/N/K match; otherwise it explicitly reports that the
shape is unmatched and falls back to the 8192³ `GemmEx` peak without
interpolation.

### MLPerf

[MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) is the
model-level falsification step. Pick published results or reproduce a workload
only when the system, GPU count, scenario, accuracy target, and software
configuration are comparable.

For each validation case, publish:

```text
napkin prediction
observed MLPerf throughput or latency
prediction / observation ratio
dominant term predicted by the model
explanation when the error is outside the stated tolerance
```

The goal is not to fit the napkin constants to MLPerf. The goal is to discover
where the simple model is missing utilization, communication, batching, or
software overhead.

## Status of the checked-in profiles

Datasheet ceilings live in `profiles/<id>.json` and carry their own `source`
notes. Measured calibrations live in `profiles/<id>.measured.json`, produced by
`napkin-bench` on Modal:

- H100: [profile](profiles/h100.json) · [measured](profiles/h100.measured.json)
- A100: [profile](profiles/a100.json) · [measured](profiles/a100.measured.json)
- NCCL Tests collectives: [H100 × 2](profiles/nccl-tests/h100x2) ·
  [H100 × 8](profiles/nccl-tests/h100x8) (a different Modal host: driver
  610.57.04 and a PCIe Gen 4 host link, versus 580.95.05 and Gen 5 for × 2)

NVBandwidth v0.10.0 evidence is ingested with per-metric provenance from
`profiles/raw/*.nvbandwidth.json` and the corresponding run metadata.

The `napkin-profile` types are the shared contract between the estimator
crate and the `napkin-bench` producer. The measured files must be replaced
metric-by-metric by the source policy above before being called canonical.

`modal/bench.py` is therefore a thin launcher for a calibration harness. Future
work should add runners/parsers around pinned upstream tools instead of
extending the homegrown measurement loops.

Prediction-versus-observation cases live under `validation/<case>/`: `case.json`
declares the workload and tolerance, while the raw observation and `run.json`
record the measured evidence. `napkin validate` builds the GPU-free report in
`VALIDATION.md`; collect or refresh observations with the opt-in
`modal run modal/validate.py`.

## Merge a GEMM shape suite without replacing scalar calibration

Run a full benchmark and merge only its cuBLASLt shape records into the
existing measured profile:

```sh
modal run modal/bench.py --gpu h100 --merge-shapes
```

The run is preserved at `target/modal/<gpu>.run.json`. `napkin merge-shapes`
requires successful run and shape checks plus matching device name and memory
size; it copies only the provenance-bearing `gemm_shapes` suite. The profile's
generic cuBLAS `GemmEx` rate, scalar samples, checks, and NVML state remain from
their original capture. Quick runs cannot be merged.

## Development checks

GPU-free checks run locally and in CI:

```sh
uv run --only-dev --locked prek
```
