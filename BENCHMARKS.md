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

The `napkin::profile` types are the shared contract between the estimator
crate and the `napkin-bench` producer. The measured files must be replaced
metric-by-metric by the source policy above before being called canonical.

`modal/bench.py` is therefore a thin launcher for a calibration harness. Future
work should add runners/parsers around pinned upstream tools instead of
extending the homegrown measurement loops.

## Development checks

GPU-free checks run locally and in CI:

```sh
sh scripts/check.sh
```
