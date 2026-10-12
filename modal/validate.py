"""Run the prediction-versus-observation validation cases on one Modal GPU.

Usage:
    modal run modal/validate.py

Opt-in and never part of CI. Each case in validation/<case>/case.json declares
its workload, the matching `napkin estimate` inputs and its tolerance before any
run. This script measures the jobs on one pinned H100 (`H100!`) and writes the
raw tool output plus run.json (command, tool versions, date, driver, CUDA,
Modal request, PCIe link, GPU time) next to each case. `napkin validate`
renders VALIDATION.md from those committed files without a GPU.

Observations come from maintained tools: cuBLAS through `torch.mm` timed by
`torch.utils.benchmark.Timer`, NVIDIA NVBandwidth for host copies, and
torch.profiler to count kernels. `nvidia-smi` samples clocks, power and the
PCIe link every 100 ms while each case runs, because links downtrain when idle
and sustained GEMMs drop clocks. `--disableAffinity` is passed to NVBandwidth
because the Modal sandbox rejects NVML's CPU-affinity call.
"""

import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import modal

GPU_REQUEST = "H100!"
TORCH = "2.8.0"
TORCH_INDEX = "https://download.pytorch.org/whl/cu129"
NVB_REPO = "https://github.com/NVIDIA/nvbandwidth"
NVB_TAG = "v0.10.0"
NVB_COMMIT = "82fc4e8c6afa0babb8687793678f615b3b8d793e"
NVB_BINARY = "/opt/nvbandwidth/build/nvbandwidth"
NVB_ARGS = [
    "--format",
    "json",
    "--disableAffinity",
    "--bufferSize",
    "1024",
    "--loopCount",
    "16",
    "--testSamples",
    "7",
    "--testcase",
    "host_to_device_memcpy_ce",
    "device_to_host_memcpy_ce",
]
GEMM_SHAPE = (16384, 28672, 8192)
GEMM_STMT = "torch.mm(a, b, out=c)"
TINY_OPS = 10_000
TINY_STMT = f"for _ in range({TINY_OPS}):\n    x.add_(1.0)"
GPU_FIELDS = "index,name,uuid,pci.bus_id,driver_version,memory.total,vbios_version"
MONITOR_FIELDS = ",".join(
    [
        "timestamp",
        "clocks.sm",
        "clocks.mem",
        "power.draw",
        "temperature.gpu",
        "clocks_event_reasons.active",
        "pcie.link.gen.current",
        "pcie.link.width.current",
        "pcie.link.gen.max",
        "pcie.link.width.max",
    ]
)

ROOT = Path(__file__).resolve().parent.parent
CASES = ROOT / "validation"

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        f"git clone {NVB_REPO} /opt/nvbandwidth",
        f"git -C /opt/nvbandwidth checkout --detach {NVB_COMMIT}",
        f"test $(git -C /opt/nvbandwidth rev-parse {NVB_TAG}^{{commit}}) = {NVB_COMMIT}",
        "cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build -DCMAKE_BUILD_TYPE=Release",
        "make -C /opt/nvbandwidth/build -j8",
    )
    .pip_install(f"torch=={TORCH}", "numpy", index_url=TORCH_INDEX)
)

app = modal.App("napkin-validation-issue12", image=image)


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


class Monitor:
    """Samples nvidia-smi every 100 ms while a case runs; keeps the raw CSV."""

    def __enter__(self) -> "Monitor":
        self.proc = subprocess.Popen(
            ["nvidia-smi", f"--query-gpu={MONITOR_FIELDS}", "--format=csv", "-lms", "100"],
            stdout=subprocess.PIPE,
            text=True,
        )
        self.csv = ""
        return self

    def __exit__(self, *_: object) -> None:
        self.proc.terminate()
        self.csv, _ = self.proc.communicate()

    def summary(self) -> dict[str, object]:
        rows = [
            [cell.strip() for cell in line.split(",")] for line in self.csv.strip().splitlines()[1:]
        ]

        def numbers(column: int) -> list[float]:
            return [float(r[column].split()[0]) for r in rows if r[column].split()[0][0].isdigit()]

        sm, power = numbers(1), numbers(3)
        return {
            "samples": len(rows),
            "sm_clock_mhz": {"min": min(sm), "max": max(sm)} if sm else None,
            "power_w": {"min": min(power), "max": max(power)} if power else None,
            "pcie_gen_peak": max(numbers(6), default=None),
            "pcie_width_peak": max(numbers(7), default=None),
            "pcie_gen_max": max(numbers(8), default=None),
            "pcie_width_max": max(numbers(9), default=None),
        }


def _timer_files(measurement: object) -> dict[str, str]:
    raw = {
        "stmt": measurement.task_spec.stmt,
        "number_per_run": measurement.number_per_run,
        "raw_times_s": measurement.raw_times,
        "times_s": measurement.times,
        "median_s": measurement.median,
    }
    return {"timer.txt": str(measurement) + "\n", "timer.json": json.dumps(raw, indent=2) + "\n"}


def _gemm() -> tuple[dict[str, str], dict[str, object]]:
    import torch
    from torch.utils.benchmark import Timer

    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    m, n, k = GEMM_SHAPE
    a = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(k, n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        torch.mm(a, b, out=c)
    torch.cuda.synchronize()
    timer = Timer(stmt=GEMM_STMT, globals={"torch": torch, "a": a, "b": b, "c": c})
    with Monitor() as monitor:
        measurement = timer.blocked_autorange(min_run_time=10.0)
    files = _timer_files(measurement) | {"nvidia-smi.csv": monitor.csv}
    run = {
        "tool": "torch.utils.benchmark.Timer.blocked_autorange(min_run_time=10.0) over torch.mm",
        "command": GEMM_STMT,
        "setup": "; ".join(
            [
                f"a = randn({m}, {k}), b = randn({k}, {n}), c = empty({m}, {n}), bf16 on cuda",
                "allow_bf16_reduced_precision_reduction = False",
                "3 s warmup",
            ]
        ),
        "monitor": monitor.summary(),
    }
    return files, run


def _pcie() -> tuple[dict[str, str], dict[str, object]]:
    version = subprocess.run([NVB_BINARY, "--version"], capture_output=True, text=True)
    with Monitor() as monitor:
        out = subprocess.run([NVB_BINARY, *NVB_ARGS], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"nvbandwidth exited {out.returncode}: {out.stderr}{out.stdout}")
    files = {"nvbandwidth.json": out.stdout, "nvidia-smi.csv": monitor.csv}
    run = {
        "tool": "nvbandwidth",
        "repository": NVB_REPO,
        "tag": NVB_TAG,
        "commit": NVB_COMMIT,
        "version_output": version.stdout.strip(),
        "command": ["nvbandwidth", *NVB_ARGS],
        "exit_code": out.returncode,
        "stderr": out.stderr,
        "monitor": monitor.summary(),
    }
    return files, run


def _tiny() -> tuple[dict[str, str], dict[str, object]]:
    import torch
    from torch.profiler import ProfilerActivity, profile
    from torch.utils.benchmark import Timer

    x = torch.zeros(1, device="cuda", dtype=torch.float32)
    for _ in range(TINY_OPS):
        x.add_(1.0)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(TINY_OPS):
            x.add_(1.0)
        torch.cuda.synchronize()
    kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    timer = Timer(stmt=TINY_STMT, globals={"x": x})
    with Monitor() as monitor:
        measurement = timer.blocked_autorange(min_run_time=5.0)
    table = prof.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    files = _timer_files(measurement) | {
        "profiler.txt": table + "\n",
        "nvidia-smi.csv": monitor.csv,
    }
    run = {
        "tool": "torch.utils.benchmark.Timer.blocked_autorange(min_run_time=5.0); torch.profiler",
        "command": TINY_STMT,
        "setup": "x = zeros(1), fp32 on cuda; one untimed pass of the loop as warmup",
        "profiled_cuda_kernels": len(kernels),
        "monitor": monitor.summary(),
    }
    return files, run


@app.function(gpu=GPU_REQUEST, cpu=8.0, memory=32768, timeout=30 * 60)
def run_cases() -> dict[str, dict[str, str]]:
    import importlib.metadata

    import torch

    started = time.monotonic()
    gpus = [
        dict(zip(GPU_FIELDS.split(","), [cell.strip() for cell in line.split(",")], strict=True))
        for line in _smi(f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader").strip().splitlines()
    ]
    common = {
        "modal_request": GPU_REQUEST,
        "image": "nvidia/cuda:12.9.1-devel-ubuntu24.04",
        "gpus": gpus,
        "driver": gpus[0]["driver_version"],
        "cuda_driver": _smi().split("CUDA Version:")[1].split()[0],
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cublas": importlib.metadata.version("nvidia-cublas-cu12"),
        "clocks_query": _smi("-q", "-d", "CLOCK"),
    }
    results: dict[str, dict[str, str]] = {}
    for case, run_case in [
        ("h100-gemm-bf16", _gemm),
        ("h100-pcie-roundtrip", _pcie),
        ("h100-tiny-kernels", _tiny),
    ]:
        case_started = time.monotonic()
        captured_at = _now()
        files, run = run_case()
        run = {"case": case, "captured_at": captured_at} | run | common
        run["case_seconds"] = round(time.monotonic() - case_started, 1)
        run["container_seconds_so_far"] = round(time.monotonic() - started, 1)
        results[case] = files | {"run.json": json.dumps(run, indent=2) + "\n"}
    return results


@app.local_entrypoint()
def main() -> None:
    for case in ("h100-gemm-bf16", "h100-pcie-roundtrip", "h100-tiny-kernels"):
        if not (CASES / case / "case.json").exists():
            raise FileNotFoundError(f"declare {CASES / case / 'case.json'} before running")
    started = time.monotonic()
    results = run_cases.remote()
    print(f"remote call took {time.monotonic() - started:.1f} s")
    for case, files in results.items():
        for name, text in files.items():
            (CASES / case / name).write_text(text)
            print(f"Wrote {CASES / case / name}")
