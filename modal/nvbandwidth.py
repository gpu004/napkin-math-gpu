"""Run pinned NVIDIA NVBandwidth on Modal GPUs.

Usage:
    modal run modal/nvbandwidth.py --gpu h100

Writes the verbatim JSON to profiles/raw/<gpu>.nvbandwidth.json and the exact
command, Modal request, GPU identity, PCIe link and topology to
profiles/raw/<gpu>.nvbandwidth.run.json. `napkin` parses both at build time.

PCIe links downtrain when idle, so the link is sampled every 100 ms while the
copies run and the highest generation and width seen per GPU are kept.
`nvidia-smi topo -m` and `nvlink --status` are recorded with their exit codes
because some Modal hosts do not expose them. `--disableAffinity` is passed
because the Modal sandbox rejects NVML's `nvmlDeviceSetCpuAffinity`, which
otherwise aborts the first host testcase; host buffers are then not NUMA-pinned.
"""

import json
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path

import modal

GPUS = {"a100": "A100-80GB:2", "h100": "H100!:2", "b200": "B200:2", "b300": "B300:2"}

REPO = "https://github.com/NVIDIA/nvbandwidth"
TAG = "v0.10.0"
COMMIT = "82fc4e8c6afa0babb8687793678f615b3b8d793e"
BINARY = "/opt/nvbandwidth/build/nvbandwidth"
TESTCASES = [
    "host_to_device_memcpy_ce",
    "device_to_host_memcpy_ce",
    "host_to_device_memcpy_sm",
    "device_to_host_memcpy_sm",
    "device_to_device_memcpy_read_ce",
    "device_to_device_memcpy_write_ce",
    "device_to_device_memcpy_read_sm",
    "device_to_device_memcpy_write_sm",
    "host_device_latency_sm",
    "device_to_device_latency_sm",
]
ARGS = [
    "--format",
    "json",
    "--disableAffinity",
    "--bufferSize",
    "512",
    "--testSamples",
    "7",
    "--testcase",
    *TESTCASES,
]
GPU_FIELDS = "index,name,uuid,pci.bus_id,driver_version,memory.total"
LINK_FIELDS = (
    "index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max"
)

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        f"git clone {REPO} /opt/nvbandwidth",
        f"git -C /opt/nvbandwidth checkout --detach {COMMIT}",
        f"test $(git -C /opt/nvbandwidth rev-parse {TAG}^{{commit}}) = {COMMIT}",
        "cmake -S /opt/nvbandwidth -B /opt/nvbandwidth/build -DCMAKE_BUILD_TYPE=Release",
        "make -C /opt/nvbandwidth/build -j8",
    )
)

app = modal.App("napkin-nvbandwidth-issue9", image=image)


def _smi(*args: str) -> str:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=True).stdout


def _smi_probe(*args: str) -> dict[str, object]:
    out = subprocess.run(["nvidia-smi", *args], capture_output=True, text=True)
    return {
        "command": ["nvidia-smi", *args],
        "exit_code": out.returncode,
        "stdout": out.stdout,
        "stderr": out.stderr,
    }


def _csv(text: str) -> list[list[str]]:
    return [[cell.strip() for cell in line.split(",")] for line in text.strip().splitlines()]


def _poll_links(stop: threading.Event, peak: dict[int, list[int]]) -> None:
    while not stop.wait(0.1):
        rows = _csv(_smi(f"--query-gpu={LINK_FIELDS}", "--format=csv,noheader,nounits"))
        for index, *values in rows:
            parsed = [int(v) if v.isdigit() else 0 for v in values]
            best = peak.setdefault(int(index), [0, 0, 0, 0])
            peak[int(index)] = [max(a, b) for a, b in zip(best, parsed, strict=True)]


def _run(gpu_request: str) -> dict[str, object]:
    version = subprocess.run([BINARY, "--version"], capture_output=True, text=True).stdout.strip()
    gpus = [
        dict(zip(GPU_FIELDS.split(","), row, strict=True))
        for row in _csv(_smi(f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader"))
    ]
    topology = [_smi_probe("topo", "-m"), _smi_probe("nvlink", "--status")]
    peak: dict[int, list[int]] = {}
    stop = threading.Event()
    poller = threading.Thread(target=_poll_links, args=(stop, peak))
    poller.start()
    captured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = subprocess.run([BINARY, *ARGS], capture_output=True, text=True)
    stop.set()
    poller.join()
    links = [
        {
            "index": index,
            "pcie_gen": gen,
            "pcie_gen_max": gen_max,
            "pcie_width": width,
            "pcie_width_max": width_max,
        }
        for index, (gen, gen_max, width, width_max) in sorted(peak.items())
    ]
    return {
        "stdout": out.stdout,
        "stderr": out.stderr,
        "run": {
            "tool": "nvbandwidth",
            "repository": REPO,
            "tag": TAG,
            "commit": COMMIT,
            "version_output": version,
            "command": ["nvbandwidth", *ARGS],
            "exit_code": out.returncode,
            "gpu_request": gpu_request,
            "captured_at": captured_at,
            "gpus": gpus,
            "pcie_links": links,
            "topology": topology,
        },
    }


FUNCS = {
    name: app.function(
        gpu=request,
        cpu=8.0,
        memory=16384,
        timeout=30 * 60,
        name=f"nvbandwidth_{name}",
    )(_run)
    for name, request in GPUS.items()
}


@app.local_entrypoint()
def main(gpu: str = "h100") -> None:
    if gpu not in FUNCS:
        raise ValueError(f"--gpu must be one of: {sorted(FUNCS)}")
    result = FUNCS[gpu].remote(GPUS[gpu])
    print(result["stderr"])
    run = result["run"]
    if run["exit_code"] != 0:
        print(result["stdout"])
        raise RuntimeError(f"nvbandwidth exited {run['exit_code']}")
    raw = ROOT / "profiles" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / f"{gpu}.nvbandwidth.json").write_text(result["stdout"])
    (raw / f"{gpu}.nvbandwidth.run.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Wrote {raw / f'{gpu}.nvbandwidth.json'}")
