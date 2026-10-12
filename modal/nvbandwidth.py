"""Run pinned NVIDIA NVBandwidth on Modal GPUs.

Usage:
    modal run modal/nvbandwidth.py --gpu h100

Writes the verbatim NVBandwidth stdout to
profiles/nvbandwidth/<gpu>x2/nvbandwidth.json, provenance to run.json, and
topology probes to topo.txt. `napkin` parses the evidence at build time.

PCIe links downtrain when idle, so the link is sampled every 100 ms while the
copies run and the highest generation and width seen per GPU are kept.
`nvidia-smi topo -m` and `nvlink --status` are recorded with their exit codes
because some Modal hosts do not expose them. `--disableAffinity` is passed
because the Modal sandbox rejects NVML's `nvmlDeviceSetCpuAffinity`, which
otherwise aborts the first host testcase; host buffers are then not NUMA-pinned.
"""

import hashlib
import json
import subprocess
import sys
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


def _probe(cmd: list[str]) -> str:
    out = subprocess.run(cmd, capture_output=True, text=True)
    return f"$ {' '.join(cmd)}\n# exit {out.returncode}\n{out.stdout}{out.stderr}"


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
    topo = "\n".join(
        _probe(cmd)
        for cmd in (
            ["nvidia-smi", "topo", "-m"],
            ["nvidia-smi", "topo", "-p2p", "n"],
            ["nvidia-smi", "nvlink", "-s"],
        )
    )
    peak: dict[int, list[int]] = {}
    stop = threading.Event()
    poller = threading.Thread(target=_poll_links, args=(stop, peak))
    poller.start()
    captured_at = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    out = subprocess.run([BINARY, *ARGS], capture_output=True)
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
        "stderr": out.stderr.decode("utf-8", errors="replace"),
        "topo": topo,
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
        print(result["stdout"].decode("utf-8", errors="replace"))
        raise RuntimeError(f"nvbandwidth exited {run['exit_code']}")
    measured = json.loads((ROOT / "profiles" / f"{gpu}.measured.json").read_text())
    expected = (measured.get("pcie_gen"), measured.get("pcie_width"))
    links = run["pcie_links"]
    mismatches = [
        f"GPU{link['index']} sampled Gen{link['pcie_gen']} x{link['pcie_width']}"
        for link in links
        if (link["pcie_gen"], link["pcie_width"]) != expected
    ]
    if len(links) != len(run["gpus"]) or not links or mismatches:
        details = ", ".join(mismatches) or "missing sampled link data"
        print(
            f"PCIe mismatch for {gpu}: expected Gen{expected[0]} x{expected[1]}; {details}",
            file=sys.stderr,
        )
        raise RuntimeError("NVBandwidth evidence not written due to PCIe mismatch")

    directory = ROOT / "profiles" / "nvbandwidth" / f"{gpu}x2"
    topo = result["topo"]
    if not topo.endswith("\n"):
        topo += "\n"
    files = {
        "nvbandwidth.json": result["stdout"],
        "run.json": (json.dumps(run, indent=2) + "\n").encode(),
        "topo.txt": topo.encode(),
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, contents in files.items():
        path = directory / name
        path.write_bytes(contents)
        relative_path = path.relative_to(ROOT).as_posix()
        print(f"sha256 {hashlib.sha256(contents).hexdigest()} {relative_path}")
