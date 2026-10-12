"""Run pinned NVIDIA NCCL Tests on Modal and keep the raw evidence.

Usage:
    modal run modal/nccl.py --gpu h100 --count 2

Writes profiles/nccl-tests/<gpu>x<count>/: run.json (provenance), topo.txt
(`nvidia-smi topo -m`, `topo -p2p n`, `nvlink -s`; Modal's sandbox fails
`topo -m`, so the other two and NCCL's own graph log stand in for it),
nccl_topology.txt (NCCL_DEBUG=INFO INIT,GRAPH for one tiny AllReduce), and
the stdout plus `-J` JSON of each collective.

The topology label in REQUESTS is checked against NCCL's graph log: "nvlink"
means one node where NCCL built every ring over NVLink (graph type NVL) and
moved data GPU-to-GPU via P2P, never through host memory or the network.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import modal

NCCL_TESTS_REPO = "https://github.com/NVIDIA/nccl-tests"
NCCL_TESTS_TAG = "v2.21.1"
NCCL_TESTS_COMMIT = "afd59abab774a6a4b492eb344bd0c845a0b7bb4d"
CUDA_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu24.04"
GENCODE = " ".join(f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in ("80", "90", "100", "103"))
COLLECTIVES = ("all_reduce_perf", "all_gather_perf", "reduce_scatter_perf")
SWEEP = ["-b", "8", "-e", "4G", "-f", "2", "-w", "5", "-n", "20", "-c", "1"]
DTYPE = "bfloat16"
REDOP = "sum"
REQUESTS = {
    ("h100", 2): ("H100!:2", "nvlink"),
    ("h100", 8): ("H100!:8", "nvlink"),
    ("b200", 8): ("B200:8", "nvlink"),
}

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git", "build-essential")
    .run_commands(
        f"git clone {NCCL_TESTS_REPO} /opt/nccl-tests",
        f"cd /opt/nccl-tests && git checkout {NCCL_TESTS_COMMIT}",
        f'cd /opt/nccl-tests && make -j MPI=0 NVCC_GENCODE="{GENCODE}"',
    )
)

app = modal.App("napkin-nccl-tests-e9f37d", image=image)


def _capture(cmd: list[str], env: dict[str, str] | None = None) -> str:
    import subprocess

    out = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if out.returncode != 0:
        raise RuntimeError(f"{cmd[0]} exited {out.returncode}: {out.stderr}")
    return out.stdout


def _probe(cmd: list[str]) -> str:
    import subprocess

    out = subprocess.run(cmd, capture_output=True, text=True)
    return f"$ {' '.join(cmd)}\n# exit {out.returncode}\n{out.stdout}{out.stderr}"


def _run(count: int, gpu_request: str, topology: str) -> dict[str, str]:
    import os

    env = {
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
        "NCCL_DEBUG": "VERSION",
    }
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]
    files = {
        "topo.txt": "\n".join(
            _probe(cmd)
            for cmd in (
                ["nvidia-smi", "topo", "-m"],
                ["nvidia-smi", "topo", "-p2p", "n"],
                ["nvidia-smi", "nvlink", "-s"],
            )
        ),
        "nccl_topology.txt": _capture(
            [
                "/opt/nccl-tests/build/all_reduce_perf",
                "-b",
                "8",
                "-e",
                "8",
                "-g",
                str(count),
            ],
            {**env, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH"},
        ),
    }
    graph = files["nccl_topology.txt"]
    if topology == "nvlink" and (
        "type NVL" not in graph
        or "via P2P" not in graph
        or "via NET" in graph
        or "via SHM" in graph
    ):
        raise RuntimeError("NCCL did not build an all-NVLink P2P topology; see nccl_topology.txt")
    commands = {}
    for name in COLLECTIVES:
        json_path = f"/tmp/{name}.json"
        cmd = [
            f"/opt/nccl-tests/build/{name}",
            *SWEEP,
            "-g",
            str(count),
            "-d",
            DTYPE,
            "-o",
            REDOP,
            "-J",
            json_path,
        ]
        files[f"{name}.txt"] = _capture(cmd, env)
        files[f"{name}.json"] = Path(json_path).read_text()
        commands[name] = " ".join(cmd).replace("/opt/nccl-tests/build/", "./build/")
    smi = _capture(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
            "--format=csv",
        ]
    )
    nccl_pkg = _capture(["dpkg-query", "-W", "libnccl2", "libnccl-dev"])
    files["run.json"] = json.dumps(
        {
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "modal_request": gpu_request,
            "gpu_count": count,
            "rank_layout": f"1 process, 1 thread, {count} GPUs (ranks 0..{count - 1} on devices 0..{count - 1})",
            "topology": topology,
            "topology_evidence": "nccl_topology.txt: NCCL graph type NVL, every channel via P2P",
            "dtype": DTYPE,
            "redop": REDOP,
            "nccl_tests": {
                "repo": NCCL_TESTS_REPO,
                "tag": NCCL_TESTS_TAG,
                "commit": NCCL_TESTS_COMMIT,
                "build": f'make MPI=0 NVCC_GENCODE="{GENCODE}"',
            },
            "image": CUDA_IMAGE,
            "nccl_packages": nccl_pkg.strip().splitlines(),
            "nvidia_smi": smi.strip().splitlines(),
            "env": sorted(f"{k}={v}" for k, v in env.items() if "VISIBLE" not in k),
            "commands": commands,
        },
        indent=2,
    )
    return files


FUNCS = {
    key: app.function(
        gpu=request[0],
        cpu=8.0,
        memory=16384,
        timeout=30 * 60,
        name=f"nccl_{key[0]}x{key[1]}",
    )(_run)
    for key, request in REQUESTS.items()
}


@app.local_entrypoint()
def main(gpu: str = "h100", count: int = 2) -> None:
    key = (gpu, count)
    if key not in FUNCS:
        raise ValueError(f"--gpu/--count must be one of: {sorted(FUNCS)}")
    files = FUNCS[key].remote(count, *REQUESTS[key])
    out = ROOT / "profiles" / "nccl-tests" / f"{gpu}x{count}"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text if text.endswith("\n") else text + "\n")
    print(f"Wrote {out}")
