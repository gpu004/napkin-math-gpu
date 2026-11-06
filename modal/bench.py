"""Launch napkin-bench on Modal GPUs.

Usage:
    modal run modal/bench.py --gpu h100
    modal run modal/bench.py --gpu a100 --quick

    modal run modal/bench.py --gpu h100 --repeat 3

Full runs write profiles/<gpu>.measured.json. Quick runs print to stdout only.
Full --merge-shapes runs preserve scalar profile data and write target/modal/<gpu>.run.json.
--repeat N starts N full runs at once, one per container (max_inputs=1, or Modal
may feed every run to one container and so one machine), and writes each to
profiles/repeats/<gpu>/ with the host it ran on; the canonical profile is left
alone.
"""

import json
import subprocess
from pathlib import Path

import modal

GPUS = {"a100": "A100-80GB:2", "h100": "H100!:2", "b200": "B200:2", "b300": "B300:2"}

ROOT = Path(__file__).resolve().parent.parent

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.1-devel-ubuntu24.04", add_python="3.12")
    .apt_install("curl", "build-essential")
    .run_commands(
        "curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain stable --profile minimal"
    )
    .add_local_file(ROOT / "Cargo.toml", "/app/Cargo.toml", copy=True)
    .add_local_file(ROOT / "Cargo.lock", "/app/Cargo.lock", copy=True)
    .add_local_dir(ROOT / "crates", "/app/crates", copy=True)
    .add_local_dir(ROOT / "profiles", "/app/profiles", copy=True)
    .run_commands("cd /app && /root/.cargo/bin/cargo build --release -p napkin-bench")
)

app = modal.App("gpu-napkin-math", image=image)


def _run(quick: bool, gpu_request: str) -> str:
    import subprocess
    import sys

    cmd = ["/app/target/release/napkin-bench", "--gpu-request", gpu_request]
    if quick:
        cmd.append("--quick")
    out = subprocess.run(cmd, capture_output=True, text=True)
    sys.stderr.write(out.stderr)
    if out.returncode != 0:
        raise RuntimeError(f"napkin-bench exited {out.returncode}")
    return out.stdout


def _run_on_host(gpu_request: str) -> dict:
    """A full run plus what identifies the machine it ran on.

    Modal exposes no machine ID inside a container. GPU UUIDs are unique per
    physical GPU, so two runs with different UUIDs used different GPUs; two
    2-GPU containers can still share one 8-GPU node unless the provider or
    region differs.
    """
    import os

    measured = json.loads(_run(False, gpu_request))
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=True,
    )
    host = {
        "cloud_provider": os.environ["MODAL_CLOUD_PROVIDER"],
        "region": os.environ["MODAL_REGION"],
        "task_id": os.environ["MODAL_TASK_ID"],
        "gpu_uuids": sorted(line.strip() for line in smi.stdout.splitlines() if line.strip()),
    }
    return {"host": host, "measured": measured}


def _function(name: str, request: str, body, **options):
    return app.function(
        gpu=request,
        cpu=8.0,
        memory=8192,
        timeout=20 * 60,
        name=name,
        **options,
    )(body)


FUNCS = {name: _function(f"bench_{name}", request, _run) for name, request in GPUS.items()}
REPEAT_FUNCS = {
    name: _function(f"repeat_{name}", request, _run_on_host, max_inputs=1)
    for name, request in GPUS.items()
}


def _write_repeats(gpu: str, repeats: int) -> None:
    out_dir = ROOT / "profiles" / "repeats" / gpu
    out_dir.mkdir(parents=True, exist_ok=True)
    for run in REPEAT_FUNCS[gpu].map([GPUS[gpu]] * repeats):
        stamp = run["measured"]["captured_at"].replace(":", "")
        out = out_dir / f"{stamp}-{run['host']['task_id']}.json"
        out.write_text(json.dumps(run, indent=2) + "\n")
        print(f"Wrote {out}")


@app.local_entrypoint()
def main(
    gpu: str = "h100", quick: bool = False, merge_shapes: bool = False, repeat: int = 0
) -> None:
    if gpu not in FUNCS:
        raise ValueError(f"--gpu must be one of: {sorted(FUNCS)}")
    if quick and merge_shapes:
        raise ValueError("--merge-shapes cannot be used with --quick")
    if repeat:
        if quick or merge_shapes:
            raise ValueError("--repeat records full runs; drop --quick and --merge-shapes")
        _write_repeats(gpu, repeat)
        return
    result = FUNCS[gpu].remote(quick, GPUS[gpu])
    if quick:
        print(result)
        return
    if merge_shapes:
        run_file = ROOT / "target" / "modal" / f"{gpu}.run.json"
        run_file.parent.mkdir(parents=True, exist_ok=True)
        run_file.write_text(result)
        subprocess.run(
            [
                "cargo",
                "run",
                "-q",
                "-p",
                "napkin",
                "--",
                "merge-shapes",
                "--from",
                str(run_file.relative_to(ROOT)),
                str((ROOT / "profiles" / f"{gpu}.measured.json").relative_to(ROOT)),
            ],
            cwd=ROOT,
            check=True,
        )
        return
    out = ROOT / "profiles" / f"{gpu}.measured.json"
    out.write_text(result)
    print(f"Wrote {out}")
