"""Launch napkin-bench on Modal GPUs.

Usage:
    modal run modal/bench.py --gpu h100
    modal run modal/bench.py --gpu a100 --quick

Full runs write profiles/<gpu>.measured.json. Quick runs print to stdout only.
"""

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


def _run(quick: bool) -> str:
    import subprocess

    cmd = ["/app/target/release/napkin-bench"]
    if quick:
        cmd.append("--quick")
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out.stdout


FUNCS = {
    name: app.function(
        gpu=request,
        cpu=8.0,
        memory=8192,
        timeout=20 * 60,
        name=f"bench_{name}",
        serialized=True,
    )(_run)
    for name, request in GPUS.items()
}


@app.local_entrypoint()
def main(gpu: str = "h100", quick: bool = False) -> None:
    if gpu not in FUNCS:
        raise ValueError(f"--gpu must be one of: {sorted(FUNCS)}")
    result = FUNCS[gpu].remote(quick)
    if quick:
        print(result)
        return
    out = ROOT / "profiles" / f"{gpu}.measured.json"
    out.write_text(result)
    print(f"Wrote {out}")
