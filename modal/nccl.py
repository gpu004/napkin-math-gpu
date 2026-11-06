"""Run pinned NVIDIA NCCL Tests on Modal and keep the raw evidence.

Usage:
    modal run modal/nccl.py --gpu h100 --count 2
    modal run modal/nccl.py --gpu h100 --count 16   # two 8-GPU machines over RDMA

Writes profiles/nccl-tests/<gpu>x<count>/: run.json (provenance), topo.txt
(`nvidia-smi topo -m`, `topo -p2p n`, `nvlink -s` per machine; Modal's sandbox
fails `topo -m`, so the other two and NCCL's own graph log stand in for it),
nccl_topology.txt (NCCL_DEBUG=INFO for one tiny AllReduce), and the stdout
plus `-J` JSON of each collective.

The topology label in REQUESTS is checked against NCCL's log (TOPOLOGIES):
"nvlink" means one node where NCCL built every ring over NVLink (graph type
NVL) and moved data GPU-to-GPU via P2P, never through host memory or the
network. "rdma" means a Modal multi-node cluster (modal.clustered, rdma=True)
where NCCL chose its verbs transport ("Using network IB", InfiniBand or RoCE)
and moved data between machines via NET/IB. NCCL silently falls back to TCP
sockets when it cannot open the IB devices, so the check refuses a run that
would have measured Ethernet.

One machine runs one process driving every GPU with `-g <count>`. A cluster
gang-schedules one container per machine, and NCCL Tests needs MPI to start
ranks in several processes, so the cluster image builds it with MPI=1 against
Ubuntu's OpenMPI. Rank 0 hands the other ranks an ssh key through a
modal.Dict, then starts one process per machine with mpirun, each driving its
own GPUs. OpenMPI carries only start-up and result gathering, over TCP on the
cluster's IPv6 interface, so NCCL alone touches the RDMA NICs. Modal gives
every container in a cluster the same NCCL_HOSTID, which would make NCCL see
GPU 0 on every machine as one device ("Duplicate GPU detected"), so each MPI
process sets its own. Rank 0 publishes "done" even when it fails, so the
other ranks stop waiting instead of holding their GPUs until the timeout.
"""

import json
import shlex
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import modal

NCCL_TESTS_REPO = "https://github.com/NVIDIA/nccl-tests"
NCCL_TESTS_TAG = "v2.21.1"
NCCL_TESTS_COMMIT = "afd59abab774a6a4b492eb344bd0c845a0b7bb4d"
CUDA_IMAGE = "nvidia/cuda:12.9.1-devel-ubuntu24.04"
GENCODE = " ".join(f"-gencode=arch=compute_{sm},code=sm_{sm}" for sm in ("80", "90", "100", "103"))
MPI_HOME = "/usr/lib/x86_64-linux-gnu/openmpi"
SINGLE_BUILD = f'make -j MPI=0 NVCC_GENCODE="{GENCODE}"'
MPI_BUILD = f'make -j MPI=1 MPI_HOME={MPI_HOME} NVCC_GENCODE="{GENCODE}"'
BUILD_DIR = "/opt/nccl-tests/build"
COLLECTIVES = ("all_reduce_perf", "all_gather_perf", "reduce_scatter_perf")
SWEEP = ["-b", "8", "-e", "4G", "-f", "2", "-w", "5", "-n", "20", "-c", "1"]
DTYPE = "bfloat16"
REDOP = "sum"
SMI_FIELDS = (
    "index",
    "name",
    "driver_version",
    "pci.bus_id",
    "pcie.link.gen.current",
    "pcie.link.width.current",
)
TOPO_PROBES = (
    ["nvidia-smi", "topo", "-m"],
    ["nvidia-smi", "topo", "-p2p", "n"],
    ["nvidia-smi", "nvlink", "-s"],
)
SSH_PORT = 2222
SSH_KEY = "/root/.ssh/napkin_ed25519"
HOSTID_WRAPPER = 'export NCCL_HOSTID=napkin-node-$OMPI_COMM_WORLD_RANK; exec "$@"'

ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Request:
    """One NCCL Tests run: the Modal GPU request per machine and how many machines."""

    modal_gpu: str
    nodes: int
    topology: str


REQUESTS = {
    ("h100", 2): Request("H100!:2", 1, "nvlink"),
    ("h100", 8): Request("H100!:8", 1, "nvlink"),
    ("b200", 8): Request("B200:8", 1, "nvlink"),
    ("h100", 16): Request("H100!:8", 2, "rdma"),
}


@dataclass(frozen=True)
class Topology:
    """What NCCL's log must and must not say before a run earns its label."""

    debug_subsys: str
    required: tuple[str, ...]
    forbidden: tuple[str, ...]
    evidence: str
    refusal: str

    def check(self, log: str) -> None:
        if any(line not in log for line in self.required) or any(
            line in log for line in self.forbidden
        ):
            raise RuntimeError(f"{self.refusal}; see nccl_topology.txt")


TOPOLOGIES = {
    "nvlink": Topology(
        debug_subsys="INIT,GRAPH",
        required=("type NVL", "via P2P"),
        forbidden=("via NET", "via SHM"),
        evidence="nccl_topology.txt: NCCL graph type NVL, every channel via P2P",
        refusal="NCCL did not build an all-NVLink P2P topology",
    ),
    "rdma": Topology(
        debug_subsys="INIT,GRAPH,NET",
        required=("Using network IB", "via NET/IB"),
        forbidden=("Using network Socket", "via NET/Socket"),
        evidence=" ".join(
            [
                "nccl_topology.txt: NCCL 'Using network IB',",
                "channels between machines via NET/IB, never NET/Socket",
            ]
        ),
        refusal=" ".join(
            [
                "NCCL did not move data between machines over NET/IB (InfiniBand or RoCE);",
                "an EFA cluster needs the aws-ofi-nccl plugin and its own topology label",
            ]
        ),
    ),
}


@dataclass(frozen=True)
class Launch:
    """How one run starts NCCL Tests and what it records about that.

    `command` turns a binary name, its arguments and the environment into the
    argv to execute: the binary itself on one machine, an mpirun line across
    several.
    """

    command: Callable[[str, list[str], dict[str, str]], list[str]]
    gpus_per_process: int
    build: str
    modal_request: str
    rank_layout: str
    record: dict[str, object] = field(default_factory=dict)
    env_notes: tuple[str, ...] = ()


def _nccl_tests_image(packages: tuple[str, ...], build: str) -> modal.Image:
    return (
        modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
        .apt_install("git", "build-essential", *packages)
        .run_commands(
            f"git clone {NCCL_TESTS_REPO} /opt/nccl-tests",
            f"cd /opt/nccl-tests && git checkout {NCCL_TESTS_COMMIT}",
            f"cd /opt/nccl-tests && {build}",
        )
    )


image = _nccl_tests_image((), SINGLE_BUILD)
cluster_image = _nccl_tests_image(
    (
        "iproute2",
        "openssh-server",
        "openssh-client",
        "openmpi-bin",
        "libopenmpi-dev",
        "libibverbs1",
        "ibverbs-providers",
        "libhwloc15",
        "libnl-route-3-200",
    ),
    MPI_BUILD,
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


def _base_env() -> dict[str, str]:
    import os

    env = {
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
        "NCCL_DEBUG": "VERSION",
    }
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]
    return env


def _node_facts() -> dict[str, str]:
    """What each machine reports about itself; rank 0 gathers every machine's."""
    return {
        "topo": "\n".join(_probe(cmd) for cmd in TOPO_PROBES),
        "smi": _capture(
            ["nvidia-smi", f"--query-gpu={','.join(SMI_FIELDS)}", "--format=csv,noheader"]
        ),
    }


def _record(
    gpu: str, count: int, launch: Launch, env: dict[str, str], nodes: list[dict[str, str]]
) -> dict[str, str]:
    """Runs the topology probe and the sweep, and returns every evidence file."""
    topology_label = REQUESTS[(gpu, count)].topology
    topology = TOPOLOGIES[topology_label]
    probe_env = {**env, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": topology.debug_subsys}
    probe_args = ["-b", "8", "-e", "8", "-g", str(launch.gpus_per_process)]
    log = _capture(launch.command("all_reduce_perf", probe_args, probe_env), probe_env)
    topology.check(log)
    files = {
        "topo.txt": nodes[0]["topo"]
        if len(nodes) == 1
        else "\n".join(f"## node {rank}\n{node['topo']}" for rank, node in enumerate(nodes)),
        "nccl_topology.txt": log,
    }
    commands = {}
    for name in COLLECTIVES:
        json_path = f"/tmp/{name}.json"
        args = [*SWEEP, "-g", str(launch.gpus_per_process), "-d", DTYPE, "-o", REDOP]
        cmd = launch.command(name, [*args, "-J", json_path], env)
        files[f"{name}.txt"] = _capture(cmd, env)
        files[f"{name}.json"] = Path(json_path).read_text()
        commands[name] = shlex.join(cmd).replace(f"{BUILD_DIR}/", "./build/")
    nccl_pkg = _capture(["dpkg-query", "-W", "libnccl2", "libnccl-dev"])
    files["run.json"] = json.dumps(
        {
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "modal_request": launch.modal_request,
            "gpu_count": count,
            "rank_layout": launch.rank_layout,
            **launch.record,
            "topology": topology_label,
            "topology_evidence": topology.evidence,
            "dtype": DTYPE,
            "redop": REDOP,
            "nccl_tests": {
                "repo": NCCL_TESTS_REPO,
                "tag": NCCL_TESTS_TAG,
                "commit": NCCL_TESTS_COMMIT,
                "build": launch.build,
            },
            "image": CUDA_IMAGE,
            "nccl_packages": nccl_pkg.strip().splitlines(),
            "nvidia_smi": [
                ", ".join(SMI_FIELDS),
                *(row for node in nodes for row in node["smi"].strip().splitlines()),
            ],
            "env": sorted(
                [f"{k}={v}" for k, v in env.items() if "VISIBLE" not in k] + [*launch.env_notes]
            ),
            "commands": commands,
        },
        indent=2,
    )
    return files


def _run(gpu: str, count: int) -> dict[str, str]:
    launch = Launch(
        command=lambda binary, args, _env: [f"{BUILD_DIR}/{binary}", *args],
        gpus_per_process=count,
        build=SINGLE_BUILD,
        modal_request=REQUESTS[(gpu, count)].modal_gpu,
        rank_layout=f"1 process, 1 thread, {count} GPUs (ranks 0..{count - 1} on devices 0..{count - 1})",
    )
    return _record(gpu, count, launch, _base_env(), [_node_facts()])


def _interface_for(address: str) -> str:
    """Names the interface holding the container address Modal handed out,
    e.g. the IPv6 i6pn address that carries the clusters' TCP control plane."""
    import ipaddress

    wanted = ipaddress.ip_address(address)
    for link in json.loads(_capture(["ip", "-j", "addr", "show"])):
        for info in link.get("addr_info", []):
            if ipaddress.ip_address(info["local"]) == wanted:
                return link["ifname"]
    raise RuntimeError(f"no interface holds the cluster address {address}")


def _start_sshd(public_key: str) -> None:
    import subprocess

    ssh_dir = Path("/root/.ssh")
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    authorized = ssh_dir / "authorized_keys"
    authorized.write_text(public_key.strip() + "\n")
    authorized.chmod(0o600)
    Path("/run/sshd").mkdir(exist_ok=True)
    config = Path("/etc/ssh/sshd_config_napkin")
    settings = [
        f"Port {SSH_PORT}",
        "PermitRootLogin prohibit-password",
        "PasswordAuthentication no",
        "KbdInteractiveAuthentication no",
        "UsePAM no",
    ]
    config.write_text("\n".join(settings) + "\n")
    _capture(["ssh-keygen", "-A"])
    with open("/tmp/sshd.log", "w") as log:
        subprocess.Popen(
            ["/usr/sbin/sshd", "-D", "-e", "-f", str(config)],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def _wait_for(state: modal.Dict, key: str, timeout_s: float = 600.0):
    import time

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if key in state:
            return state[key]
        if "done" in state:
            raise RuntimeError(f"rank 0 finished before publishing {key!r}")
        time.sleep(2)
    raise TimeoutError(f"cluster peer never published {key!r}")


def _share_ssh_key(state: modal.Dict) -> None:
    Path(SSH_KEY).parent.mkdir(mode=0o700, exist_ok=True)
    _capture(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", SSH_KEY])
    public_key = Path(f"{SSH_KEY}.pub").read_text()
    _start_sshd(public_key)
    state["public_key"] = public_key


def _write_ssh_config(hosts: list[str], ips: list[str]) -> None:
    entries = (
        [
            f"Host {host}",
            f"HostName {ip}",
            f"Port {SSH_PORT}",
            "User root",
            f"IdentityFile {SSH_KEY}",
            "StrictHostKeyChecking no",
            "UserKnownHostsFile /dev/null",
            "LogLevel ERROR",
        ]
        for host, ip in zip(hosts, ips, strict=True)
    )
    Path("/root/.ssh/config").write_text("".join("\n  ".join(entry) + "\n" for entry in entries))
    Path("/tmp/hostfile").write_text("".join(f"{host} slots=1\n" for host in hosts))


def _mpirun(
    nodes: int, interface: str, binary: str, args: list[str], env: dict[str, str]
) -> list[str]:
    forward = sorted(key for key in env if not key.startswith("OMPI_"))
    return [
        "mpirun",
        "--hostfile",
        "/tmp/hostfile",
        "-np",
        str(nodes),
        "--map-by",
        "ppr:1:node",
        "--bind-to",
        "none",
        "--mca",
        "pml",
        "ob1",
        "--mca",
        "btl",
        "self,tcp",
        "--mca",
        "btl_tcp_if_include",
        interface,
        "--mca",
        "oob_tcp_if_include",
        interface,
        *(arg for key in forward for arg in ("-x", key)),
        "bash",
        "-c",
        HOSTID_WRAPPER,
        "_",
        f"{BUILD_DIR}/{binary}",
        *args,
    ]


def _run_cluster(gpu: str, count: int, state: modal.Dict) -> dict[str, str]:
    cluster = modal.Cluster.from_context()
    ips = cluster.container_ips()
    nodes = REQUESTS[(gpu, count)].nodes
    if len(ips) != nodes:
        raise RuntimeError(f"expected {nodes} machines, got {ips}")
    rank = cluster.container_rank()
    interface = _interface_for(ips[rank])
    if rank != 0:
        _start_sshd(_wait_for(state, "public_key"))
        state[f"node{rank}"] = {**_node_facts(), "interface": interface}
        _wait_for(state, "done", timeout_s=3 * 60 * 60)
        return {}
    try:
        return _lead_cluster(gpu, count, state, interface, ips)
    finally:
        state["done"] = True


def _lead_cluster(
    gpu: str, count: int, state: modal.Dict, interface: str, ips: list[str]
) -> dict[str, str]:
    hosts = [f"node{rank}" for rank in range(len(ips))]
    _share_ssh_key(state)
    _write_ssh_config(hosts, ips)
    nodes = [
        {**_node_facts(), "interface": interface},
        *(_wait_for(state, host) for host in hosts[1:]),
    ]
    if any(node["interface"] != interface for node in nodes):
        raise RuntimeError(f"machines disagree on the cluster interface: {nodes}")

    per_node = count // len(hosts)
    machines = ", ".join(
        f"{rank * per_node}..{(rank + 1) * per_node - 1} on machine {rank}"
        for rank in range(len(hosts))
    )
    env = {
        **_base_env(),
        "NCCL_SOCKET_FAMILY": "AF_INET6",
        "NCCL_SOCKET_IFNAME": interface,
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    launch = Launch(
        command=partial(_mpirun, len(hosts), interface),
        gpus_per_process=per_node,
        build=MPI_BUILD,
        modal_request=" ".join(
            [REQUESTS[(gpu, count)].modal_gpu, f"× {len(hosts)} (modal.clustered, rdma=True)"]
        ),
        rank_layout=" ".join(
            [
                f"{len(hosts)} processes, one per machine, 1 thread, {per_node} GPUs each",
                f"(ranks {machines})",
            ]
        ),
        record={
            "nodes": len(hosts),
            "processes": len(hosts),
            "mpi_packages": _capture(["dpkg-query", "-W", "openmpi-bin"]).strip().splitlines(),
        },
        env_notes=("NCCL_HOSTID=napkin-node-<MPI rank>",),
    )
    return _record(gpu, count, launch, env, nodes)


def _function(key: tuple[str, int], request: Request) -> modal.Function:
    name = f"nccl_{key[0]}x{key[1]}"
    common = {"gpu": request.modal_gpu, "cpu": 8.0, "memory": 16384, "name": name}
    if request.nodes == 1:
        return app.function(timeout=30 * 60, **common)(_run)
    return app.function(image=cluster_image, timeout=60 * 60, **common)(
        modal.clustered(size=request.nodes, rdma=True)(_run_cluster)
    )


FUNCS = {key: _function(key, request) for key, request in REQUESTS.items()}


@app.local_entrypoint()
def main(gpu: str = "h100", count: int = 2) -> None:
    key = (gpu, count)
    if key not in REQUESTS:
        raise ValueError(f"--gpu/--count must be one of: {sorted(REQUESTS)}")
    if REQUESTS[key].nodes == 1:
        files = FUNCS[key].remote(gpu, count)
    else:
        with modal.Dict.ephemeral() as state:
            files = FUNCS[key].remote(gpu, count, state)
    out = ROOT / "profiles" / "nccl-tests" / f"{gpu}x{count}"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text if text.endswith("\n") else text + "\n")
    print(f"Wrote {out}")
