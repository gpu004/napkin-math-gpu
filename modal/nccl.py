"""Run pinned NVIDIA NCCL Tests on Modal and keep the raw evidence.

Usage:
    modal run modal/nccl.py --gpu h100 --count 2
    modal run modal/nccl.py --gpu h100 --count 16   # two 8-GPU machines over RDMA

Writes profiles/nccl-tests/<gpu>x<count>/: run.json (provenance), topo.txt
(`nvidia-smi topo -m`, `topo -p2p n`, `nvlink -s`; Modal's sandbox fails
`topo -m`, so the other two and NCCL's own graph log stand in for it),
nccl_topology.txt (NCCL_DEBUG=INFO INIT,GRAPH for one tiny AllReduce), and
the stdout plus `-J` JSON of each collective.

The topology label in REQUESTS is checked against NCCL's graph log: "nvlink"
means one node where NCCL built every ring over NVLink (graph type NVL) and
moved data GPU-to-GPU via P2P, never through host memory or the network.
"rdma" means a Modal multi-node cluster (modal.clustered, rdma=True) where
NCCL chose its verbs transport ("Using network IB", InfiniBand or RoCE) and
moved data between machines via NET/IB, never falling back to TCP sockets.

A multi-node run gang-schedules one container per machine. NCCL Tests needs
MPI to start ranks in several processes, so the clustered image builds it with
MPI=1 against Ubuntu's OpenMPI. Rank 1 accepts an ssh key from rank 0 through a
modal.Dict, and rank 0 starts one process per machine with mpirun, each
driving its 8 GPUs with `-g 8`. OpenMPI carries only start-up and result
gathering, over TCP on the cluster's IPv6 interface, so NCCL alone touches the
RDMA NICs. NCCL silently falls back to TCP sockets when it cannot open the IB
devices, so the "rdma" check above refuses a run that would measure Ethernet.
Modal gives every container in a cluster the same NCCL_HOSTID, which would make
NCCL see GPU 0 on both machines as one device ("Duplicate GPU detected"), so
each MPI process sets its own. Rank 0 publishes "done" even when it fails, so
rank 1 stops waiting instead of holding its 8 GPUs until the timeout.
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
CLUSTER_REQUESTS = {
    ("h100", 16): ("H100!:8", 2, "rdma"),
}
GPUS_PER_NODE = 8
MPI_HOME = "/usr/lib/x86_64-linux-gnu/openmpi"
SSH_PORT = 2222
SSH_KEY = "/root/.ssh/napkin_ed25519"
HOSTID_WRAPPER = 'export NCCL_HOSTID=napkin-node-$OMPI_COMM_WORLD_RANK; exec "$@"'

ROOT = Path(__file__).resolve().parent.parent

NCCL_TESTS_CLONE = (
    f"git clone {NCCL_TESTS_REPO} /opt/nccl-tests",
    f"cd /opt/nccl-tests && git checkout {NCCL_TESTS_COMMIT}",
)
image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install("git", "build-essential")
    .run_commands(
        *NCCL_TESTS_CLONE,
        f'cd /opt/nccl-tests && make -j MPI=0 NVCC_GENCODE="{GENCODE}"',
    )
)
MPI_BUILD = f'make MPI=1 MPI_HOME={MPI_HOME} NVCC_GENCODE="{GENCODE}"'
cluster_image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .apt_install(
        "git",
        "build-essential",
        "iproute2",
        "openssh-server",
        "openssh-client",
        "openmpi-bin",
        "libopenmpi-dev",
        "libibverbs1",
        "ibverbs-providers",
        "libhwloc15",
        "libnl-route-3-200",
    )
    .run_commands(*NCCL_TESTS_CLONE, f"cd /opt/nccl-tests && {MPI_BUILD}")
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


def _node_facts() -> dict[str, str]:
    """What each machine reports about itself; rank 0 gathers both."""
    return {
        "topo": "\n".join(
            _probe(cmd)
            for cmd in (
                ["nvidia-smi", "topo", "-m"],
                ["nvidia-smi", "topo", "-p2p", "n"],
                ["nvidia-smi", "nvlink", "-s"],
                ["ip", "-brief", "addr", "show"],
            )
        ),
        "smi": _capture(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,pci.bus_id,pcie.link.gen.current,pcie.link.width.current",
                "--format=csv,noheader",
            ]
        ),
    }


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


def _check_rdma(graph: str) -> None:
    if (
        "Using network IB" not in graph
        or "via NET/IB" not in graph
        or "Using network Socket" in graph
        or "via NET/Socket" in graph
    ):
        raise RuntimeError(
            "NCCL did not move data between machines over NET/IB (InfiniBand or RoCE); an EFA"
            + " cluster needs the aws-ofi-nccl plugin and its own topology label"
        )


def _run_cluster(
    count: int, gpu_request: str, nodes: int, topology: str, state: modal.Dict
) -> dict[str, str]:
    cluster = modal.Cluster.from_context()
    ips = cluster.container_ips()
    if len(ips) != nodes or nodes != 2:
        raise RuntimeError(f"expected a 2-machine cluster, got {ips}")
    rank = cluster.container_rank()
    interface = _interface_for(ips[rank])
    if rank != 0:
        _start_sshd(_wait_for(state, "public_key"))
        state["node1"] = {**_node_facts(), "interface": interface}
        _wait_for(state, "done", timeout_s=3 * 60 * 60)
        return {}

    try:
        return _lead_cluster(count, gpu_request, nodes, topology, state, interface, ips)
    finally:
        state["done"] = True


def _lead_cluster(
    count: int,
    gpu_request: str,
    nodes: int,
    topology: str,
    state: modal.Dict,
    interface: str,
    ips: list[str],
) -> dict[str, str]:
    import os

    Path(SSH_KEY).parent.mkdir(mode=0o700, exist_ok=True)
    _capture(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", SSH_KEY])
    public_key = Path(f"{SSH_KEY}.pub").read_text()
    _start_sshd(public_key)
    state["public_key"] = public_key
    node0 = {**_node_facts(), "interface": interface}
    node1 = _wait_for(state, "node1")
    if node1["interface"] != interface:
        raise RuntimeError(f"machines disagree on the cluster interface: {interface} vs {node1}")
    hosts = [f"node{rank}" for rank in range(nodes)]
    Path("/root/.ssh/config").write_text(
        "".join(
            "\n  ".join(
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
            )
            + "\n"
            for host, ip in zip(hosts, ips, strict=True)
        )
    )
    Path("/tmp/hostfile").write_text("".join(f"{host} slots=1\n" for host in hosts))

    env = {
        "PATH": "/usr/local/cuda/bin:/usr/bin:/bin",
        "LD_LIBRARY_PATH": "/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu",
        "NCCL_DEBUG": "VERSION",
        "NCCL_SOCKET_FAMILY": "AF_INET6",
        "NCCL_SOCKET_IFNAME": interface,
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    for key in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        if key in os.environ:
            env[key] = os.environ[key]

    def mpirun(binary: str, args: list[str], forward: list[str]) -> list[str]:
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
            f"/opt/nccl-tests/build/{binary}",
            *args,
        ]

    forward = sorted(key for key in env if not key.startswith("OMPI_"))
    probe_env = {**env, "NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,GRAPH,NET"}
    files = {
        "topo.txt": f"## node 0\n{node0['topo']}\n## node 1\n{node1['topo']}",
        "nccl_topology.txt": _capture(
            mpirun(
                "all_reduce_perf",
                ["-b", "8", "-e", "8", "-g", str(GPUS_PER_NODE)],
                [*forward, "NCCL_DEBUG_SUBSYS"],
            ),
            probe_env,
        ),
    }
    if topology == "rdma":
        _check_rdma(files["nccl_topology.txt"])
    commands = {}
    for name in COLLECTIVES:
        json_path = f"/tmp/{name}.json"
        args = [*SWEEP, "-g", str(GPUS_PER_NODE), "-d", DTYPE, "-o", REDOP, "-J", json_path]
        cmd = mpirun(name, args, forward)
        files[f"{name}.txt"] = _capture(cmd, env)
        files[f"{name}.json"] = Path(json_path).read_text()
        commands[name] = " ".join(f"'{arg}'" if " " in arg else arg for arg in cmd).replace(
            "/opt/nccl-tests/build/", "./build/"
        )
    smi_header = (
        "index, name, driver_version, pci.bus_id, pcie.link.gen.current, pcie.link.width.current"
    )
    nccl_pkg = _capture(["dpkg-query", "-W", "libnccl2", "libnccl-dev"])
    mpi_pkg = _capture(["dpkg-query", "-W", "openmpi-bin"])
    files["run.json"] = json.dumps(
        {
            "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "modal_request": f"{gpu_request} × {nodes} (modal.clustered, rdma=True)",
            "gpu_count": count,
            "nodes": nodes,
            "processes": nodes,
            "rank_layout": " ".join(
                [
                    f"{nodes} processes, one per machine, 1 thread, {GPUS_PER_NODE} GPUs each",
                    f"(ranks 0..{GPUS_PER_NODE - 1} on machine 0,",
                    f"{GPUS_PER_NODE}..{count - 1} on machine 1)",
                ]
            ),
            "topology": topology,
            "topology_evidence": " ".join(
                [
                    "nccl_topology.txt: NCCL 'Using network IB',",
                    "channels between machines via NET/IB, never NET/Socket",
                ]
            ),
            "dtype": DTYPE,
            "redop": REDOP,
            "nccl_tests": {
                "repo": NCCL_TESTS_REPO,
                "tag": NCCL_TESTS_TAG,
                "commit": NCCL_TESTS_COMMIT,
                "build": MPI_BUILD,
            },
            "image": CUDA_IMAGE,
            "nccl_packages": nccl_pkg.strip().splitlines(),
            "mpi_packages": mpi_pkg.strip().splitlines(),
            "nvidia_smi": [
                smi_header,
                *node0["smi"].strip().splitlines(),
                *node1["smi"].strip().splitlines(),
            ],
            "env": sorted(
                [f"{k}={v}" for k, v in env.items() if "VISIBLE" not in k]
                + ["NCCL_HOSTID=napkin-node-<MPI rank>"]
            ),
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
CLUSTER_FUNCS = {
    key: app.function(
        image=cluster_image,
        gpu=request[0],
        cpu=8.0,
        memory=16384,
        timeout=60 * 60,
        name=f"nccl_{key[0]}x{key[1]}",
    )(modal.clustered(size=request[1], rdma=True)(_run_cluster))
    for key, request in CLUSTER_REQUESTS.items()
}


@app.local_entrypoint()
def main(gpu: str = "h100", count: int = 2) -> None:
    key = (gpu, count)
    if key in FUNCS:
        files = FUNCS[key].remote(count, *REQUESTS[key])
    elif key in CLUSTER_FUNCS:
        with modal.Dict.ephemeral() as state:
            files = CLUSTER_FUNCS[key].remote(count, *CLUSTER_REQUESTS[key], state)
    else:
        raise ValueError(f"--gpu/--count must be one of: {sorted([*FUNCS, *CLUSTER_FUNCS])}")
    out = ROOT / "profiles" / "nccl-tests" / f"{gpu}x{count}"
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text if text.endswith("\n") else text + "\n")
    print(f"Wrote {out}")
