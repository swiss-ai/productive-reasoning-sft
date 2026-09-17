from __future__ import annotations

import json
import os
import socket
import subprocess
import time
from pathlib import Path
from types import TracebackType


class RayCluster:
    """Start Ray locally or form one cluster from one process per Slurm node."""

    def __init__(self, *, run_dir: Path, cpus_per_node: int, gpus_per_node: int) -> None:
        self.run_dir = run_dir
        self.cpus_per_node = cpus_per_node
        self.gpus_per_node = gpus_per_node
        self._manages_ray = False

    def __enter__(self) -> RayCluster:
        if os.environ.get("RAY_ADDRESS"):
            self._connect(os.environ["RAY_ADDRESS"])
            return self
        if not os.environ.get("SLURM_JOB_ID"):
            self._connect(None)
            return self
        self._start_slurm()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        import ray

        ray.shutdown()
        if self._manages_ray:
            subprocess.run(["ray", "stop", "--force"], check=False)

    @staticmethod
    def _connect(address: str | None) -> None:
        import ray

        ray.init(address=address, ignore_reinit_error=True)

    def _start_slurm(self) -> None:
        node_id = int(os.environ.get("SLURM_NODEID", "0"))
        nodes = _slurm_nodes()
        if not nodes:
            raise RuntimeError("SLURM_JOB_ID is set but SLURM_JOB_NODELIST is unavailable")
        broadcast_dir = Path(os.environ.get("RAY_PORT_BROADCAST_DIR", self.run_dir / "cluster"))
        broadcast_dir.mkdir(parents=True, exist_ok=True)
        port_file = broadcast_dir / f"ray_{os.environ['SLURM_JOB_ID']}.json"
        subprocess.run(["ray", "stop", "--force"], check=False, capture_output=True)

        if node_id == 0:
            head_ip = socket.gethostbyname(nodes[0])
            port = _free_port()
            command = [
                "ray",
                "start",
                "--head",
                f"--node-ip-address={head_ip}",
                f"--port={port}",
                f"--num-cpus={self.cpus_per_node}",
                f"--num-gpus={self.gpus_per_node}",
                "--include-dashboard=false",
                "--disable-usage-stats",
            ]
            subprocess.run(command, check=True)
            temporary = port_file.with_suffix(f".tmp.{os.getpid()}")
            temporary.write_text(json.dumps({"address": f"{head_ip}:{port}"}), encoding="utf-8")
            temporary.replace(port_file)
            self._manages_ray = True
            self._connect(f"{head_ip}:{port}")
            _wait_for_nodes(len(nodes))
            return

        deadline = time.monotonic() + 600
        while not port_file.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for Ray head metadata: {port_file}")
            time.sleep(0.5)
        address = json.loads(port_file.read_text(encoding="utf-8"))["address"]
        local_ip = socket.gethostbyname(socket.gethostname())
        code = subprocess.call(
            [
                "ray",
                "start",
                f"--address={address}",
                f"--node-ip-address={local_ip}",
                f"--num-cpus={self.cpus_per_node}",
                f"--num-gpus={self.gpus_per_node}",
                "--disable-usage-stats",
                "--block",
            ]
        )
        raise SystemExit(code)


def _slurm_nodes() -> list[str]:
    expression = os.environ.get("SLURM_JOB_NODELIST") or os.environ.get("SLURM_NODELIST")
    if not expression:
        return []
    result = subprocess.run(
        ["scontrol", "show", "hostnames", expression],
        check=True,
        capture_output=True,
        text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return int(sock.getsockname()[1])


def _wait_for_nodes(expected: int, timeout: float = 600.0) -> None:
    import ray

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        alive = sum(1 for node in ray.nodes() if node.get("Alive"))
        if alive >= expected:
            return
        time.sleep(1.0)
    raise TimeoutError(f"only {alive}/{expected} Ray nodes joined within {timeout:.0f}s")


def available_gpu_replicas(tensor_parallel_size: int) -> int:
    import ray

    gpus = int(ray.cluster_resources().get("GPU", 0))
    replicas = gpus // tensor_parallel_size
    if replicas < 1:
        raise RuntimeError(
            f"Ray reports {gpus} GPUs, insufficient for tensor_parallel_size={tensor_parallel_size}"
        )
    return replicas
