"""RunPod lifecycle for remote runs: launch, find, pull, terminate, sweep.

The one property that matters: **nothing this module starts can be left running.**
The session that launches a pod can die (it did twice in one week), so no single
process is trusted to stop one. Four guards, each independent of the others:

| guard | where | stops a pod when |
|---|---|---|
| job      | on the pod (``cloud/pod_job.sh``)     | the job ends and its results have been pulled, or a bounded wait for the pull runs out |
| deadline | on the pod (its start command)        | its absolute deadline passes, whatever the job is doing |
| sweeper  | this box (systemd timer, ``sweep``)   | it is past its deadline plus a grace period, or carries no deadline |
| budget   | this box, before launch (``plan``)    | it would never start: rate × hours above the session ceiling or the balance |

**Ownership is by name.** Every pod created here is named ``taz-<tag>-dl<unix
deadline>``. The sweeper acts on that prefix only, so the account's other pods are
never touched, and the deadline travels *with the pod*: a lost ledger, a wiped
scratchpad or a different machine running the sweep cannot strand one.

**Terminate, never stop.** A stopped pod still bills for its disk. Results come back
to this box (rsync over SSH, pulled from here: the pod cannot reach a box behind
NAT), and then the pod is deleted.

Standard library only: this runs from a systemd timer on a box where the project's
venv may be mid-rebuild, and it has to keep working then.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REST = "https://rest.runpod.io/v1"
GRAPHQL = "https://api.runpod.io/graphql"
KEY_FILE = Path("~/.config/runpod/api_key").expanduser()
STATE_DIR = Path("~/.local/state/taz-cloud").expanduser()
LEDGER = STATE_DIR / "ledger.jsonl"
SSH_KEY = Path("~/.ssh/runpod_ed25519").expanduser()
LIMITS_FILE = Path("~/.config/taz-cloud/limits.json").expanduser()

PREFIX = "taz-"
_NAME = re.compile(r"^taz-(?P<tag>[a-z0-9][a-z0-9_-]{0,40})-dl(?P<deadline>\d{10})$")

#: The image's own start: its entrypoint is nvidia_entrypoint.sh, its CMD /start.sh
#: (read from the registry config). The deadline watchdog is prepended to the CMD,
#: so it exists from the container's first second and SSH comes up exactly as the
#: image intends.
DEFAULT_IMAGE = "runpod/pytorch:1.4.1-cu1281-torch2121-ubuntu2404"
IMAGE_CMD = "/start.sh"


class CloudError(RuntimeError):
    """Anything that should stop a launch or be reported by a sweep."""


class BudgetError(CloudError):
    """A launch the limits refuse."""


# ---------------------------------------------------------------------------
# Names, deadlines, limits: pure, and what the unit tests pin
# ---------------------------------------------------------------------------


def pod_name(tag: str, deadline: int) -> str:
    name = f"{PREFIX}{tag}-dl{int(deadline)}"
    if not _NAME.match(name):
        raise ValueError(f"tag {tag!r} gives an unparseable pod name {name!r}")
    return name


def parse_deadline(name: str) -> int | None:
    """The deadline a pod's name carries, or None if the name is not ours-and-valid."""
    m = _NAME.match(name or "")
    return int(m.group("deadline")) if m else None


def is_ours(name: str) -> bool:
    return (name or "").startswith(PREFIX)


@dataclass(frozen=True)
class Limits:
    """Ceilings a launch must fit under. Read from ``~/.config/taz-cloud/limits.json``."""

    max_session_cost: float = 10.0  # USD: rate x hours for one pod
    max_hours: float = 30.0
    balance_margin: float = 5.0  # USD left untouched on the account

    @classmethod
    def load(cls, path: Path | None = None) -> Limits:
        path = path or LIMITS_FILE
        if not path.exists():
            return cls()
        return cls(**json.loads(path.read_text()))


def plan(rate_per_hr: float, hours: float, balance: float, limits: Limits) -> float:
    """Worst-case cost of a launch, or BudgetError. The guard that runs before anything bills."""
    if hours <= 0:
        raise BudgetError("hours must be positive")
    if hours > limits.max_hours:
        raise BudgetError(f"{hours} h exceeds the {limits.max_hours} h ceiling")
    worst = rate_per_hr * hours
    if worst > limits.max_session_cost:
        raise BudgetError(
            f"${rate_per_hr:.3f}/h x {hours} h = ${worst:.2f} exceeds the "
            f"${limits.max_session_cost:.2f} session ceiling"
        )
    if worst > balance - limits.balance_margin:
        raise BudgetError(
            f"${worst:.2f} worst case leaves less than the ${limits.balance_margin:.2f} "
            f"margin on a ${balance:.2f} balance"
        )
    return worst


def sweep_decisions(
    pods: list[dict[str, Any]], now: float, grace_s: int = 300
) -> list[tuple[str, str, str]]:
    """``(pod_id, name, reason)`` for every pod of ours the sweeper must terminate.

    Ours means the ``taz-`` prefix and nothing else: anything else on the account is
    the account holder's and is never touched. Of ours, terminate a pod past its
    deadline plus ``grace_s`` (the pod's own watchdog fires at the deadline; this is
    its backstop) and any pod whose name carries no parseable deadline, because
    nothing would ever stop it.
    """
    out = []
    for pod in pods:
        name = pod.get("name") or ""
        if not is_ours(name):
            continue
        deadline = parse_deadline(name)
        if deadline is None:
            out.append((pod["id"], name, "ours by prefix but no parseable deadline"))
        elif now > deadline + grace_s:
            out.append((pod["id"], name, f"{int(now - deadline)} s past its deadline"))
    return out


def watchdog_start_cmd(enabled: bool = True) -> list[str]:
    """The pod's start command: the deadline watchdog in the background, then the image's CMD.

    At ``TAZ_DEADLINE`` the watchdog deletes its own pod, retrying, by the pod-scoped
    ``runpodctl`` first and the REST API second. ``enabled=False`` exists for one
    purpose: proving the sweeper catches a pod whose own watchdog never fires.
    """
    if not enabled:
        return ["bash", "-c", f"exec {IMAGE_CMD}"]
    watchdog = (
        'deadline="${TAZ_DEADLINE:?}"; '
        'while [ "$(date +%s)" -lt "$deadline" ]; do sleep 20; done; '
        'echo "$(date -Is) deadline reached: deleting pod $RUNPOD_POD_ID" >> /taz-watchdog.log; '
        "for i in 1 2 3 4 5 6; do "
        'runpodctl remove pod "$RUNPOD_POD_ID" >> /taz-watchdog.log 2>&1 && break; '
        'curl -fsS -X DELETE "https://rest.runpod.io/v1/pods/$RUNPOD_POD_ID" '
        '-H "Authorization: Bearer $RUNPOD_API_KEY" -A "taz-cloud/1.0 (+tinyalphazero)" '
        ">> /taz-watchdog.log 2>&1 && break; "
        "sleep 30; done"
    )
    return ["bash", "-c", f"( {watchdog} ) & exec {IMAGE_CMD}"]


# ---------------------------------------------------------------------------
# The API
# ---------------------------------------------------------------------------


class RunPod:
    """Thin REST/GraphQL client. The key is read from a file, never from argv or env."""

    def __init__(self, key_file: Path = KEY_FILE, timeout: float = 30.0) -> None:
        self._key = key_file.read_text().strip()
        self.timeout = timeout

    def _request(self, method: str, url: str, body: Any = None) -> tuple[int, Any]:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._key}",
                # rest.runpod.io sits behind Cloudflare, which refuses urllib's default
                # User-Agent outright (403, error code 1010).
                "User-Agent": "taz-cloud/1.0 (+tinyalphazero)",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                raw = r.read()
                return r.status, (json.loads(raw) if raw else None)
        except urllib.error.HTTPError as e:
            raw = e.read()
            try:
                detail: Any = json.loads(raw)
            except ValueError:
                detail = raw[:300].decode(errors="replace")
            return e.code, detail

    def _gql(self, query: str) -> dict[str, Any]:
        status, d = self._request("POST", GRAPHQL, {"query": query})
        if status != 200 or not isinstance(d, dict) or d.get("errors"):
            raise CloudError(f"GraphQL {status}: {d}")
        return d["data"]

    # -- account -----------------------------------------------------------
    def balance(self) -> dict[str, float]:
        m = self._gql("query { myself { clientBalance currentSpendPerHr } }")["myself"]
        return {"balance": float(m["clientBalance"]), "spend_per_hr": float(m["currentSpendPerHr"])}

    def gpu_prices(self) -> list[dict[str, Any]]:
        q = (
            "query { gpuTypes { id displayName memoryInGb securePrice communityPrice "
            "lowestPrice(input: {gpuCount: 1}) { stockStatus } } }"
        )
        return self._gql(q)["gpuTypes"]

    # -- pods --------------------------------------------------------------
    def list_pods(self) -> list[dict[str, Any]]:
        status, d = self._request("GET", f"{REST}/pods")
        if status != 200 or not isinstance(d, list):
            raise CloudError(f"GET /pods {status}: {d}")
        return d

    def get_pod(self, pod_id: str) -> dict[str, Any] | None:
        status, d = self._request("GET", f"{REST}/pods/{pod_id}")
        if status == 404:
            return None
        if status != 200:
            raise CloudError(f"GET /pods/{pod_id} {status}: {d}")
        return d

    def create_pod(self, body: dict[str, Any]) -> dict[str, Any]:
        status, d = self._request("POST", f"{REST}/pods", body)
        if status not in (200, 201) or not isinstance(d, dict) or "id" not in d:
            raise CloudError(f"POST /pods {status}: {d}")
        return d

    def terminate_pod(self, pod_id: str) -> bool:
        """Delete a pod. True if it was deleted or was already gone."""
        status, d = self._request("DELETE", f"{REST}/pods/{pod_id}")
        if status in (200, 204, 404):
            return True
        raise CloudError(f"DELETE /pods/{pod_id} {status}: {d}")


# ---------------------------------------------------------------------------
# Ledger, SSH
# ---------------------------------------------------------------------------


def ledger(event: str, **fields: Any) -> None:
    """Append-only record of every launch, termination and sweep, on this box."""
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "event": event, **fields}
    with LEDGER.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def ssh_target(pod: dict[str, Any]) -> tuple[str, int] | None:
    """(public ip, port) for full SSH, or None while the pod is still coming up."""
    ip = pod.get("publicIp") or ""
    port = (pod.get("portMappings") or {}).get("22")
    return (ip, int(port)) if ip and port else None


def ssh_base(ip: str, port: int) -> list[str]:
    return [
        "ssh",
        "-i",
        str(SSH_KEY),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        f"UserKnownHostsFile={STATE_DIR / 'known_hosts'}",
        "-o",
        "ConnectTimeout=15",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "BatchMode=yes",
        "-o",
        "ForwardAgent=no",
        f"root@{ip}",
    ]


@dataclass
class LaunchSpec:
    """What a launch asks RunPod for. Defaults: terminate-only, no pod volume, full SSH."""

    tag: str
    gpu_type: str
    hours: float
    cloud: str = "COMMUNITY"
    image: str = DEFAULT_IMAGE
    vcpu_per_gpu: int = 16
    ram_per_gpu: int = 32
    container_disk_gb: int = 60
    watchdog: bool = True
    # Host CUDA versions to accept, or None for no filter. An image's own
    # NVIDIA_REQUIRE_CUDA is what actually binds; the API's filter matched no host
    # at all on 2026-10-24 for ["12.8", "12.9", "13.0"], so it is opt-in.
    allowed_cuda: tuple[str, ...] | None = None

    def body(self, deadline: int) -> dict[str, Any]:
        return {
            "name": pod_name(self.tag, deadline),
            "imageName": self.image,
            "gpuTypeIds": [self.gpu_type],
            "gpuCount": 1,
            "cloudType": self.cloud,
            "computeType": "GPU",
            "containerDiskInGb": self.container_disk_gb,
            "volumeInGb": 0,  # no pod volume: it would bill after a stop, and we never stop
            "minVCPUPerGPU": self.vcpu_per_gpu,
            "minRAMPerGPU": self.ram_per_gpu,
            "ports": ["22/tcp"],
            "supportPublicIp": True,  # full SSH (rsync) needs a public IP, not the proxy
            "dockerStartCmd": watchdog_start_cmd(self.watchdog),
            "env": {"TAZ_DEADLINE": str(deadline)},
            **({"allowedCudaVersions": list(self.allowed_cuda)} if self.allowed_cuda else {}),
        }


def as_dict(spec: LaunchSpec) -> dict[str, Any]:
    return asdict(spec)


def environment_has_key() -> bool:
    return KEY_FILE.exists() and os.access(KEY_FILE, os.R_OK)
