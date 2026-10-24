#!/usr/bin/env python3
"""RunPod pods for remote runs, with nothing left running (``tinyalphazero.cloud``).

    python scripts/runpod_pod.py prices --max-price 0.5
    python scripts/runpod_pod.py launch --gpu "RTX 4090" --hours 2 --tag lifecycle   # dry run
    python scripts/runpod_pod.py launch --gpu "RTX 4090" --hours 2 --tag lifecycle --yes
    python scripts/runpod_pod.py wait-ssh POD_ID
    python scripts/runpod_pod.py ssh POD_ID -- nvidia-smi
    python scripts/runpod_pod.py push POD_ID local/path /root/path
    python scripts/runpod_pod.py pull POD_ID /root/path local/path
    python scripts/runpod_pod.py status
    python scripts/runpod_pod.py terminate POD_ID          # or --all-ours
    python scripts/runpod_pod.py sweep                     # what the systemd timer runs

``launch`` without ``--yes`` prints the plan and the worst-case cost and creates
nothing. Only pods named ``taz-…`` are ever listed, terminated or swept.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time

from tinyalphazero import cloud


def _gpu_lookup(rp: cloud.RunPod, wanted: str, cloud_type: str) -> tuple[str, float]:
    """(gpu type id, listed $/h) for ``wanted``, matched on id or display name."""
    for g in rp.gpu_prices():
        if wanted in (g["id"], g["displayName"]):
            price = g["securePrice"] if cloud_type == "SECURE" else g["communityPrice"]
            if not price:
                raise cloud.CloudError(f"{g['displayName']} has no {cloud_type} price listed")
            return g["id"], float(price)
    raise cloud.CloudError(f"no GPU type matches {wanted!r}")


def cmd_prices(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    rows = []
    for g in rp.gpu_prices():
        for kind, price in (("COMMUNITY", g["communityPrice"]), ("SECURE", g["securePrice"])):
            if price and float(price) <= args.max_price:
                stock = (g.get("lowestPrice") or {}).get("stockStatus")
                rows.append((float(price), kind, g["displayName"], g["memoryInGb"], stock))
    for price, kind, name, vram, stock in sorted(rows):
        print(f"${price:<6.2f} {kind:9s} {name:28s} {vram:>4} GB  stock={stock}")
    return 0


def cmd_balance(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    print(json.dumps(rp.balance()))
    return 0


def cmd_launch(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    limits = cloud.Limits.load()
    gpu_id, listed = _gpu_lookup(rp, args.gpu, args.cloud)
    account = rp.balance()
    worst = cloud.plan(listed, args.hours, account["balance"], limits)
    deadline = int(time.time() + args.hours * 3600)
    spec = cloud.LaunchSpec(
        tag=args.tag,
        gpu_type=gpu_id,
        hours=args.hours,
        cloud=args.cloud,
        vcpu_per_gpu=args.vcpu,
        ram_per_gpu=args.ram,
        container_disk_gb=args.disk,
        watchdog=not args.no_watchdog,
        image=args.image,
        allowed_cuda=tuple(args.cuda.split(",")) if args.cuda else None,
    )
    body = spec.body(deadline)
    print(
        f"plan: {gpu_id} ({args.cloud}) at listed ${listed:.3f}/h, {args.hours} h cap "
        f"-> worst case ${worst:.2f}; balance ${account['balance']:.2f}\n"
        f"      name {body['name']}  deadline {time.strftime('%H:%M:%S %Z', time.localtime(deadline))}"
        f"  vCPU>={args.vcpu} RAM>={args.ram}GB disk {args.disk}GB  watchdog={'on' if spec.watchdog else 'OFF'}\n"
        f"      image {args.image}  cuda filter {args.cuda or 'none'}"
    )
    if not args.yes:
        print("dry run: nothing created (pass --yes to launch)")
        return 0

    pod = rp.create_pod(body)
    cloud.ledger(
        "launch", pod_id=pod["id"], name=body["name"], deadline=deadline, spec=cloud.as_dict(spec)
    )
    actual = float(pod.get("costPerHr") or listed)
    try:
        cloud.plan(actual, args.hours, account["balance"], limits)
    except cloud.BudgetError as e:
        rp.terminate_pod(pod["id"])
        cloud.ledger("terminate", pod_id=pod["id"], reason=f"actual rate refused: {e}")
        print(
            f"terminated {pod['id']} at once: the quoted rate ${actual:.3f}/h breaks the budget ({e})"
        )
        return 2
    print(
        f"launched {pod['id']} {body['name']} at ${actual:.3f}/h (vCPU {pod.get('vcpuCount')}, RAM {pod.get('memoryInGb')} GB)"
    )
    return 0


def _target(rp: cloud.RunPod, pod_id: str) -> tuple[str, int]:
    pod = rp.get_pod(pod_id)
    if pod is None:
        raise cloud.CloudError(f"no pod {pod_id}")
    t = cloud.ssh_target(pod)
    if t is None:
        raise cloud.CloudError(f"pod {pod_id} has no public SSH endpoint yet")
    return t


def cmd_wait_ssh(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    end = time.time() + args.timeout
    while time.time() < end:
        pod = rp.get_pod(args.pod)
        if pod is None:
            raise cloud.CloudError(f"pod {args.pod} is gone")
        t = cloud.ssh_target(pod)
        if t is not None:
            ok = subprocess.run([*cloud.ssh_base(*t), "true"], capture_output=True, timeout=60)
            if ok.returncode == 0:
                print(f"ssh ready: {' '.join(cloud.ssh_base(*t))}")
                return 0
        time.sleep(10)
    print(f"ssh not ready after {args.timeout} s", file=sys.stderr)
    return 1


def cmd_ssh(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    return subprocess.run([*cloud.ssh_base(*_target(rp, args.pod)), *args.command]).returncode


def _rsync(rp: cloud.RunPod, pod_id: str, src: str, dst: str, up: bool) -> int:
    ip, port = _target(rp, pod_id)
    ssh = " ".join(cloud.ssh_base(ip, port)[:-1])  # the -e command, without the user@host
    remote = f"root@{ip}:"
    a, b = (src, remote + dst) if up else (remote + src, dst)
    return subprocess.run(["rsync", "-az", "--partial", "-e", ssh, a, b]).returncode


def cmd_push(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    return _rsync(rp, args.pod, args.src, args.dst, up=True)


def cmd_pull(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    return _rsync(rp, args.pod, args.src, args.dst, up=False)


def cmd_status(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    now = time.time()
    ours = [p for p in rp.list_pods() if cloud.is_ours(p.get("name", ""))]
    for p in ours:
        dl = cloud.parse_deadline(p["name"])
        left = f"{(dl - now) / 60:+.0f} min to deadline" if dl else "NO DEADLINE"
        print(f"{p['id']}  {p['name']}  {p.get('desiredStatus')}  ${p.get('costPerHr')}/h  {left}")
    print(f"{len(ours)} pod(s) of ours; account {json.dumps(rp.balance())}")
    return 0


def cmd_terminate(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    targets = (
        [p["id"] for p in rp.list_pods() if cloud.is_ours(p.get("name", ""))]
        if args.all_ours
        else [args.pod]
    )
    for pod_id in targets:
        pod = rp.get_pod(pod_id)
        if pod is not None and not cloud.is_ours(pod.get("name", "")):
            raise cloud.CloudError(f"{pod_id} ({pod.get('name')}) is not ours; refusing")
        rp.terminate_pod(pod_id)
        cloud.ledger("terminate", pod_id=pod_id, reason="requested")
        print(f"terminated {pod_id}")
    return 0


def cmd_sweep(rp: cloud.RunPod, args: argparse.Namespace) -> int:
    pods = rp.list_pods()
    failures = 0
    for pod_id, name, reason in cloud.sweep_decisions(pods, time.time(), args.grace):
        try:
            rp.terminate_pod(pod_id)
            cloud.ledger("sweep_terminate", pod_id=pod_id, name=name, reason=reason)
            msg = f"taz sweeper terminated {name}: {reason}"
        except cloud.CloudError as e:
            failures += 1
            cloud.ledger("sweep_failed", pod_id=pod_id, name=name, reason=reason, error=str(e))
            msg = f"taz sweeper FAILED to terminate {name}: {e}"
        print(msg)
        subprocess.run(["notify-send", "-u", "critical", "RunPod sweeper", msg], check=False)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prices")
    p.add_argument("--max-price", type=float, default=1.0)
    sub.add_parser("balance")
    p = sub.add_parser("launch")
    p.add_argument("--gpu", required=True, help="GPU type id or display name, e.g. 'RTX 4090'")
    p.add_argument("--hours", type=float, required=True, help="hard deadline, from now")
    p.add_argument("--tag", required=True, help="lowercase [a-z0-9_-]")
    p.add_argument("--cloud", choices=["COMMUNITY", "SECURE"], default="COMMUNITY")
    p.add_argument("--vcpu", type=int, default=16, help="minimum vCPUs")
    p.add_argument("--ram", type=int, default=32, help="minimum RAM, GB")
    p.add_argument("--disk", type=int, default=60, help="container disk, GB")
    p.add_argument("--no-watchdog", action="store_true", help="lifecycle tests only")
    p.add_argument("--image", default=cloud.DEFAULT_IMAGE)
    p.add_argument("--cuda", default=None, help="comma list of host CUDA versions to accept")
    p.add_argument("--yes", action="store_true", help="actually create the pod")
    p = sub.add_parser("wait-ssh")
    p.add_argument("pod")
    p.add_argument("--timeout", type=int, default=900)
    p = sub.add_parser("ssh")
    p.add_argument("pod")
    p.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("push", "pull"):
        p = sub.add_parser(name)
        p.add_argument("pod")
        p.add_argument("src")
        p.add_argument("dst")
    sub.add_parser("status")
    p = sub.add_parser("terminate")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("pod", nargs="?")
    g.add_argument("--all-ours", action="store_true")
    p = sub.add_parser("sweep")
    p.add_argument("--grace", type=int, default=300, help="seconds past a deadline before acting")
    args = ap.parse_args(argv)
    if getattr(args, "command", None) and args.command[:1] == ["--"]:
        args.command = args.command[1:]

    rp = cloud.RunPod()
    try:
        return globals()[f"cmd_{args.cmd.replace('-', '_')}"](rp, args)
    except cloud.CloudError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
