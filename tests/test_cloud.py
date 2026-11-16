"""RunPod lifecycle logic (``tinyalphazero.cloud``): the parts that decide, without the network.

What must never happen is a pod left running, or a pod that is not ours touched.
Both are decided by pure functions — the name/deadline convention, the budget, the
sweeper's verdicts and the launch body — so they are pinned here; the live
lifecycle is proven against the real API separately (RUNLOG / benchmarks note).
"""

from __future__ import annotations

import pytest

from tinyalphazero import cloud


def test_names_carry_their_deadline_and_round_trip() -> None:
    name = cloud.pod_name("p2g", 1792800000)
    assert name == "taz-p2g-dl1792800000"
    assert cloud.parse_deadline(name) == 1792800000
    assert cloud.is_ours(name)


@pytest.mark.parametrize("bad", ["P2G", "has space", "", "x" * 41 + "y", "-lead"])
def test_a_tag_that_would_make_an_unparseable_name_is_refused(bad: str) -> None:
    with pytest.raises(ValueError):
        cloud.pod_name(bad, 1792800000)


def test_the_sweeper_never_touches_a_pod_that_is_not_ours() -> None:
    pods = [
        {"id": "a", "name": "my comfy pod"},
        {"id": "b", "name": "tazzy-dl0000000001"},  # looks close; is not the prefix
        {"id": "c", "name": "TAZ-x-dl1000000000"},  # case matters
    ]
    assert cloud.sweep_decisions(pods, now=9e9) == []


def test_the_sweeper_terminates_ours_past_deadline_plus_grace_only() -> None:
    dl = 1_800_000_000
    pods = [
        {"id": "early", "name": cloud.pod_name("t", dl)},
        {"id": "late", "name": cloud.pod_name("u", dl - 400)},
        {"id": "nodl", "name": "taz-orphan"},
    ]
    verdicts = {
        pid: reason for pid, _name, reason in cloud.sweep_decisions(pods, now=dl + 200, grace_s=300)
    }
    assert set(verdicts) == {"late", "nodl"}, verdicts  # 'early' is inside its grace
    assert "no parseable deadline" in verdicts["nodl"]


def test_budget_refuses_over_ceiling_over_hours_and_into_the_margin() -> None:
    limits = cloud.Limits(max_session_cost=10.0, max_hours=30.0, balance_margin=5.0)
    assert cloud.plan(0.34, 20, balance=49.98, limits=limits) == pytest.approx(6.8)
    with pytest.raises(cloud.BudgetError, match="session ceiling"):
        cloud.plan(0.89, 20, balance=49.98, limits=limits)
    with pytest.raises(cloud.BudgetError, match="ceiling"):
        cloud.plan(0.10, 31, balance=49.98, limits=limits)
    with pytest.raises(cloud.BudgetError, match="margin"):
        cloud.plan(0.34, 20, balance=10.0, limits=limits)
    with pytest.raises(cloud.BudgetError):
        cloud.plan(0.34, 0, balance=49.98, limits=limits)


def test_launch_body_is_terminate_only_full_ssh_and_names_its_deadline() -> None:
    """The v2 CreatePodRequest (v1 retired 2026-11-15; validated against RunPod's OpenAPI)."""
    spec = cloud.LaunchSpec(tag="lifecycle", gpu_type="NVIDIA RTX A5000", hours=0.25)
    body = spec.body(deadline=1_800_000_000)
    assert "mounts" not in body, "a pod volume bills after a stop"
    assert body["ports"] == ["22/tcp"] and body["startSsh"] is True  # ssh.direct, for rsync
    assert body["env"]["TAZ_DEADLINE"] == "1800000000"
    assert cloud.parse_deadline(body["name"]) == 1_800_000_000
    assert body["gpu"] == {
        "id": "NVIDIA RTX A5000", "count": 1, "minVcpuCountPerGpu": 16, "minRamPerGpu": 32,
    }, "the default 2 vCPUs cannot run the actors"  # fmt: skip
    assert body["cmd"] == cloud.watchdog_start_cmd(True) and body["image"] == cloud.DEFAULT_IMAGE
    v1_fields = {"imageName", "gpuTypeIds", "volumeInGb", "supportPublicIp", "dockerStartCmd"}
    assert not v1_fields & set(body), "v2 refuses unknown fields with a 422"


class _FakeApi(cloud.RunPod):
    """``RunPod`` with canned v2 responses keyed by (method, url)."""

    def __init__(self, responses: dict[tuple[str, str], tuple[int, object]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, str]] = []

    def _request(self, method: str, url: str, body: object = None) -> tuple[int, object]:
        self.calls.append((method, url))
        return self.responses[(method, url)]


def test_v2_lists_every_page_and_leaves_terminated_pods_out() -> None:
    base = f"{cloud.REST}/pods"
    api = _FakeApi({
        ("GET", base): (200, {"pods": [{"id": "a", "status": "RUNNING"}, {"id": "b", "status": "TERMINATED"}],
                              "pagination": {"hasNextPage": True, "nextCursor": "c/1"}}),
        ("GET", base + "?cursor=c/1"): (200, {"pods": [{"id": "c", "status": "EXITED"}],
                                              "pagination": {"hasNextPage": False, "nextCursor": None}}),
    })  # fmt: skip
    assert [p["id"] for p in api.list_pods()] == ["a", "c"]
    assert cloud.REST == "https://api.runpod.io/v2"
    with pytest.raises(cloud.CloudError):  # a v1-shaped bare list is an error, not an empty account
        _FakeApi({("GET", base): (200, [{"id": "a"}])}).list_pods()


def test_v2_get_terminate_and_ssh_endpoint() -> None:
    base = f"{cloud.REST}/pods"
    running = {
        "id": "a", "status": "RUNNING",
        "ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22}, "direct": {"host": "1.2.3.4", "port": 40022}},
    }  # fmt: skip
    api = _FakeApi({
        ("GET", base + "/a"): (200, running),
        ("GET", base + "/t"): (200, {"id": "t", "status": "TERMINATED"}),
        ("GET", base + "/gone"): (404, {"title": "Not Found", "status": 404}),
        ("DELETE", base + "/a"): (204, None),
        ("DELETE", base + "/gone"): (404, {"title": "Not Found", "status": 404}),
        ("DELETE", base + "/x"): (500, {"title": "Internal", "status": 500}),
    })  # fmt: skip
    assert api.get_pod("a") == running and api.get_pod("t") is None and api.get_pod("gone") is None
    assert api.terminate_pod("a") and api.terminate_pod("gone")
    with pytest.raises(cloud.CloudError):
        api.terminate_pod("x")
    assert cloud.ssh_target(running) == ("1.2.3.4", 40022)  # direct, never the shell-only proxy
    assert (
        cloud.ssh_target({"ssh": {"proxy": {"host": "ssh.runpod.io", "port": 22}, "direct": None}})
        is None
    )
    assert cloud.ssh_target({"id": "starting"}) is None


def test_watchdog_runs_before_the_image_start_and_can_be_disabled_for_tests() -> None:
    on = cloud.watchdog_start_cmd(True)
    assert on[:2] == ["bash", "-c"]
    assert "TAZ_DEADLINE" in on[2] and on[2].rstrip().endswith(f"exec {cloud.IMAGE_CMD}")
    assert "runpodctl remove pod" in on[2] and "https://api.runpod.io/v2/pods/" in on[2]
    assert "rest.runpod.io" not in on[2]
    off = cloud.watchdog_start_cmd(False)
    assert "runpodctl" not in off[2] and off[2] == f"exec {cloud.IMAGE_CMD}"


def test_cuda_filter_is_opt_in_and_the_watchdog_names_itself_to_cloudflare() -> None:
    """The API's CUDA filter matched no host on 2026-10-24, and Cloudflare refuses
    anonymous-looking clients, so: no filter unless asked, and a User-Agent on curl."""
    body = cloud.LaunchSpec(tag="t", gpu_type="g", hours=1).body(deadline=1_800_000_000)
    assert "allowedCudaVersions" not in body["gpu"]
    pinned = cloud.LaunchSpec(tag="t", gpu_type="g", hours=1, allowed_cuda=("12.8",))
    assert pinned.body(deadline=1_800_000_000)["gpu"]["allowedCudaVersions"] == ["12.8"]
    assert '-A "taz-cloud/1.0' in cloud.watchdog_start_cmd(True)[2]


def test_the_pod_job_deletes_itself_through_v2() -> None:
    from pathlib import Path

    job = (Path(__file__).resolve().parents[1] / "cloud" / "pod_job.sh").read_text()
    assert "https://api.runpod.io/v2/pods/$POD_ID" in job and "rest.runpod.io" not in job
