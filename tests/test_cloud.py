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
    spec = cloud.LaunchSpec(tag="lifecycle", gpu_type="NVIDIA RTX A5000", hours=0.25)
    body = spec.body(deadline=1_800_000_000)
    assert body["volumeInGb"] == 0, "a pod volume bills after a stop"
    assert body["ports"] == ["22/tcp"] and body["supportPublicIp"] is True
    assert body["env"]["TAZ_DEADLINE"] == "1800000000"
    assert cloud.parse_deadline(body["name"]) == 1_800_000_000
    assert body["minVCPUPerGPU"] == 16, "the default 2 vCPUs cannot run the actors"


def test_watchdog_runs_before_the_image_start_and_can_be_disabled_for_tests() -> None:
    on = cloud.watchdog_start_cmd(True)
    assert on[:2] == ["bash", "-c"]
    assert "TAZ_DEADLINE" in on[2] and on[2].rstrip().endswith(f"exec {cloud.IMAGE_CMD}")
    assert "runpodctl remove pod" in on[2] and "rest.runpod.io/v1/pods" in on[2]
    off = cloud.watchdog_start_cmd(False)
    assert "runpodctl" not in off[2] and off[2] == f"exec {cloud.IMAGE_CMD}"


def test_cuda_filter_is_opt_in_and_the_watchdog_names_itself_to_cloudflare() -> None:
    """The API's CUDA filter matched no host on 2026-10-24, and Cloudflare refuses
    anonymous-looking clients, so: no filter unless asked, and a User-Agent on curl."""
    body = cloud.LaunchSpec(tag="t", gpu_type="g", hours=1).body(deadline=1_800_000_000)
    assert "allowedCudaVersions" not in body
    pinned = cloud.LaunchSpec(tag="t", gpu_type="g", hours=1, allowed_cuda=("12.8",))
    assert pinned.body(deadline=1_800_000_000)["allowedCudaVersions"] == ["12.8"]
    assert '-A "taz-cloud/1.0' in cloud.watchdog_start_cmd(True)[2]
