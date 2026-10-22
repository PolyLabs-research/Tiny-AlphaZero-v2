"""The .gitignore must never swallow a record the project depends on.

Four separate incidents across two sessions, each caught by hand after the fact:

  BLOCKED.md              ignored outright, contradicting AGENTS.md's own rule
  bare `data/`            ignored runs/data/ as a directory, so git could not
                          descend to reach the meta.json negations
  benchmarks/results.jsonl ignored, making a dated SHA-stamped record unversioned
  runs/**                 swallowed runs/ANCHORS.sha256 on the commit that added it

Whack-a-mole was losing, so this closes the class: the must-track list below is
the contract, and `git check-ignore` is the judge. Add to MUST_TRACK whenever a
new kind of record starts mattering.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# Paths (or glob patterns) that must never be ignored. Patterns are expanded
# against the working tree; a pattern matching nothing is itself a failure,
# because a record that has vanished is as bad as one that is ignored.
MUST_TRACK: list[str] = [
    "BLOCKED-*.md",
    "benchmarks/results.jsonl",
    "runs/ANCHORS.sha256",
    "runs/data/*/meta.json",
    "runs/*/metrics.jsonl",
    "runs/*/ladder.jsonl",  # the anchor ladder IS the M3 verdict
    "runs/p2_b/annotations.jsonl",  # why a timing row is anomalous
    "runs/*/config.yaml",  # the resolved config a run's rows are read against
    "tests/suites/*.epd",
    "TRAINING_STRATEGY.md",
    "PROGRESS.md",
    # The mate-in-2 tuning suite the eval-profile sweep was selected on (E10).
    "runs/data/*/mate_in_*.epd",
    # p2_c/p2_d record kinds (REVIEW-2026-10-21 P7): the Track 1 convertibility
    # scores, the registered-statistic replications and their verdict, and the
    # pair-score replay that found the skill-0 nondeterminism.
    "runs/p2_*/track1_ckpt_iter*.json",
    "runs/p2_d/p2_verdict_reps.json",
    "runs/p2_*/reps_iter*.jsonl",
    "runs/p2_c/ladder_pairscores.jsonl",
]

# Records that do not exist yet but must be reachable when they do. MUST_TRACK
# needs a file on disk, which cannot cover a rule written *before* the run that
# produces it — and "we'll remember to add the negation later" is how three of the
# four incidents above happened. `git check-ignore -q` answers for a hypothetical
# path, so the rule can be tested the day it is written.
MUST_REACH: list[str] = [
    "runs/p2_a/iterations.jsonl",  # a Phase 2 campaign's only durable narrative
    "runs/p2_a/provenance.json",  # the SHA and fingerprint that make it readable
    "runs/p2_b/ladder.jsonl",  # p2_b's passes, before p2_b has run
    "runs/p2_b/config.yaml",
    "runs/p2_a/ladder_diagnostic.jsonl",  # diagnostic passes, excluded from a verdict
    "runs/p2_a/triage_mate2_iter0050.json",  # the decomposition RUNLOG-p2_a cites
    "runs/p2_a/s4_flip_iter0050.json",
    # The next campaign's copies of the p2_c/p2_d record kinds, and a fresh
    # tuning suite harvested under a new name.
    "runs/p2_e/track1_ckpt_iter0020.json",
    "runs/p2_e/p2_verdict_reps.json",
    "runs/p2_e/reps_iter0020_r1.jsonl",
    "runs/p2_e/ladder_pairscores.jsonl",
    "runs/data/mate2_tune_fresh/mate_in_2.epd",
    # p2_e's bridge: p2_d/0020 re-measured on the fixed code (PREREG-p2_e Step 0).
    "runs/p2_e_bridge/ladder.jsonl",
    "runs/p2_e_bridge/reps_iter0020_r1.jsonl",
    "runs/p2_e_bridge/p2_verdict_reps.json",
]

# The counterpart: things that must STAY ignored. A well-meant negation that
# accidentally starts tracking 75 MB checkpoints is the opposite failure.
MUST_IGNORE: list[str] = [
    "runs/phase1_anchor.pt",
    "runs/phase1_c/final.pt",
    "runs/data/mate2_tune_500/keymoves.dat",  # the .epd negation must not widen to data
    ".venv/bin/python",
]


def _is_git_repo() -> bool:
    return (REPO / ".git").exists()


def _check_ignored(path: str) -> bool:
    """True if git ignores ``path``. Uses check-ignore's exit status (0 = ignored)."""
    result = subprocess.run(
        ["git", "-C", str(REPO), "check-ignore", "-q", path],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


pytestmark = pytest.mark.skipif(
    not _is_git_repo(), reason="not a git repository — nothing to check"
)


@pytest.mark.parametrize("pattern", MUST_TRACK)
def test_must_track_paths_are_not_ignored(pattern: str) -> None:
    matches = sorted(REPO.glob(pattern))
    assert matches, f"{pattern!r} matched no file — the record it names is missing"

    ignored = [
        str(p.relative_to(REPO)) for p in matches if _check_ignored(str(p.relative_to(REPO)))
    ]
    assert not ignored, (
        f"{pattern!r} is gitignored: {ignored}\n"
        "These are project records, not build output. Add a negation to .gitignore."
    )


@pytest.mark.parametrize("path", MUST_REACH)
def test_future_records_are_already_reachable(path: str) -> None:
    """The negation must be in place before the run that produces the record."""
    assert not _check_ignored(path), (
        f"{path!r} would be gitignored. A run that writes it would produce a record "
        "git silently drops — add a negation to .gitignore."
    )


@pytest.mark.parametrize("path", MUST_IGNORE)
def test_must_ignore_paths_stay_ignored(path: str) -> None:
    """Guard the other direction: negations must not start tracking large blobs."""
    if not (REPO / path).exists():
        pytest.skip(f"{path} not present in this working tree")
    assert _check_ignored(path), (
        f"{path!r} is NOT ignored — a .gitignore negation has gone too wide. "
        "Checkpoints and the venv must never enter the repository."
    )
