"""Where a run's provenance comes from — one implementation, read once.

Two rules, both learned the expensive way (REVIEW-2026-10-21 C2/D2, B4):

* **The git SHA is read once per process, at launch, and stamped from that
  value.** Reading ``git rev-parse HEAD`` at every write stamps whatever happens
  to be checked out at that moment, and a 12-hour campaign keeps running the code
  it was launched from while commits land in the tree. p2_b's rows carry four
  SHAs from one process, two of them for code that writes a schema version the
  rows do not have; every p2_d row names a commit made 24 minutes after launch.
  Callers that write rows or checkpoints take the SHA as a parameter and get it
  from the run state, not from this module, every time after the first.
* **Anchored to the source tree, never the cwd**, and absence is explicit:
  ``"no-git"`` is a sentinel a reader can test for, ``"unknown"`` was a value
  that looked like a measurement.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

#: What ``git_sha`` returns when there is no answer: git absent, not a
#: repository, or no commits yet. A sentinel, deliberately not a plausible SHA.
NO_GIT_SENTINEL = "no-git"

#: The repository this package lives in (``src/tinyalphazero/provenance.py`` →
#: two parents up). ``git -C`` against this, so the answer does not depend on
#: where the process was launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]


def git_sha(repo_root: Path | str = REPO_ROOT, short: bool = True) -> str:
    """HEAD of ``repo_root`` (short by default), or :data:`NO_GIT_SENTINEL`."""
    args = ["git", "-C", str(repo_root), "rev-parse"]
    if short:
        args.append("--short")
    args.append("HEAD")
    try:
        return subprocess.check_output(args, stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.SubprocessError):
        return NO_GIT_SENTINEL


def git_dirty(repo_root: Path | str = REPO_ROOT) -> bool | None:
    """Does the tree carry uncommitted changes to tracked files? None if unknown.

    A SHA names the code only when the tree matches it; a dirty flag next to the
    SHA is what lets a reader tell "ran 4710690" from "ran 4710690 plus an edit".
    Untracked files are ignored: run artefacts and scratch do not change the code.
    """
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return bool(out.strip())


def sha256_file(path: Path | str, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file's bytes, streamed (anchor checkpoints are ~78 MB)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            block = fh.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
