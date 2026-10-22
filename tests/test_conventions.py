"""AGENTS.md conventions that are cheap to assert and expensive to rediscover.

Each of these was found violated by REVIEW-2026-10-21 (E11, C10): printing
CLIs inside ``src/``, ``sys.path`` surgery in tests. A grep in a test keeps
them from coming back without anyone noticing.
"""

from __future__ import annotations

import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src" / "tinyalphazero"
TESTS = REPO / "tests"

_PRINT = re.compile(r"^\s*print\(")
_MAIN_GUARD = re.compile(r'^if __name__ == "__main__":')
# Anchored on the statement, not a substring: a bare substring test matches the
# string literal in this file's own check (AGENTS.md §5).
_SYS_PATH = re.compile(r"^\s*sys\.path(\.(insert|append|extend)\(|\s*\+?=)")


def _py_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def test_library_code_does_not_print_or_carry_a_cli() -> None:
    """AGENTS.md §4: library code raises; only scripts/ prints."""
    offenders = []
    for path in _py_files(SRC):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if _PRINT.match(line) or _MAIN_GUARD.match(line) or "import argparse" in line:
                offenders.append(f"{path.relative_to(REPO)}:{lineno}: {line.strip()}")
    assert not offenders, "\n".join(offenders)


def test_tests_do_not_mutate_sys_path() -> None:
    """AGENTS.md §3: sys.path mutation is banned — fix the packaging instead."""
    offenders = [
        f"{path.relative_to(REPO)}:{lineno}"
        for path in _py_files(TESTS)
        for lineno, line in enumerate(path.read_text().splitlines(), 1)
        if _SYS_PATH.match(line)
    ]
    assert not offenders, "\n".join(offenders)
