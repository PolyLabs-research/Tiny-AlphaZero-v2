#!/usr/bin/env python3
"""Markov-invariant fuzz of the board encoding (spec §3.1, §9.4; M0).

    python scripts/fuzz_encoding.py --fuzz 1000000 --seed 0

Walks random games and checks that two positions with the same encoding have the
same legal-move set and the same draw-claim status. The check itself is
``tinyalphazero.encoding.fuzz_markov_invariant`` (it raises on a violation); this
wrapper prints the summary. It used to be ``python -m tinyalphazero.encoding``
(REVIEW-2026-10-21 E11: no ``print()`` in ``src/``).
"""

from __future__ import annotations

import argparse
import sys

from tinyalphazero.encoding import fuzz_markov_invariant


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fuzz", type=int, required=True, metavar="N", help="positions to walk")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    stats = fuzz_markov_invariant(args.fuzz, seed=args.seed)
    print(
        f"Fuzz complete: {stats['positions']} positions, {stats['unique']} unique encodings, "
        f"{stats['collisions']} same-encoding collisions — all invariants hold."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
