"""The Phase 2 per-iteration log row (spec §7, chunk 9) — one definition.

`runs/<name>/iterations.jsonl` is the campaign's only durable narrative: a 24h run
produces ~50 of these rows and nothing else that survives the process. Chunk 11's
dashboard is specified as a pure reader with zero imports from training code
(AGENTS.md §4), which only works if "what a row contains" is written down
somewhere both sides agree on. That is this module: the writer builds rows through
:func:`iteration_row`, the reader validates them through :func:`validate`, and a
field that exists in one and not the other is a test failure rather than a silent
``KeyError`` at hour nineteen.

**The wall-clock split is the point of the schema, not a nicety.** Spec erratum 6
replaced §10's unreachable 3,000 sims/s target with a budget of ~35 minutes per
iteration. A budget is only auditable if every iteration says where its minutes
went, so ``selfplay_s + train_s + snapshot_s + other_s == total_s`` exactly, by
construction: ``other_s`` is the residual, not an estimate. When an iteration runs
long the row already says which stage did it.

**None means "not measured".** ``false_resign_rate`` is None until a
resign-disabled game has actually crossed the threshold, and
``mean_plies_decisive`` is None in an iteration with no decisive game. A 0.0 in
either slot would read as a measurement, and the resignation rate is a calibration
number people will act on.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1

#: Fields whose value may legitimately be ``null`` in the JSONL.
NULLABLE = frozenset({"false_resign_rate", "mean_plies_decisive"})

#: The wall-clock split must account for the whole iteration.
_SPLIT = ("selfplay_s", "train_s", "snapshot_s", "other_s")
_SPLIT_TOLERANCE_S = 0.02  # rounding only; the residual is computed, not measured


@dataclass
class IterationRow:
    """One iteration of the Phase 2 loop."""

    # -- identity ---------------------------------------------------------
    iteration: int
    git_sha: str

    # -- generation -------------------------------------------------------
    games: int
    positions: int
    mean_plies: float
    mean_plies_decisive: float | None
    decisive_rate: float
    draw_rate: float
    # Draw causes are reported separately because they mean different things: a
    # ply_cap draw is an unresolved game, a draw_claim is a repetition/50-move
    # agreement the search understood, and `auto` is stalemate or insufficient
    # material. A single draw rate hides a shift between them, which is exactly
    # what a draw-glut looks like as it starts.
    draw_rate_ply_cap: float
    draw_rate_draw_claim: float
    draw_rate_auto: float
    resign_rate: float
    false_resign_rate: float | None
    # Two entropies at move 1, in nats, because one number cannot answer both
    # questions. ``move1_prior_entropy`` is spec §8's collapse detector: has the
    # policy stopped considering alternatives. ``move1_policy_entropy`` is the
    # sharpness of the improved policy, i.e. of the training target, and it falls
    # as soon as the value head can separate two moves — σ multiplies q̂ by
    # ``(c_visit + max_N)`` ≈ 70. Reading the second as the first would report a
    # working value head as a collapsing policy.
    move1_policy_entropy: float
    move1_prior_entropy: float
    pool_moves_frac: float  # fraction of stored positions whose mover was a pool opponent

    # -- learning ---------------------------------------------------------
    train_steps: int
    skipped_steps: int
    loss: float
    policy_loss: float
    promo_loss: float
    value_loss: float
    legality_loss: float
    grad_norm: float
    lr: float
    replay_size: int
    pool_size: int

    # -- throughput and the wall-clock split ------------------------------
    sims: int
    sims_per_s: float
    selfplay_s: float
    train_s: float
    snapshot_s: float
    other_s: float
    total_s: float

    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Declared field order — the reader's contract.
FIELDS: tuple[str, ...] = tuple(f.name for f in fields(IterationRow))


def iteration_row(**kwargs: Any) -> dict[str, Any]:
    """Build a validated row. Raises on a missing, unknown, or non-summing field.

    Callers pass the three measured stage times and ``total_s``; ``other_s`` is
    derived here so the split cannot be made to sum by adjusting a stage.
    """
    kwargs.setdefault(
        "other_s",
        round(
            kwargs["total_s"] - kwargs["selfplay_s"] - kwargs["train_s"] - kwargs["snapshot_s"], 4
        ),
    )
    row = IterationRow(**kwargs).to_dict()
    validate(row)
    return row


def validate(row: dict[str, Any]) -> None:
    """Raise ``ValueError`` unless ``row`` is exactly one well-formed schema row."""
    missing = [k for k in FIELDS if k not in row]
    unknown = [k for k in row if k not in FIELDS]
    if missing or unknown:
        raise ValueError(f"log row schema mismatch: missing={missing} unknown={unknown}")

    nulls = [k for k in FIELDS if row[k] is None and k not in NULLABLE]
    if nulls:
        raise ValueError(f"log row has null in non-nullable field(s): {nulls}")

    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"log row schema_version {row['schema_version']} != {SCHEMA_VERSION}")

    split = sum(float(row[k]) for k in _SPLIT)
    if abs(split - float(row["total_s"])) > _SPLIT_TOLERANCE_S:
        raise ValueError(
            f"wall-clock split does not account for the iteration: "
            f"{' + '.join(_SPLIT)} = {split:.4f} vs total_s = {row['total_s']:.4f}"
        )


def append_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one validated row to a JSONL log."""
    validate(row)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL log. Blank lines are skipped; a torn final line is dropped.

    A killed process can leave a partial last line, and refusing to read the whole
    file because of it would make the dashboard the second casualty of the crash.
    """
    rows: list[dict[str, Any]] = []
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if i == len(lines) - 1:
                break  # torn tail from a kill; everything before it is intact
            raise
    return rows
