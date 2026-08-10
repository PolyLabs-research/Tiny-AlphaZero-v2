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

**And absence carries a reason.** Where a value can be legitimately missing for
more than one cause, the row also carries a non-nullable note saying which — see
``LadderEvalRow.calibration_note``. This is the structural form of the rule above:
a row may lack a measurement, but it may never be silent about why. The reader
these logs are written for is someone months from now with no access to the
conversation that produced them, and "ece: null" alone leaves them to decide for
themselves whether the value head was uncalibrated or simply never asked.
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
    _append(path, row)


def _append(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# The anchor ladder (spec §8, chunk 10)
#
# Two row types share `ladder.jsonl` and are told apart by `row_type`, because
# they answer different questions at different cadences: one row per
# (checkpoint, anchor) matchup, and one row per checkpoint for everything that
# does not involve an opponent. Splitting them into two files would make "what
# did this checkpoint score" a join.
# ---------------------------------------------------------------------------

MATCH_ROW = "match"
EVAL_ROW = "eval"

#: Nullable in a match row: an anchor that could not be run leaves the CI absent
#: rather than zero, and ``iteration`` is None for a standalone evaluation of a
#: checkpoint that is not part of a campaign (the Phase 1 anchor, for instance).
_MATCH_NULLABLE = frozenset({"elo_lo", "elo_hi", "iteration"})
#: Nullable in an eval row: a suite or metric that is not present in this working
#: tree reports None, which reads as "not measured" rather than "measured as 0".
_EVAL_NULLABLE = frozenset(
    {"mate3_rate", "mate3_solved", "mate3_total", "ece", "calibration_n", "iteration"}
)


@dataclass
class LadderMatchRow:
    """One (checkpoint, anchor) matchup: the Elo curve M3 is judged on."""

    row_type: str
    checkpoint: str
    iteration: int | None
    anchor: str
    games: int
    wins: int
    draws: int
    losses: int
    score: float
    elo: float
    elo_lo: float | None
    elo_hi: float | None
    elo_clamped: bool
    draw_rate: float
    mean_plies: float
    match_sims: int
    max_plies: int
    seconds: float
    git_sha: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LadderEvalRow:
    """One checkpoint's opponent-free measurements: suites, watchlist, calibration."""

    row_type: str
    checkpoint: str
    iteration: int | None
    suite_sims: int
    mate2_solved: int
    mate2_total: int
    mate2_rate: float
    mate2_baseline: float  # the pinned M3 no-regress number
    mate2_no_regress: bool  # rate >= baseline (within one position)
    mate2_m3_clause: bool  # rate >= 0.95 absolute
    mate2_failed: list
    mate3_solved: int | None
    mate3_total: int | None
    mate3_rate: float | None
    watchlist: list  # [{id, verdict, solved, value_on_key_child, baseline_value}]
    ece: float | None
    calibration_n: int | None
    calibration_bins: list
    # Why `ece` is what it is, in words, always present. Calibration draws its
    # positions from the *anchor* matchup, so a partial pass (`--anchors random`)
    # legitimately measures nothing — and a bare null leaves a reader to guess
    # whether the value head was uncalibrated or simply never asked. The house
    # rule is that None means "not measured"; this says which.
    calibration_note: str
    book_move1_entropy: float
    seconds: float
    git_sha: str
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MATCH_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LadderMatchRow))
EVAL_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LadderEvalRow))


def _check(row: dict[str, Any], declared: tuple[str, ...], nullable: frozenset[str]) -> None:
    missing = [k for k in declared if k not in row]
    unknown = [k for k in row if k not in declared]
    if missing or unknown:
        raise ValueError(f"ladder row schema mismatch: missing={missing} unknown={unknown}")
    nulls = [k for k in declared if row[k] is None and k not in nullable]
    if nulls:
        raise ValueError(f"ladder row has null in non-nullable field(s): {nulls}")
    if row["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"ladder row schema_version {row['schema_version']} != {SCHEMA_VERSION}")


def validate_ladder(row: dict[str, Any]) -> None:
    """Raise unless ``row`` is a well-formed ladder row of either type."""
    kind = row.get("row_type")
    if kind == MATCH_ROW:
        _check(row, MATCH_FIELDS, _MATCH_NULLABLE)
    elif kind == EVAL_ROW:
        _check(row, EVAL_FIELDS, _EVAL_NULLABLE)
    else:
        raise ValueError(
            f"unknown ladder row_type {kind!r}; expected {MATCH_ROW!r} or {EVAL_ROW!r}"
        )


def ladder_match_row(**kwargs: Any) -> dict[str, Any]:
    row = LadderMatchRow(row_type=MATCH_ROW, **kwargs).to_dict()
    validate_ladder(row)
    return row


def ladder_eval_row(**kwargs: Any) -> dict[str, Any]:
    row = LadderEvalRow(row_type=EVAL_ROW, **kwargs).to_dict()
    validate_ladder(row)
    return row


def append_ladder_row(path: str | Path, row: dict[str, Any]) -> None:
    """Append one validated ladder row.

    Written the moment a matchup finishes, not at the end of the pass. That is
    what makes a ladder pass resumable: a kill costs the matchup in flight and
    nothing already measured. Same ordering argument as the replay ring — data
    lands before anything points at it.
    """
    validate_ladder(row)
    _append(path, row)


def completed_matchups(path: str | Path) -> set[tuple[str, str]]:
    """``(checkpoint, anchor)`` pairs already in the log — never recomputed."""
    p = Path(path)
    if not p.exists():
        return set()
    return {(r["checkpoint"], r["anchor"]) for r in read_jsonl(p) if r.get("row_type") == MATCH_ROW}


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
