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

**Versions belong to row families, and a validator knows every version it has
ever written.** Until 2026-10-21 one ``SCHEMA_VERSION`` served both the iteration
row and the ladder rows, so a ladder-only change to 3 re-stamped byte-identical
iteration rows as 3 and made the validator reject the 70 unchanged rows of p2_a
and p2_b (REVIEW-2026-10-21 P10). Each family now has its own constant, and the
validators accept any version in that family's history against the shape that
version actually had — a committed row never becomes invalid because the code
moved on.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# The iteration row
# ---------------------------------------------------------------------------

#: The version new iteration rows are stamped with. The *shape* has never
#: changed — versions 1, 2 and 3 are one shape (see the module docstring) — so
#: this stays at 3 rather than restarting the count, and a reader never sees a
#: version go backwards in a log.
ITERATION_SCHEMA_VERSION = 3

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

    schema_version: int = ITERATION_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: Declared field order — the reader's contract.
FIELDS: tuple[str, ...] = tuple(f.name for f in fields(IterationRow))

#: Every version an iteration row has ever carried, and the shape it had. One
#: shape, three numbers — the history the module docstring explains.
_ITERATION_SHAPES: dict[int, tuple[str, ...]] = {1: FIELDS, 2: FIELDS, 3: FIELDS}


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
    """Raise ``ValueError`` unless ``row`` is a well-formed iteration row of a known version."""
    version = row.get("schema_version")
    if version not in _ITERATION_SHAPES:
        raise ValueError(
            f"log row schema_version {version!r} is not one this reader knows "
            f"({sorted(_ITERATION_SHAPES)})"
        )
    declared = _ITERATION_SHAPES[version]
    missing = [k for k in declared if k not in row]
    unknown = [k for k in row if k not in declared]
    if missing or unknown:
        raise ValueError(f"log row schema mismatch: missing={missing} unknown={unknown}")

    nulls = [k for k in declared if row[k] is None and k not in NULLABLE]
    if nulls:
        raise ValueError(f"log row has null in non-nullable field(s): {nulls}")

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
    """Append one row as its own line — after making sure the previous line ended.

    A process killed mid-write leaves a torn fragment with no newline. Appending
    straight after it would glue the new row onto the fragment: the combined line
    fails to parse, and because it is now the *last* line :func:`read_jsonl`
    drops it silently — so the matchup just measured vanishes from the log, and
    the next resume measures it again and makes the corrupt line non-final, at
    which point the whole file stops reading (REVIEW-2026-10-21 D1). The torn
    fragment carried no information a reader could use, so it is cut before the
    append; everything before it is intact by construction.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _repair_torn_tail(path)
    with path.open("a") as fh:
        fh.write(json.dumps(row) + "\n")


def _repair_torn_tail(path: Path) -> None:
    """Truncate a trailing partial line (no newline) left by a kill mid-write."""
    if not path.exists():
        return
    with path.open("r+b") as fh:
        fh.seek(0, os.SEEK_END)
        size = fh.tell()
        if size == 0:
            return
        fh.seek(size - 1)
        if fh.read(1) == b"\n":
            return
        fh.seek(0)
        data = fh.read()
        fh.seek(data.rfind(b"\n") + 1)
        fh.truncate()


# ---------------------------------------------------------------------------
# The anchor ladder (spec §8, chunk 10)
#
# Two row types share `ladder.jsonl` and are told apart by `row_type`, because
# they answer different questions at different cadences: one row per
# (checkpoint, anchor) matchup, and one row per checkpoint for everything that
# does not involve an opponent. Splitting them into two files would make "what
# did this checkpoint score" a join.
# ---------------------------------------------------------------------------

#: The version new ladder rows are stamped with. History, so a reader can place
#: any committed row (and :func:`validate_ladder` accepts each against its own
#: shape):
#:
#: * 1 — chunk 10; p2_a and p2_b's protocol passes.
#: * 2 — ``role`` and ``produced`` (2026-08-11); p2_b's Stockfish rows, p2_c.
#: * 3 — ``pair_scores`` (2026-08-13); p2_d. ``nondeterministic`` was added to
#:   this version after p2_d's rows were written, so for version 3 the field is
#:   *optional* on read: the 17 committed rows that lacked it were backfilled on
#:   2026-10-21 (RUNLOG-p2_d.md, errata), but a version number cannot be made to
#:   mean two shapes retroactively, and this one does.
#: * 4 — ``nondeterministic`` required; ``config_fingerprint`` on both row
#:   types and ``anchor_digest`` on match rows (REVIEW-2026-10-21 P1, D3, D6).
LADDER_SCHEMA_VERSION = 4

MATCH_ROW = "match"
EVAL_ROW = "eval"

# What a row is *for*, and how it was produced. Two axes, deliberately not one
# enum: p2_a's passes at iterations 30/40/50 were backfilled offline and are
# nonetheless the protocol passes M3 was judged on, so "backfill" is a statement
# about provenance and "protocol" a statement about standing. Collapsing them
# would have forced the five-pass record to describe three of its own rows as
# something other than what they are.
#
# `role` is the one that stops a misreading. A diagnostic row — the 0048 pass, a
# climbed Stockfish rung — announces that no gate reads it, in the row, where a
# future reader meets it, rather than in a PREREG paragraph they may never open.
ROLE_PROTOCOL = "protocol"  # a pass a gate or verdict may read
ROLE_DIAGNOSTIC = "diagnostic"  # instrumentation; no gate reads it, ever
ROLES = frozenset({ROLE_PROTOCOL, ROLE_DIAGNOSTIC})

PRODUCED_INLINE = "inline"  # by the campaign's own --ladder-every hook
PRODUCED_OFFLINE = "offline"  # by scripts/eval_ladder.py against a stored checkpoint
PRODUCED = frozenset({PRODUCED_INLINE, PRODUCED_OFFLINE})

#: Rows written before schema_version 2 carry neither field. They are not
#: silently defaulted on read: :func:`ladder_role` states what absence means and
#: where the answer actually lives for those files.

#: Nullable in a match row: an anchor that could not be run leaves the CI absent
#: rather than zero, ``iteration`` is None for a standalone evaluation of a
#: checkpoint that is not part of a campaign (the Phase 1 anchor, for instance),
#: and ``anchor_digest`` is None for every opponent that is not a checkpoint —
#: random, greedy and an engine have no file to hash.
_MATCH_NULLABLE = frozenset({"elo_lo", "elo_hi", "iteration", "anchor_digest"})
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
    # The resolved config the pass ran under, as ``sha256`` of its saved YAML
    # (see ``config.config_fingerprint``). An offline backfill and an inline pass
    # that disagree on ``eval_m`` or the match sims are different measurements
    # of the same checkpoint, and without this field nothing in the row could
    # tell them apart afterwards (REVIEW-2026-10-21 D6).
    config_fingerprint: str
    # sha256 of the opponent checkpoint's bytes for a network anchor, None for
    # every other family. The ladder's zero point is "the Phase 1 anchor", and
    # the anchor is a 78 MB untracked file at a path two versions have lived at
    # (runs/ANCHORS.sha256); the row records which one it actually played
    # (REVIEW-2026-10-21 D3).
    anchor_digest: str | None
    # Does the opponent randomize its own play? A row that says so cannot be
    # read as a reproducible measurement, and a single row of a nondeterministic
    # rung cannot be compared to another single row -- it needs repetitions.
    # Same lineage as `role` and `calibration_note`: the row announces what kind
    # of fact it is, rather than leaving the reader to know. Required, with no
    # default: "deterministic" must be something the opponent *declared*, never
    # what a missing attribute fell back to (REVIEW-2026-10-21 P3).
    nondeterministic: bool
    schema_version: int = LADDER_SCHEMA_VERSION
    role: str = ROLE_PROTOCOL
    produced: str = PRODUCED_OFFLINE
    # The learner's mean score over each opening's two colour-swapped games —
    # the unit paired_bootstrap_elo already resamples within a matchup. Stored
    # (v3) because comparing two *passes* needs the per-opening vector, and
    # p2_c discarded it: the aggregate cannot support a paired-difference test,
    # so "pass 15 beat pass 5" had to be demoted to descriptive after the fact
    # (RUNLOG-p2_c.md §2). ~100 floats per row.
    pair_scores: list = field(default_factory=list)

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
    config_fingerprint: str  # as on LadderMatchRow
    schema_version: int = LADDER_SCHEMA_VERSION
    role: str = ROLE_PROTOCOL
    produced: str = PRODUCED_OFFLINE

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


MATCH_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LadderMatchRow))
EVAL_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(LadderEvalRow))

# -- the shape each ladder version actually had ---------------------------------
# Built by subtraction from the current dataclasses so a field added to the
# dataclass is a field added to version 4 and to nothing earlier.
_V2_ADDED = frozenset({"role", "produced"})
_V3_ADDED = frozenset({"pair_scores"})
_V4_ADDED_MATCH = frozenset({"nondeterministic", "config_fingerprint", "anchor_digest"})
_V4_ADDED_EVAL = frozenset({"config_fingerprint"})

_MATCH_V1 = tuple(k for k in MATCH_FIELDS if k not in _V2_ADDED | _V3_ADDED | _V4_ADDED_MATCH)
_MATCH_V2 = tuple(k for k in MATCH_FIELDS if k not in _V3_ADDED | _V4_ADDED_MATCH)
_MATCH_V3 = tuple(k for k in MATCH_FIELDS if k not in _V4_ADDED_MATCH)
_EVAL_V1 = tuple(k for k in EVAL_FIELDS if k not in _V2_ADDED | _V4_ADDED_EVAL)
_EVAL_V2 = tuple(k for k in EVAL_FIELDS if k not in _V4_ADDED_EVAL)

#: version -> (required fields, optional fields). Optional exists for exactly one
#: reason: version 3 was written both without and with ``nondeterministic``.
_MATCH_SHAPES: dict[int, tuple[tuple[str, ...], frozenset[str]]] = {
    1: (_MATCH_V1, frozenset()),
    2: (_MATCH_V2, frozenset()),
    3: (_MATCH_V3, frozenset({"nondeterministic"})),
    4: (MATCH_FIELDS, frozenset()),
}
_EVAL_SHAPES: dict[int, tuple[tuple[str, ...], frozenset[str]]] = {
    1: (_EVAL_V1, frozenset()),
    2: (_EVAL_V2, frozenset()),
    3: (_EVAL_V2, frozenset()),
    4: (EVAL_FIELDS, frozenset()),
}


def _check(
    row: dict[str, Any],
    shapes: dict[int, tuple[tuple[str, ...], frozenset[str]]],
    nullable: frozenset[str],
) -> None:
    version = row.get("schema_version")
    if version not in shapes:
        raise ValueError(
            f"ladder row schema_version {version!r} is not one this reader knows ({sorted(shapes)})"
        )
    required, optional = shapes[version]
    missing = [k for k in required if k not in row]
    unknown = [k for k in row if k not in required and k not in optional]
    if missing or unknown:
        raise ValueError(
            f"ladder row (schema {version}) mismatch: missing={missing} unknown={unknown}"
        )
    nulls = [k for k in row if row[k] is None and k not in nullable]
    if nulls:
        raise ValueError(f"ladder row has null in non-nullable field(s): {nulls}")
    if "role" in row and row["role"] not in ROLES:
        raise ValueError(f"ladder row role {row['role']!r} not in {sorted(ROLES)}")
    if "produced" in row and row["produced"] not in PRODUCED:
        raise ValueError(f"ladder row produced {row['produced']!r} not in {sorted(PRODUCED)}")
    if "nondeterministic" in row and not isinstance(row["nondeterministic"], bool):
        raise ValueError(
            f"ladder row nondeterministic must be a bool, got {row['nondeterministic']!r}"
        )


def validate_ladder(row: dict[str, Any]) -> None:
    """Raise unless ``row`` is a well-formed ladder row of either type, of any known version."""
    kind = row.get("row_type")
    if kind == MATCH_ROW:
        _check(row, _MATCH_SHAPES, _MATCH_NULLABLE)
    elif kind == EVAL_ROW:
        _check(row, _EVAL_SHAPES, _EVAL_NULLABLE)
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


def ladder_role(row: dict[str, Any]) -> str | None:
    """``row``'s role, or None if it predates the field (schema_version < 2).

    None is returned rather than defaulted to ``protocol``, because the two
    pre-v2 files answer differently and neither answer is in the row: every row
    in a campaign's ``ladder.jsonl`` is a protocol pass, and every row in
    ``ladder_diagnostic.jsonl`` is diagnostic — the distinction lived in the file
    name before it lived in the row. A caller that needs a role for an old row
    must therefore say which file it came from, which is exactly the thought this
    field exists to make unnecessary going forward.
    """
    return row.get("role")


def completed_matchups(path: str | Path) -> set[tuple[str, str]]:
    """``(checkpoint, anchor)`` pairs already in the log — the coarse view.

    Resumption does *not* skip on this (see :func:`ladder_key`): a 20-game smoke
    row and a 200-game protocol row are the same pair and different measurements.
    """
    p = Path(path)
    if not p.exists():
        return set()
    return {(r["checkpoint"], r["anchor"]) for r in read_jsonl(p) if r.get("row_type") == MATCH_ROW}


def match_key(
    checkpoint: str, anchor: str, games: int, match_sims: int, max_plies: int, role: str
) -> tuple:
    """The identity a resume skips a matchup on: the pair *and* its protocol.

    ``(checkpoint, anchor)`` alone let a 20-game smoke row, a different-sims
    diagnostic, or a pass at a different ply cap block the protocol row for the
    same pair — and ``scripts/eval_ladder.py`` writes beside the checkpoint by
    default, i.e. into the campaign's own log (REVIEW-2026-10-21 D5). A row that
    predates ``role`` (schema < 2) is a protocol row: every such row lives in a
    campaign's ``ladder.jsonl`` (see :func:`ladder_role`).
    """
    return (MATCH_ROW, checkpoint, anchor, int(games), int(match_sims), int(max_plies), role)


def eval_key(checkpoint: str, suite_sims: int, role: str) -> tuple:
    """The identity a resume skips the opponent-free row on (REVIEW-2026-10-21 D4)."""
    return (EVAL_ROW, checkpoint, int(suite_sims), role)


def ladder_key(row: dict[str, Any]) -> tuple:
    """The resume key of an existing row, of any version."""
    role = row.get("role") or ROLE_PROTOCOL
    if row.get("row_type") == MATCH_ROW:
        return match_key(
            row["checkpoint"],
            row["anchor"],
            row["games"],
            row["match_sims"],
            row["max_plies"],
            role,
        )
    return eval_key(row["checkpoint"], row["suite_sims"], role)


def completed_keys(path: str | Path) -> set[tuple]:
    """Every :func:`ladder_key` already in the log — what a pass skips on."""
    p = Path(path)
    if not p.exists():
        return set()
    return {ladder_key(r) for r in read_jsonl(p) if r.get("row_type") in (MATCH_ROW, EVAL_ROW)}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL log. Blank lines are skipped; a torn final line is dropped.

    A killed process can leave a partial last line, and refusing to read the whole
    file because of it would make the dashboard the second casualty of the crash.
    (The writer also cuts that fragment before its next append, so the torn line
    can never end up in the middle of the file — see :func:`_append`.)
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
