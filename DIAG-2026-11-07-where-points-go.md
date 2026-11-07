# DIAG 2026-11-07 — where the points go against skill 5

**Status: plan written before any game was graded (2026-11-07 14:40 UTC).** Results go in §5, below the line;
nothing above it changes after the report is read, except by a dated addendum.

## 1. Question

Do the standing model's losses against `stockfish:5` come from **one recurring cause**, a single kind of mistake in a
single phase of the game, or from many?

Why ask: Michael raised write-ups of LLM-written weight edits ("weight-space programming", 2026-11). In them, an
edit fixed a network that had a single, localized fault (a car policy that stalled at a ramp; Qwen3.5-0.8B's
reasoning loops) and failed where there was none (Gemma 270M's maths: 6.8% → 7.7%). An edit, or targeted training
data, is only a candidate here if our losses have that shape. The breakdown is also an input to p2_t's teacher
set (PREREG-p2_t), whichever way it falls.

This is a diagnostic, not a round. No lever is tested; no decision row rests on it beyond the reading rule in §3.

## 2. Instrument

- **Checkpoint:** `runs/p2_s_cont/ckpt_iter0030.pt`, the standing line's endpoint and p2_t's start.
- **Games:** two reps against `stockfish:5` at the instrument's settings (`eval_match_sims` 48, Stockfish 16 at
  10,000 nodes, skill 5, the frozen 100-opening book played twice colour-swapped, the config's seed, 300-ply cap):
  400 games, every move recorded, and the model's own root value at each of its moves
  (`scripts/where_points_go.py play`; `MatchResult.moves`). Played under `gpu hold` on this box at 14:16–14:35 UTC:
  rep 1 0.3125 (W34 D57 L109), rep 2 0.3625 (W47 D51 L102). The instrument's six reps of this checkpoint averaged
  0.330.
- **Grading:** every position of every game read by Stockfish 16 at **100,000 nodes**, one thread, hash cleared
  before each position, with the game's move history (`where_points_go.py annotate`, two worker processes; this box
  trips on all-core CPU load). A position's value is the engine's **expected score for the side to move**,
  `win + draw / 2` from its WDL output; a position the rules end is scored by the rule.
- **A learner move's drop** is its expected score before the move minus after it, from the learner's side. A
  **mistake** drops ≥ 0.10, a **blunder** ≥ 0.20.
- **Kinds** (each mistake gets the first that fits):
  1. `allowed_mate`: after the move the engine finds a forced mate against the learner, and before it did not;
  2. `missed_mate`: before the move the engine had a forced mate for the learner, and the move gives it up;
  3. `material`: along the engine's line from the position after the move, the learner ends 8 plies on at
     least 2 pawns worse than along its best line from the position before (P1 N3 B3 R5 Q9);
  4. `slip_while_winning`: the learner stood at ≥ 0.75 before the move;
  5. `other`: the rest. These are drops no 8-ply line turns into material or mate.
- **Phases:** `endgame` when both sides' knights, bishops, rooks and queens add up to ≤ 26 pawns; otherwise
  `opening` for the first 16 plies from the book position (itself 4 plies in), `middlegame` after.

## 3. Reading rule (registered here, before the report)

The **cell** is kind × phase. **One recurring cause = the largest cell holds ≥ 50% of the expected score the
learner lost on mistakes.**

- **If one cell holds ≥ 50%:** it is a candidate for a targeted fix, by an edit or by data aimed at that cell. That
  fix becomes a proposal for Michael, as its own registered A/B. Nothing is edited on the strength of this
  diagnostic alone.
- **If no cell does:** the losses are spread out, and the edit idea is closed for this model at this rung. The
  breakdown is reported, and feeds p2_t's teacher set as information only.

Also reported, with no rule attached:
- the share by kind and by phase;
- thrown wins: games that reached ≥ 0.85 and did not win, with their final result;
- blunders the search did not see: the model's own root value at the move is ≥ 0.25 above the engine's
  value after it;
- the opponent's gifts;
- the sum of drops against the score.

## 4. Limits, stated before

- The engine's expected score is what engine-level play would make of the position, not what this matchup makes of
  it. A position "worth 0.6" may score worse for a 48-simulation model against skill 5. Drops are read as the
  engine's grading, not as this matchup's probabilities.
- 100,000 nodes is about depth 16. That is far beyond both players, but the line used for `material` is that
  search's line, and its first 8 plies are what is counted.
- One checkpoint, 400 games. The kinds are heuristics with a fixed order, so a mistake that is both material and
  slip-while-winning counts as material.
- The mate-in-2 positions that fail at every measure (`mate2_124`–`129`, RUNLOG-p2_s §3) were checked before this
  plan, and they are not evidence of a pattern: the six are variants of one position (Black's king on a5, mate by
  Nb6 or Rb4), from one source. `mate2_091` and `mate2_196` are unrelated to them and to each other.

---

## 5. Results (graded 14:41–15:15 UTC, read 15:16)

`runs/diag_points/p2_s_cont_0030/where_points_go.json`: 400 games, 135 points (0.3375), 31,328 learner moves.

**Reading rule: no recurring cause.** The largest cell, `other/middlegame`, holds **29.7%** of the expected score the
learner lost on mistakes, under the 50% bar. **The weight-edit idea is closed for this model at this rung.**

| kind | share of the 829 lost on mistakes | mistakes |
|---|---|---|
| `other` | 49.4% | 1,542 |
| `slip_while_winning` | 30.9% | 746 |
| `material` | 18.5% | 374 |
| `allowed_mate` | 1.1% | 15 |
| `missed_mate` | 0.1% | 1 |

By phase: middlegame 61.6%, opening 20.3%, endgame 18.0%. The cells after the largest: `slip_while_winning/middlegame`
18.5%, `material/middlegame` 12.5%, `other/opening` 11.6%, `other/endgame` 8.1%, and every other cell under 7%.

Readings with no rule attached:

1. **The games swing hard both ways.** The learner made 2,678 mistakes (6.7 a game), 1,778 of them blunders (4.4 a
   game), and lost 829 in expected score on them (955 on all drops). Skill 5 gave back **875** on its own mistakes.
   The game starts at 0.50 on average and ends at 0.34: the score is the small difference of two large sums, not
   the work of a few decisive errors.
2. **Thrown wins: 198 of the 400 games reached ≥ 0.85 for the learner and did not win.** They lost 159 points:
   120 lost by checkmate, 19.5 drawn by claim, 11 by the ply cap and 8.5 drawn by rule. The engine's 0.85 is engine
   play's number, not this matchup's (§4), but the number of games and of points is large either way.
3. **The search did not see two blunders in three.** At 1,169 of the 1,778 blunders (66%), the model's own root
   value at the move was at least 0.25 above the engine's value after it. The losses are misjudged positions, more
   than lines the search saw and played anyway.
4. Allowed and missed mates are 1.2% of the loss together. The mate-in-2 failures (§4) have no counterpart in play.

**For p2_t:** the losses are broad: positional and tactical misjudgement in the middlegame, and failure to convert.
An edit cannot target that, by this diagnostic's own rule. A teacher that labels the model's own positions with a
stronger evaluation and stronger move choices is aimed at that breadth. Reading 3, the value misjudgement, is the
part its value target addresses. This is information for the PREREG, not a registered expectation.
