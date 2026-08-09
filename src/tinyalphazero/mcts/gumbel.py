"""Gumbel root selection and Sequential Halving (spec §6.2, Danihelka et al. 2022).

Why Gumbel rather than PUCT + Dirichlet: at 48 sims on a ~30-branching game,
classic visit counts are barely past the prior, and Gumbel's guarantees are
designed for exactly that regime. It also deletes two hyperparameter families
(Dirichlet noise, temperature schedule) that v1 carried in two slightly
different copies. PUCT is retained behind ``cfg.mcts.root_mode`` as the control.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from tinyalphazero.config import MCTSConfig

# Priors arrive as probabilities; log(0) must not become -inf and poison the
# Gumbel argmax, so floor it well below any prior a softmax can produce.
LOG_PRIOR_FLOOR = -1e4


def gumbel_noise(rng: np.random.Generator, n: int) -> np.ndarray:
    """i.i.d. Gumbel(0,1) samples via inverse CDF."""
    u = rng.random(n)
    return -np.log(-np.log(np.clip(u, 1e-12, 1.0 - 1e-12))).astype(np.float32)


def prior_logits(prior: np.ndarray) -> np.ndarray:
    """log of a probability vector, floored so zero-prior actions stay finite."""
    with np.errstate(divide="ignore"):
        return np.where(prior > 0, np.log(np.maximum(prior, 1e-45)), LOG_PRIOR_FLOOR).astype(
            np.float32
        )


def sigma(q: np.ndarray, max_n: int, cfg: MCTSConfig) -> np.ndarray:
    """σ(q) = (c_visit + max_N) · c_scale · q  (spec §6.2)."""
    return ((cfg.c_visit + float(max_n)) * cfg.c_scale * q).astype(np.float32)


def halving_phase_budgets(num_sims: int, m: int) -> list[int]:
    """Total simulations allotted to each Sequential Halving phase.

    Phases = ``ceil(log2(m))``, the number of halvings needed to reach a single
    survivor. Within a phase the budget is split evenly across survivors; the
    **final phase absorbs whatever is left**, which is what makes the total come
    out exactly equal to ``num_sims`` for every ``(num_sims, m)`` rather than
    only for the ones that divide evenly.

    Worked example, n=48, m=16 — 4 phases, survivors 16 → 8 → 4 → 2 → 1:

    ======  =========  ==================  =========  =========
    phase   survivors  visits per action   phase use  remaining
    ======  =========  ==================  =========  =========
    1       16         1                   16         32
    2        8         1                    8         24
    3        4         3                   12         12
    4        2         6 (absorbs rest)    12          0
    ======  =========  ==================  =========  =========

    Per-phase visits are ``max(1, num_sims // (phases * survivors))`` so every
    considered action is sampled at least once before any elimination — with
    n=48, m=16 the naive ``48 // (4*16) = 0`` would eliminate half the set on no
    evidence at all. The caller must pass ``m <= num_sims``; otherwise that
    guarantee is unaffordable.
    """
    if m <= 0 or num_sims <= 0:
        return []
    if m == 1:
        return [num_sims]
    if m > num_sims:
        raise ValueError(f"m={m} exceeds num_sims={num_sims}; cap m before calling")

    phases = max(1, math.ceil(math.log2(m)))
    budgets: list[int] = []
    remaining = num_sims
    survivors = m

    for phase in range(phases):
        if phase == phases - 1:
            budgets.append(remaining)
            remaining = 0
            break
        per_action = max(1, num_sims // (phases * survivors))
        use = min(per_action * survivors, remaining)
        budgets.append(use)
        remaining -= use
        survivors = max(1, (survivors + 1) // 2)
        if remaining <= 0:
            budgets.extend([0] * (phases - phase - 1))
            remaining = 0
            break

    if remaining > 0:  # pragma: no cover - defensive
        budgets[-1] += remaining
    assert sum(budgets) == num_sims, (sum(budgets), num_sims)
    return budgets


def allocate_within_phase(budget: int, n_survivors: int) -> list[int]:
    """Split a phase budget evenly across survivors, remainder to the top-ranked.

    Survivors are assumed pre-sorted best-first, so the remainder favours the
    actions currently believed strongest. Sums to ``budget`` exactly.
    """
    if n_survivors <= 0 or budget <= 0:
        return [0] * max(n_survivors, 0)
    base, extra = divmod(budget, n_survivors)
    return [base + (1 if i < extra else 0) for i in range(n_survivors)]


def force_include_proven_wins(
    tree: Any, root: int, considered: list[int], rank_score: np.ndarray
) -> list[int]:
    """Ensure exactly-known wins are in the considered set.

    Measured, not assumed: with uniform stub priors, top-m by ``g + logits`` is
    a uniform random m-subset, so on the mate-in-1 suite (median 43 legal moves,
    m=16) the mating move is simply not considered most of the time. Built as
    specified first, the plan's "Gumbel at 48 sims still solves mate-in-1 100%"
    gate scored **46/100**; with this it scores 100/100. Discarding a proven win
    in favour of a sampled estimate makes the search strictly weaker than its
    own knowledge.

    Swaps rather than appends, so ``len(considered)`` is unchanged and the
    halving schedule's budget arithmetic still lands exactly on num_sims.
    """
    sl = tree.edge_slice(root)
    wins = [
        i
        for i in range(int(tree.num_edges[root]))
        if tree.terminal[sl.start + i] and -tree.terminal_value[sl.start + i] >= 1.0
    ]
    missing = [w for w in wins if w not in considered]
    if not missing:
        return considered

    out = list(considered)
    # Evict the weakest considered actions that are not themselves wins.
    evictable = sorted((a for a in out if a not in wins), key=lambda a: rank_score[a])
    for win, victim in zip(missing, evictable, strict=False):
        out[out.index(victim)] = win
    for win in missing[len(evictable) :]:
        if win not in out:
            out.append(win)
    return out


def form_considered_set(
    tree: Any,
    root: int,
    cfg: MCTSConfig,
    rng: np.random.Generator | None,
    num_sims: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """The root's ``(g, logits, considered)`` — for **both** search drivers.

    ``Search.run_gumbel`` and ``BatchRunner._finish_root`` each derived this
    independently, in code that had to be read side by side to confirm it
    agreed. Review S1: one definition, two consumers, neither re-deriving any
    part of it.

    ``rng`` may be None under the evaluation profile, which draws no noise —
    ``g ≡ 0`` then makes the considered set the deterministic top-m by raw
    logits, at no RNG cost.

    Takes ``num_sims`` because the considered-set size is
    ``min(gumbel_m, n_actions, num_sims)``: without the third term a 6-sim
    search would consider 16 actions it cannot afford to sample even once,
    which is the guarantee ``halving_phase_budgets`` is built on.
    """
    sl = tree.edge_slice(root)
    n_actions = int(tree.num_edges[root])
    logits = prior_logits(tree.prior[sl])
    if cfg.root_noise:
        if rng is None:
            raise ValueError("root_noise=True requires an rng")
        g = gumbel_noise(rng, n_actions)
    else:
        g = np.zeros(n_actions, dtype=np.float32)

    m = min(cfg.gumbel_m, n_actions, max(num_sims, 1))
    order = np.argsort(-(g + logits), kind="stable")
    considered = force_include_proven_wins(tree, root, list(order[:m]), g + logits)
    return g, logits, considered


def halve_survivors(survivors: list[int], scores: np.ndarray) -> list[int]:
    """Keep the better half by ``scores``, rounding up — the halving step itself."""
    return sorted(survivors, key=lambda a: -scores[a])[
        : max(1, (len(survivors) + 1) // 2)
    ]


def phase_visit_queue(budget: int, n_survivors: int) -> list[int]:
    """Flatten an allocation into the order the visits are actually spent.

    Survivor-local indices, best-ranked first, each repeated by its allocation:
    ``allocate_within_phase(5, 3) == [2, 2, 1]`` becomes ``[0, 0, 1, 1, 2]``.

    Exists because the batched driver cannot loop over the allocation the way
    the synchronous one does — it hands out one forced edge per call and resumes
    later. It previously improvised with ``survivors[sims_run % k]``, a
    round-robin, which agrees with the allocation only when the budget divides
    evenly across survivors. At ``m=5, sims=6`` the schedule is ``[5, 1, 0]``
    and phase 2's single visit belongs to the *best* survivor; the round-robin
    gave it to ``survivors[5 % 3]``, the worst of the three. Every divergence
    the parity probe found was a remainder landing on the wrong action.
    """
    return [
        local
        for local, visits in enumerate(allocate_within_phase(budget, n_survivors))
        for _ in range(visits)
    ]


def improved_policy(
    logits: np.ndarray, completed_q: np.ndarray, max_n: int, cfg: MCTSConfig
) -> np.ndarray:
    """softmax(logits + σ(completed_q)) over **all** legal root actions (§6.2).

    ``completed_q`` must already carry exact values on terminal actions and the
    root value estimate on unvisited ones — see ``Search._root_completed_q``.
    Blending an estimate where a truth is available is the failure mode this
    signature exists to make hard: it degrades the policy target silently,
    because the softmax still looks perfectly well-formed.
    """
    z = logits + sigma(completed_q, max_n, cfg)
    z = z - z.max()
    p = np.exp(z)
    total = p.sum()
    return (p / total).astype(np.float32) if total > 0 else np.full_like(p, 1.0 / len(p))


def dirichlet_mixed_prior(
    prior: np.ndarray, rng: np.random.Generator, cfg: MCTSConfig
) -> np.ndarray:
    """PUCT ablation control: (1-frac)·prior + frac·Dirichlet(α) (v1 semantics)."""
    if len(prior) == 0:
        return prior
    noise = rng.dirichlet([cfg.dirichlet_alpha] * len(prior)).astype(np.float32)
    return ((1.0 - cfg.dirichlet_frac) * prior + cfg.dirichlet_frac * noise).astype(np.float32)


def visit_count_policy(visits: np.ndarray, ply: int, cfg: MCTSConfig) -> np.ndarray:
    """PUCT ablation control: visit counts under the v1 temperature schedule.

    T = 1.0 before ``temperature_ply_cutoff``, 0.1 after.
    """
    if visits.sum() == 0:
        return np.full(len(visits), 1.0 / max(len(visits), 1), dtype=np.float32)
    temp = 1.0 if ply < cfg.temperature_ply_cutoff else 0.1
    scaled = np.power(visits.astype(np.float64), 1.0 / temp)
    return (scaled / scaled.sum()).astype(np.float32)
