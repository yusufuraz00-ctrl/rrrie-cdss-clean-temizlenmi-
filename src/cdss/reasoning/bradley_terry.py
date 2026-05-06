"""W7.3 K.5 — Bradley-Terry pairwise MLE.

Given pairwise win counts ``w_ij`` (number of times candidate i was judged
better than candidate j), estimate latent strengths ``s_i`` such that

    P(i beats j) = s_i / (s_i + s_j)

Maximum-likelihood solution via Zermelo's fixed-point iteration:

    s_i^(t+1) = W_i / Σ_j (n_ij / (s_i^(t) + s_j^(t)))

where ``W_i = Σ_j w_ij`` (total wins of i) and ``n_ij = w_ij + w_ji``
(comparisons between the pair). Pure-Python; no scipy required. Strengths
are renormalized after each iteration so the largest is 1.0 (gauge fix).

Pairwise win counts whose total comparisons ``n_ij`` is zero contribute
nothing. Candidates with zero wins receive a small floor (1e-3) to avoid
collapse to zero.

References:
    Bradley & Terry (1952), Zermelo (1929), MedArena Stanford (2025).
"""

from __future__ import annotations

from typing import Mapping


def _participants(wins: Mapping[tuple[str, str], int | float]) -> list[str]:
    seen: list[str] = []
    seen_set: set[str] = set()
    for (i, j), _ in wins.items():
        for label in (i, j):
            label = str(label).strip()
            if label and label not in seen_set:
                seen.append(label)
                seen_set.add(label)
    return seen


def bt_mle(
    wins: Mapping[tuple[str, str], int | float],
    *,
    iters: int = 50,
    tol: float = 1e-6,
    floor: float = 1e-3,
) -> dict[str, float]:
    """Estimate Bradley-Terry strengths via Zermelo fixed-point iteration.

    Args:
        wins: dict keyed by ``(winner, loser)`` → win count. ``(i, j) -> w_ij``
              where i beat j. Symmetric pair ``(j, i) -> w_ji`` separate.
        iters: max iterations (Zermelo converges geometrically).
        tol: convergence threshold on max relative change.
        floor: lower bound on strengths (prevents zero-collapse).

    Returns:
        Normalized ``{label: strength}`` with ``max(strength) == 1.0``.
    """
    labels = _participants(wins)
    if not labels:
        return {}
    n = len(labels)
    if n == 1:
        return {labels[0]: 1.0}

    # Total wins per i and pair comparison counts.
    W: dict[str, float] = {lbl: 0.0 for lbl in labels}
    pair_n: dict[tuple[str, str], float] = {}
    for (i, j), w in wins.items():
        wi = float(w or 0.0)
        if wi <= 0.0:
            continue
        i_s, j_s = str(i), str(j)
        W[i_s] = W.get(i_s, 0.0) + wi
        key = (i_s, j_s) if i_s < j_s else (j_s, i_s)
        pair_n[key] = pair_n.get(key, 0.0) + wi
    # pair_n must include w_ji as well — accumulate both directions.
    # (above loop did include both because (i,j) and (j,i) pair-sort to same key)

    s: dict[str, float] = {lbl: 1.0 for lbl in labels}
    for _ in range(int(max(1, iters))):
        s_new: dict[str, float] = {}
        for i in labels:
            denom = 0.0
            for j in labels:
                if i == j:
                    continue
                key = (i, j) if i < j else (j, i)
                n_ij = pair_n.get(key, 0.0)
                if n_ij <= 0.0:
                    continue
                denom += n_ij / max(1e-12, s[i] + s[j])
            if denom <= 0.0:
                # i never compared OR all pair counts zero — keep prior strength.
                s_new[i] = max(floor, s[i])
            else:
                wi = max(0.0, W.get(i, 0.0))
                s_new[i] = max(floor, wi / denom) if wi > 0.0 else floor

        # Gauge fix: scale so max == 1.0 (matches reporting convention).
        m = max(s_new.values()) if s_new else 1.0
        if m > 0.0:
            s_new = {k: v / m for k, v in s_new.items()}

        # Convergence check (max relative change).
        delta = max(abs(s_new[k] - s[k]) for k in labels)
        s = s_new
        if delta < float(tol):
            break

    return s


def tournament_rank(
    strengths: Mapping[str, float],
) -> list[tuple[str, float]]:
    """Sort candidates by strength descending. Stable on ties (lex)."""
    items = [(str(k), float(v)) for k, v in strengths.items()]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items


def win_probability(s_i: float, s_j: float) -> float:
    """Bradley-Terry win probability of i over j given strengths."""
    s_i = max(1e-9, float(s_i))
    s_j = max(1e-9, float(s_j))
    return s_i / (s_i + s_j)


def aggregate_judge_votes(
    judge_outputs: list[Mapping[tuple[str, str], int | float]],
) -> dict[tuple[str, str], float]:
    """Sum win counts across N independent judges (different prompt framings).

    Each judge_output is ``{(winner, loser): vote_weight}``. Sum.
    """
    out: dict[tuple[str, str], float] = {}
    for judge in judge_outputs:
        for k, v in judge.items():
            out[k] = out.get(k, 0.0) + float(v or 0.0)
    return out


__all__ = ["bt_mle", "tournament_rank", "win_probability", "aggregate_judge_votes"]
