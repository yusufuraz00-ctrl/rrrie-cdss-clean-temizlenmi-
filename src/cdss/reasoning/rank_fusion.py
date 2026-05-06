"""Rank-based ensemble fusion for swarm worker outputs (W6.1 Module J.1).

Replaces the per-candidate `max(score)` aggregation in `_fold_wave_into_belief`
with three layered, peer-reviewed fusion rules combined by a tunable convex
mixture. Pure numeric — no LLM, no I/O.

Layered fusion
--------------

1. **Reciprocal Rank Fusion (RRF, Cormack 2009)** — rank-stable, score-scale
   invariant. Robust when worker score scales differ (Qwen vs Groq vs Gemini).

       RRF(d) = Σ_i 1 / (k + rank_i(d))         k = 60 (literature default)

   Candidates absent from worker i's ranking contribute 0 from that worker.

2. **Trust-Weighted Borda (TeamMedAgents 2025, arXiv 2508.08115)** —
   incorporates per-(profile, family) reliability and family-specialist
   hierarchical weights.

       Borda_i(d) = (N - rank_i(d)) / max(1, N - 1)        ∈ [0, 1]
       TWB(d)     = Σ_i τ_i · w_hier(profile_i) · Borda_i(d)

   Final TWB normalized by Σ_i τ_i · w_hier so TWB(d) ∈ [0, 1].

3. **Likelihood mean** — temperature-ensemble Beta posterior mean from
   `likelihood_ensemble.py`. Direct probabilistic signal: P(evidence | h).

Final fused score:

       score(d) = α · RRF(d) + β · TWB(d) + γ · L̄(d)

with default (α, β, γ) = (0.4, 0.4, 0.2). All three components live on [0, 1]
after their respective normalizations so the mixture is interpretable.

This module exposes:
  - `reciprocal_rank_fusion(rankings, k=60)`
  - `trust_weighted_borda(rankings, trust, hier_weights)`
  - `fuse(rrf, twb, likelihood, alpha=0.4, beta=0.4, gamma=0.2)`
  - `aggregate_worker_outputs(...)` — one-call convenience entry point

Design contract: input rankings are already canonicalized via
`normalize_candidate_label`; this module is downstream of label validation.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from src.cdss.core import thresholds as clinical_thresholds

# Cormack 2009 — RRF k constant. Slightly damps top-rank dominance so that a
# strong consensus across many workers can outweigh a single top-1 outlier.
# Defaults are read from `config/clinical_thresholds.json` at call time so they
# remain tunable without code changes; the literals here are the safe fallbacks
# used if the config file is missing.
_RRF_K_DEFAULT = clinical_thresholds.get_int("fusion.rrf_k", 60)

# Convex-mixture defaults. Tunable via `fusion.alpha_rrf` etc. in the
# clinical-thresholds config.
_ALPHA_DEFAULT = clinical_thresholds.get_float("fusion.alpha_rrf", 0.4)   # weight of RRF
_BETA_DEFAULT = clinical_thresholds.get_float("fusion.beta_twb", 0.4)    # weight of trust-weighted Borda
_GAMMA_DEFAULT = clinical_thresholds.get_float("fusion.gamma_likelihood", 0.2)  # weight of likelihood mean


# -----------------------------------------------------------------
# 1. Reciprocal Rank Fusion
# -----------------------------------------------------------------

def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    *,
    k: int = _RRF_K_DEFAULT,
    normalize: bool = True,
) -> dict[str, float]:
    """Cormack-style RRF over a list of per-worker ranked-id lists.

    Each `rankings[i]` is the i-th worker's candidate ids in descending score
    order (top-1 first). Empty lists are tolerated. Output is keyed by id;
    value is the RRF score, optionally min-max normalized to [0, 1].
    """
    scores: dict[str, float] = {}
    if not rankings:
        return scores
    k_val = max(1, int(k))
    for ranks in rankings:
        if not ranks:
            continue
        for r, dx in enumerate(ranks, start=1):
            if not dx:
                continue
            scores[str(dx)] = scores.get(str(dx), 0.0) + 1.0 / (k_val + r)
    if not scores or not normalize:
        return scores
    lo = min(scores.values())
    hi = max(scores.values())
    if hi <= lo:
        return {d: 1.0 for d in scores}
    return {d: (v - lo) / (hi - lo) for d, v in scores.items()}


# -----------------------------------------------------------------
# 2. Trust-Weighted Borda
# -----------------------------------------------------------------

def trust_weighted_borda(
    rankings: Sequence[Sequence[str]],
    *,
    profiles: Sequence[str],
    trust: Mapping[str, float] | None = None,
    hier_weights: Mapping[str, float] | None = None,
    cold_trust: float = 0.8,
    cold_hier: float = 1.0,
) -> dict[str, float]:
    """TeamMedAgents Borda weighted by per-profile trust + hierarchy boost.

    `rankings[i]` is worker i's candidate-id list in descending score order
    and `profiles[i]` is that worker's profile name (e.g. "general",
    "cardiology"). `trust[profile]` ∈ [0, 1]; `hier_weights[profile]` boosts
    family-matched specialists.

    Defaults:
      - missing trust → `cold_trust=0.8` (TeamMedAgents cold-start)
      - missing hier  → `cold_hier=1.0` (no boost)

    Output is normalized to [0, 1] by total worker weight so it composes
    cleanly with RRF and likelihood-mean components in `fuse()`.
    """
    if not rankings or len(rankings) != len(profiles):
        return {}
    trust_map = dict(trust or {})
    hier_map = dict(hier_weights or {})
    total_w = 0.0
    raw: dict[str, float] = {}
    for ranks, prof in zip(rankings, profiles):
        if not ranks:
            continue
        n = len(ranks)
        denom = max(1, n - 1)
        tau = float(trust_map.get(prof, cold_trust))
        whier = float(hier_map.get(prof, cold_hier))
        worker_weight = max(0.0, tau) * max(0.0, whier)
        if worker_weight <= 0.0:
            continue
        total_w += worker_weight
        for r, dx in enumerate(ranks, start=1):
            if not dx:
                continue
            borda = (n - r) / denom  # rank-1 → 1.0; rank-N → 0.0
            raw[str(dx)] = raw.get(str(dx), 0.0) + worker_weight * borda
    if total_w <= 0.0 or not raw:
        return raw
    return {d: v / total_w for d, v in raw.items()}


# -----------------------------------------------------------------
# 3. Convex mixture
# -----------------------------------------------------------------

def fuse(
    rrf: Mapping[str, float],
    twb: Mapping[str, float],
    likelihood: Mapping[str, float] | None = None,
    *,
    alpha: float = _ALPHA_DEFAULT,
    beta: float = _BETA_DEFAULT,
    gamma: float = _GAMMA_DEFAULT,
) -> dict[str, float]:
    """Convex mixture α·RRF + β·TWB + γ·likelihood_mean.

    Components default to 0 for ids absent from a layer; weights renormalize
    so missing components don't shrink the surviving signal.
    """
    a, b, c = max(0.0, float(alpha)), max(0.0, float(beta)), max(0.0, float(gamma))
    has_lik = bool(likelihood)
    if not has_lik:
        c = 0.0
    s = a + b + c
    if s <= 0.0:
        # Degenerate weights — fall back to RRF only.
        return dict(rrf)
    a, b, c = a / s, b / s, c / s
    keys = set(rrf or {}) | set(twb or {})
    if has_lik:
        keys |= set(likelihood or {})
    out: dict[str, float] = {}
    for d in keys:
        v = a * float((rrf or {}).get(d, 0.0)) + b * float((twb or {}).get(d, 0.0))
        if has_lik:
            v += c * float((likelihood or {}).get(d, 0.0))
        out[d] = v
    return out


# -----------------------------------------------------------------
# Convenience: one call from `_fold_wave_into_belief`
# -----------------------------------------------------------------

def aggregate_worker_outputs(
    *,
    worker_results: Sequence[Sequence[tuple[str, float]]],
    profiles: Sequence[str],
    trust: Mapping[str, float] | None = None,
    hier_weights: Mapping[str, float] | None = None,
    likelihood: Mapping[str, float] | None = None,
    alpha: float = _ALPHA_DEFAULT,
    beta: float = _BETA_DEFAULT,
    gamma: float = _GAMMA_DEFAULT,
    rrf_k: int = _RRF_K_DEFAULT,
) -> dict[str, float]:
    """One-call fusion entry point.

    `worker_results[i]` is `[(id, score), ...]` for worker i; this function
    sorts them descending by score to obtain the rank list, then applies
    RRF + TWB + (optional) likelihood mixture.
    """
    if not worker_results:
        return {}
    rankings: list[list[str]] = []
    for items in worker_results:
        sorted_items = sorted(items or [], key=lambda kv: float(kv[1] or 0.0), reverse=True)
        rankings.append([str(d) for d, _s in sorted_items if d])
    rrf = reciprocal_rank_fusion(rankings, k=rrf_k)
    twb = trust_weighted_borda(
        rankings,
        profiles=list(profiles or []),
        trust=trust,
        hier_weights=hier_weights,
    )
    return fuse(rrf, twb, likelihood, alpha=alpha, beta=beta, gamma=gamma)


def severity_weighted(
    fused: Mapping[str, float],
    severity: Mapping[str, float] | None,
    *,
    delta: float = 0.25,
) -> dict[str, float]:
    """Multiplicative miss-cost prior on the fused posterior.

    Implements the expected-cost re-rank ``score' = score · (1 + δ · severity)``.
    Properties:

    * Monotonic in posterior — equal-severity candidates keep their order.
    * Cannot promote a label not present in ``fused`` (severity is a weight,
      not a creator).
    * Missing severity falls back to 0.3 (matches the default in
      ``bayes_update.miss_risk``); ``δ`` is a single calibration knob.

    See plan item 4 (cross-case engineering improvements 2026-04-26).
    """

    if not fused:
        return {}
    d = max(0.0, float(delta or 0.0))
    if d <= 0.0 or not severity:
        return dict(fused)
    out: dict[str, float] = {}
    for label, score in fused.items():
        sev_raw = severity.get(label)
        if sev_raw is None:
            sev_raw = severity.get(str(label).lower())
        sev = float(sev_raw) if sev_raw is not None else 0.3
        sev = max(0.0, min(1.0, sev))
        out[str(label)] = float(score) * (1.0 + d * sev)
    return out


__all__ = [
    "reciprocal_rank_fusion",
    "trust_weighted_borda",
    "fuse",
    "aggregate_worker_outputs",
    "severity_weighted",
]
