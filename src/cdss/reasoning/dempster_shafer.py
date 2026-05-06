"""Dempster-Shafer evidence fusion (W2 Module E).

Captures ignorance explicitly. Bayesian posterior assumes evidence sources are
independent AND exhaustive (every mass must land on a singleton). Real clinical
signals violate both: two swarm agents share a prompt (not independent); a
finding may be uninformative (mass on the whole frame of discernment Ω = "I
don't know").

Sparse representation.
  mass: dict[str, float]      # key = '|'-sorted hypothesis ids, or 'OMEGA'
  Ω (universal set / total ignorance) encoded as the literal string 'OMEGA'.
  Singleton {h} encoded as the id string 'h' (no separator).
  Set {h1, h2} encoded as 'h1|h2' (sorted alphabetically).

Matches the `ds_mass: dict[str, float]` field on `DiagnosticBelief`.

Math:
  Combination (Dempster's rule, orthogonal sum m1 ⊕ m2):
    m(A) = Σ_{B∩C=A, A≠∅} m1(B)·m2(C) / (1 − K)
    K    = Σ_{B∩C=∅}      m1(B)·m2(C)         (conflict)
  K ∈ [0,1]. K close to 1 → sources fundamentally disagree; combination is
  unstable and the result is routed to the premise-conflict resolver instead.

  Belief / Plausibility bracket:
    Bel(A) = Σ_{B⊆A} m(B)        (lower bound on P(A))
    Pl(A)  = Σ_{B∩A≠∅} m(B)      (upper bound; = 1 − Bel(Ā) when mass
                                   is normalized over singletons+Ω only)

All pure. No LLM, no I/O. `combine` is O(|m1|·|m2|).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

OMEGA = "OMEGA"  # universal set (total ignorance)
_SEP = "|"

# Conflict above this threshold → do NOT combine; caller should invoke premise-
# conflict resolver (MedLA-style) or fall back to single-source Bayesian view.
CONFLICT_REJECT_THRESHOLD = 0.60


# ---------------------------------------------------------------------------
# Key <-> frozenset encoding
# ---------------------------------------------------------------------------

def _decode(key: str) -> frozenset[str] | None:
    """String key → frozenset of hypothesis ids. OMEGA → None (sentinel)."""
    if key == OMEGA:
        return None
    if not key:
        return frozenset()  # empty set (contradiction); mass here is illegal
    return frozenset(p for p in key.split(_SEP) if p)


def _encode(fs: frozenset[str] | None, *, universe: frozenset[str] | None = None) -> str:
    """frozenset → canonical sorted key. None or full-universe → OMEGA."""
    if fs is None:
        return OMEGA
    if universe is not None and fs == universe:
        return OMEGA
    if not fs:
        # Empty set: should never carry mass post-normalization.
        return ""
    return _SEP.join(sorted(fs))


def _intersect(a: frozenset[str] | None, b: frozenset[str] | None) -> frozenset[str] | None:
    """Set intersection with Ω as identity element."""
    if a is None:
        return b
    if b is None:
        return a
    return a & b


# ---------------------------------------------------------------------------
# Mass constructors
# ---------------------------------------------------------------------------

def _normalize(mass: Mapping[str, float]) -> dict[str, float]:
    """Drop empty-set / negative / zero mass; renormalize to Σ = 1."""
    cleaned: dict[str, float] = {}
    for k, v in mass.items():
        if k == "" or v is None:
            continue
        f = float(v)
        if f <= 0.0:
            continue
        cleaned[k] = cleaned.get(k, 0.0) + f
    total = sum(cleaned.values())
    if total <= 0.0:
        return {OMEGA: 1.0}
    return {k: v / total for k, v in cleaned.items()}


def mass_from_singletons(
    probs: Mapping[str, float],
    *,
    ignorance: float = 0.0,
) -> dict[str, float]:
    """Construct m from a probability distribution over singletons.

    `ignorance` ∈ [0,1] reserves mass on Ω (how much the source "doesn't know").
    Common use: LLM prior with confidence c → mass_from_singletons(..., ignorance=1-c).
    """
    ign = max(0.0, min(1.0, float(ignorance)))
    scale = 1.0 - ign
    raw: dict[str, float] = {}
    total = sum(max(0.0, float(v)) for v in probs.values())
    if total <= 0.0:
        return {OMEGA: 1.0}
    for h, p in probs.items():
        f = max(0.0, float(p)) / total
        if f <= 0.0:
            continue
        raw[str(h)] = f * scale
    if ign > 0.0:
        raw[OMEGA] = raw.get(OMEGA, 0.0) + ign
    return _normalize(raw)


def mass_from_swarm_vote(
    candidate_scores: Mapping[str, float],
    *,
    agreement: float = 0.5,
) -> dict[str, float]:
    """Build m from one swarm agent's score row.

    `agreement` is the agent's self-reported (or empirical) reliability.
    Low agreement → more mass on Ω (epistemic caution).
    """
    agree = max(0.0, min(1.0, float(agreement)))
    return mass_from_singletons(candidate_scores, ignorance=1.0 - agree)


def mass_from_prototype(
    prototype_match: Mapping[str, float],
    *,
    match_strength: float = 0.5,
) -> dict[str, float]:
    """Build m from prototype similarity scores. `match_strength` ∈ [0,1]."""
    return mass_from_singletons(prototype_match, ignorance=1.0 - max(0.0, min(1.0, float(match_strength))))


def uniform_ignorance() -> dict[str, float]:
    """m(Ω) = 1.0 — prior before any evidence arrives."""
    return {OMEGA: 1.0}


# ---------------------------------------------------------------------------
# Combination rule
# ---------------------------------------------------------------------------

def conflict(m1: Mapping[str, float], m2: Mapping[str, float]) -> float:
    """Dempster conflict K = Σ_{B∩C=∅} m1(B)·m2(C). Returns ∈ [0,1]."""
    k = 0.0
    for kb, vb in m1.items():
        b = _decode(kb)
        for kc, vc in m2.items():
            c = _decode(kc)
            inter = _intersect(b, c)
            # Ω ∩ anything ≠ ∅ (None); only explicit empty frozenset is conflict.
            if inter is not None and len(inter) == 0:
                k += float(vb) * float(vc)
    return max(0.0, min(1.0, k))


def combine(
    m1: Mapping[str, float],
    m2: Mapping[str, float],
    *,
    reject_threshold: float = CONFLICT_REJECT_THRESHOLD,
) -> tuple[dict[str, float], float]:
    """Dempster's orthogonal sum m1 ⊕ m2.

    Returns (combined_mass, K). If K ≥ reject_threshold, caller SHOULD NOT
    trust the combined result (premise-conflict); still returned for audit.

    The combined mass is normalized by 1/(1 − K); on K≈1 we fall back to Ω.
    """
    accum: dict[str, float] = {}
    k = 0.0
    for kb, vb in m1.items():
        b = _decode(kb)
        vb_f = float(vb)
        for kc, vc in m2.items():
            c = _decode(kc)
            vc_f = float(vc)
            inter = _intersect(b, c)
            if inter is not None and len(inter) == 0:
                k += vb_f * vc_f
                continue
            key = _encode(inter)
            accum[key] = accum.get(key, 0.0) + vb_f * vc_f

    if k >= 1.0 - 1e-9:
        return {OMEGA: 1.0}, 1.0

    denom = 1.0 - k
    combined = {key: v / denom for key, v in accum.items() if v > 0.0}
    return _normalize(combined), float(k)


def combine_many(
    masses: Iterable[Mapping[str, float]],
    *,
    reject_threshold: float = CONFLICT_REJECT_THRESHOLD,
) -> tuple[dict[str, float], float]:
    """Left-fold combine over a sequence; returns (fused, max_pairwise_K)."""
    it = iter(masses)
    try:
        fused = dict(next(it))
    except StopIteration:
        return uniform_ignorance(), 0.0
    max_k = 0.0
    for m in it:
        fused, k = combine(fused, m, reject_threshold=reject_threshold)
        if k > max_k:
            max_k = k
    return fused, max_k


# ---------------------------------------------------------------------------
# Belief / Plausibility
# ---------------------------------------------------------------------------

def belief_plausibility(
    mass: Mapping[str, float],
    hypothesis: str,
) -> tuple[float, float]:
    """Return (Bel, Pl) bracket for singleton `hypothesis`.

      Bel({h}) = Σ_{B ⊆ {h}} m(B) = m({h})       (singletons only)
      Pl({h})  = Σ_{B ∩ {h} ≠ ∅} m(B)            (includes Ω, any superset)
    """
    bel = 0.0
    pl = 0.0
    target = frozenset({str(hypothesis)})
    for k, v in mass.items():
        s = _decode(k)
        vf = float(v)
        # Bel: B ⊆ target (only singleton {h} or ∅). B=None (Ω) not ⊆ finite.
        if s is not None and s.issubset(target):
            bel += vf
        # Pl: B ∩ target ≠ ∅. Ω intersects everything → count its mass.
        if s is None or (len(s & target) > 0):
            pl += vf
    return (max(0.0, min(1.0, bel)), max(0.0, min(1.0, pl)))


def belief_plausibility_set(
    mass: Mapping[str, float],
    hypotheses: Iterable[str],
) -> tuple[float, float]:
    """Bel/Pl bracket for a *set* A of hypotheses."""
    target = frozenset(str(h) for h in hypotheses)
    if not target:
        return (0.0, 0.0)
    bel = 0.0
    pl = 0.0
    for k, v in mass.items():
        s = _decode(k)
        vf = float(v)
        if s is not None and s.issubset(target):
            bel += vf
        if s is None or (len(s & target) > 0):
            pl += vf
    return (max(0.0, min(1.0, bel)), max(0.0, min(1.0, pl)))


def bracket_per_hypothesis(
    mass: Mapping[str, float],
    hypotheses: Iterable[str],
) -> dict[str, tuple[float, float]]:
    """Convenience: {h: (Bel, Pl)} for a list of candidate hypotheses."""
    out: dict[str, tuple[float, float]] = {}
    for h in hypotheses:
        out[str(h)] = belief_plausibility(mass, h)
    return out


def pignistic_transform(mass: Mapping[str, float]) -> dict[str, float]:
    """Smets' pignistic probability: split non-singleton mass uniformly.

    Produces a decision-ready probability distribution from a mass function.
    BetP({h}) = Σ_{B ∋ h} m(B) / |B|.  Ω is skipped (infinite support);
    callers wanting Ω-smoothing should add uniform noise after the transform.
    """
    collected: dict[str, float] = {}
    members: set[str] = set()
    for k in mass:
        s = _decode(k)
        if s is not None:
            members.update(s)
    for k, v in mass.items():
        s = _decode(k)
        vf = float(v)
        if s is None or len(s) == 0:
            continue
        share = vf / float(len(s))
        for h in s:
            collected[h] = collected.get(h, 0.0) + share
    total = sum(collected.values())
    if total <= 0.0:
        return {}
    return {h: v / total for h, v in collected.items()}


# ---------------------------------------------------------------------------
# Fusion with Bayesian posterior (weighted mix)
# ---------------------------------------------------------------------------

def fuse_with_bayes(
    bayes_posterior: Mapping[str, float],
    mass: Mapping[str, float],
    *,
    kappa: float = 0.6,
) -> dict[str, float]:
    """Weighted combination of Bayesian posterior and pignistic mass.

    `kappa` ∈ [0,1] is Bayesian weight. 1.0 = ignore D-S; 0.0 = ignore Bayes.
    Plan calls for κ=0.6 Bayes + 0.4 D-S-center-of-mass when independence
    violated (two correlated swarms).
    """
    k = max(0.0, min(1.0, float(kappa)))
    betp = pignistic_transform(mass)
    keys = set(bayes_posterior) | set(betp)
    raw = {h: k * float(bayes_posterior.get(h, 0.0)) + (1.0 - k) * float(betp.get(h, 0.0)) for h in keys}
    total = sum(raw.values())
    if total <= 0.0:
        return dict(bayes_posterior)
    return {h: v / total for h, v in raw.items()}


__all__ = [
    "OMEGA",
    "CONFLICT_REJECT_THRESHOLD",
    "mass_from_singletons",
    "mass_from_swarm_vote",
    "mass_from_prototype",
    "uniform_ignorance",
    "conflict",
    "combine",
    "combine_many",
    "belief_plausibility",
    "belief_plausibility_set",
    "bracket_per_hypothesis",
    "pignistic_transform",
    "fuse_with_bayes",
]
