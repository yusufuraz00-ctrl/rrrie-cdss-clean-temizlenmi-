"""Decision-theoretic utility U(b) over a DiagnosticBelief (W3 Module D.2).

    U(b) = -H(b_S)                           # information utility
           − λ · miss_risk(b)                # safety penalty (severity × P(wrong tx))
           + μ · calibration_tightness(b)    # decisiveness bonus
           − ν · budget_spent_frac           # urgency

Coefficients (λ, μ, ν) loaded from `CdssRuntimePolicy`. Default (0.8, 0.2, 0.05)
per plan.

Tightness proxy: 1 − (|top-k with mass ≥ 0.05| / |active_candidates|). Sharp
posterior → small k → tightness near 1.

Everything here is pure numeric. No LLM. Unit-testable.
"""

from __future__ import annotations

from typing import Mapping

from ..contracts.models import DiagnosticBelief
from .belief_propagation import entropy as _entropy
from .bayes_update import miss_risk as _miss_risk


def calibration_tightness(
    posterior: Mapping[str, float],
    *,
    mass_threshold: float = 0.05,
) -> float:
    """Fraction of probability concentrated in a minimal cover.

    Returns value ∈ [0, 1]. Sharp (max≈1) → 1.0. Uniform over N → ≈1/N.
    Formally: 1 − (support_count / total_candidates) where support_count is
    the count of hypotheses with mass ≥ `mass_threshold`.
    """
    if not posterior:
        return 0.0
    total = len(posterior)
    if total <= 0:
        return 0.0
    support = sum(1 for v in posterior.values() if float(v) >= float(mass_threshold))
    if support <= 0:
        # No hypothesis above threshold → posterior is noise; zero tightness.
        return 0.0
    return max(0.0, 1.0 - (support / total))


def utility(
    belief: DiagnosticBelief,
    *,
    severity: Mapping[str, float] | None = None,
    treatment_correct: Mapping[str, float] | None = None,
    miss_risk_weight: float = 0.8,
    tightness_weight: float = 0.2,
    budget_weight: float = 0.05,
    budget_spent_frac: float = 0.0,
) -> float:
    """Composite utility evaluated at species level.

    `severity`, `treatment_correct` feed `miss_risk`. Both default to empty →
    `miss_risk` uses its internal defaults (sev=0.3, tx=0.7) per hypothesis.

    `budget_spent_frac ∈ [0,1]`: wall-clock / tokens / dollars consumed so far
    divided by the total allowance. Higher → urgency dominates.
    """
    species_post = dict(belief.species_posterior or {})
    h_s = _entropy(species_post) if species_post else 0.0
    mr = _miss_risk(belief, severity=severity or {}, treatment_correct=treatment_correct or {}) if species_post else 0.0
    tight = calibration_tightness(species_post)
    budget = max(0.0, min(1.0, float(budget_spent_frac)))

    u = (
        -float(h_s)
        - float(miss_risk_weight) * float(mr)
        + float(tightness_weight) * float(tight)
        - float(budget_weight) * budget
    )
    return round(float(u), 6)


def utility_delta(
    before: DiagnosticBelief,
    after: DiagnosticBelief,
    **kwargs,
) -> float:
    """ΔU = U(after) − U(before). Positive = action improved the belief."""
    return round(float(utility(after, **kwargs) - utility(before, **kwargs)), 6)


def utility_components(
    belief: DiagnosticBelief,
    *,
    severity: Mapping[str, float] | None = None,
    treatment_correct: Mapping[str, float] | None = None,
    budget_spent_frac: float = 0.0,
) -> dict[str, float]:
    """Return the individual utility components for audit / UI display."""
    species_post = dict(belief.species_posterior or {})
    return {
        "entropy": round(float(_entropy(species_post)) if species_post else 0.0, 6),
        "miss_risk": round(
            float(_miss_risk(belief, severity=severity or {}, treatment_correct=treatment_correct or {}))
            if species_post else 0.0,
            6,
        ),
        "tightness": round(calibration_tightness(species_post), 6),
        "budget_spent": round(max(0.0, min(1.0, float(budget_spent_frac))), 6),
    }


__all__ = [
    "utility",
    "utility_delta",
    "utility_components",
    "calibration_tightness",
]
