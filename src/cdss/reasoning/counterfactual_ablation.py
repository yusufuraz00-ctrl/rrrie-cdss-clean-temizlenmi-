"""W7.2 K.3 — Counterfactual finding-ablation probe.

Given a posterior built from a sequence of evidence deltas, compute, for each
top-N influential finding, the posterior we would have observed if that finding
had never been seen. A finding whose ablation flips top-1 is a *pivot finding*
— the diagnosis hangs on it. If two or more pivot findings are negative
findings (absences), the case is "diagnosis hangs on absence-of-evidence",
which the controller surfaces for clinician review.

Math is exact for the log-additive Bayes update used by ``bayes_update``:

    log α_h^{post} = log α_h^{prior} + Σ_i log P(e_i | h)

Removing finding f means subtracting log P(e_f | h) for each h. We invert
each delta's aggregated likelihoods and renormalize.

Pure numeric — no LLM, no I/O.
"""

from __future__ import annotations

import math

from ..contracts.models import DiagnosticBelief, EvidenceDelta


_LOG_P_MIN = math.log(1e-6)


def _normalize_log(log_scores: dict[str, float]) -> dict[str, float]:
    if not log_scores:
        return {}
    m = max(log_scores.values())
    exps = {k: math.exp(max(_LOG_P_MIN, v - m)) for k, v in log_scores.items()}
    z = sum(exps.values())
    if z <= 0.0:
        n = len(log_scores)
        return {k: 1.0 / n for k in log_scores}
    return {k: v / z for k, v in exps.items()}


def _ablate_one(
    posterior: dict[str, float],
    delta: EvidenceDelta,
) -> dict[str, float]:
    """Return posterior with `delta` removed.

    Approach: posterior holds normalized P(h). Multiply by 1/P(e|h) per
    hypothesis (subtract log) and renormalize.
    """
    if not posterior:
        return {}
    log_p: dict[str, float] = {}
    for h, p in posterior.items():
        p_safe = max(1e-9, float(p))
        log_p[h] = math.log(p_safe)
    for h, lik in (delta.likelihoods or {}).items():
        lik_safe = max(1e-6, min(1.0 - 1e-6, float(lik)))
        if h in log_p:
            log_p[h] -= math.log(lik_safe)
    return _normalize_log(log_p)


def _top1(posterior: dict[str, float]) -> str:
    if not posterior:
        return ""
    return max(posterior.items(), key=lambda kv: kv[1])[0]


def _influence_score(delta: EvidenceDelta) -> float:
    """Information gain proxy: |H_before - H_after|.

    Larger drop = more informative finding.
    """
    h_b = float(delta.entropy_before or 0.0)
    h_a = float(delta.entropy_after or 0.0)
    return abs(h_b - h_a)


def find_pivot_findings(
    belief: DiagnosticBelief,
    *,
    level: int = 2,
    top_n: int = 5,
    negative_finding_marker: str = "neg:",
) -> dict:
    """Run ablation across the top-N most informative deltas at ``level``.

    A *pivot finding* is one whose removal would change the top-1 hypothesis.

    Returns:
        {
          "pivot_findings": [{"finding", "rank_change", "is_negative"}, ...],
          "top1_current": str,
          "absence_dependent": bool,   # True if ≥2 pivots are negative.
        }
    """
    if level == 0:
        post = dict(belief.family_posterior or {})
    elif level == 1:
        post = dict(belief.genus_posterior or {})
    else:
        post = dict(belief.species_posterior or {})

    if not post:
        return {"pivot_findings": [], "top1_current": "", "absence_dependent": False}

    deltas: list[EvidenceDelta] = [
        d for d in (belief.evidence_log or []) if int(d.level) == int(level)
    ]
    if not deltas:
        return {"pivot_findings": [], "top1_current": _top1(post), "absence_dependent": False}

    # Rank deltas by influence (entropy drop) and keep top_n.
    deltas_ranked = sorted(deltas, key=_influence_score, reverse=True)[: max(1, int(top_n))]

    cur_top1 = _top1(post)
    pivots: list[dict] = []
    for delta in deltas_ranked:
        ablated = _ablate_one(post, delta)
        if not ablated:
            continue
        new_top1 = _top1(ablated)
        if new_top1 != cur_top1 and new_top1:
            finding_label = str(delta.finding or "").strip() or delta.source
            is_neg = bool(finding_label) and (
                negative_finding_marker.lower() in finding_label.lower()
                or finding_label.lower().startswith("neg")
            )
            pivots.append(
                {
                    "finding": finding_label,
                    "from_top1": cur_top1,
                    "to_top1": new_top1,
                    "influence": round(_influence_score(delta), 4),
                    "is_negative": is_neg,
                }
            )

    absence_dependent = sum(1 for p in pivots if p.get("is_negative")) >= 2
    return {
        "pivot_findings": pivots,
        "top1_current": cur_top1,
        "absence_dependent": absence_dependent,
    }


__all__ = ["find_pivot_findings"]
