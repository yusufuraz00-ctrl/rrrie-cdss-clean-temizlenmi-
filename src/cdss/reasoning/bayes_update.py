"""Bayesian posterior update for hierarchical diagnostic belief (W1 Module B.2).

Given a likelihood matrix P(evidence | hypothesis) for a batch of findings and
the current DiagnosticBelief, emit an updated belief with:

  log α_h^{t+1} = log α_h^t + Σ_i log P(e_i | h)

Posterior mean is then α / Σα (Dirichlet conjugate). We work in log-space for
numerical stability — a finding with likelihood 1e-20 does not underflow.

Updates apply to the `active_level` of the belief. Down-propagation (from
belief_propagation.propagate_down) should follow so parent-consistency holds.

Everything here is pure numeric — no LLM, no I/O.
"""

from __future__ import annotations

import math
from typing import Mapping

from ..contracts.models import DiagnosticBelief, EvidenceDelta
from .belief_propagation import (
    dirichlet_mean,
    entropy,
    propagate_down,
)
from .likelihood_ensemble import LikelihoodEstimate

_LOG_P_MIN = math.log(1e-6)  # -13.8, guard against −inf accumulation.


def _log_alpha_from_alpha(alpha: Mapping[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in alpha.items():
        v = max(1e-6, float(v))
        out[k] = math.log(v)
    return out


def _normalize_log(log_scores: Mapping[str, float]) -> dict[str, float]:
    """log-sum-exp normalization → probability distribution."""
    if not log_scores:
        return {}
    m = max(log_scores.values())
    exps = {k: math.exp(max(_LOG_P_MIN, v - m)) for k, v in log_scores.items()}
    z = sum(exps.values())
    if z <= 0.0:
        n = len(log_scores)
        return {k: 1.0 / n for k in log_scores}
    return {k: v / z for k, v in exps.items()}


def update_posterior(
    belief: DiagnosticBelief,
    *,
    level: int,
    likelihoods: Mapping[str, Mapping[str, LikelihoodEstimate | float]],
    parent_of_genus: dict[str, str] | None = None,
    parent_of_species: dict[str, str] | None = None,
    source: str = "swarm",
) -> DiagnosticBelief:
    """Multi-finding Bayes update on one hierarchy level.

    `likelihoods` is {finding: {hypothesis: LikelihoodEstimate or float_mean}}.
    Hypotheses absent from the likelihood row keep their prior α unchanged
    (equivalent to log-likelihood 0; ML estimate of "no information").
    """
    if level == 0:
        alpha_field = "family_alpha"
        post_field = "family_posterior"
    elif level == 1:
        alpha_field = "genus_alpha"
        post_field = "genus_posterior"
    else:
        alpha_field = "species_alpha"
        post_field = "species_posterior"

    prior_alpha = dict(getattr(belief, alpha_field) or {})
    if not prior_alpha:
        # No candidates at this level yet — seed from likelihood rows.
        prior_alpha = {h: 1.0 for row in likelihoods.values() for h in row.keys()}

    prior_post = dirichlet_mean(prior_alpha)
    entropy_before = entropy(prior_post)

    log_alpha = _log_alpha_from_alpha(prior_alpha)

    per_finding_likelihood: dict[str, dict[str, float]] = {}
    per_finding_variance: dict[str, dict[str, float]] = {}

    for finding, row in likelihoods.items():
        lik_row: dict[str, float] = {}
        var_row: dict[str, float] = {}
        for h, est in row.items():
            if isinstance(est, LikelihoodEstimate):
                p = float(est.mean)
                v = float(est.variance)
            else:
                p = float(est)
                v = 0.0
            p = max(1e-6, min(1.0 - 1e-6, p))
            lik_row[h] = p
            var_row[h] = v
            # Hypothesis not yet in log_alpha → initialize flat (α=1 → log 0).
            if h not in log_alpha:
                log_alpha[h] = 0.0
            log_alpha[h] += math.log(p)
        per_finding_likelihood[finding] = lik_row
        per_finding_variance[finding] = var_row

    # Convert log_alpha → posterior via log-sum-exp.
    posterior = _normalize_log(log_alpha)
    # α carried forward: exp(log_alpha − max) as relative concentration.
    max_la = max(log_alpha.values()) if log_alpha else 0.0
    # Keep α well-scaled: multiply by prior α-sum so concentration doesn't drift.
    total_prior = max(1.0, sum(prior_alpha.values()))
    new_alpha = {k: max(1e-3, math.exp(v - max_la) * total_prior) for k, v in log_alpha.items()}

    entropy_after = entropy(posterior)

    updates: dict[str, object] = {
        alpha_field: new_alpha,
        post_field: posterior,
        "active_level": int(level),
        "step": int(belief.step) + 1,
    }

    # Record one aggregated evidence delta (collapses per-finding updates into one
    # audit record to keep evidence_log bounded).
    aggregated_likelihoods: dict[str, float] = {}
    aggregated_variance: dict[str, float] = {}
    hyp_keys = set()
    for row in per_finding_likelihood.values():
        hyp_keys.update(row.keys())
    for h in hyp_keys:
        probs = [per_finding_likelihood[f].get(h, 1.0) for f in per_finding_likelihood]
        vars_ = [per_finding_variance[f].get(h, 0.0) for f in per_finding_likelihood]
        if probs:
            aggregated_likelihoods[h] = float(sum(probs) / len(probs))
        if vars_:
            aggregated_variance[h] = float(sum(vars_) / len(vars_))

    delta = EvidenceDelta(
        source=source,
        finding=",".join(list(per_finding_likelihood.keys())[:6]),
        step=int(belief.step) + 1,
        level=int(level),
        likelihoods=aggregated_likelihoods,
        likelihood_var=aggregated_variance,
        posterior_before=prior_post,
        posterior_after=posterior,
        entropy_before=entropy_before,
        entropy_after=entropy_after,
        meta={"findings_count": len(per_finding_likelihood)},
    )
    evidence_log = list(belief.evidence_log or []) + [delta]
    updates["evidence_log"] = evidence_log

    # Track species-level entropy progression for MCTS utility later.
    if level == 2:
        history = list(belief.entropy_history or []) + [round(entropy_after, 6)]
        updates["entropy_history"] = history

    updated = belief.model_copy(update=updates)

    # Parent-consistency down-propagation if maps given.
    if parent_of_genus is not None and parent_of_species is not None:
        updated = propagate_down(
            updated,
            parent_of_genus=parent_of_genus,
            parent_of_species=parent_of_species,
        )
    return updated


def top_k(
    posterior: Mapping[str, float],
    k: int = 5,
    *,
    min_prob: float = 0.0,
) -> list[tuple[str, float]]:
    items = [(str(h), float(p)) for h, p in posterior.items() if float(p) >= float(min_prob)]
    items.sort(key=lambda x: x[1], reverse=True)
    return items[: max(1, int(k))]


def miss_risk(
    belief: DiagnosticBelief,
    *,
    severity: Mapping[str, float],
    treatment_correct: Mapping[str, float] | None = None,
) -> float:
    """Expected regret if we stopped now.

    miss_risk = Σ_h P(h) · severity(h) · [1 − P(correct treatment | h)]

    severity ∈ [0, 1], treatment_correct ∈ [0, 1] (prob system picks right tx).
    Missing hypotheses treated as severity=0.3, treatment_correct=0.7 (defaults).
    """
    tc = dict(treatment_correct or {})
    total = 0.0
    for h, p in (belief.species_posterior or {}).items():
        sev = float(severity.get(h, 0.3))
        tc_h = float(tc.get(h, 0.7))
        total += float(p) * max(0.0, sev) * max(0.0, 1.0 - min(1.0, tc_h))
    return round(total, 6)


__all__ = [
    "update_posterior",
    "top_k",
    "miss_risk",
]
