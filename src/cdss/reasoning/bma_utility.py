"""Bayesian-Model-Averaged expected utility (W4 Module G).

Replaces the old point-estimate OUTCOME step (`_run_test_time_simulation` in
`state_machine.py`) with a principled decision-theoretic average over the
top-k posterior. For each candidate action `a ∈ {admit, discharge, treat}`:

    E[U | a] = Σ_{h ∈ top-k(b_S)} P(h) · E[U(a, h)]

with `E[U(a, h)] = E[benefit | a, h] − γ · E[harm | a, h]`.

`γ` (harm weight) scales with `arbitration._severity_factor(h)` so rare-lethal
hypotheses dominate the decision even when posterior mass is small.

Multi-horizon rollout:

    E[event | h] = Σ_{t ∈ {7d, 30d, 90d}} δ^{t/30} · p_event(h, t)

`δ = 0.95`. Horizons come from curated `mortality_priors.json`. Missing
entries default to tier-based fallbacks (EMERGENCY=0.3, URGENT=0.15,
ROUTINE=0.05) so the module is well-defined even on uncurated species.

Pure math + one table lookup. No LLM here; the LLM counterfactual-discharge
probe runs in `state_machine.py` alongside and injects additional harm
estimates when available.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

_log = logging.getLogger("rrrie-cdss")

_DEFAULT_DISCOUNT = 0.95
_HORIZON_DAYS = {"e7d": 7, "e30d": 30, "e90d": 90}
_DEFAULT_TIER_RATES = {
    "EMERGENCY": {"e7d": 0.35, "e30d": 0.55, "e90d": 0.70},
    "URGENT":    {"e7d": 0.10, "e30d": 0.25, "e90d": 0.40},
    "ROUTINE":   {"e7d": 0.02, "e30d": 0.08, "e90d": 0.18},
    "UNKNOWN":   {"e7d": 0.08, "e30d": 0.20, "e90d": 0.35},
}


@dataclass
class PriorRow:
    """Per-hypothesis outcome priors (bare probabilities per horizon)."""

    hypothesis: str
    e7d: float = 0.0
    e30d: float = 0.0
    e90d: float = 0.0
    benefit_if_treated: float = 0.5   # P(adverse averted | correct treatment)
    harm_if_untreated: float = 0.5    # P(adverse given discharge/no-treat)
    harm_if_treated_wrong: float = 0.1  # iatrogenic risk of wrong treatment
    tier: str = "UNKNOWN"

    @classmethod
    def from_dict(cls, h: str, d: Mapping[str, Any]) -> "PriorRow":
        return cls(
            hypothesis=str(h),
            e7d=float(d.get("e7d", 0.0) or 0.0),
            e30d=float(d.get("e30d", 0.0) or 0.0),
            e90d=float(d.get("e90d", 0.0) or 0.0),
            benefit_if_treated=float(d.get("benefit_if_treated", 0.5) or 0.5),
            harm_if_untreated=float(d.get("harm_if_untreated", 0.5) or 0.5),
            harm_if_treated_wrong=float(d.get("harm_if_treated_wrong", 0.1) or 0.1),
            tier=str(d.get("tier", "UNKNOWN") or "UNKNOWN").upper(),
        )


@dataclass
class DispositionRecommendation:
    action: str
    e_utility: float
    e_benefit: float
    e_harm: float
    per_horizon: dict[str, float] = field(default_factory=dict)
    per_hypothesis: dict[str, float] = field(default_factory=dict)
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "e_utility": round(self.e_utility, 4),
            "e_benefit": round(self.e_benefit, 4),
            "e_harm": round(self.e_harm, 4),
            "per_horizon": {k: round(v, 4) for k, v in self.per_horizon.items()},
            "per_hypothesis": {k: round(v, 4) for k, v in self.per_hypothesis.items()},
            "rationale": self.rationale,
        }


# -----------------------------------------------------------------
# Asset loading
# -----------------------------------------------------------------

def load_mortality_priors(path: str | Path | None = None) -> dict[str, PriorRow]:
    """Load curated per-hypothesis priors from JSON asset."""
    if path is None:
        path = Path(__file__).resolve().parents[3] / "data" / "cdss" / "knowledge" / "mortality_priors.json"
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.warning("[BMA] failed to read mortality_priors.json: %s", exc)
        return {}
    rows = data.get("priors", {}) or {}
    out: dict[str, PriorRow] = {}
    for hyp, row in rows.items():
        if not isinstance(row, Mapping):
            continue
        out[str(hyp)] = PriorRow.from_dict(str(hyp), row)
    return out


def tier_fallback(tier: str | None) -> PriorRow:
    """Return a synthetic PriorRow when an explicit entry is missing."""
    t = (str(tier or "UNKNOWN")).upper()
    base = _DEFAULT_TIER_RATES.get(t, _DEFAULT_TIER_RATES["UNKNOWN"])
    return PriorRow(
        hypothesis=f"<fallback:{t}>",
        e7d=base["e7d"],
        e30d=base["e30d"],
        e90d=base["e90d"],
        benefit_if_treated=0.55,
        harm_if_untreated=base["e30d"],
        harm_if_treated_wrong=0.10 if t == "EMERGENCY" else 0.05,
        tier=t,
    )


# -----------------------------------------------------------------
# Multi-horizon expected event
# -----------------------------------------------------------------

def _horizon_sum(row: PriorRow, *, discount: float = _DEFAULT_DISCOUNT) -> tuple[float, dict[str, float]]:
    """Discounted sum Σ_t δ^{t/30} · p_event(t)."""
    per: dict[str, float] = {}
    total = 0.0
    for key, days in _HORIZON_DAYS.items():
        p = float(getattr(row, key, 0.0) or 0.0)
        w = math.pow(max(0.01, float(discount)), days / 30.0)
        contrib = w * p
        per[key] = contrib
        total += contrib
    return total, per


# -----------------------------------------------------------------
# Action-conditional utility
# -----------------------------------------------------------------

def _utility_for_action(row: PriorRow, action: str) -> tuple[float, float]:
    """Return (benefit, harm) for a given disposition against one hypothesis.

    Actions:
      - `admit_treat_correct`  — patient gets correct targeted care.
      - `discharge`            — patient leaves without treatment.
      - `admit_observe`        — admit but delay definitive treatment.
      - `treat_empiric`        — blind treatment without confirmed dx.
    """
    a = str(action or "").lower()
    if a in ("admit", "admit_treat", "admit_treat_correct"):
        return row.benefit_if_treated, row.harm_if_treated_wrong * 0.5
    if a == "discharge":
        return 0.0, row.harm_if_untreated
    if a in ("admit_observe", "observe"):
        return row.benefit_if_treated * 0.6, row.harm_if_untreated * 0.3
    if a in ("treat_empiric", "empiric"):
        return row.benefit_if_treated * 0.8, row.harm_if_treated_wrong
    # Unknown action — zero benefit, fallback harm.
    return 0.0, row.harm_if_untreated * 0.5


def bma_expected_utility(
    species_posterior: Mapping[str, float],
    action: str,
    priors: Mapping[str, PriorRow],
    *,
    severity_factor: Mapping[str, float] | None = None,
    discount: float = _DEFAULT_DISCOUNT,
    harm_weight_gamma: float = 2.0,
    top_k: int = 5,
    tier_of: Mapping[str, str] | None = None,
) -> DispositionRecommendation:
    """Compute E[U | action] via BMA over the posterior's top-k.

    `severity_factor[h]` (default 1.0) multiplies γ per hypothesis — rare-
    lethal dx uplift comes from `arbitration._severity_factor`.
    """
    ranked = sorted((species_posterior or {}).items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
    if not ranked:
        return DispositionRecommendation(action=action, e_utility=0.0, e_benefit=0.0, e_harm=0.0)

    total_p = sum(max(0.0, float(p)) for _h, p in ranked) or 1.0
    e_benefit = 0.0
    e_harm = 0.0
    per_horizon_total = {k: 0.0 for k in _HORIZON_DAYS}
    per_hyp_util: dict[str, float] = {}

    for h, p_raw in ranked:
        p = max(0.0, float(p_raw)) / total_p
        row = priors.get(h)
        if row is None:
            tier = (tier_of or {}).get(h, "UNKNOWN")
            row = tier_fallback(tier)
        benefit, harm = _utility_for_action(row, action)
        sev = float((severity_factor or {}).get(h, 1.0))
        gamma_eff = float(harm_weight_gamma) * sev
        # Multi-horizon discount shapes the harm (events accumulate over time).
        _, per_h = _horizon_sum(row, discount=discount)
        harm_discounted = sum(per_h.values()) / max(1e-6, sum(per_h.values()) or 1.0) * harm
        utility = benefit - gamma_eff * harm_discounted
        e_benefit += p * benefit
        e_harm += p * gamma_eff * harm_discounted
        per_hyp_util[h] = round(utility, 4)
        for k, v in per_h.items():
            per_horizon_total[k] += p * v

    e_util = e_benefit - e_harm
    rationale = (
        f"BMA over top-{len(ranked)} with γ_eff scaled by severity; "
        f"discount δ={discount:.2f} across {list(_HORIZON_DAYS.keys())}."
    )
    return DispositionRecommendation(
        action=action,
        e_utility=e_util,
        e_benefit=e_benefit,
        e_harm=e_harm,
        per_horizon=per_horizon_total,
        per_hypothesis=per_hyp_util,
        rationale=rationale,
    )


# -----------------------------------------------------------------
# Decision rule — Admit-if-ambiguous
# -----------------------------------------------------------------

def decide_disposition(
    species_posterior: Mapping[str, float],
    priors: Mapping[str, PriorRow],
    *,
    severity_factor: Mapping[str, float] | None = None,
    tier_of: Mapping[str, str] | None = None,
    discount: float = _DEFAULT_DISCOUNT,
    harm_weight_gamma: float = 2.0,
    top_k: int = 5,
    discharge_harm_threshold: float = 0.25,
    candidate_actions: list[str] | None = None,
) -> dict[str, Any]:
    """Run BMA for all candidate actions; return ranking + chosen disposition.

    Admit-if-ambiguous rule: if `E[harm | discharge] > discharge_harm_threshold`
    OR `E[utility | discharge] < E[utility | admit]`, choose `admit`. Otherwise
    pick argmax utility.
    """
    actions = list(candidate_actions or ["admit_treat_correct", "discharge", "admit_observe", "treat_empiric"])
    recs: list[DispositionRecommendation] = []
    for a in actions:
        recs.append(
            bma_expected_utility(
                species_posterior,
                a,
                priors,
                severity_factor=severity_factor,
                tier_of=tier_of,
                discount=discount,
                harm_weight_gamma=harm_weight_gamma,
                top_k=top_k,
            )
        )
    recs_by_action = {r.action: r for r in recs}
    discharge = recs_by_action.get("discharge")
    admit = recs_by_action.get("admit_treat_correct") or recs_by_action.get("admit_observe")

    # Admit-if-ambiguous rule.
    chosen: DispositionRecommendation = max(recs, key=lambda r: r.e_utility)
    reason = "argmax_utility"
    if discharge is not None and admit is not None:
        if discharge.e_harm > discharge_harm_threshold or discharge.e_utility < admit.e_utility:
            chosen = admit
            reason = (
                f"admit_if_ambiguous: E[harm|discharge]={discharge.e_harm:.2f}"
                f" (thresh={discharge_harm_threshold}), E[U|discharge]={discharge.e_utility:.2f}"
                f" < E[U|admit]={admit.e_utility:.2f}"
            )

    return {
        "chosen": chosen.as_dict(),
        "chosen_reason": reason,
        "ranked": [r.as_dict() for r in sorted(recs, key=lambda r: r.e_utility, reverse=True)],
    }


# -----------------------------------------------------------------
# Counterfactual discharge harm (LLM-free quick estimate)
# -----------------------------------------------------------------

def counterfactual_discharge_harm(
    species_posterior: Mapping[str, float],
    priors: Mapping[str, PriorRow],
    *,
    tier_of: Mapping[str, str] | None = None,
    top_k: int = 5,
    discount: float = _DEFAULT_DISCOUNT,
) -> dict[str, Any]:
    """'If we discharged now, what is the expected adverse-event load?'

    Returns dict with overall BMA harm + per-horizon + worst-offender hypothesis.
    Mirrors the LLM counterfactual_discharge probe when the LLM is unavailable.
    """
    ranked = sorted((species_posterior or {}).items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(top_k))]
    per_h: dict[str, dict[str, float]] = {}
    total_harm = 0.0
    total_p = sum(max(0.0, float(p)) for _h, p in ranked) or 1.0
    per_horizon: dict[str, float] = {k: 0.0 for k in _HORIZON_DAYS}
    for h, p_raw in ranked:
        p = max(0.0, float(p_raw)) / total_p
        row = priors.get(h) or tier_fallback((tier_of or {}).get(h, "UNKNOWN"))
        total, horizons = _horizon_sum(row, discount=discount)
        per_h[h] = {"p": round(p, 4), "harm_sum": round(total, 4), **{k: round(v, 4) for k, v in horizons.items()}}
        total_harm += p * total
        for k, v in horizons.items():
            per_horizon[k] += p * v
    worst = max(per_h.items(), key=lambda kv: kv[1]["harm_sum"]) if per_h else ("", {"harm_sum": 0.0})
    return {
        "expected_harm": round(total_harm, 4),
        "per_horizon": {k: round(v, 4) for k, v in per_horizon.items()},
        "per_hypothesis": per_h,
        "worst_offender": {"hypothesis": worst[0], "harm_sum": worst[1].get("harm_sum", 0.0)},
    }


__all__ = [
    "PriorRow",
    "DispositionRecommendation",
    "load_mortality_priors",
    "tier_fallback",
    "bma_expected_utility",
    "decide_disposition",
    "counterfactual_discharge_harm",
]
