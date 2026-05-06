"""Policy gates for adaptive pipeline depth and abstention.

Two pure decision functions live here:

- ``should_skip_deep_simulation(signals)`` — returns True when the case is
  stable enough that the heavy `BACKWARD_SIMULATION` and
  `OUTCOME_SIMULATION` stages can be skipped without losing reasoning
  quality. Triggers on the conjunction of: confident top-1 score, large
  margin to runner-up, high inline-grounding pass rate, no red flags, and
  adequate evidence coverage. All thresholds live in
  ``config/clinical_thresholds.json`` (section ``adaptive_depth``).

- ``abstain_or_escalate(signals)`` — returns an ``AbstentionDecision``
  recommending abstention when the fused margin is below
  ``abstain_margin_max`` *and* inline grounding risk is above
  ``abstain_grounding_risk_min``. This is the "I don't know" path: better
  to escalate to objective confirmation than to force a top-1.

Both functions consume a typed ``GateSignals`` dataclass so call sites do
not depend on the full ``WorkflowState``. The functions are pure and have
no side effects — wiring into the state machine is the caller's job.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.cdss.core import thresholds as clinical_thresholds


@dataclass(frozen=True)
class GateSignals:
    """Minimal signal bundle the policy gates need."""

    top1_score: float = 0.0
    runner_up_score: float = 0.0
    grounding_pass_rate: float = 1.0
    grounding_risk: float = 0.0
    evidence_coverage: float = 0.0
    has_red_flags: bool = False
    evidence_starvation: bool = False
    urgency: str = "ROUTINE"

    @property
    def margin(self) -> float:
        return max(0.0, float(self.top1_score) - float(self.runner_up_score))


@dataclass(frozen=True)
class AbstentionDecision:
    abstain: bool
    reason: str = ""
    margin: float = 0.0
    grounding_risk: float = 0.0


def should_skip_deep_simulation(signals: GateSignals) -> bool:
    """Return True when the deep-simulation stages can be safely skipped.

    Skipping is conservative by design — every condition must hold:

    - top-1 score is at least ``adaptive_depth.skip_top1_score_min``,
    - margin to runner-up is at least ``adaptive_depth.skip_margin_min``,
    - inline grounding pass rate is at least ``adaptive_depth.skip_grounding_pass_min``,
    - evidence coverage is at least ``adaptive_depth.skip_evidence_coverage_min``,
    - there are no red flags or starvation flags,
    - urgency is ROUTINE.
    """
    if signals.has_red_flags or signals.evidence_starvation:
        return False
    if (signals.urgency or "ROUTINE").upper() != "ROUTINE":
        return False
    top_min = clinical_thresholds.get_float("adaptive_depth.skip_top1_score_min", 0.70)
    margin_min = clinical_thresholds.get_float("adaptive_depth.skip_margin_min", 0.25)
    grounding_min = clinical_thresholds.get_float("adaptive_depth.skip_grounding_pass_min", 0.90)
    coverage_min = clinical_thresholds.get_float("adaptive_depth.skip_evidence_coverage_min", 0.60)
    return (
        signals.top1_score >= top_min
        and signals.margin >= margin_min
        and signals.grounding_pass_rate >= grounding_min
        and signals.evidence_coverage >= coverage_min
    )


def abstain_or_escalate(signals: GateSignals) -> AbstentionDecision:
    """Decide whether the system should abstain rather than force a top-1.

    Abstain when the margin is below ``abstain_margin_max`` and either
    grounding risk is high OR the evidence retrieval is starved. This
    keeps the "I don't know" path explicit, auditable, and surfaceable
    on the result UI.
    """
    margin_max = clinical_thresholds.get_float("adaptive_depth.abstain_margin_max", 0.10)
    risk_min = clinical_thresholds.get_float("adaptive_depth.abstain_grounding_risk_min", 0.40)

    margin = signals.margin
    risk = float(signals.grounding_risk or 0.0)
    starved = bool(signals.evidence_starvation)

    if margin < margin_max and (risk >= risk_min or starved):
        if starved and risk >= risk_min:
            reason = "low_margin_and_grounding_and_starvation"
        elif starved:
            reason = "low_margin_and_evidence_starvation"
        else:
            reason = "low_margin_and_high_grounding_risk"
        return AbstentionDecision(
            abstain=True,
            reason=reason,
            margin=round(margin, 4),
            grounding_risk=round(risk, 4),
        )

    return AbstentionDecision(abstain=False, reason="", margin=round(margin, 4), grounding_risk=round(risk, 4))


def signals_from_state(
    *,
    differential: Any,
    inline_grounding_pass_rate: float | None,
    evidence_coverage: float,
    risk_profile: Any,
    retrieval_stats: Any | None,
) -> GateSignals:
    """Construct ``GateSignals`` from typed pipeline objects.

    Keeps the call site at the state-machine level small and readable, and
    isolates downstream changes to a single helper.
    """
    cands = list(getattr(differential, "candidates", []) or [])
    top = float(getattr(cands[0], "score", 0.0) or 0.0) if cands else 0.0
    runner = float(getattr(cands[1], "score", 0.0) or 0.0) if len(cands) > 1 else 0.0
    pass_rate = float(inline_grounding_pass_rate) if inline_grounding_pass_rate is not None else 1.0
    grounding_risk = max(0.0, min(1.0, 1.0 - pass_rate))
    urgency = str(getattr(risk_profile, "urgency", "") or "")
    if hasattr(urgency, "value"):  # pydantic Enum compatibility
        urgency = urgency.value
    has_red_flags = bool(getattr(risk_profile, "vital_alerts", None) or getattr(risk_profile, "escalation_reasons", None))
    starvation = bool(getattr(retrieval_stats, "evidence_starvation_flag", False)) if retrieval_stats is not None else False

    return GateSignals(
        top1_score=top,
        runner_up_score=runner,
        grounding_pass_rate=pass_rate,
        grounding_risk=grounding_risk,
        evidence_coverage=float(evidence_coverage or 0.0),
        has_red_flags=has_red_flags,
        evidence_starvation=starvation,
        urgency=urgency.upper() if isinstance(urgency, str) else "ROUTINE",
    )


__all__ = [
    "AbstentionDecision",
    "GateSignals",
    "abstain_or_escalate",
    "should_skip_deep_simulation",
    "signals_from_state",
]
