"""General objective-confirmation planning for unresolved diagnostic risk."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.cdss.contracts.models import (
    DifferentialSet,
    EvidenceNeed,
    ExplanationGraph,
    RiskProfile,
    StructuredFindings,
    UrgencyTier,
)
from src.cdss.core import thresholds as clinical_thresholds


@dataclass(frozen=True)
class ObjectiveConfirmationPlan:
    required: bool = False
    evidence_needs: list[EvidenceNeed] = field(default_factory=list)
    blocking_findings: list[str] = field(default_factory=list)
    rationale: list[str] = field(default_factory=list)
    actions: list[str] = field(default_factory=list)


def _candidate_margin(differential: DifferentialSet) -> float:
    if len(differential.candidates or []) < 2:
        return 1.0
    return round(float(differential.candidates[0].score or 0.0) - float(differential.candidates[1].score or 0.0), 3)


def plan_objective_confirmation(
    findings: StructuredFindings,
    differential: DifferentialSet,
    explanation_graph: ExplanationGraph,
    risk_profile: RiskProfile,
) -> ObjectiveConfirmationPlan:
    blocking_findings = list(
        dict.fromkeys(
            [
                *list(explanation_graph.unexplained_findings or []),
                *[
                    link.finding
                    for link in explanation_graph.links
                    if link.critical and link.explanation_type == "still_unexplained"
                ],
            ]
        )
    )
    critical_open = bool(explanation_graph.critical_unexplained_count or blocking_findings)
    close_margin = clinical_thresholds.get_float("confirmation_planner.close_ranking_margin_lt", 0.20)
    close_ranking = _candidate_margin(differential) < close_margin
    high_risk = risk_profile.urgency != UrgencyTier.ROUTINE or bool(findings.red_flags)
    required = bool(high_risk or critical_open or close_ranking)
    if not required:
        return ObjectiveConfirmationPlan(required=False)

    top = differential.candidates[0].label if differential.candidates else "active_differential"
    rival = differential.candidates[1].label if len(differential.candidates or []) > 1 else ""
    target_finding = blocking_findings[0] if blocking_findings else (findings.red_flags[0] if findings.red_flags else "top candidate")
    high_priority = clinical_thresholds.get_float("confirmation_planner.high_priority_score", 0.92)
    default_priority = clinical_thresholds.get_float("confirmation_planner.default_priority_score", 0.72)
    high_info_gain = clinical_thresholds.get_float("confirmation_planner.high_information_gain", 0.82)
    base_info_gain = clinical_thresholds.get_float("confirmation_planner.baseline_information_gain", 0.65)
    needs = [
        EvidenceNeed(
            objective=f"objective confirmation for {target_finding}",
            decision_target="discriminate" if rival else "confirm",
            rationale="Narrative alone cannot safely close a high-risk or under-explained presentation.",
            hypothesis_label=top,
            priority=high_priority if high_risk else default_priority,
            active_state=top,
            unresolved_critical_finding=target_finding,
            rival_mechanism=rival,
            desired_discriminator="targeted exam, vital verification, labs, imaging, or specialist assessment as clinically indicated",
            decision_relevance=0.9,
            expected_information_gain=high_info_gain if close_ranking or critical_open else base_info_gain,
        )
    ]
    return ObjectiveConfirmationPlan(
        required=True,
        evidence_needs=needs,
        blocking_findings=blocking_findings[:6],
        rationale=[
            "high_risk_context" if high_risk else "",
            "critical_findings_unexplained" if critical_open else "",
            "top_candidates_close" if close_ranking else "",
        ],
        actions=[
            "obtain_objective_confirmation_for_unresolved_findings",
            "use_targeted_exam_labs_imaging_or_specialist_assessment",
        ],
    )
