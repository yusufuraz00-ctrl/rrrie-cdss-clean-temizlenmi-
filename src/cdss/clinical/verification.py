"""Verification policy for the vNext DDx + Safety surface."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.cdss.clinical.confirmation_planner import plan_objective_confirmation
from src.cdss.clinical.grounding import validate_narrative_grounding
from src.cdss.clinical.mechanism_fit import score_mechanism_completeness
from src.cdss.clinical.phenotype import has_dominant_phenotype_signature
from src.cdss.contracts.models import (
    CandidateScoreBreakdown,
    ContradictionCluster,
    DecisionStatus,
    DifferentialSet,
    EvidenceBundle,
    ExplanationGraph,
    GateEvaluation,
    InterventionSafetyAssessment,
    ModelSupportSignals,
    OODAssessment,
    RetrievalRankingStats,
    RiskProfile,
    StructuredFindings,
    UncertaintyAxes,
    UrgencyTier,
    VerificationIssue,
    VerificationReport,
)
from src.cdss.runtime.policy import load_runtime_policy


def _is_generic_label(label: str) -> bool:
    return str(label or "").endswith("_process")


@dataclass(frozen=True)
class _VerificationThresholdProfile:
    diagnosis_uncertainty_max: float
    epistemic_gate_min: float
    closure_coverage_min: float
    closure_contradiction_max: float
    uncertainty_revise_threshold: float
    evidence_gap_threshold: float
    reasoning_incomplete_threshold: float


_BUILTIN_VERIFICATION_PROFILES: dict[str, dict] = {
    "classic": dict(
        diagnosis_uncertainty_max=0.9,
        epistemic_gate_min=0.44,
        closure_coverage_min=0.24,
        closure_contradiction_max=0.24,
        uncertainty_revise_threshold=0.72,
        evidence_gap_threshold=0.24,
        reasoning_incomplete_threshold=0.28,
    ),
    "low_risk": dict(
        diagnosis_uncertainty_max=0.84,
        epistemic_gate_min=0.4,
        closure_coverage_min=0.34,
        closure_contradiction_max=0.42,
        uncertainty_revise_threshold=0.76,
        evidence_gap_threshold=0.25,
        reasoning_incomplete_threshold=0.55,
    ),
    "default": dict(
        diagnosis_uncertainty_max=0.8,
        epistemic_gate_min=0.44,
        closure_coverage_min=0.5,
        closure_contradiction_max=0.32,
        uncertainty_revise_threshold=0.72,
        evidence_gap_threshold=0.35,
        reasoning_incomplete_threshold=0.55,
    ),
}


def _load_verification_profiles() -> dict[str, _VerificationThresholdProfile]:
    """Load profiles from config/verification_profiles.json; fall back to built-ins."""
    config_path = Path(__file__).resolve().parents[4] / "config" / "verification_profiles.json"
    raw: dict[str, dict] = _BUILTIN_VERIFICATION_PROFILES
    if config_path.exists():
        try:
            data = json.loads(config_path.read_text(encoding="utf-8"))
            raw = {k: v for k, v in data.get("profiles", {}).items() if isinstance(v, dict)}
            if not raw:
                raw = _BUILTIN_VERIFICATION_PROFILES
        except Exception:
            raw = _BUILTIN_VERIFICATION_PROFILES
    return {
        name: _VerificationThresholdProfile(**{k: float(v) for k, v in profile.items()})
        for name, profile in raw.items()
    }


_VERIFICATION_THRESHOLD_TABLE: dict[str, _VerificationThresholdProfile] = _load_verification_profiles()


def _clamp(value: float, *, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _unique_items(items: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(cleaned)
    return output


def _resolve_threshold_profile(
    *,
    runtime_policy,
    classic_signature_case: bool,
    is_low_risk_case: bool,
    risk_profile: RiskProfile,
    top_score: float,
    evidence: EvidenceBundle,
    explanation_graph: ExplanationGraph,
    ood_assessment: OODAssessment,
) -> tuple[_VerificationThresholdProfile, str]:
    if classic_signature_case:
        profile = _VERIFICATION_THRESHOLD_TABLE["classic"]
        profile_name = "classic"
    elif is_low_risk_case:
        profile = _VERIFICATION_THRESHOLD_TABLE["low_risk"]
        profile_name = "low_risk"
    else:
        profile = _VERIFICATION_THRESHOLD_TABLE["default"]
        profile_name = "default"

    if not runtime_policy.verification_adaptive_gates_enabled:
        return profile, profile_name

    high_risk_context = (
        risk_profile.urgency != UrgencyTier.ROUTINE
        or ood_assessment.ood_score >= 0.55
        or explanation_graph.critical_unexplained_count > 0
    )
    low_risk_coherent_context = (
        is_low_risk_case
        and top_score >= 0.58
        and evidence.contradiction_mass <= 0.18
        and ood_assessment.ood_score < 0.42
        and explanation_graph.critical_unexplained_count == 0
    )

    if high_risk_context:
        return _VerificationThresholdProfile(
            diagnosis_uncertainty_max=_clamp(profile.diagnosis_uncertainty_max - 0.04, minimum=0.65, maximum=0.95),
            epistemic_gate_min=_clamp(profile.epistemic_gate_min + 0.03, minimum=0.3, maximum=0.7),
            closure_coverage_min=_clamp(profile.closure_coverage_min + 0.06, minimum=0.2, maximum=0.82),
            closure_contradiction_max=_clamp(profile.closure_contradiction_max - 0.05, minimum=0.14, maximum=0.7),
            uncertainty_revise_threshold=_clamp(profile.uncertainty_revise_threshold - 0.04, minimum=0.6, maximum=0.9),
            evidence_gap_threshold=_clamp(profile.evidence_gap_threshold + 0.04, minimum=0.15, maximum=0.65),
            reasoning_incomplete_threshold=_clamp(
                profile.reasoning_incomplete_threshold + (0.06 if risk_profile.urgency != UrgencyTier.ROUTINE else 0.0),
                minimum=0.2,
                maximum=0.75,
            ),
        ), f"{profile_name}:high_risk_adapted"

    if low_risk_coherent_context:
        return _VerificationThresholdProfile(
            diagnosis_uncertainty_max=_clamp(profile.diagnosis_uncertainty_max + 0.05, minimum=0.65, maximum=0.93),
            epistemic_gate_min=_clamp(profile.epistemic_gate_min - 0.04, minimum=0.32, maximum=0.7),
            closure_coverage_min=_clamp(profile.closure_coverage_min - 0.08, minimum=0.22, maximum=0.82),
            closure_contradiction_max=_clamp(profile.closure_contradiction_max + 0.08, minimum=0.14, maximum=0.58),
            uncertainty_revise_threshold=_clamp(profile.uncertainty_revise_threshold + 0.06, minimum=0.6, maximum=0.86),
            evidence_gap_threshold=_clamp(profile.evidence_gap_threshold - 0.05, minimum=0.18, maximum=0.65),
            reasoning_incomplete_threshold=_clamp(profile.reasoning_incomplete_threshold - 0.08, minimum=0.2, maximum=0.75),
        ), f"{profile_name}:low_risk_coherent_adapted"

    return profile, profile_name


def _classic_signature_case(
    findings: StructuredFindings,
    risk_profile: RiskProfile,
    differential: DifferentialSet,
    evidence: EvidenceBundle,
    explanation_graph: ExplanationGraph,
    model_support: ModelSupportSignals,
) -> bool:
    if not differential.candidates:
        return False
    top = differential.candidates[0]
    if _is_generic_label(top.label):
        return False
    margin = (top.score - differential.candidates[1].score) if len(differential.candidates) > 1 else top.score
    breakdown = model_support.score_breakdowns.get(top.label, CandidateScoreBreakdown())
    phenotype_support = has_dominant_phenotype_signature(findings)
    top_score_threshold = 0.78 if risk_profile.urgency == UrgencyTier.EMERGENCY else 0.72
    return bool(
        top.score >= top_score_threshold
        and margin >= 0.08
        and evidence.contradiction_mass <= 0.24
        and explanation_graph.critical_unexplained_count == 0
        and (phenotype_support or float(breakdown.story_fit or 0.0) >= 0.6 or float(breakdown.simulation_score or 0.0) >= 0.62)
    )


def _build_contradiction_clusters(
    *,
    findings: StructuredFindings,
    differential: DifferentialSet,
    evidence: EvidenceBundle,
    explanation_graph: ExplanationGraph,
) -> list[ContradictionCluster]:
    if evidence.contradiction_mass < 0.18:
        return []

    hypothesis_labels = [item.label for item in differential.candidates[:2] if str(item.label or "").strip()]
    finding_signals = [
        *[item for item in explanation_graph.contradictory_findings[:4] if str(item or "").strip()],
        *[item for item in findings.red_flags[:3] if str(item or "").strip()],
    ]
    if not finding_signals:
        finding_signals = [item for item in findings.positive_findings[:3] if str(item or "").strip()]

    severity = "minor"
    if evidence.contradiction_mass >= 0.42:
        severity = "critical"
    elif evidence.contradiction_mass >= 0.26:
        severity = "major"

    resolution_cost = 1
    if evidence.contradiction_mass >= 0.3:
        resolution_cost += 1
    if evidence.coverage < 0.5:
        resolution_cost += 1
    if len(finding_signals) >= 3:
        resolution_cost += 1
    resolution_cost = max(1, min(4, resolution_cost))

    cluster_id = "primary_conflict_cluster"
    if hypothesis_labels:
        cluster_id = "pair_conflict_" + "_vs_".join(label for label in hypothesis_labels[:2])

    return [
        ContradictionCluster(
            cluster_id=cluster_id,
            hypothesis_labels=hypothesis_labels,
            finding_signals=finding_signals[:6],
            severity=severity,
            resolution_cost=resolution_cost,
            remediable=bool(hypothesis_labels) and evidence.coverage < 0.96,
        )
    ]


def _decision_reason_category(decision: DecisionStatus, decision_path: list[str], issues: list[VerificationIssue]) -> str:
    path = " ".join(str(item or "").strip().lower() for item in decision_path)
    issue_types = {str(item.issue_type or "").strip().lower() for item in issues}
    if decision == DecisionStatus.URGENT_ESCALATION:
        return "safety_escalation"
    if decision == DecisionStatus.ABSTAIN:
        return "insufficient_signal"
    if decision == DecisionStatus.REVISE:
        if "specificity_gap" in issue_types:
            return "specificity_gap"
        if "pregnancy_context_gap" in issue_types:
            return "context_gap"
        if "evidence_gap" in issue_types or "high_conflict" in path:
            return "evidence_deficit"
        if "distribution_shift_warning" in issue_types:
            return "distribution_shift"
        if "safety_gate_failed" in path or "unsafe" in path:
            return "safety_risk"
        return "epistemic_uncertainty"
    if decision == DecisionStatus.PRELIMINARY:
        if "low_risk_supported_preliminary" in path:
            return "low_risk_coherent"
        if "classic_signature_with_all_core_gates" in path:
            return "high_specificity_alignment"
        return "provisional"
    return "provisional"


def build_verification_report(
    findings: StructuredFindings,
    risk_profile: RiskProfile,
    differential: DifferentialSet,
    evidence: EvidenceBundle,
    intervention_safety: InterventionSafetyAssessment | None = None,
    explanation_graph: ExplanationGraph | None = None,
    model_support: ModelSupportSignals | None = None,
    ood_assessment: OODAssessment | None = None,
    retrieval_stats: RetrievalRankingStats | None = None,
) -> VerificationReport:
    """Convert current state into verification gates and disposition."""
    issues: list[VerificationIssue] = []
    intervention_safety = intervention_safety or InterventionSafetyAssessment()
    explanation_graph = explanation_graph or ExplanationGraph()
    model_support = model_support or ModelSupportSignals()
    ood_assessment = ood_assessment or OODAssessment()
    classic_signature_case = _classic_signature_case(
        findings,
        risk_profile,
        differential,
        evidence,
        explanation_graph,
        model_support,
    )
    runtime_policy = load_runtime_policy()
    top_score = float(differential.candidates[0].score or 0.0) if differential.candidates else 0.0
    second_score = float(differential.candidates[1].score or 0.0) if len(differential.candidates) > 1 else 0.0

    # 5. Somut Ayırıcı Test İsteme (Discriminator Matrix) Bütçesi
    # ATOM modundayken (veyahut genel kural olarak) aradaki fark %20'den azsa katı engel koy.
    if differential.candidates and len(differential.candidates) > 1:
        if (top_score - second_score) < 0.20:
            if not evidence.evidence_needs and not intervention_safety.blocked_interventions:
                issues.append(
                    VerificationIssue(
                        severity="major",
                        issue_type="missing_discriminator",
                        detail=f"Top candidate ({differential.candidates[0].label}) is insufficiently separated from rival ({differential.candidates[1].label}). Discriminator budget exhausted; further tests required before definitive closure."
                    )
                )

    is_low_risk_case = (
        risk_profile.urgency == UrgencyTier.ROUTINE
        and not risk_profile.blocked_actions
        and not risk_profile.workflow_guards
        and len(findings.red_flags) == 0
    )
    epistemic_score = 0.72 if classic_signature_case else (0.68 if is_low_risk_case else 0.58)
    explicit_unsafe_plan = False
    top_label = differential.candidates[0].label if differential.candidates else ""
    grounding_report = validate_narrative_grounding(findings)
    if grounding_report.hallucination_risk_score > 0:
        issue_details = "; ".join(
            f"{issue.field}:{issue.item}" for issue in grounding_report.issues[:3]
        )
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="grounding_hallucination_risk",
                detail=(
                    "Some structured findings are not traceable to explicit narrative evidence; "
                    f"unknown values must not be treated as denied. {issue_details}"
                ).strip(),
            )
        )
        epistemic_score -= min(0.16, max(0.04, grounding_report.hallucination_risk_score * 0.2))

    # Low-evidence-grounding gate: surface retrieval failure rather than let
    # the system silently claim "evidence-grounded" with zero items.
    if retrieval_stats is not None and getattr(retrieval_stats, "evidence_starvation_flag", False):
        starved = list(retrieval_stats.starved_candidates or [])[:5]
        starved_detail = (
            "No supporting evidence retrieved for: " + ", ".join(starved)
            if starved
            else "Retrieval returned no items for the active differential."
        )
        issues.append(
            VerificationIssue(
                severity="medium",
                issue_type="low_evidence_grounding",
                detail=(
                    f"{starved_detail} The system cannot ground these candidates against external sources; "
                    "treat reasoning as narrative-only and require objective confirmation before closure."
                ),
            )
        )
        epistemic_score -= 0.08
    elif not (evidence.items or []) and (differential.candidates or []):
        issues.append(
            VerificationIssue(
                severity="medium",
                issue_type="low_evidence_grounding",
                detail=(
                    "Evidence bundle is empty for the active differential. Treat reasoning as narrative-only "
                    "and require objective confirmation before closure."
                ),
            )
        )
        epistemic_score -= 0.05

    confirmation_plan = plan_objective_confirmation(
        findings=findings,
        differential=differential,
        explanation_graph=explanation_graph,
        risk_profile=risk_profile,
    )
    confirmation_required = bool(confirmation_plan.required)
    if confirmation_required:
        issues.append(
            VerificationIssue(
                severity="major" if risk_profile.urgency == UrgencyTier.ROUTINE else "critical",
                issue_type="objective_confirmation_required",
                detail=(
                    "; ".join(item for item in confirmation_plan.rationale if item)
                    or "Objective confirmation is required before safe closure."
                ),
            )
        )
        existing_need_keys = {
            (
                str(need.objective or "").strip().lower(),
                str(need.hypothesis_label or "").strip().lower(),
                str(need.unresolved_critical_finding or "").strip().lower(),
            )
            for need in evidence.evidence_needs
        }
        for need in confirmation_plan.evidence_needs:
            key = (
                str(need.objective or "").strip().lower(),
                str(need.hypothesis_label or "").strip().lower(),
                str(need.unresolved_critical_finding or "").strip().lower(),
            )
            if key not in existing_need_keys:
                evidence.evidence_needs.append(need)
                existing_need_keys.add(key)
        epistemic_score -= 0.08 if risk_profile.urgency == UrgencyTier.ROUTINE else 0.12

    mechanism_score = (
        score_mechanism_completeness(differential.candidates[0], findings, risk_profile)
        if differential.candidates
        else None
    )
    if mechanism_score and mechanism_score.closure_blockers:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="mechanism_incomplete",
                detail=(
                    "Leading candidate lacks a complete mechanism across timing, risk context, "
                    f"severity, and confirmation axes: {', '.join(mechanism_score.closure_blockers[:4])}."
                ),
            )
        )
        epistemic_score -= 0.06
    critical_state_active = bool(explanation_graph.state_frames) and (
        risk_profile.urgency != UrgencyTier.ROUTINE
        or explanation_graph.critical_unexplained_count > 0
        or any(
            marker in state
            for state in explanation_graph.state_frames
            for marker in ("hazard", "device_reliability", "contradiction", "physiologic_instability")
        )
    )
    required_action_pool = {
        *risk_profile.required_actions,
        *intervention_safety.required_concurrent_actions,
    }
    unresolved_hazard = (risk_profile.urgency != UrgencyTier.ROUTINE or critical_state_active) and evidence.coverage < 0.42
    missing_concurrent_action = (risk_profile.urgency != UrgencyTier.ROUTINE or critical_state_active) and not required_action_pool and evidence.coverage < 0.5

    thresholds, adaptive_profile = _resolve_threshold_profile(
        runtime_policy=runtime_policy,
        classic_signature_case=classic_signature_case,
        is_low_risk_case=is_low_risk_case,
        risk_profile=risk_profile,
        top_score=top_score,
        evidence=evidence,
        explanation_graph=explanation_graph,
        ood_assessment=ood_assessment,
    )
    diagnosis_uncertainty_max = thresholds.diagnosis_uncertainty_max
    epistemic_gate_min = thresholds.epistemic_gate_min
    closure_coverage_min = thresholds.closure_coverage_min
    closure_contradiction_max = thresholds.closure_contradiction_max
    uncertainty_revise_threshold = thresholds.uncertainty_revise_threshold
    evidence_gap_threshold = thresholds.evidence_gap_threshold
    reasoning_incomplete_threshold = thresholds.reasoning_incomplete_threshold

    if not findings.positive_findings:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="insufficient_findings",
                detail="Narrative did not yield enough positive findings for a confident differential.",
            )
        )
        epistemic_score -= 0.25

    if risk_profile.urgency == UrgencyTier.EMERGENCY:
        issues.append(
            VerificationIssue(
                severity="critical",
                issue_type="high_risk_case",
                detail="High-risk presentation requires clinician review and objective workup.",
            )
        )
        epistemic_score -= 0.08

    demographics = dict(findings.demographics or {})
    sex_value = str(demographics.get("sex", "") or "").strip().lower()
    reproductive_context = sex_value in {"female", "f"} and any(
        marker in str(item or "").lower()
        for item in [*findings.positive_findings[:10], *findings.red_flags[:6], *findings.raw_segments[:8]]
        for marker in ("pelvic", "lower abdominal", "missed period", "menses")
    )
    pregnancy_context_documented = any(
        marker in str(item or "").lower()
        for item in [*findings.positive_findings[:10], *findings.negative_findings[:10], *findings.input_context[:10], *findings.raw_segments[:12]]
        for marker in ("pregnan", "beta hcg", "b-hcg", "hcg")
    )
    if reproductive_context and not pregnancy_context_documented:
        issues.append(
            VerificationIssue(
                severity="major" if risk_profile.urgency == UrgencyTier.ROUTINE else "critical",
                issue_type="pregnancy_context_gap",
                detail="Reproductive-age pelvic presentation is missing explicit pregnancy-status confirmation, weakening safe closure.",
            )
        )
        epistemic_score -= 0.06 if risk_profile.urgency == UrgencyTier.ROUTINE else 0.1

    if not differential.candidates:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="no_ranked_differential",
                detail="Differential engine returned no usable syndromic candidates.",
            )
        )
        epistemic_score -= 0.2
    elif top_label.endswith("_process") and not is_low_risk_case:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="specificity_gap",
                detail="Current anchor remains at family/process level and needs disease-level refinement before reliable closure.",
            )
        )
        epistemic_score -= 0.08

    evidence_gap_active = evidence.coverage < evidence_gap_threshold
    if classic_signature_case and top_score >= 0.84 and evidence.coverage >= 0.18:
        evidence_gap_active = False
    if evidence_gap_active:
        issues.append(
            VerificationIssue(
                severity="minor" if (is_low_risk_case or classic_signature_case) else "major",
                issue_type="evidence_gap",
                detail="Current structured data does not cover enough candidate-specific evidence needs.",
            )
        )
        epistemic_score -= 0.04 if classic_signature_case else (0.06 if is_low_risk_case else 0.12)
    elif evidence.coverage > 0.7:
        epistemic_score += 0.08

    reasoning_incomplete_active = risk_profile.urgency != UrgencyTier.ROUTINE and evidence.coverage < reasoning_incomplete_threshold
    if classic_signature_case and top_score >= 0.86 and evidence.contradiction_mass <= 0.24:
        reasoning_incomplete_active = False
    if reasoning_incomplete_active:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="reasoning_incomplete",
                detail="High-risk case remains under-explained; more targeted evidence gathering is needed before stable closure.",
            )
        )
        epistemic_score -= 0.04 if classic_signature_case else 0.08

    if evidence.contradiction_mass > (0.3 if classic_signature_case else (0.26 if is_low_risk_case else 0.2)):
        issues.append(
            VerificationIssue(
                severity="major" if evidence.contradiction_mass < 0.4 else "critical",
                issue_type="candidate_conflict",
                detail="Top candidates remain too close together for a stable autonomous interpretation.",
            )
        )
        epistemic_score -= min(0.06 if classic_signature_case else (0.1 if is_low_risk_case else 0.18), evidence.contradiction_mass)

    if len(findings.red_flags) >= 2:
        epistemic_score -= 0.08

    if explicit_unsafe_plan:
        issues.append(
            VerificationIssue(
                severity="critical",
                issue_type="contraindicated_intervention",
                detail="Planned intervention conflicts with the current safety gate and should be blocked.",
            )
        )
        epistemic_score -= 0.08

    if unresolved_hazard and not classic_signature_case:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="unresolved_hazard",
                detail="High-risk trajectory remains under-explained for safe closure.",
            )
        )
    if missing_concurrent_action:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="missing_concurrent_action",
                detail="High-risk context still lacks concurrent stabilizing actions or objective confirmation steps.",
            )
        )

    if differential.uncertainty > (0.82 if is_low_risk_case else 0.75):
        epistemic_score -= 0.05 if is_low_risk_case else 0.08
    if model_support.model_disagreement >= (0.28 if classic_signature_case else 0.18):
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="model_disagreement",
                detail="Support-side calibration disagrees materially with the current differential ranking.",
            )
        )
        epistemic_score -= 0.08
    if ood_assessment.ood_score >= 0.58:
        issues.append(
            VerificationIssue(
                severity="major",
                issue_type="distribution_shift_warning",
                detail="Case appears outside the stable operating distribution for autonomous closure.",
            )
        )
        epistemic_score -= 0.1
    elif is_low_risk_case and model_support.low_risk_closure_support >= 0.42 and ood_assessment.ood_score < 0.32:
        epistemic_score += 0.06
    if is_low_risk_case and model_support.low_risk_closure_support >= 0.56 and evidence.coverage >= 0.34:
        epistemic_score += 0.06
    if critical_state_active and required_action_pool:
        epistemic_score += 0.04

    diagnosis_gate = bool(differential.candidates) and differential.uncertainty <= diagnosis_uncertainty_max
    safety_score = max(
        0.0,
        min(
            1.0,
            round(
                0.94
                - (0.46 if explicit_unsafe_plan else 0.0)
                - (0.18 if unresolved_hazard else 0.0)
                - (0.12 if missing_concurrent_action else 0.0),
                2,
            ),
        ),
    )
    if critical_state_active and required_action_pool:
        safety_score = min(0.99, round(safety_score + 0.08, 2))
    safety_gate = safety_score >= 0.45 and not explicit_unsafe_plan
    epistemic_score = max(0.0, min(1.0, round(epistemic_score, 2)))
    epistemic_gate = epistemic_score >= epistemic_gate_min
    closure_gate = (
        model_support.low_risk_closure_support >= 0.5
        if is_low_risk_case
        else (evidence.coverage >= closure_coverage_min and evidence.contradiction_mass <= closure_contradiction_max)
    )
    if confirmation_required or grounding_report.hallucination_risk_score > 0:
        closure_gate = False

    diagnosis_score = _clamp01(1.0 - differential.uncertainty)
    diagnosis_threshold = _clamp01(1.0 - diagnosis_uncertainty_max)
    closure_score = (
        _clamp01(model_support.low_risk_closure_support)
        if is_low_risk_case
        else _clamp01(evidence.coverage - max(0.0, evidence.contradiction_mass - closure_contradiction_max))
    )
    closure_threshold = 0.5 if is_low_risk_case else _clamp01(closure_coverage_min)

    recommended_next_actions = _unique_items(
        [
            *(
                [
                    "obtain_objective_discriminator_data",
                    "expand_differential_with_targeted_retrieval",
                ]
                if not diagnosis_gate
                else []
            ),
            *(
                [
                    "stabilize_risk_pathway_before_closure",
                    "resolve_intervention_contraindications",
                ]
                if not safety_gate
                else []
            ),
            *(
                [
                    "increase_evidence_coverage_for_top_candidates",
                    "challenge_anchor_with_alternative_hypotheses",
                ]
                if not epistemic_gate
                else []
            ),
            *(
                [
                    "close_unresolved_contradictions",
                    "collect_missing_exclusion_signals",
                ]
                if not closure_gate
                else []
            ),
            *( ["obtain_targeted_objective_workup"] if evidence_gap_active else [] ),
            *confirmation_plan.actions,
            *( ["confirm_pregnancy_status_with_beta_hcg"] if reproductive_context and not pregnancy_context_documented else [] ),
            *( ["document_and_execute_required_concurrent_actions"] if missing_concurrent_action else [] ),
            *( ["escalate_for_high_risk_manual_review"] if unresolved_hazard else [] ),
        ]
    )

    gate_evaluations = [
        GateEvaluation(
            gate_id="diagnosis_gate",
            score=round(diagnosis_score, 2),
            threshold=round(diagnosis_threshold, 2),
            passed=diagnosis_gate,
            confidence_margin=round(diagnosis_score - diagnosis_threshold, 2),
            remediable=bool(differential.candidates) and evidence.coverage < 0.92,
            rationale=[
                f"uncertainty={round(differential.uncertainty, 2)}",
                f"threshold_max={round(diagnosis_uncertainty_max, 2)}",
            ],
            next_actions=([] if diagnosis_gate else ["obtain_objective_discriminator_data"]),
        ),
        GateEvaluation(
            gate_id="safety_gate",
            score=round(safety_score, 2),
            threshold=0.45,
            passed=safety_gate,
            confidence_margin=round(safety_score - 0.45, 2),
            remediable=not explicit_unsafe_plan or bool(required_action_pool),
            rationale=[
                f"unresolved_hazard={str(unresolved_hazard).lower()}",
                f"missing_concurrent_action={str(missing_concurrent_action).lower()}",
            ],
            next_actions=([] if safety_gate else ["stabilize_risk_pathway_before_closure"]),
        ),
        GateEvaluation(
            gate_id="epistemic_gate",
            score=round(epistemic_score, 2),
            threshold=round(epistemic_gate_min, 2),
            passed=epistemic_gate,
            confidence_margin=round(epistemic_score - epistemic_gate_min, 2),
            remediable=evidence_gap_active or reasoning_incomplete_active or model_support.model_disagreement >= 0.18,
            rationale=[
                f"evidence_gap_active={str(evidence_gap_active).lower()}",
                f"reasoning_incomplete_active={str(reasoning_incomplete_active).lower()}",
                f"model_disagreement={round(model_support.model_disagreement, 2)}",
            ],
            next_actions=([] if epistemic_gate else ["increase_evidence_coverage_for_top_candidates"]),
        ),
        GateEvaluation(
            gate_id="closure_gate",
            score=round(closure_score, 2),
            threshold=round(closure_threshold, 2),
            passed=closure_gate,
            confidence_margin=round(closure_score - closure_threshold, 2),
            remediable=evidence.coverage < 0.9 or evidence.contradiction_mass > 0.0,
            rationale=[
                f"coverage={round(evidence.coverage, 2)}",
                f"contradiction_mass={round(evidence.contradiction_mass, 2)}",
                f"closure_contradiction_max={round(closure_contradiction_max, 2)}",
            ],
            next_actions=([] if closure_gate else ["close_unresolved_contradictions"]),
        ),
    ]

    adaptive_thresholds = {
        "diagnosis_uncertainty_max": round(diagnosis_uncertainty_max, 2),
        "epistemic_gate_min": round(epistemic_gate_min, 2),
        "closure_coverage_min": round(closure_coverage_min, 2),
        "closure_contradiction_max": round(closure_contradiction_max, 2),
        "uncertainty_revise_threshold": round(uncertainty_revise_threshold, 2),
        "evidence_gap_threshold": round(evidence_gap_threshold, 2),
        "reasoning_incomplete_threshold": round(reasoning_incomplete_threshold, 2),
    }

    epistemic_uncertainty = _clamp01(1.0 - epistemic_score)
    aleatoric_uncertainty = _clamp01(
        (evidence.contradiction_mass * 0.58)
        + (min(1.0, len(findings.red_flags) / 4.0) * 0.22)
        + (min(1.0, differential.uncertainty) * 0.2)
    )
    distribution_shift_uncertainty = _clamp01(ood_assessment.ood_score)
    overall_uncertainty = _clamp01(
        (epistemic_uncertainty * 0.46)
        + (aleatoric_uncertainty * 0.32)
        + (distribution_shift_uncertainty * 0.22)
    )
    uncertainty_axes = UncertaintyAxes(
        epistemic=round(epistemic_uncertainty, 2),
        aleatoric=round(aleatoric_uncertainty, 2),
        distribution_shift=round(distribution_shift_uncertainty, 2),
        overall=round(overall_uncertainty, 2),
    )
    contradiction_clusters = _build_contradiction_clusters(
        findings=findings,
        differential=differential,
        evidence=evidence,
        explanation_graph=explanation_graph,
    )

    # Composite reliability score — weighted multi-signal metric (P1 fix)
    # Combines epistemic quality, evidence coverage, safety, contradiction burden, and overall uncertainty
    reliability = round(
        _clamp01(
            (epistemic_score * 0.42)
            + (min(1.0, evidence.coverage) * 0.24)
            + (safety_score * 0.14)
            + ((1.0 - min(1.0, evidence.contradiction_mass * 2.5)) * 0.12)
            + ((1.0 - overall_uncertainty) * 0.08)
        ),
        2,
    )
    empty_signal_case = (
        risk_profile.urgency == UrgencyTier.ROUTINE
        and not findings.positive_findings
        and not differential.candidates
        and len(findings.red_flags) == 0
    )

    decision_path: list[str] = []
    if empty_signal_case:
        decision_path.append("empty_signal_case")
        decision = DecisionStatus.ABSTAIN
        disposition = "abstain_and_collect_more_data"
    elif risk_profile.urgency == UrgencyTier.EMERGENCY:
        decision_path.append("emergency_urgency_hard_escalation")
        decision = DecisionStatus.URGENT_ESCALATION
        disposition = "urgent_escalation"
    elif classic_signature_case and diagnosis_gate and safety_gate and epistemic_gate:
        decision_path.append("classic_signature_with_all_core_gates")
        decision = DecisionStatus.PRELIMINARY
        disposition = "preliminary_ddx"
    elif not safety_gate and risk_profile.urgency != UrgencyTier.ROUTINE:
        decision_path.append("safety_gate_failed_non_routine")
        decision = DecisionStatus.REVISE
        disposition = "revise_after_more_evidence"
    elif risk_profile.manual_review_required or evidence.contradiction_mass >= 0.45 or ood_assessment.ood_score >= 0.58:
        decision_path.append("manual_review_or_high_conflict_or_ood")
        decision = DecisionStatus.REVISE
        disposition = "revise_after_more_evidence"
    elif not diagnosis_gate and not epistemic_gate:
        decision_path.append("diagnosis_and_epistemic_gates_failed")
        if risk_profile.urgency != UrgencyTier.ROUTINE or evidence.coverage < 0.72:
            decision_path.append("revise_due_to_risk_or_low_coverage")
            decision = DecisionStatus.REVISE
            disposition = "revise_after_more_evidence"
        else:
            decision_path.append("abstain_due_to_low_confidence_without_high_risk")
            decision = DecisionStatus.ABSTAIN
            disposition = "abstain_and_collect_more_data"
    elif not epistemic_gate or differential.uncertainty > uncertainty_revise_threshold:
        decision_path.append("epistemic_gate_failed_or_high_uncertainty")
        decision = DecisionStatus.REVISE
        disposition = "revise_after_more_evidence"
    elif is_low_risk_case and model_support.low_risk_closure_support >= 0.56 and ood_assessment.ood_score < 0.42:
        decision_path.append("low_risk_supported_preliminary")
        decision = DecisionStatus.PRELIMINARY
        disposition = "low_risk_preliminary_ddx"
    else:
        decision_path.append("default_preliminary_path")
        decision = DecisionStatus.PRELIMINARY
        disposition = "preliminary_ddx"

    decision_reason_category = _decision_reason_category(decision, decision_path, issues)

    return VerificationReport(
        decision=decision,
        reliability_score=reliability,
        epistemic_score=epistemic_score,
        safety_score=safety_score,
        evidence_coverage=evidence.coverage,
        contradiction_mass=evidence.contradiction_mass,
        issues=issues,
        recommended_disposition=disposition,
        diagnosis_gate=diagnosis_gate,
        safety_gate=safety_gate,
        epistemic_gate=epistemic_gate,
        closure_gate=closure_gate,
        adaptive_profile=adaptive_profile,
        decision_reason_category=decision_reason_category,
        adaptive_thresholds=adaptive_thresholds,
        gate_evaluations=gate_evaluations,
        decision_path=decision_path,
        recommended_next_actions=recommended_next_actions,
        uncertainty_axes=uncertainty_axes,
        contradiction_clusters=contradiction_clusters,
    )
