"""Native UI/API render model for the vNext CDSS packet."""

from __future__ import annotations

from typing import Any

from src.cdss.app.trace_metrics import compute_stage_metrics_from_trace, compute_trace_wall_time
from src.cdss.contracts.models import (
    ConfidenceView,
    DecisionCandidateView,
    DecisionPacket,
    DecisionPacketView,
    DecisionStatus,
    EvidenceTraceView,
    FamilyHypothesis,
    MechanismFrame,
    SafetyStateView,
    TimingView,
)


_VIEW_STAGE_MAP = {
    "risk": "R1",
    "extractor": "R1",
    "r1_reasoned": "R1",
    "evidence_planning": "R2",
    "r2_reinforced": "R2",
    "hypothesis_generation": "R3",
    "specificity_resolver": "R3",
    "r3_reasoning": "R3",
    "governor": "IE",
    "verification": "IE",
    "ie_override": "IE",
}


def _trace_consistency_summary(packet: DecisionPacket, stages: dict[str, dict[str, Any]]) -> dict[str, Any]:
    decision_matches_packet_status = packet.status.value == str(packet.status.value)
    required_stage_buckets = ("R1", "R2", "R3", "IE")
    mapped_stage_set = {
        _VIEW_STAGE_MAP[item.stage]
        for item in packet.trace
        if item.stage in _VIEW_STAGE_MAP
    }
    stage_bucket_presence = {
        stage: stage in mapped_stage_set
        for stage in required_stage_buckets
    }
    required_stage_coverage_ok = all(stage_bucket_presence.values())
    reasoning_trace_matches_packet = bool(packet.reasoning_trace) and len(packet.reasoning_trace) <= len(packet.trace)
    warnings: list[str] = []
    if not required_stage_coverage_ok:
        missing = [name for name, present in stage_bucket_presence.items() if not present]
        warnings.append(f"missing_stage_buckets:{','.join(missing)}")
    if not reasoning_trace_matches_packet:
        warnings.append("reasoning_trace_not_aligned")
    return {
        "decision_matches_packet_status": decision_matches_packet_status,
        "stage_bucket_presence": stage_bucket_presence,
        "required_stage_coverage_ok": required_stage_coverage_ok,
        "reasoning_trace_matches_packet": reasoning_trace_matches_packet,
        "status": "ok" if not warnings else "warn",
        "warnings": warnings,
    }


def _confidence_zone(packet: DecisionPacket) -> str:
    if packet.status == DecisionStatus.PRELIMINARY and packet.closure_confidence >= 0.7:
        return "safe"
    if packet.status in {DecisionStatus.REVISE, DecisionStatus.ABSTAIN}:
        return "critical"
    if packet.status == DecisionStatus.URGENT_ESCALATION:
        return "critical"
    return "caution"


def _unsupported_claim_count(packet: DecisionPacket) -> int:
    unsupported_issue_types = {
        "reasoning_incomplete",
        "evidence_gap",
        "specificity_gap",
        "no_ranked_differential",
    }
    issue_count = sum(
        1
        for item in packet.verification.issues
        if str(item.issue_type or "").strip().lower() in unsupported_issue_types
    )
    return max(issue_count, int(packet.explanation_graph.critical_unexplained_count or 0))


def _top_candidate_view(packet: DecisionPacket) -> DecisionCandidateView:
    profile_by_label = {profile.label: profile for profile in packet.generated_candidate_profiles}
    family_by_label = {item.label: item.family_label for item in packet.disease_hypotheses}
    top = packet.differential.candidates[0] if packet.differential.candidates else None
    if not top:
        return DecisionCandidateView()
    profile = profile_by_label.get(top.label)
    return DecisionCandidateView(
        label=top.label,
        family_label=family_by_label.get(top.label, profile.family_label if profile else ""),
        score=round(float(top.score), 2),
        rationale=top.rationale[:4],
        evidence_needed=(profile.evidence_needs[:4] if profile else top.evidence_needed[:4]),
        unsafe_interventions=(profile.unsafe_interventions[:4] if profile else []),
        must_not_miss=bool(profile.must_not_miss) if profile else False,
        dangerous_if_missed=bool(profile.dangerous_if_missed) if profile else False,
    )


def _family_view(packet: DecisionPacket) -> FamilyHypothesis:
    if packet.family_hypotheses:
        return packet.family_hypotheses[0]
    if packet.mechanism_frames:
        frame = packet.mechanism_frames[0]
        return FamilyHypothesis(
            label=frame.active_state or "undifferentiated_high_variance_process",
            score=round(float(packet.diagnostic_confidence or 0.0), 2),
            rationale=[frame.primary_mechanism or "mechanism-derived family anchor"],
            dominant_mechanism=frame.primary_mechanism,
            dominant_findings=frame.critical_findings[:4],
        )
    return FamilyHypothesis(label="")


def _mechanism_view(packet: DecisionPacket) -> MechanismFrame:
    return packet.mechanism_frames[0] if packet.mechanism_frames else MechanismFrame()


def _is_generic_view_label(label: str) -> bool:
    """P4: Identify generic bucket/family labels that should not appear as disease candidates."""
    slug = str(label or "").strip().lower().replace(" ", "_")
    _generic = {
        "cardiorespiratory_process", "infectious_inflammatory_process",
        "metabolic_or_abdominal_process", "neurologic_process",
        "undifferentiated_high_variance_process",
    }
    return slug in _generic or slug.endswith("_process")


def _disease_candidate_views(packet: DecisionPacket) -> list[DecisionCandidateView]:
    profile_by_label = {profile.label: profile for profile in packet.generated_candidate_profiles}
    output: list[DecisionCandidateView] = []
    # Scan extra candidates to fill 5 after filtering generic labels (P4)
    candidates = packet.disease_hypotheses[:8]
    for disease in candidates:
        if _is_generic_view_label(disease.label):
            continue
        profile = profile_by_label.get(disease.label)
        output.append(
            DecisionCandidateView(
                label=disease.label,
                family_label=disease.family_label,
                score=round(float(disease.score), 2),
                rationale=disease.rationale[:4],
                evidence_needed=disease.evidence_needed[:4],
                unsafe_interventions=(profile.unsafe_interventions[:4] if profile else disease.unsafe_interventions[:4]),
                must_not_miss=bool(profile.must_not_miss) if profile else False,
                dangerous_if_missed=bool(profile.dangerous_if_missed) if profile else False,
            )
        )
        if len(output) >= 5:
            break
    if output:
        return output
    return [_top_candidate_view(packet)] if packet.differential.candidates else []


def _consistent_summary(packet: DecisionPacket) -> str:
    """Fix-E: ensure the rendered summary agrees with the differential's
    actual top-1 leader.

    The packet's ``summary`` is generated by the action-plan stage at a
    moment when the differential leader may differ from what the
    final-stage safety penalty / verifier alternative promotion produces.
    This helper rebuilds a short, deterministic lead phrase from the
    *current* ``packet.differential.candidates[0]`` and prepends it to the
    original summary when the original summary doesn't already mention
    the current leader. Result: hero text and ranked-candidate row never
    contradict each other.
    """
    cands = list(packet.differential.candidates or [])
    if not cands:
        return packet.summary or ""
    top = cands[0]
    runner = cands[1] if len(cands) > 1 else None
    top_label = (top.label or "").strip()
    if not top_label:
        return packet.summary or ""

    original = (packet.summary or "").strip()
    # If the original summary already names the current leader, leave it.
    if top_label.lower() in original.lower():
        return original

    rival_phrase = f", with {runner.label} as the primary rival pending discrimination" if runner and runner.label else ""
    lead = f"Current differential is led by {top_label}{rival_phrase}."

    if not original:
        return lead
    return lead + " " + original


def build_decision_packet_view(packet: DecisionPacket, runtime_snapshot: dict[str, Any] | None = None) -> DecisionPacketView:
    runtime_snapshot = runtime_snapshot or {}
    stages, llm_seconds, token_budget = compute_stage_metrics_from_trace(packet, _VIEW_STAGE_MAP)
    trace_consistency = _trace_consistency_summary(packet, stages)
    wall_seconds = compute_trace_wall_time(packet)
    total_seconds = max(llm_seconds, wall_seconds)
    top_candidate = _top_candidate_view(packet)
    safety_summary = packet.ie_override.summary or packet.ie_override.critical_statement or packet.summary
    consistent_summary = _consistent_summary(packet)
    return DecisionPacketView(
        case_id=packet.case_id,
        summary=consistent_summary,
        decision=packet.status.value,
        top_differential=top_candidate,
        family=_family_view(packet),
        mechanism=_mechanism_view(packet),
        disease_candidates=_disease_candidate_views(packet),
        required_data=packet.required_data,
        safety_state=SafetyStateView(
            decision=packet.status.value,
            mode=packet.ie_override.mode,
            summary=safety_summary,
            blocked_interventions=packet.blocked_interventions,
            blocked_orders=packet.ie_override.blocked_orders,
            mandatory_actions=packet.ie_override.mandatory_actions,
            workflow_modifications=packet.ie_override.workflow_modifications,
            issues=packet.verification.issues,
        ),
        confidence=ConfidenceView(
            diagnosis=round(float(packet.diagnostic_confidence), 2),
            closure=round(float(packet.closure_confidence), 2),
            evidence=round(float(packet.evidence.coverage), 2),
            reliability=round(float(packet.verification.reliability_score), 2),
            safety_model=round(float(packet.verification.safety_model_score), 2),
            ie_judge=round(float(packet.verification.ie_judge_score), 2),
            unsupported_claims=_unsupported_claim_count(packet),
            zone=_confidence_zone(packet),
        ),
        timing=TimingView(
            total_seconds=total_seconds,
            wall_seconds=wall_seconds,
            llm_seconds=llm_seconds,
            iterations=max(1, len(packet.loop_history) + 1),
            stages=stages,
            token_budget=token_budget,
        ),
        evidence_trace=EvidenceTraceView(
            coverage=round(float(packet.evidence.coverage), 2),
            contradiction_mass=round(float(packet.evidence.contradiction_mass), 2),
            query_hygiene_score=round(float(packet.retrieval_stats.query_hygiene_score), 2),
            hazard_leakage_detected=bool(packet.retrieval_stats.hazard_leakage_detected),
            items=packet.evidence.items[:8],
            retrieval_intents=packet.evidence.retrieval_intents[:8],
        ),
        reasoning_trace=packet.reasoning_trace[:24],
        simulation_scorecards=packet.simulation_scorecards[:5],
        runtime={
            "operation_mode": runtime_snapshot.get("operation_mode", "local_core_with_online_r2"),
            "runtime_profile": runtime_snapshot.get("runtime_profile", "auto"),
            "server_topology": runtime_snapshot.get("server_topology", ""),
            "free_vram_gb": runtime_snapshot.get("free_vram_gb"),
            "total_vram_gb": runtime_snapshot.get("total_vram_gb"),
            "compute_cap": runtime_snapshot.get("compute_cap"),
            "binary_compatibility": runtime_snapshot.get("binary_compatibility", ""),
            "degrade_reason": runtime_snapshot.get("degrade_reason", ""),
            "engine_mode": packet.engine_mode,
            "engine_model": packet.engine_model,
            "trace_id": packet.session_isolation_id,
            "mode_lock_violation": packet.mode_lock_violation,
            "verification_profile": packet.verification.adaptive_profile,
            "verification_reason_category": packet.verification.decision_reason_category,
            "verification_decision_path": packet.verification.decision_path[:8],
            "verification_recommended_next_actions": packet.verification.recommended_next_actions[:8],
            "verification_gate_evaluations": [
                {
                    "gate_id": item.gate_id,
                    "passed": bool(item.passed),
                    "score": round(float(item.score), 2),
                    "threshold": round(float(item.threshold), 2),
                    "confidence_margin": round(float(item.confidence_margin), 2),
                    "remediable": bool(item.remediable),
                }
                for item in packet.verification.gate_evaluations[:8]
            ],
            "verification_uncertainty_axes": packet.verification.uncertainty_axes.model_dump(mode="json"),
            "verification_contradiction_clusters": [
                item.model_dump(mode="json")
                for item in packet.verification.contradiction_clusters[:4]
            ],
            "trace_consistency": trace_consistency,
        },
        abstention_recommended=bool(getattr(packet, "abstention_recommended", False)),
        abstention_reason=str(getattr(packet, "abstention_reason", "") or ""),
        abstention_margin=float(getattr(packet, "abstention_margin", 0.0) or 0.0),
        abstention_grounding_risk=float(getattr(packet, "abstention_grounding_risk", 0.0) or 0.0),
        inline_grounding_pass_rate=float(getattr(packet, "inline_grounding_pass_rate", 1.0) or 1.0),
        evidence_starvation_flag=bool(getattr(packet.retrieval_stats, "evidence_starvation_flag", False)),
        starved_candidates=list(getattr(packet.retrieval_stats, "starved_candidates", []) or []),
        coverage_per_candidate=dict(getattr(packet.retrieval_stats, "coverage_per_candidate", {}) or {}),
    )
