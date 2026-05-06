"""Typed intervention safety engine without local disease-specific heuristics."""

from __future__ import annotations

from src.cdss.contracts.models import (
    FactGraph,
    HypothesisFrontier,
    InterventionDecision,
    InterventionSafetyAssessment,
    InterventionSet,
    PatientInput,
    RiskProfile,
    StructuredFindings,
)
from src.cdss.knowledge.registry import load_syndrome_registry
from src.cdss.text_normalization import ascii_fold


def _identifier(value: object) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in ascii_fold(str(value or "")).strip()).strip("_")


def _unique(values: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = _identifier(item)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
    return output


def _ctx_map(findings: StructuredFindings) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for raw in findings.input_context:
        text = str(raw or "").strip()
        if not text:
            continue
        prefix = "signal"
        value = text
        if ":" in text:
            head, tail = text.split(":", 1)
            prefix = _identifier(head) or "signal"
            value = tail or head
        value_id = _identifier(value) or _identifier(text)
        if not value_id:
            continue
        mapping.setdefault(prefix, [])
        if value_id not in mapping[prefix]:
            mapping[prefix].append(value_id)
    return mapping


def _med_tokens(value: str) -> frozenset[str]:
    """Significant tokens (≥3 chars) from a medication label for overlap matching."""
    return frozenset(t for t in _identifier(value).split("_") if len(t) >= 3)


def _token_overlap(a: str, b: str) -> bool:
    """True when the two strings share at least one significant token."""
    return bool(_med_tokens(a) & _med_tokens(b))


def _patient_current_medications(
    findings: StructuredFindings,
    fact_graph: FactGraph | None,
) -> list[str]:
    """Extract medications the patient is *currently taking* from structured data.

    Sources (in priority order):
    1. fact_graph nodes with category == 'medication' and polarity == 'positive'
    2. findings.medications list (LLM-extracted, may include prescribed or reported)
    """
    meds: list[str] = []
    seen: set[str] = set()
    if fact_graph:
        for node in fact_graph.nodes:
            if str(node.category or "").strip().lower() != "medication":
                continue
            if str(node.polarity or "positive").strip().lower() == "negative":
                continue
            key = _identifier(node.label)
            if key and key not in seen:
                seen.add(key)
                meds.append(key)
    for raw in list(findings.medications or []):
        key = _identifier(raw)
        if key and key not in seen:
            seen.add(key)
            meds.append(key)
    return meds


def build_intervention_safety_assessment(
    findings: StructuredFindings,
    risk_profile: RiskProfile,
    frontier: HypothesisFrontier,
    interventions: InterventionSet,
    patient_input: PatientInput | None = None,
    fact_graph: FactGraph | None = None,
) -> InterventionSafetyAssessment:
    del patient_input
    ctx = _ctx_map(findings)
    registry = load_syndrome_registry()
    profile_unsafe: list[str] = []
    profile_required: list[str] = []
    if registry:
        for node in frontier.hypotheses[:3]:
            profile = registry.by_id(node.label)
            if not profile:
                continue
            profile_unsafe.extend(profile.unsafe_interventions[:4])
            profile_required.extend(profile.immediate_actions[:4])

    # Cross-check patient's current medications against profile_unsafe.
    # This catches cases where the patient self-reported a contraindicated drug
    # (e.g. ibuprofen in dengue) that is not part of any CDSS-proposed intervention.
    patient_meds = _patient_current_medications(findings, fact_graph)
    for med in patient_meds:
        for unsafe in profile_unsafe:
            if _token_overlap(med, unsafe):
                # Add the patient's reported medication to the block pool so it
                # surfaces as a contraindicated_intervention verification issue.
                if med not in profile_unsafe:
                    profile_unsafe.append(med)
                break

    blocked_pool = set(_unique([*risk_profile.blocked_actions, *ctx.get("blocked_order", []), *profile_unsafe]))
    required_pool = _unique([*risk_profile.required_actions, *ctx.get("required_action", []), *ctx.get("workflow", []), *profile_required])
    contradiction_active = bool(ctx.get("contradiction") or ctx.get("device_reliability"))
    hard_conflict_active = bool(blocked_pool or ctx.get("causal_loop"))
    prerequisite_context_active = bool(required_pool or ctx.get("data_request") or ctx.get("pathway_fit") or ctx.get("time_window"))

    decisions: list[InterventionDecision] = []
    blocked: list[str] = []
    allowed: list[str] = []
    required_concurrent: list[str] = _unique(
        [
            *required_pool,
            *(["objective_confirmation_before_irreversible_intervention"] if contradiction_active else []),
        ]
    )
    unsafe_without_exclusion: list[str] = []

    for item in interventions.items:
        label = _identifier(item.label)
        classification = "allowed"
        rationale = "No typed safety conflict detected for the current intervention."

        if label in blocked_pool:
            classification = "blocked"
            rationale = "Intervention conflicts with the current typed safety gate."
        elif hard_conflict_active and contradiction_active and not prerequisite_context_active:
            classification = "unsafe_without_exclusion"
            rationale = "Contradictory or potentially deceptive data requires targeted exclusion before proceeding."
        elif contradiction_active or prerequisite_context_active:
            classification = "allowed_with_prerequisites"
            rationale = "Intervention remains pathway-compatible but should proceed only after concurrent objective confirmation and workflow prerequisites are completed."

        decisions.append(
            InterventionDecision(
                intervention=item.label,
                classification=classification,
                rationale=rationale,
                linked_hypotheses=[candidate.label for candidate in frontier.hypotheses[:3]],
            )
        )
        if classification == "blocked":
            blocked.append(label)
        elif classification == "unsafe_without_exclusion":
            unsafe_without_exclusion.append(label)
        else:
            allowed.append(label)

    present_labels = {_identifier(item.intervention) for item in decisions}
    for item in sorted(blocked_pool):
        if item in present_labels:
            continue
        decisions.append(
            InterventionDecision(
                intervention=item,
                classification="blocked",
                rationale="Typed safety state carries a hard block for this order even if it was not materialized in the intervention list.",
                linked_hypotheses=[candidate.label for candidate in frontier.hypotheses[:3]],
            )
        )
        blocked.append(item)

    return InterventionSafetyAssessment(
        decisions=decisions,
        blocked_interventions=_unique(blocked),
        allowed_interventions=_unique(allowed),
        required_concurrent_actions=_unique(required_concurrent),
        unsafe_without_exclusion=_unique(unsafe_without_exclusion),
    )
