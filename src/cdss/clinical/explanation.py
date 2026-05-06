"""Clinical explanation graph and multi-axis confidence helpers."""

from __future__ import annotations

import re

from src.cdss.clinical.phenotype import phenotype_labels, phenotype_query_terms, phenotype_state_frames
from src.cdss.contracts.models import (
    DetectiveAtom,
    DetectiveEdge,
    ExplanationGraph,
    ExplanationLink,
    FactGraph,
    HypothesisFrontier,
    InterventionSafetyAssessment,
    InterventionSet,
    ModelSupportSignals,
    OODAssessment,
    ProvenanceEnvelope,
    ReliabilitySignals,
    RetrievalRankingStats,
    RiskProfile,
    StructuredFindings,
    UrgencyTier,
)
from src.cdss.text_normalization import ascii_fold


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ascii_fold(str(value or "")).lower()).strip("_")


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{3,}", ascii_fold(str(value or "")).lower()) if token}


def _ctx_rows(findings: StructuredFindings) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in findings.input_context:
        text = str(raw or "").strip()
        if not text:
            continue
        prefix = "signal"
        value = text
        if ":" in text:
            head, tail = text.split(":", 1)
            prefix = _slug(head) or "signal"
            value = tail or head
        value_slug = _slug(value) or _slug(text)
        if value_slug:
            rows.append((prefix, value_slug))
    return rows


def _story_context(findings: StructuredFindings) -> list[str]:
    if findings.context_lanes:
        return list(dict.fromkeys(findings.context_lanes.get("patient_narrative", [])[:8]))
    values: list[str] = []
    for prefix, value in _ctx_rows(findings):
        if prefix in {
            "story_frame",
            "baseline_context",
            "event_context",
            "clinician_anchor",
            "patient_concern",
            "alternative_explanation",
        }:
            values.append(value)
    return list(dict.fromkeys(values))[:8]


def _top_profiles(frontier: HypothesisFrontier) -> list[tuple[str, set[str], bool]]:
    output: list[tuple[str, set[str], bool]] = []
    for node in frontier.hypotheses[:3]:
        terms = _tokenize(node.label.replace("_", " "))
        for item in node.rationale[:6]:
            terms |= _tokenize(item)
        output.append((node.label, terms, bool(node.must_not_miss or node.dangerous_if_missed)))
    return output


def _event_context(findings: StructuredFindings) -> set[str]:
    return {
        token
        for item in [*findings.exposures, *findings.timeline, *findings.raw_segments, *_story_context(findings), *phenotype_labels(findings, limit=8)]
        for token in _tokenize(item)
    }


def _treatment_context(interventions: InterventionSet, findings: StructuredFindings) -> set[str]:
    return {
        token
        for item in [*(node.label for node in interventions.items), *findings.planned_interventions]
        for token in _tokenize(item)
    }


def _token_overlap_score(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    overlap = len(left & right)
    return overlap / max(1, min(6, len(left | right)))


def _detective_base_confidence(category: str) -> float:
    return {
        "finding": 0.72,
        "hypothesis": 0.76,
        "state": 0.66,
        "context": 0.58,
        "intervention": 0.54,
    }.get(category, 0.5)


def _build_detective_atoms(
    findings: StructuredFindings,
    frontier: HypothesisFrontier,
    interventions: InterventionSet,
    state_frames: list[str],
) -> list[DetectiveAtom]:
    rows: list[tuple[str, str, list[str]]] = []
    rows.extend(("finding", item, [item]) for item in [*findings.positive_findings, *findings.red_flags][:18] if str(item or "").strip())
    rows.extend(("context", item, [item]) for item in [*findings.exposures[:8], *findings.timeline[:8], *_story_context(findings)[:8]])
    rows.extend(("hypothesis", node.label, [node.label, *node.rationale[:2]]) for node in frontier.hypotheses[:5])
    rows.extend(("intervention", item.label, [item.label]) for item in interventions.items[:8])
    rows.extend(("intervention", item, [item]) for item in findings.planned_interventions[:8])
    rows.extend(("state", frame, [frame]) for frame in state_frames[:8])

    atoms: list[DetectiveAtom] = []
    seen: set[str] = set()
    for index, (category, label, evidence) in enumerate(rows, start=1):
        normalized = _slug(label)
        if not normalized:
            continue
        key = f"{category}:{normalized}"
        if key in seen:
            continue
        seen.add(key)
        token_bonus = min(0.18, len(_tokenize(label)) * 0.03)
        atoms.append(
            DetectiveAtom(
                atom_id=f"{category}_{index}",
                label=normalized,
                category=category,
                confidence=round(min(0.95, _detective_base_confidence(category) + token_bonus), 2),
                evidence=[item for item in evidence if str(item or "").strip()][:3],
            )
        )
    return atoms


def _build_detective_edges(atoms: list[DetectiveAtom]) -> list[DetectiveEdge]:
    relation_profiles: dict[tuple[str, str], tuple[str, float]] = {
        ("finding", "hypothesis"): ("supports_hypothesis", 0.28),
        ("finding", "context"): ("linked_to_event_context", 0.2),
        ("finding", "intervention"): ("possible_treatment_effect", 0.22),
        ("finding", "state"): ("state_consistent_signal", 0.24),
        ("hypothesis", "state"): ("mechanistic_alignment", 0.18),
        ("context", "state"): ("contextual_state_bridge", 0.16),
        ("intervention", "state"): ("state_treatment_constraint", 0.18),
    }

    edges: list[DetectiveEdge] = []
    for i, left in enumerate(atoms):
        left_tokens = _tokenize(left.label)
        for right in atoms[i + 1 :]:
            if left.category == right.category and left.category not in {"finding", "context"}:
                continue
            pair = tuple(sorted((left.category, right.category)))
            profile = relation_profiles.get(pair)
            if not profile:
                continue
            relation, bias = profile
            overlap_score = _token_overlap_score(left_tokens, _tokenize(right.label))
            edge_weight = min(0.99, bias + (overlap_score * 0.72))
            if edge_weight < 0.34:
                continue
            edges.append(
                DetectiveEdge(
                    source_atom_id=left.atom_id,
                    target_atom_id=right.atom_id,
                    relation=relation,
                    weight=round(edge_weight, 2),
                    rationale=f"{left.category} and {right.category} share discriminative signal overlap.",
                )
            )
    return edges


def _detective_hubs(atoms: list[DetectiveAtom], edges: list[DetectiveEdge], *, limit: int = 5) -> list[str]:
    score_by_atom = {atom.atom_id: 0.0 for atom in atoms}
    for edge in edges:
        score_by_atom[edge.source_atom_id] = score_by_atom.get(edge.source_atom_id, 0.0) + float(edge.weight)
        score_by_atom[edge.target_atom_id] = score_by_atom.get(edge.target_atom_id, 0.0) + float(edge.weight)
    atom_by_id = {atom.atom_id: atom for atom in atoms}
    ranked = sorted(score_by_atom.items(), key=lambda item: item[1], reverse=True)
    hubs: list[str] = []
    for atom_id, _ in ranked:
        atom = atom_by_id.get(atom_id)
        if not atom:
            continue
        hubs.append(atom.label)
        if len(hubs) >= limit:
            break
    return hubs


def _is_critical_finding(finding: str, findings: StructuredFindings) -> bool:
    text = str(finding or "").strip()
    if not text:
        return False
    lowered = text.lower()
    red_flag_pool = {item.lower() for item in findings.red_flags}
    if lowered in red_flag_pool:
        return True
    return lowered.startswith("system_alert:")


def _merged_vitals(findings: StructuredFindings, fact_graph: FactGraph) -> dict[str, str]:
    merged = {str(key): str(value) for key, value in (findings.derived_vitals or {}).items()}
    for node in fact_graph.nodes:
        if node.category != "vital":
            continue
        if node.label not in merged and str(node.value or "").strip():
            merged[node.label] = str(node.value)
    return merged


def _vital_state_frames(findings: StructuredFindings, fact_graph: FactGraph) -> list[str]:
    frames: list[str] = []
    vitals = _merged_vitals(findings, fact_graph)

    def _to_float(value: str) -> float | None:
        try:
            text = str(value or "").replace(",", ".").replace("%", "").strip()
            return float(text) if text else None
        except ValueError:
            return None

    spo2 = _to_float(vitals.get("spo2") or vitals.get("o2_sat") or "")
    sbp = _to_float(vitals.get("sbp") or vitals.get("systolic_bp") or "")
    hr = _to_float(vitals.get("heart_rate") or vitals.get("hr") or vitals.get("pulse") or "")
    rr = _to_float(vitals.get("respiratory_rate") or vitals.get("rr") or "")

    if spo2 is not None and spo2 < 92:
        frames.append("physiologic_instability_state__hypoxemia")
    if sbp is not None and sbp < 90:
        frames.append("physiologic_instability_state__shock")
    elif sbp is not None and sbp <= 100:
        frames.append("physiologic_alert_state__borderline_hypotension")
    if hr is not None and hr >= 130:
        frames.append("physiologic_instability_state__marked_tachycardia")
    elif hr is not None and hr >= 100:
        frames.append("physiologic_alert_state__tachycardia")
    if rr is not None and rr >= 30:
        frames.append("physiologic_instability_state__marked_tachypnea")
    elif rr is not None and rr >= 22:
        frames.append("physiologic_alert_state__tachypnea")
    return frames


def derive_state_frames(
    findings: StructuredFindings,
    fact_graph: FactGraph,
    interventions: InterventionSet | None = None,
) -> list[str]:
    del interventions
    frames: list[str] = []
    for prefix, value in _ctx_rows(findings):
        if prefix in {
            "hazard",
            "pathway_fit",
            "masquerade_risk",
            "causal_loop",
            "device_reliability",
            "contradiction",
        }:
            frames.append(f"{prefix}_state__{value}")
    frames.extend(phenotype_state_frames(findings))
    frames.extend(_vital_state_frames(findings, fact_graph))
    if findings.red_flags:
        frames.append("high_risk_signal_state")
    return list(dict.fromkeys(frame for frame in frames if frame))[:8]


def derive_dangerous_treatment_assumptions(
    findings: StructuredFindings,
    fact_graph: FactGraph,
    interventions: InterventionSet,
    state_frames: list[str],
) -> list[str]:
    del fact_graph
    assumptions: list[str] = []
    for prefix, value in _ctx_rows(findings):
        if prefix in {"working_diagnosis", "clinician_anchor", "story_frame", "baseline_context", "event_context", "patient_concern"}:
            continue
        if prefix == "blocked_order":
            assumptions.append(f"blocked order remains unsafe without override clearance: {value}")
        elif prefix == "causal_loop":
            assumptions.append(f"planned treatment may reinforce an active causal loop: {value}")
        elif prefix == "device_reliability":
            assumptions.append(f"device-derived reassurance may be misleading: {value}")
        elif prefix == "contradiction":
            assumptions.append(f"contradictory case state requires objective confirmation: {value}")
        elif prefix == "masquerade_risk":
            assumptions.append(f"apparent syndrome may mask a more dangerous process: {value}")
        elif prefix == "hazard":
            assumptions.append(f"high-risk trajectory flagged by typed context: {value}")
    if not assumptions and interventions.items and any(
        marker in frame
        for frame in state_frames
        for marker in ("hazard", "device_reliability", "contradiction", "physiologic_instability")
    ):
        assumptions.append("intervention timing must stay aligned with the active safety state before execution")
    if not assumptions and state_frames:
        assumptions.append("active state remains insufficiently excluded for autonomous treatment closure")
    return list(dict.fromkeys(assumptions))[:6]


def _story_builder(
    *,
    findings: StructuredFindings,
    state_frames: list[str],
    unexplained: list[ExplanationLink],
    dangerous_treatment_assumptions: list[str],
    detective_hubs: list[str],
) -> tuple[str, list[str], list[str], str]:
    primary_process = state_frames[0] if state_frames else ""
    secondary_processes = state_frames[1:4]
    epistemic_gaps = [item.finding for item in unexplained[:5]]
    story_context = _story_context(findings)
    phenotype_terms = phenotype_query_terms(findings, limit=3)
    if primary_process:
        summary = f"Primary typed state is {primary_process.replace('_', ' ')}."
    else:
        summary = findings.summary[:220] or "No stable typed state identified yet."
    if story_context:
        summary += f" Narrative context: {', '.join(item.replace('_', ' ') for item in story_context[:3])}."
    if phenotype_terms:
        summary += f" Phenotype signature: {', '.join(item.replace('_', ' ') for item in phenotype_terms[:3])}."
    if secondary_processes:
        summary += f" Secondary states: {', '.join(item.replace('_', ' ') for item in secondary_processes)}."
    if dangerous_treatment_assumptions:
        summary += " Treatment assumptions remain safety-sensitive."
    if detective_hubs:
        summary += f" Investigation hubs: {', '.join(item.replace('_', ' ') for item in detective_hubs[:3])}."
    if epistemic_gaps:
        summary += f" Key unresolved findings: {', '.join(epistemic_gaps[:3])}."
    return primary_process, secondary_processes, epistemic_gaps, summary


def _edge_adjacency(edges: list[DetectiveEdge]) -> dict[str, list[tuple[str, float, str]]]:
    adjacency: dict[str, list[tuple[str, float, str]]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source_atom_id, []).append((edge.target_atom_id, float(edge.weight), edge.relation))
        adjacency.setdefault(edge.target_atom_id, []).append((edge.source_atom_id, float(edge.weight), edge.relation))
    return adjacency


def build_explanation_graph(
    findings: StructuredFindings,
    frontier: HypothesisFrontier,
    interventions: InterventionSet,
    fact_graph: FactGraph,
) -> ExplanationGraph:
    event_terms = _event_context(findings)
    treatment_terms = _treatment_context(interventions, findings)
    state_frames = derive_state_frames(findings, fact_graph, interventions)
    background_terms = {token for item in findings.exposures[:6] for token in _tokenize(item)}
    state_terms = {token for item in state_frames[:6] for token in _tokenize(item)}
    detective_atoms = _build_detective_atoms(findings, frontier, interventions, state_frames)
    detective_edges = _build_detective_edges(detective_atoms)
    detective_hubs = _detective_hubs(detective_atoms, detective_edges)
    atom_by_id = {atom.atom_id: atom for atom in detective_atoms}
    adjacency = _edge_adjacency(detective_edges)
    finding_atom_by_label = {
        atom.label: atom.atom_id
        for atom in detective_atoms
        if atom.category == "finding"
    }

    links: list[ExplanationLink] = []
    all_findings = list(dict.fromkeys([*findings.positive_findings, *findings.red_flags]))

    for finding in all_findings:
        finding_terms = _tokenize(finding)
        critical = _is_critical_finding(finding, findings)

        scored_candidates: list[tuple[float, str, str, str]] = []
        finding_atom_id = finding_atom_by_label.get(_slug(finding), "")
        if finding_atom_id:
            for neighbor_id, weight, relation in adjacency.get(finding_atom_id, []):
                neighbor = atom_by_id.get(neighbor_id)
                if not neighbor:
                    continue
                if neighbor.category == "hypothesis":
                    scored_candidates.append((weight + 0.1, "disease_explained", neighbor.label, f"Detective-board relation: {relation}."))
                elif neighbor.category == "context":
                    scored_candidates.append((weight + 0.06, "event_explained", "triggering_event_or_exposure", f"Detective-board relation: {relation}."))
                elif neighbor.category == "intervention":
                    scored_candidates.append((weight + 0.08, "treatment_explained", "planned_or_recent_intervention", f"Detective-board relation: {relation}."))
                elif neighbor.category == "state":
                    scored_candidates.append((weight + 0.07, "state_explained", neighbor.label, f"Detective-board relation: {relation}."))

        signal_profiles = [
            ("event_explained", "triggering_event_or_exposure", event_terms, 0.22),
            ("treatment_explained", "planned_or_recent_intervention", treatment_terms, 0.2),
            ("background_explained", "background_context", background_terms, 0.16),
            ("state_explained", state_frames[0] if state_frames else "active_state_context", state_terms, 0.18),
        ]
        for explanation_type, target, terms, bias in signal_profiles:
            overlap_score = _token_overlap_score(finding_terms, terms)
            if overlap_score <= 0:
                continue
            scored_candidates.append(
                (
                    bias + (overlap_score * 0.72),
                    explanation_type,
                    target,
                    f"Adaptive signal overlap score={round(overlap_score, 2)} with {explanation_type} profile.",
                )
            )

        scored_candidates.sort(key=lambda item: item[0], reverse=True)
        best_score, explanation_type, target, rationale = scored_candidates[0] if scored_candidates else (0.0, "", "", "")

        if best_score >= 0.34:
            confidence = min(0.94, round(0.3 + (best_score * 0.74), 2))
        elif len(finding_terms) <= 1:
            explanation_type = "incidental"
            target = "low_specificity_finding"
            confidence = 0.35
            rationale = "Finding is too nonspecific to materially change the current differential."
        else:
            explanation_type = "still_unexplained"
            target = ""
            confidence = 0.18
            rationale = "Finding remains insufficiently explained by the current frontier."

        links.append(
            ExplanationLink(
                finding=finding,
                explanation_type=explanation_type,
                target=target,
                confidence=round(confidence, 2),
                rationale=rationale,
                critical=critical,
                provenance_envelope=ProvenanceEnvelope(
                    source_stage="explanation_graph",
                    source_type="reasoning_link",
                    source_texts=[finding],
                    evidence_refs=[target] if target else [],
                    derivation=explanation_type,
                    confidence_basis=rationale,
                ),
            )
        )

    explained_count = len([item for item in links if item.explanation_type != "still_unexplained"])
    unexplained = [item for item in links if item.explanation_type == "still_unexplained"]
    critical_unexplained = [item for item in unexplained if item.critical]
    contradictory_findings = [
        item.finding
        for item in links
        if item.explanation_type == "still_unexplained" and item.critical
    ][:6]
    total = len(links) or 1
    dangerous_treatment_assumptions = derive_dangerous_treatment_assumptions(findings, fact_graph, interventions, state_frames)
    primary_process, secondary_processes, epistemic_gaps, story_summary = _story_builder(
        findings=findings,
        state_frames=state_frames,
        unexplained=unexplained,
        dangerous_treatment_assumptions=dangerous_treatment_assumptions,
        detective_hubs=detective_hubs,
    )
    return ExplanationGraph(
        links=links,
        detective_atoms=detective_atoms,
        detective_edges=detective_edges,
        detective_hubs=detective_hubs,
        explained_count=explained_count,
        unexplained_count=len(unexplained),
        critical_unexplained_count=len(critical_unexplained),
        coverage=round(explained_count / total, 2),
        state_frames=state_frames,
        unexplained_findings=[item.finding for item in unexplained][:8],
        contradictory_findings=contradictory_findings,
        dangerous_treatment_assumptions=dangerous_treatment_assumptions,
        primary_process=primary_process,
        secondary_processes=secondary_processes,
        epistemic_gaps=epistemic_gaps,
        story_summary=story_summary,
    )


def build_reliability_axes(
    *,
    findings: StructuredFindings,
    risk_profile: RiskProfile,
    frontier: HypothesisFrontier,
    fact_graph: FactGraph,
    explanation_graph: ExplanationGraph,
    intervention_safety: InterventionSafetyAssessment,
    evidence_coverage: float,
    contradiction_mass: float,
    verification_reliability: float,
    model_support: ModelSupportSignals,
    ood_assessment: OODAssessment,
    retrieval_stats: RetrievalRankingStats,
) -> ReliabilitySignals:
    semantic_confidence = 0.0
    if fact_graph.nodes:
        semantic_confidence = min(1.0, round(sum(node.confidence for node in fact_graph.nodes) / len(fact_graph.nodes), 2))
    intervention_risk_conflict = 1.0 if intervention_safety.blocked_interventions else 0.0
    critical_total = max(1, len([item for item in explanation_graph.links if item.critical]))
    critical_unexplained_ratio = round(explanation_graph.critical_unexplained_count / critical_total, 2)
    state_coherence = max(
        0.0,
        min(
            1.0,
            round(
                (0.28 if explanation_graph.state_frames else 0.0)
                + (explanation_graph.coverage * 0.44)
                + ((1.0 - critical_unexplained_ratio) * 0.28),
                2,
            ),
        ),
    )
    explanation_completeness = max(0.0, min(1.0, round((explanation_graph.coverage * 0.7) + ((1.0 - critical_unexplained_ratio) * 0.3), 2)))

    top_support = 0.0
    top_label = frontier.anchor_hypothesis
    if top_label:
        top_support = model_support.candidate_support.get(top_label, 0.0)
    independent_consensus = min(
        1.0,
        max(0.0, (top_support * 0.45) + (retrieval_stats.citation_alignment * 0.25) + ((1.0 - model_support.model_disagreement) * 0.3)),
    )

    diagnostic_fit = max(
        0.0,
        min(
            1.0,
            round(
                (explanation_graph.coverage * 0.4)
                + (state_coherence * 0.12)
                + ((1.0 - frontier.frontier_entropy) * 0.28)
                + (semantic_confidence * 0.16)
                + (independent_consensus * 0.16)
                - (critical_unexplained_ratio * 0.18),
                2,
            ),
        ),
    )
    explicit_unsafe_plan = 0.0
    unresolved_hazard = 1.0 if explanation_graph.critical_unexplained_count > 0 and risk_profile.urgency != UrgencyTier.ROUTINE else 0.0
    missing_concurrent_action = 1.0 if intervention_safety.decisions and intervention_safety.required_concurrent_actions and not intervention_safety.allowed_interventions else 0.0
    action_safety = max(
        0.0,
        min(
            1.0,
            round(
                0.92
                - (explicit_unsafe_plan * 0.42)
                - (unresolved_hazard * 0.16)
                - (missing_concurrent_action * 0.12)
                - (critical_unexplained_ratio * 0.16)
                - (0.08 if risk_profile.urgency == UrgencyTier.EMERGENCY else 0.0),
                2,
            ),
        ),
    )
    evidence_sufficiency = max(
        0.0,
        min(
            1.0,
            round(
                (evidence_coverage * 0.7)
                + (retrieval_stats.specificity_gain * 0.2)
                + (retrieval_stats.novelty_gain * 0.1),
                2,
            ),
        ),
    )
    distribution_stability = max(0.0, min(1.0, round(1.0 - ood_assessment.ood_score, 2)))
    closure_readiness = max(
        0.0,
        min(
            1.0,
            round(
                (diagnostic_fit * 0.28)
                + (action_safety * 0.18)
                + (evidence_sufficiency * 0.2)
                + (explanation_completeness * 0.08)
                + (distribution_stability * 0.16)
                + (verification_reliability * 0.18)
                - (contradiction_mass * 0.18)
                - (critical_unexplained_ratio * 0.2),
                2,
            ),
        ),
    )

    reasons: list[str] = []
    if explanation_graph.critical_unexplained_count:
        reasons.append(f"{explanation_graph.critical_unexplained_count} critical findings remain under-explained")
    if explanation_graph.state_frames:
        reasons.append(f"active typed states: {', '.join(explanation_graph.state_frames[:3])}")
    if evidence_sufficiency < 0.55:
        reasons.append("evidence sufficiency remains limited for stable closure")
    if retrieval_stats.novelty_gain < 0.12 and evidence_coverage < 0.7:
        reasons.append("additional retrieval is yielding low novelty")
    if model_support.model_disagreement >= 0.18:
        reasons.append("independent model families still disagree materially")
    if not reasons:
        reasons.append("current frontier is reasonably coherent for the available evidence")

    overall = max(
        0.0,
        min(
            1.0,
            round(
                (diagnostic_fit * 0.26)
                + (action_safety * 0.14)
                + (closure_readiness * 0.26)
                + (evidence_sufficiency * 0.14)
                + (distribution_stability * 0.12)
                + (independent_consensus * 0.08),
                2,
            ),
        ),
    )
    return ReliabilitySignals(
        semantic_confidence=semantic_confidence,
        frontier_entropy=frontier.frontier_entropy,
        evidence_coverage=evidence_coverage,
        contradiction_mass=contradiction_mass,
        intervention_risk_conflict=intervention_risk_conflict,
        explanation_coverage=explanation_graph.coverage,
        state_coherence=state_coherence,
        explanation_completeness=explanation_completeness,
        critical_unexplained_ratio=critical_unexplained_ratio,
        diagnostic_fit=diagnostic_fit,
        action_safety=action_safety,
        closure_readiness=closure_readiness,
        evidence_sufficiency=evidence_sufficiency,
        distribution_stability=distribution_stability,
        consensus_strength=round(independent_consensus, 2),
        stop_reason=reasons[0],
        stop_reasons=reasons,
        overall=overall,
    )
