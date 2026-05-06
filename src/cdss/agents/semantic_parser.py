"""LLM-first semantic parser that materializes a typed fact graph without static cue banks."""

from __future__ import annotations

from src.cdss.agents.extractor import ExtractorAgent
from src.cdss.contracts.models import DecisionTrace, FactGraph, FactNode, GraphEdge, InterventionNode, InterventionSet, PatientInput, ProvenanceEnvelope, StructuredFindings
from src.cdss.core.state import StatePatch
from src.cdss.runtime.policy import load_runtime_policy
from src.cdss.text_normalization import ascii_fold


def _identifier(value: object) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in ascii_fold(str(value or "")).strip()).strip("_")


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
            prefix = _identifier(head) or "signal"
            value = tail or head
        value_id = _identifier(value) or _identifier(text)
        if value_id:
            rows.append((prefix, value_id))
    return rows


def _ctx_category(prefix: str) -> tuple[str, str]:
    if prefix in {"working_diagnosis", "clinician_anchor"}:
        return "authority_claim", "clinician_context"
    if prefix in {"external_evidence", "literature", "verified_evidence"}:
        return "authority_claim", "external_context"
    return "context_frame", "typed_context"


def _build_fact_graph(findings: StructuredFindings, *, include_static_patterns: bool = False) -> FactGraph:
    del include_static_patterns
    nodes: list[FactNode] = []
    edges: list[GraphEdge] = []
    counter = 0

    def add_node(
        label: str,
        category: str,
        *,
        value: str = "",
        polarity: str = "positive",
        confidence: float = 0.75,
        temporal: str = "",
        source_text: str = "",
        provenance: str = "narrative",
    ) -> str:
        nonlocal counter
        counter += 1
        node_id = f"fact_{counter}"
        nodes.append(
            FactNode(
                node_id=node_id,
                label=label,
                category=category,
                value=value,
                polarity=polarity,
                confidence=confidence,
                temporal=temporal,
                source_text=source_text or label,
                provenance=provenance,
                provenance_envelope=ProvenanceEnvelope(
                    source_stage="extractor",
                    source_type=provenance or "narrative",
                    source_texts=[(source_text or label).strip()][:1],
                    derivation="semantic_parser_projection",
                    confidence_basis="source_text_grounding",
                ),
            )
        )
        return node_id

    timeline_root = ""
    if findings.timeline:
        timeline_root = add_node("timeline_context", "timeline", value="; ".join(findings.timeline), confidence=0.82)

    episodic_markers = {"paroxysmal", "episodic", "intermittent", "recurrent", "attacks", "waves"}

    finding_nodes: list[str] = []
    for item in findings.positive_findings:
        normalized_item = str(item or "").lower()
        temporal_tag = "cyclic" if any(marker in normalized_item for marker in episodic_markers) else "present"
        category = "episodic_finding" if temporal_tag == "cyclic" else "finding"
        node_id = add_node(item, category, confidence=0.82, temporal=temporal_tag, source_text=item)
        finding_nodes.append(node_id)
        if timeline_root:
            edges.append(GraphEdge(source_id=node_id, target_id=timeline_root, relation="occurs_with", confidence=0.7))

    for item in findings.negative_findings:
        add_node(item, "finding", polarity="negative", confidence=0.78, temporal="absent", source_text=item)

    risk_nodes: list[str] = []
    for item in findings.red_flags:
        risk_nodes.append(add_node(item, "risk_marker", confidence=0.88, source_text=item))

    diagnostic_nodes: list[str] = []
    for item in findings.suspected_conditions:
        diagnostic_nodes.append(add_node(item, "diagnostic_context", confidence=0.76, source_text=item))

    for item in findings.exposures:
        add_node(item, "exposure", confidence=0.84, source_text=item)

    for item in findings.medications:
        add_node(item, "medication", confidence=0.8, source_text=item)

    for prefix, value in _ctx_rows(findings):
        category, provenance = _ctx_category(prefix)
        add_node(value, category, confidence=0.74, source_text=f"{prefix}:{value}", provenance=provenance)

    phenotype_nodes: list[str] = []
    for atom in findings.phenotype_atoms:
        if not str(atom.label or "").strip():
            continue
        phenotype_nodes.append(
            add_node(
                atom.label,
                "semantic_pattern",
                confidence=float(atom.confidence or 0.72),
                source_text="; ".join(atom.evidence[:3]) or atom.label,
                provenance=atom.provenance or "phenotype_compiler",
            )
        )

    for key, value in (findings.derived_vitals or {}).items():
        add_node(_identifier(key), "vital", value=str(value), confidence=0.88, source_text=f"{key}={value}")

    for risk_id in risk_nodes[:4]:
        for target in finding_nodes[:4]:
            edges.append(GraphEdge(source_id=risk_id, target_id=target, relation="highlights", confidence=0.62))
        for target in diagnostic_nodes[:3]:
            edges.append(GraphEdge(source_id=risk_id, target_id=target, relation="raises_risk_of", confidence=0.58))

    for diagnostic_id in diagnostic_nodes[:4]:
        for target in finding_nodes[:5]:
            edges.append(GraphEdge(source_id=diagnostic_id, target_id=target, relation="explains", confidence=0.56))

    for phenotype_id in phenotype_nodes[:6]:
        for target in finding_nodes[:6]:
            edges.append(GraphEdge(source_id=phenotype_id, target_id=target, relation="patterns_with", confidence=0.6))

    return FactGraph(nodes=nodes, edges=edges, summary=findings.summary)


def _build_interventions(findings: StructuredFindings) -> InterventionSet:
    items: list[InterventionNode] = []
    for index, item in enumerate(findings.planned_interventions, start=1):
        items.append(
            InterventionNode(
                node_id=f"intervention_{index}",
                label=item,
                status="planned",
                confidence=0.78,
                source_text=item,
            )
        )
    return InterventionSet(items=items)


class SemanticParserAgent:
    """Primary semantic parse stage for v3 transition."""

    def __init__(self, extractor: ExtractorAgent | None = None) -> None:
        self._extractor = extractor or ExtractorAgent()
        self._policy = load_runtime_policy()

    async def run(self, patient_input: PatientInput) -> StatePatch:
        extractor_patch = await self._extractor.run(patient_input)
        findings = extractor_patch.findings or StructuredFindings()
        fact_graph = _build_fact_graph(findings, include_static_patterns=self._policy.static_pattern_map_enabled)
        interventions = _build_interventions(findings)
        extractor_payload = dict((extractor_patch.trace[0].payload if extractor_patch.trace else {}) or {})
        metrics = dict(extractor_payload.get("metrics", {}) or {})
        return StatePatch(
            status="semantic_parse_ready",
            active_stage="semantic_parse",
            fact_graph=fact_graph,
            interventions=interventions,
            findings=findings,
            trace=[
                DecisionTrace(
                    stage="extractor",
                    message="Semantic fact graph parsed from patient narrative.",
                    payload={
                        "fact_node_count": len(fact_graph.nodes),
                        "edge_count": len(fact_graph.edges),
                        "intervention_count": len(interventions.items),
                        "metrics": metrics,
                    },
                )
            ],
        )
