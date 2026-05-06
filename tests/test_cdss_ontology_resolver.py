from src.cdss.contracts.models import FactGraph, FactNode, StructuredFindings
from src.cdss.knowledge.ontology import DiagnosisCandidateResolver, OntologyDecision

from cdss_test_support import run_async


def test_ontology_rejects_noise_label_toothpick():
    resolver = DiagnosisCandidateResolver()
    findings = StructuredFindings(
        summary="4-year-old with drooling, stridor, and fever",
        positive_findings=["drooling", "stridor", "fever"],
        red_flags=["airway compromise"],
    )
    decision = run_async(
        resolver.validate(
            label="toothpick",
            findings=findings,
            rationale=["object mention in text"],
            fact_graph=FactGraph(),
            require_external_verification=False,
        )
    )
    assert isinstance(decision, OntologyDecision)
    assert decision.accepted is False
    assert decision.reason in {"failed_structural_gate", "no_case_context_overlap"}


def test_ontology_allows_structural_candidate_when_external_unavailable(monkeypatch):
    resolver = DiagnosisCandidateResolver()

    async def _stub_verify_external(_canonical_label: str):
        return False, False

    monkeypatch.setattr(resolver, "_verify_external_ontology", _stub_verify_external)

    findings = StructuredFindings(
        summary="child with stridor, drooling, muffled voice, and high fever",
        positive_findings=["stridor", "drooling", "muffled voice"],
        red_flags=["airway compromise"],
        suspected_conditions=["epiglottitis"],
        raw_segments=["epiglottitis concern with drooling, stridor, muffled voice, and high fever"],
    )
    graph = FactGraph(
        nodes=[
            FactNode(
                node_id="n1",
                label="airway obstruction suspected",
                category="clinical_state",
            )
        ]
    )
    decision = run_async(
        resolver.validate(
            label="epiglottitis",
            findings=findings,
            rationale=["airway obstruction pattern with drooling and stridor"],
            fact_graph=graph,
            require_external_verification=True,
            validated_by_llm=True,
        )
    )
    assert decision.accepted is True
    assert decision.canonical_label == "epiglottitis"
    assert decision.reason == "validated_structural_case_context_gate"
    assert decision.externally_verified is True
