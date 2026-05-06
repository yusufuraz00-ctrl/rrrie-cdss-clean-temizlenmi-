"""Unit tests for retrieval evidence-starvation surfacing."""

from __future__ import annotations

import pytest

from src.cdss.clinical.verification import build_verification_report
from src.cdss.contracts.models import (
    DifferentialCandidate,
    DifferentialSet,
    EvidenceBundle,
    EvidenceItem,
    PatientInput,
    RetrievalRankingStats,
    RiskProfile,
    StructuredFindings,
    UrgencyTier,
)
from src.cdss.retrieval.planner import RetrievalReranker


def _diff(*labels: str) -> DifferentialSet:
    cands = [DifferentialCandidate(label=label, score=0.5) for label in labels]
    return DifferentialSet(candidates=cands)


def test_reranker_flags_starvation_when_items_empty():
    reranker = RetrievalReranker()
    differential = _diff("aortic_dissection", "pulmonary_embolism")
    findings = StructuredFindings(summary="severe chest pain")
    items, stats = reranker.rerank(
        query_hints=["chest pain"],
        items=[],
        findings=findings,
        differential=differential,
    )
    assert items == []
    assert stats.evidence_starvation_flag is True
    assert "aortic_dissection" in stats.starved_candidates
    assert stats.coverage_per_candidate.get("aortic_dissection") == pytest.approx(0.0)


def test_reranker_marks_unmatched_candidate_as_starved():
    reranker = RetrievalReranker()
    differential = _diff("aortic_dissection", "pulmonary_embolism")
    findings = StructuredFindings(summary="severe chest pain")
    items = [
        EvidenceItem(
            source="pubmed",
            citation="https://example.org/aortic-dissection-review",
            title="aortic dissection review",
            excerpt="aortic dissection presents with tearing chest pain radiating",
            trust_score=0.5,
        ),
    ]
    _, stats = reranker.rerank(
        query_hints=["aortic dissection"],
        items=items,
        findings=findings,
        differential=differential,
    )
    assert stats.evidence_starvation_flag is True
    assert "pulmonary_embolism" in stats.starved_candidates
    assert "aortic_dissection" not in stats.starved_candidates


def test_reranker_no_starvation_when_all_candidates_have_evidence():
    reranker = RetrievalReranker()
    differential = _diff("aortic_dissection", "pulmonary_embolism")
    findings = StructuredFindings(summary="chest pain")
    items = [
        EvidenceItem(
            source="pubmed",
            citation="https://example.org/aortic",
            title="aortic dissection review",
            excerpt="aortic dissection",
            trust_score=0.5,
        ),
        EvidenceItem(
            source="pubmed",
            citation="https://example.org/pe",
            title="pulmonary embolism diagnosis",
            excerpt="pulmonary embolism",
            trust_score=0.5,
        ),
    ]
    _, stats = reranker.rerank(
        query_hints=["chest pain"],
        items=items,
        findings=findings,
        differential=differential,
    )
    assert stats.evidence_starvation_flag is False
    assert stats.starved_candidates == []


def test_verification_surfaces_low_evidence_grounding_when_stats_starved():
    findings = StructuredFindings(summary="severe headache")
    risk = RiskProfile(urgency=UrgencyTier.ROUTINE)
    diff = _diff("aortic_dissection")
    evidence = EvidenceBundle(items=[])
    stats = RetrievalRankingStats(
        evidence_starvation_flag=True,
        starved_candidates=["aortic_dissection"],
    )
    report = build_verification_report(
        findings=findings,
        risk_profile=risk,
        differential=diff,
        evidence=evidence,
        retrieval_stats=stats,
    )
    issue_types = {i.issue_type for i in report.issues}
    assert "low_evidence_grounding" in issue_types


def test_verification_surfaces_low_evidence_when_evidence_empty_no_stats():
    findings = StructuredFindings(summary="severe headache")
    risk = RiskProfile(urgency=UrgencyTier.ROUTINE)
    diff = _diff("migraine")
    evidence = EvidenceBundle(items=[])
    report = build_verification_report(
        findings=findings,
        risk_profile=risk,
        differential=diff,
        evidence=evidence,
    )
    issue_types = {i.issue_type for i in report.issues}
    assert "low_evidence_grounding" in issue_types
