"""Unit tests for the inline candidate-level grounding gate."""

from __future__ import annotations

import pytest

from src.cdss.clinical.inline_grounding import (
    aggregate_pass_rate,
    gate_candidate_grounding,
    gate_differential,
)
from src.cdss.contracts.models import (
    DifferentialCandidate,
    DifferentialSet,
    GroundingVerdict,
    StructuredFindings,
)


def _findings(*segments: str, summary: str = "") -> StructuredFindings:
    return StructuredFindings(
        summary=summary or " ".join(segments),
        raw_segments=list(segments),
    )


def test_pass_when_rationale_is_in_narrative():
    findings = _findings("patient reports severe tearing chest pain radiating to back")
    cand = DifferentialCandidate(
        label="aortic_dissection",
        score=0.7,
        rationale=["severe tearing chest pain radiating to back"],
    )
    verdict = gate_candidate_grounding(cand, findings)
    assert verdict.verdict == "pass"
    assert verdict.score == pytest.approx(1.0)
    assert verdict.unsupported_claims == []


def test_drop_when_rationale_is_unsupported():
    findings = _findings("patient reports mild headache and runny nose")
    cand = DifferentialCandidate(
        label="aortic_dissection",
        score=0.7,
        rationale=[
            "tearing chest pain radiating to back",
            "wide pulse pressure",
            "asymmetric arm blood pressure",
            "syncope on standing",
        ],
    )
    verdict = gate_candidate_grounding(cand, findings)
    assert verdict.verdict == "drop"
    assert verdict.score < 0.4
    assert verdict.unsupported_claims


def test_small_sample_demotes_instead_of_dropping():
    """Sample-size guard: with fewer than min_factual_claims_for_drop
    factual claims, a candidate with 0% grounding score is demoted, not
    dropped, to avoid false-positive drops on minimal rationales."""
    findings = _findings("patient reports mild headache and runny nose")
    cand = DifferentialCandidate(
        label="aortic_dissection",
        score=0.7,
        rationale=["tearing chest pain radiating to back"],
    )
    verdict = gate_candidate_grounding(cand, findings)
    assert verdict.verdict == "demote"


def test_conceptual_rationale_passes_without_narrative_match():
    """Conceptual fragments (mechanism names, planned tests, diagnostic
    pattern labels) must NOT be checked against the narrative; if a
    candidate's rationale is entirely conceptual, the grounding gate
    must return pass with neutral score. This prevents the Batch-2
    regression where specific dx with technical-vocabulary rationales
    were dropped while generic dx with common-language rationales
    survived."""
    findings = _findings(
        "21yo college student. drank too much at frat party. slipped on wet stairs and "
        "hit head against railing. knocked out for a minute. woke up feeling fine. "
        "this morning splitting headache, vomited four times, dizzy, left arm feels weak."
    )
    cand = DifferentialCandidate(
        label="epidural_hematoma",
        score=0.7,
        rationale=[
            "Classic post-traumatic intracranial bleed with lucid interval pattern",
            "Pattern consistent with epidural hematoma",
            "Mechanism: middle meningeal artery rupture after temporal trauma",
        ],
        evidence_needed=["CT head without contrast", "neurosurgical consult", "INR/PT"],
    )
    verdict = gate_candidate_grounding(cand, findings)
    assert verdict.verdict == "pass"
    assert verdict.score == pytest.approx(1.0)
    assert verdict.checked_claims == 0


def test_factual_rationale_passes_when_in_narrative():
    """Factual rationale fragments (symptom + timing + exposure) ARE
    checked against the narrative. When all are present, the gate
    returns pass."""
    findings = _findings(
        "21yo college student. drank too much at frat party. slipped on wet stairs and "
        "hit head against railing. knocked out for a minute. woke up feeling fine. "
        "this morning splitting headache, vomited four times, dizzy, left arm feels weak."
    )
    cand = DifferentialCandidate(
        label="epidural_hematoma",
        score=0.7,
        rationale=[
            "Recent head trauma with brief loss of consciousness",
            "Severe headache with repeated vomiting",
            "Focal left arm weakness",
        ],
    )
    verdict = gate_candidate_grounding(cand, findings)
    assert verdict.verdict == "pass"
    assert verdict.checked_claims >= 1
    assert verdict.score >= 0.6


def test_evidence_needed_does_not_penalize_grounding():
    """evidence_needed items (tests, imaging, consults) are by definition
    actions that have not happened. Including them in grounding is a
    category error that systematically penalizes specific dx whose
    workup requires imaging."""
    findings = _findings("recent head trauma with severe headache")
    cand_with_workup = DifferentialCandidate(
        label="epidural_hematoma",
        score=0.7,
        rationale=["recent head trauma with severe headache"],
        evidence_needed=["CT head", "neurosurgical consult", "INR/PT"],
    )
    cand_without_workup = DifferentialCandidate(
        label="epidural_hematoma",
        score=0.7,
        rationale=["recent head trauma with severe headache"],
        evidence_needed=[],
    )
    v1 = gate_candidate_grounding(cand_with_workup, findings)
    v2 = gate_candidate_grounding(cand_without_workup, findings)
    assert v1.verdict == v2.verdict
    assert v1.score == pytest.approx(v2.score)


def test_demote_when_partial_support():
    findings = _findings(
        "patient reports tearing chest pain",
        "no mention of vital instability",
    )
    cand = DifferentialCandidate(
        label="aortic_dissection",
        score=0.7,
        rationale=[
            "tearing chest pain",
            "shock physiology with hypotension",
        ],
    )
    verdict = gate_candidate_grounding(cand, findings)
    assert verdict.verdict in {"demote", "pass"}
    assert 0.0 < verdict.score <= 1.0


def test_no_claims_passes_neutral():
    findings = _findings("anything")
    cand = DifferentialCandidate(label="x", score=0.5, rationale=[])
    verdict = gate_candidate_grounding(cand, findings)
    assert verdict.verdict == "pass"
    assert verdict.score == pytest.approx(1.0)


def test_gate_differential_drops_dropped_candidates():
    findings = _findings("patient reports severe headache for 3 days")
    cands = [
        DifferentialCandidate(
            label="migraine",
            score=0.6,
            rationale=["severe headache for 3 days"],
        ),
        DifferentialCandidate(
            label="acute_appendicitis",
            score=0.5,
            rationale=["right lower quadrant pain", "rebound tenderness", "fever and vomiting"],
        ),
    ]
    diff = DifferentialSet(candidates=cands, uncertainty=0.5)
    updated, verdicts, pass_rate = gate_differential(diff, findings)

    surviving_labels = {c.label for c in updated.candidates}
    assert "migraine" in surviving_labels
    assert verdicts["acute_appendicitis"].verdict == "drop"
    assert "acute_appendicitis" not in surviving_labels
    assert 0.0 <= pass_rate <= 1.0


def test_gate_differential_demote_reduces_score():
    findings = _findings("patient reports tearing chest pain")
    cand = DifferentialCandidate(
        label="aortic_dissection",
        score=0.8,
        rationale=["tearing chest pain", "shock physiology", "wide pulse pressure"],
    )
    diff = DifferentialSet(candidates=[cand])
    updated, verdicts, _ = gate_differential(diff, findings)
    if verdicts["aortic_dissection"].verdict == "demote":
        assert updated.candidates[0].score < 0.8


def test_gate_differential_preserves_pass_score():
    findings = _findings("severe tearing chest pain radiating to back")
    cand = DifferentialCandidate(
        label="aortic_dissection",
        score=0.85,
        rationale=["severe tearing chest pain radiating to back"],
    )
    diff = DifferentialSet(candidates=[cand])
    updated, verdicts, _ = gate_differential(diff, findings)
    assert updated.candidates[0].score == pytest.approx(0.85)
    assert updated.candidates[0].grounding_verdict == "pass"


def test_aggregate_pass_rate():
    verdicts = [
        GroundingVerdict(verdict="pass"),
        GroundingVerdict(verdict="demote"),
        GroundingVerdict(verdict="drop"),
        GroundingVerdict(verdict="pass"),
    ]
    assert aggregate_pass_rate(verdicts) == pytest.approx(0.5)
    assert aggregate_pass_rate([]) == pytest.approx(1.0)
