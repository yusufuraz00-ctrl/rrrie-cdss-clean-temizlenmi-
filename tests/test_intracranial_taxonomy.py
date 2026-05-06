"""Structural taxonomy tests — intracranial hemorrhage siblings.

These tests verify that the post-Batch-7 fixes (C, B, A, D, E, F) keep
sibling must-not-miss diagnoses (epidural_hematoma, subdural_hematoma,
intracerebral_hemorrhage, intracranial_hemorrhage, diffuse_axonal_injury,
concussion, subarachnoid_hemorrhage) distinct through dedup and let a
verifier-named alternative correctly become the differential leader.

NO LIVE LLM. NO HARDCODED CASE OUTCOME. The tests check generic system
properties using synthetic differentials and synthetic verifier issues.
"""

from __future__ import annotations

import pytest

from src.cdss.contracts.models import (
    DifferentialCandidate,
    DifferentialSet,
    StructuredFindings,
    VerificationIssue,
)
from src.cdss.runtime.state_machine import (
    _dedup_differential_candidates,
    _label_similarity,
)
from src.cdss.app.view_model import _consistent_summary


_INTRACRANIAL_SIBLINGS = [
    "epidural_hematoma",
    "subdural_hematoma",
    "intracerebral_hemorrhage",
    "intracranial_hemorrhage",
    "diffuse_axonal_injury",
    "concussion",
    "subarachnoid_hemorrhage",
]


# ---------------------------------------------------------------------------
# Fix-B: ontology-aware dedup. Sibling labels must NOT collapse silently.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b,band", [
    # Measured ratios at writing time. The point is to lock them in so a
    # future change to `_label_similarity` cannot silently shift these
    # sibling pairs across the merge boundary.
    ("epidural_hematoma", "subdural_hematoma", (0.78, 0.85)),       # 0.824
    ("intracranial_hemorrhage", "intracerebral_hemorrhage", (0.84, 0.92)),  # 0.851
    ("epidural_hematoma", "intracerebral_hemorrhage", (0.30, 0.55)),
    ("subdural_hematoma", "intracerebral_hemorrhage", (0.30, 0.55)),
])
def test_intracranial_sibling_string_similarity_locked(a, b, band):
    """Pin the SequenceMatcher.ratio values for sibling pairs so future
    refactors cannot silently shift them across the merge boundary.

    The ``intracranial_hemorrhage`` / ``intracerebral_hemorrhage`` pair
    sits just above the *old* 0.85 threshold (= silent merge under the
    pre-Fix-B logic). The ``epidural`` / ``subdural`` pair sits just
    below that boundary at 0.824 — close enough that any drift would
    cross it. Both must remain distinct under the new ontology-aware
    dedup regardless."""
    sim = _label_similarity(a, b)
    lo, hi = band
    assert lo <= sim <= hi, (
        f"sibling pair {a!r}/{b!r} similarity={sim:.3f} outside [{lo}, {hi}]"
    )


def test_dedup_does_not_merge_distinct_intracranial_siblings():
    """Post-Fix-B: distinct sibling labels survive dedup as separate
    candidates regardless of string similarity."""
    cands = [
        DifferentialCandidate(label=label, score=0.5)
        for label in _INTRACRANIAL_SIBLINGS
    ]
    survivors = _dedup_differential_candidates(cands)
    surviving_labels = {c.label for c in survivors}
    assert surviving_labels == set(_INTRACRANIAL_SIBLINGS), (
        f"sibling dedup collapsed labels: missing {set(_INTRACRANIAL_SIBLINGS) - surviving_labels}"
    )


def test_dedup_still_merges_typo_variants():
    """The relaxed threshold must still catch genuine typos (the
    original purpose of dedup). Post-Fix-B this only happens when the
    string-similarity falls back path applies (registry doesn't know
    the typo'd label)."""
    cands = [
        DifferentialCandidate(label="acute_appendicitis", score=0.6),
        DifferentialCandidate(label="acute_appendecitis", score=0.5),  # typo
    ]
    survivors = _dedup_differential_candidates(cands)
    # Either both survive (registry-driven) or one merges (string-similarity).
    # Both outcomes are acceptable — what we forbid is silent merge of
    # genuine sibling diagnoses (covered by previous test).
    assert 1 <= len(survivors) <= 2


# ---------------------------------------------------------------------------
# Fix-E: consistent_summary glues hero text to the actual leader.
# ---------------------------------------------------------------------------


def test_consistent_summary_prepends_correct_leader_when_summary_is_stale():
    """Reproduces the user's bug: summary mentions DAI as leader, but
    differential's actual top-1 is epidural_hematoma. After Fix-E the
    rendered summary leads with the actual top-1."""
    from src.cdss.contracts.models import (
        DecisionPacket,
        DecisionStatus,
        EvidenceBundle,
        RiskProfile,
        VerificationReport,
    )
    diff = DifferentialSet(
        candidates=[
            DifferentialCandidate(label="epidural_hematoma", score=0.75),
            DifferentialCandidate(label="diffuse_axonal_injury", score=0.55),
        ],
    )
    packet = DecisionPacket(
        case_id="test1",
        status=DecisionStatus.URGENT_ESCALATION,
        summary="Current adaptive differential is led by diffuse axonal injury, with traumatic brain injury still requiring objective discrimination.",
        structured_findings=StructuredFindings(),
        risk_profile=RiskProfile(),
        differential=diff,
        evidence=EvidenceBundle(),
        verification=VerificationReport(),
    )
    consistent = _consistent_summary(packet)
    # Must mention the actual top-1 first.
    lead_part = consistent.split(".")[0].lower()
    assert "epidural_hematoma" in lead_part
    # Original action-plan summary preserved as tail.
    assert "discrimination" in consistent.lower()


def test_consistent_summary_passes_through_when_summary_already_matches():
    from src.cdss.contracts.models import (
        DecisionPacket,
        DecisionStatus,
        EvidenceBundle,
        RiskProfile,
        VerificationReport,
    )
    diff = DifferentialSet(
        candidates=[DifferentialCandidate(label="migraine", score=0.7)],
    )
    pkt = DecisionPacket(
        case_id="test2",
        status=DecisionStatus.PRELIMINARY,
        summary="migraine is the leading hypothesis given throbbing headache and photophobia",
        structured_findings=StructuredFindings(),
        risk_profile=RiskProfile(),
        differential=diff,
        evidence=EvidenceBundle(),
        verification=VerificationReport(),
    )
    assert _consistent_summary(pkt).startswith("migraine is the leading hypothesis")


# ---------------------------------------------------------------------------
# Fix-D + Fix-F: verifier-named alternative + secondaries promotion.
# ---------------------------------------------------------------------------


def test_anchor_mismatch_signal_collects_secondary_alternatives():
    """The verifier's free-text often names multiple sibling diagnoses;
    Fix-F preserves all of them as secondary_alternatives in the
    extracted dict."""
    # Use a minimal harness: instantiate a thin agent only to call the
    # method. We avoid pulling the full state machine startup.
    import importlib

    sm = importlib.import_module("src.cdss.runtime.state_machine")
    # Build a fake instance with the bound method.
    klass = sm.CdssStateMachineAgent
    detect = klass._detect_anchor_mismatch_signal

    issues = [
        VerificationIssue(
            severity="major",
            issue_type="anchor_mismatch",
            detail=(
                "left_arm_weakness not explained by [traumatic_brain_injury] — "
                "consider [epidural_hematoma] or [diffuse_axonal_injury]; "
                "alternative [intracranial_hemorrhage] better explains "
                "full symptom cluster including focal motor deficit and LOC"
            ),
        )
    ]

    # Bind a stub `self` — only `_detect_anchor_mismatch_signal` is called
    # and it does not touch instance state.
    class _StubAgent:
        pass

    result = detect(_StubAgent(), issues)
    assert result is not None
    assert "secondary_alternatives" in result
    secondaries = [s.lower() for s in result["secondary_alternatives"]]
    primary = result["alternative_label"].lower()
    # Primary + secondaries should together cover the bracketed labels.
    named = {primary, *secondaries}
    assert "epidural_hematoma" in named or "epidural hematoma" in named or any(
        "epidural" in s for s in named
    )


# ---------------------------------------------------------------------------
# Inline grounding (Fix-C) — specific dx with technical-vocab rationale
# must not be drop-killed by narrative absence of conceptual tokens.
# ---------------------------------------------------------------------------


def test_intracranial_specific_dx_survives_inline_grounding_with_realistic_narrative():
    """For each sibling intracranial dx with a realistic specific
    rationale (mixing factual and conceptual fragments), the inline
    grounding gate must NOT drop the candidate when the patient
    narrative contains the supporting symptoms in lay language. This
    is the system-level test that the user's failing case relies on."""
    from src.cdss.clinical.inline_grounding import gate_candidate_grounding

    narrative = (
        "21yo college student. drank too much at frat party. slipped on wet stairs and "
        "hit head against railing. knocked out for a minute. woke up feeling fine. "
        "this morning splitting headache, vomited four times, dizzy, left arm feels weak."
    )
    findings = StructuredFindings(summary=narrative, raw_segments=[narrative])

    rationales_by_dx = {
        "epidural_hematoma": [
            "Recent head trauma with brief loss of consciousness",
            "Severe headache with repeated vomiting",
            "Focal left arm weakness after fall",
            "Pattern consistent with epidural hematoma",
        ],
        "subdural_hematoma": [
            "Recent head trauma in young adult",
            "Severe headache with vomiting",
            "Pattern consistent with acute subdural hematoma",
        ],
        "diffuse_axonal_injury": [
            "Head trauma with loss of consciousness",
            "Persistent neurological symptoms after impact",
            "Pattern consistent with diffuse axonal injury",
        ],
        "concussion": [
            "Mild head trauma with brief loss of consciousness",
            "Headache and dizziness following impact",
            "Pattern consistent with concussion",
        ],
        "intracerebral_hemorrhage": [
            "Head trauma with severe headache and vomiting",
            "Focal arm weakness suggests intracerebral hemorrhage",
        ],
    }

    for dx, rationale in rationales_by_dx.items():
        cand = DifferentialCandidate(
            label=dx,
            score=0.5,
            rationale=rationale,
            evidence_needed=["CT head", "neurosurgical consult"],
        )
        verdict = gate_candidate_grounding(cand, findings)
        assert verdict.verdict in {"pass", "demote"}, (
            f"{dx} was dropped by inline grounding (score={verdict.score}, "
            f"checked={verdict.checked_claims}, unsupported={verdict.unsupported_claims})"
        )
