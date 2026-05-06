from src.cdss.clinical.confirmation_planner import plan_objective_confirmation
from src.cdss.clinical.diagnosis_normalization import normalize_differential
from src.cdss.clinical.finding_fit import score_candidate_finding_fit
from src.cdss.clinical.grounding import validate_narrative_grounding
from src.cdss.clinical.mechanism_fit import score_mechanism_completeness
from src.cdss.clinical.safety import build_risk_profile
from src.cdss.clinical.verification import build_verification_report
from src.cdss.contracts.models import (
    DifferentialCandidate,
    DifferentialSet,
    EvidenceBundle,
    ExplanationGraph,
    ExplanationLink,
    PatientInput,
    RiskProfile,
    StructuredFindings,
    UrgencyTier,
)


def test_grounding_validator_rejects_unstated_negative_finding():
    findings = StructuredFindings(
        summary="Sudden severe lower abdominal pain with dizziness after standing.",
        positive_findings=["sudden severe lower abdominal pain", "dizziness when standing"],
        negative_findings=["no chest pain"],
        raw_segments=["Sudden severe lower abdominal pain with dizziness after standing."],
    )

    report = validate_narrative_grounding(findings)

    assert report.hallucination_risk_score > 0
    assert any(issue.field == "negative_findings" for issue in report.issues)


def test_diagnosis_normalization_merges_near_duplicate_typos_without_rule_table():
    differential = DifferentialSet(
        candidates=[
            DifferentialCandidate(label="acute_appendicitis", score=0.62, evidence_needed=["ct"]),
            DifferentialCandidate(label="acute_appendedicitis", score=0.58, evidence_needed=["cbc"]),
            DifferentialCandidate(label="acute_perforated_viscus", score=0.70),
        ]
    )

    result = normalize_differential(differential)

    labels = [candidate.label for candidate in result.differential.candidates]
    assert labels.count("acute_appendicitis") == 1
    assert "acute_appendedicitis" not in labels
    assert result.absorbed["acute_appendicitis"] == ["acute_appendedicitis"]


def test_finding_fit_penalizes_critical_orphan_findings():
    findings = StructuredFindings(
        positive_findings=[
            "self-limited nausea after food exposure",
            "orthostatic dizziness",
            "sudden severe abdominal pain",
            "referred shoulder pain",
        ],
        red_flags=["orthostatic dizziness", "sudden severe pain"],
    )
    weak = DifferentialCandidate(
        label="benign_gastrointestinal_irritation",
        rationale=["Food exposure explains nausea but does not explain hemodynamic instability."],
    )
    stronger = DifferentialCandidate(
        label="acute_hemodynamic_abdominal_process",
        rationale=[
            "Acute abdominal process explains sudden severe abdominal pain, orthostatic dizziness, and referred pain through internal irritation and instability."
        ],
    )

    weak_fit = score_candidate_finding_fit(weak, findings)
    strong_fit = score_candidate_finding_fit(stronger, findings)

    assert weak_fit.critical_unexplained
    assert weak_fit.orphan_penalty > strong_fit.orphan_penalty
    assert strong_fit.evidence_fit_score > weak_fit.evidence_fit_score


def test_mechanism_completeness_requires_timing_severity_and_confirmation():
    findings = StructuredFindings(
        positive_findings=[
            "sudden onset severe pain",
            "rapid progression over hours",
            "objective weakness",
        ],
        red_flags=["rapid progression", "objective weakness"],
        timeline=["started one hour ago", "worsened rapidly"],
        exposures=["recent medication change"],
    )
    candidate = DifferentialCandidate(
        label="acute_toxic_or_vascular_process",
        rationale=[
            "A toxic or vascular mechanism can connect recent medication exposure, sudden timing, rapid progression, severe pain, and objective weakness."
        ],
        evidence_needed=["urgent exam", "targeted labs", "imaging when indicated"],
    )

    score = score_mechanism_completeness(candidate, findings, RiskProfile(urgency=UrgencyTier.EMERGENCY))

    assert score.score >= 0.70
    assert not score.closure_blockers
    assert {"timing", "severity", "confirmation"}.issubset(set(score.satisfied_axes))


def test_safety_profile_escalates_time_sensitive_permanent_harm_pattern():
    patient = PatientInput(patient_text="rapidly progressive severe pain with new weakness")
    findings = StructuredFindings(
        positive_findings=[
            "rapidly progressive severe pain",
            "new weakness",
            "cannot walk",
        ],
        timeline=["worsened in 4 hours"],
        red_flags=["pain out of proportion", "new weakness"],
        planned_interventions=["wait until tomorrow"],
    )

    risk = build_risk_profile(patient, findings)

    assert risk.urgency == UrgencyTier.EMERGENCY
    assert "delay_objective_workup" in risk.workflow_guards
    assert "obtain_objective_testing" in risk.required_actions


def test_safety_profile_does_not_escalate_low_risk_symptoms_by_presence_alone():
    patient = PatientInput(patient_text="sore throat, runny nose, and dry cough for three days")
    findings = StructuredFindings(
        positive_findings=["sore throat", "runny nose", "dry cough"],
        timeline=["three days"],
    )

    risk = build_risk_profile(patient, findings)

    assert risk.urgency == UrgencyTier.ROUTINE
    assert not risk.manual_review_required


def test_confirmation_planner_requires_objective_discriminators_for_unresolved_critical_findings():
    findings = StructuredFindings(red_flags=["new neurologic deficit"], positive_findings=["cannot stand"])
    differential = DifferentialSet(
        candidates=[
            DifferentialCandidate(label="central_neurologic_process", score=0.68),
            DifferentialCandidate(label="peripheral_symptom_explanation", score=0.61),
        ]
    )
    graph = ExplanationGraph(
        links=[
            ExplanationLink(
                finding="cannot stand",
                explanation_type="still_unexplained",
                critical=True,
            )
        ],
        critical_unexplained_count=1,
        unexplained_findings=["cannot stand"],
    )

    plan = plan_objective_confirmation(findings, differential, graph, RiskProfile(urgency=UrgencyTier.EMERGENCY))

    assert plan.required
    assert plan.evidence_needs
    assert any("objective" in need.objective for need in plan.evidence_needs)
    assert "cannot stand" in " ".join(plan.blocking_findings)


def test_verification_report_blocks_hallucinated_negatives_and_missing_confirmation():
    findings = StructuredFindings(
        summary="Sudden severe pain with inability to stand.",
        positive_findings=["sudden severe pain", "inability to stand"],
        negative_findings=["no chest pain"],
        raw_segments=["Sudden severe pain with inability to stand."],
        red_flags=["inability to stand"],
    )
    differential = DifferentialSet(
        candidates=[DifferentialCandidate(label="acute_instability_process", score=0.72)],
        uncertainty=0.28,
    )
    graph = ExplanationGraph(
        links=[ExplanationLink(finding="inability to stand", explanation_type="still_unexplained", critical=True)],
        critical_unexplained_count=1,
        unexplained_findings=["inability to stand"],
    )

    report = build_verification_report(
        findings=findings,
        risk_profile=RiskProfile(urgency=UrgencyTier.EMERGENCY),
        differential=differential,
        evidence=EvidenceBundle(coverage=0.4),
        explanation_graph=graph,
    )

    issue_types = {issue.issue_type for issue in report.issues}
    assert "grounding_hallucination_risk" in issue_types
    assert "objective_confirmation_required" in issue_types
    assert report.decision in {"revise", "urgent_escalation"}
