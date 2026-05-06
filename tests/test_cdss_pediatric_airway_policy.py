from src.cdss.clinical.intervention_safety import build_intervention_safety_assessment
from src.cdss.clinical.safety import build_risk_profile
from src.cdss.contracts.models import HypothesisFrontier, HypothesisNode, InterventionNode, InterventionSet, PatientInput, StructuredFindings, UrgencyTier


def test_pediatric_airway_pattern_forces_emergency_risk_profile():
    patient_input = PatientInput(
        patient_text="4 year old, drooling, stridor, tripod position, high fever.",
        demographics={"age": 4, "sex": "male"},
        execution_mode="local_qwen",
    )
    findings = StructuredFindings(
        summary="Child with drooling, stridor, tripod position, and high fever.",
        positive_findings=["drooling", "stridor", "tripod positioning", "unable to swallow"],
        red_flags=["airway_compromise"],
        input_context=["blocked_order:tongue_depressor_exam"],
        planned_interventions=["tongue_depressor_exam"],
        derived_vitals={"temperature_c": 39.5, "respiratory_rate": 40, "heart_rate": 145},
    )
    risk = build_risk_profile(patient_input, findings)
    assert risk.urgency == UrgencyTier.EMERGENCY
    assert "tongue_depressor_exam" in risk.blocked_actions


def test_pediatric_airway_pattern_blocks_tongue_depressor_intervention():
    findings = StructuredFindings(
        summary="4 year old with drooling and stridor, leaning forward to breathe.",
        positive_findings=["drooling", "stridor", "forward leaning posture", "unable to swallow"],
        red_flags=["airway_compromise"],
        input_context=["blocked_order:tongue_depressor_exam"],
        planned_interventions=["tongue_depressor_exam"],
    )
    risk_profile = build_risk_profile(
        PatientInput(
            patient_text="4 year old with drooling stridor forward-leaning posture",
            demographics={"age": 4},
            execution_mode="local_qwen",
        ),
        findings,
    )
    frontier = HypothesisFrontier(
        hypotheses=[
            HypothesisNode(
                node_id="h1",
                label="acute_epiglottitis",
                canonical_label="acute_epiglottitis",
                score=0.72,
                must_not_miss=True,
                dangerous_if_missed=True,
                source="llm_free_slate",
            )
        ],
        anchor_hypothesis="acute_epiglottitis",
    )
    interventions = InterventionSet(items=[InterventionNode(node_id="i1", label="tongue_depressor_exam")])
    assessment = build_intervention_safety_assessment(
        findings=findings,
        risk_profile=risk_profile,
        frontier=frontier,
        interventions=interventions,
    )
    assert "tongue_depressor_exam" in assessment.blocked_interventions


def test_obstetric_hypertensive_crisis_blocks_ketorolac():
    findings = StructuredFindings(
        summary="32 haftalik gebe, tansiyon 175/115, visual semptom, RUQ pain.",
        positive_findings=["visual changes", "right upper quadrant pain", "edema"],
        input_context=["blocked_order:ketorolac"],
        planned_interventions=["ketorolac"],
        red_flags=["pregnancy_related_emergency"],
        demographics={"age": 24},
        derived_vitals={"sbp": 175, "dbp": 115},
    )
    patient_input = PatientInput(
        patient_text="32 haftalik gebe visual semptom RUQ pain 175/115",
        demographics={"age": 24, "sex": "female"},
        vitals={"sbp": 175, "dbp": 115},
        execution_mode="local_qwen",
    )
    frontier = HypothesisFrontier(
        hypotheses=[
            HypothesisNode(
                node_id="h1",
                label="severe_preeclampsia",
                canonical_label="severe_preeclampsia",
                score=0.8,
                must_not_miss=True,
                dangerous_if_missed=True,
                source="llm_free_slate",
            )
        ],
        anchor_hypothesis="severe_preeclampsia",
    )
    interventions = InterventionSet(items=[InterventionNode(node_id="i1", label="ketorolac")])
    assessment = build_intervention_safety_assessment(
        findings=findings,
        risk_profile=build_risk_profile(patient_input, findings),
        frontier=frontier,
        interventions=interventions,
        patient_input=patient_input,
    )
    assert "ketorolac" in assessment.blocked_interventions


def test_hit_paradox_blocks_heparin_products():
    findings = StructuredFindings(
        summary="Heparin sonrası 6. gün platelet 250000 -> 110000 ve yeni arterial tromboz.",
        positive_findings=["new arterial thrombosis", "platelet fall >50%"],
        input_context=["blocked_order:all_heparin_products"],
        planned_interventions=["heparin"],
    )
    patient_input = PatientInput(
        patient_text="Heparin 6 gun sonra platelet 250000 den 110000 dustu ve tromboz oldu",
        demographics={"age": 55, "sex": "male"},
        labs={"platelet_baseline": 250000, "platelet_current": 110000},
        execution_mode="local_qwen",
    )
    assessment = build_intervention_safety_assessment(
        findings=findings,
        risk_profile=build_risk_profile(patient_input, findings),
        frontier=HypothesisFrontier(),
        interventions=InterventionSet(items=[InterventionNode(node_id="i1", label="heparin")]),
        patient_input=patient_input,
    )
    assert "all_heparin_products" in assessment.blocked_interventions
