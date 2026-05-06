from src.cdss.runtime.canonical_state import build_canonical_state, sanitize_string_list


def test_sanitize_string_list_drops_placeholder_artifacts():
    values = ["exclusions_semi_colon", "None specified", " severe_anemia_severity ", "severe_anemia_severity"]
    assert sanitize_string_list(values) == ["severe anemia severity"]


def test_canonical_state_promotes_source_candidate_and_requires_underlying_resolution():
    state = build_canonical_state(
        anchor="iron_deficiency_anemia",
        anchor_role="manifestation",
        source_disease_resolved=False,
        phenotype_candidates=[{"diagnosis": "iron_deficiency_anemia", "confidence": 0.92}],
        source_disease_candidates=[{"diagnosis": "occult_gastrointestinal_blood_loss", "confidence": 0.71}],
        linked_evidence=[],
        required_data=["objective_vitals"],
        objective_discriminators=["objective_discriminator_for_iron_deficiency_anemia"],
        must_not_miss=[],
        contraindications=[],
        urgency="routine",
    )
    assert state.resolved_anchor == "occult gastrointestinal blood loss"
    assert state.anchor_role == "source_disease"
    assert "underlying_cause_resolution" in state.required_data
    assert "objective discriminator for iron deficiency anemia" in state.required_data
