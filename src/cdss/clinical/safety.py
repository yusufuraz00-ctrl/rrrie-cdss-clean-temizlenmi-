"""Typed-context safety policy built from objective data and open-ended LLM context."""

from __future__ import annotations

from typing import Any

from src.cdss.contracts.models import PatientInput, RiskProfile, StructuredFindings, UrgencyTier
from src.cdss.core import thresholds as clinical_thresholds
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


def _coerce_float(value: Any) -> float | None:
    try:
        text = str(value).strip().replace(",", ".").replace("%", "")
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError, AttributeError):
        return None


def _merge_vitals(patient_input: PatientInput, findings: StructuredFindings) -> dict[str, Any]:
    merged = dict(findings.derived_vitals or {})
    for key, value in (patient_input.vitals or {}).items():
        if value in (None, ""):
            continue
        merged[str(key)] = value
    return merged


def _compute_vital_alerts(vitals: dict[str, Any]) -> list[str]:
    alerts: list[str] = []
    heart_rate = _coerce_float(vitals.get("heart_rate") or vitals.get("hr") or vitals.get("pulse"))
    respiratory_rate = _coerce_float(vitals.get("respiratory_rate") or vitals.get("rr"))
    spo2 = _coerce_float(vitals.get("spo2") or vitals.get("o2_sat"))
    systolic_bp = _coerce_float(vitals.get("sbp") or vitals.get("systolic_bp"))
    temperature = _coerce_float(vitals.get("temperature_c") or vitals.get("temp_c"))

    spo2_critical = clinical_thresholds.get_float("vitals.spo2_critical_lt", 92.0)
    sbp_shock = clinical_thresholds.get_float("vitals.sbp_shock_lt", 90.0)
    sbp_borderline = clinical_thresholds.get_float("vitals.sbp_borderline_lt_eq", 100.0)
    sbp_severe_htn = clinical_thresholds.get_float("vitals.sbp_severe_htn_gte", 180.0)
    hr_marked = clinical_thresholds.get_float("vitals.hr_marked_tachy_gt", 130.0)
    hr_tachy = clinical_thresholds.get_float("vitals.hr_tachy_gte", 100.0)
    rr_marked = clinical_thresholds.get_float("vitals.rr_marked_tachypnea_gt", 30.0)
    rr_tachy = clinical_thresholds.get_float("vitals.rr_tachypnea_gte", 22.0)
    temp_hyperpyrexia = clinical_thresholds.get_float("vitals.temp_hyperpyrexia_gte", 40.0)

    if spo2 is not None and spo2 < spo2_critical:
        alerts.append(f"critical_hypoxemia_spo2_{int(spo2)}")
    if systolic_bp is not None and systolic_bp < sbp_shock:
        alerts.append(f"shock_physiology_sbp_{int(systolic_bp)}")
    elif systolic_bp is not None and systolic_bp <= sbp_borderline:
        alerts.append(f"borderline_hypotension_sbp_{int(systolic_bp)}")
    if systolic_bp is not None and systolic_bp >= sbp_severe_htn:
        alerts.append(f"severe_hypertension_sbp_{int(systolic_bp)}")
    if heart_rate is not None and heart_rate > hr_marked:
        alerts.append(f"marked_tachycardia_hr_{int(heart_rate)}")
    elif heart_rate is not None and heart_rate >= hr_tachy:
        alerts.append(f"tachycardia_hr_{int(heart_rate)}")
    if respiratory_rate is not None and respiratory_rate > rr_marked:
        alerts.append(f"marked_tachypnea_rr_{int(respiratory_rate)}")
    elif respiratory_rate is not None and respiratory_rate >= rr_tachy:
        alerts.append(f"tachypnea_rr_{int(respiratory_rate)}")
    if temperature is not None and temperature >= temp_hyperpyrexia:
        alerts.append(f"hyperpyrexia_temp_{temperature:g}")
    return alerts


def _objective_instability(vital_alerts: list[str], findings: StructuredFindings) -> bool:
    if vital_alerts:
        return True
    red_flag_pool = " ".join([*findings.red_flags, *findings.positive_findings]).lower()
    return any(term in red_flag_pool for term in ("shock", "unstable", "collapse", "unresponsive", "severe_bleeding"))


def _signal_text(patient_input: PatientInput, findings: StructuredFindings) -> str:
    return ascii_fold(
        " ".join(
            [
                str(patient_input.patient_text or ""),
                str(findings.summary or ""),
                *[str(item or "") for item in findings.positive_findings[:20]],
                *[str(item or "") for item in findings.red_flags[:12]],
                *[str(item or "") for item in findings.timeline[:10]],
                *[str(item or "") for item in findings.exposures[:10]],
                *[str(item or "") for item in findings.raw_segments[:14]],
            ]
        )
    ).lower()


def _intervention_text(findings: StructuredFindings) -> str:
    return ascii_fold(" ".join(str(item or "") for item in findings.planned_interventions[:12])).lower()


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def _time_sensitive_hazard_markers(patient_input: PatientInput, findings: StructuredFindings) -> list[str]:
    """Infer general time-sensitive hazard classes without diagnosis shortcuts."""

    text = _signal_text(patient_input, findings)
    markers: list[str] = []
    rapid = _has_any(text, ("sudden", "rapid", "progress", "worsen", "spread", "hour", "acute onset"))
    severe = _has_any(text, ("severe", "10/10", "excruciat", "agoniz", "worst", "scream", "cannot", "unable"))
    functional_loss = _has_any(text, ("cannot walk", "cannot stand", "fall", "weakness", "paraly", "unable to move", "can't move"))
    neuro_or_sensory = _has_any(text, ("confus", "altered", "numb", "weak", "hoarse", "vision", "blind", "curtain", "seizure"))
    perfusion_or_tissue = _has_any(text, ("cold", "pale", "purple", "grey", "gray", "black", "blister", "bull", "necros", "bleed"))
    systemic_instability = _has_any(text, ("fever", "chills", "dizzy", "lightheaded", "letharg", "sleepy", "tachy", "vomit"))
    catastrophic_pain_context = _has_any(text, ("tearing", "crackling", "crunchy", "rigid", "out of proportion"))

    if rapid:
        markers.append("rapid_progression")
    if severe:
        markers.append("severe_symptom_burden")
    if functional_loss:
        markers.append("functional_or_motor_loss")
    if neuro_or_sensory:
        markers.append("neurologic_or_sensory_risk")
    if perfusion_or_tissue:
        markers.append("perfusion_or_tissue_loss_risk")
    if systemic_instability:
        markers.append("systemic_instability")
    if catastrophic_pain_context:
        markers.append("catastrophic_pain_context")

    high_risk_patterns = (
        (rapid and severe and (functional_loss or neuro_or_sensory or perfusion_or_tissue))
        or (severe and catastrophic_pain_context and (systemic_instability or functional_loss or perfusion_or_tissue))
        or (perfusion_or_tissue and (functional_loss or systemic_instability))
        or (neuro_or_sensory and functional_loss and rapid)
    )
    if high_risk_patterns:
        markers.append("time_sensitive_permanent_harm_risk")
    return _unique(markers)


def _unsafe_self_treatment_markers(findings: StructuredFindings) -> list[str]:
    text = _intervention_text(findings)
    if not text:
        return []
    delay_or_home_terms = (
        "wait",
        "tomorrow",
        "weekend",
        "sleep",
        "rest",
        "home",
        "wrap",
        "ointment",
        "massage",
        "deep tissue",
        "hot water",
        "exercise",
        "antacid",
        "double",
        "increase dose",
    )
    if _has_any(text, delay_or_home_terms):
        return ["unsafe_delay_or_home_treatment_plan"]
    return []


def _base_required_actions(ctx: dict[str, list[str]]) -> list[str]:
    return _unique([*ctx.get("required_action", [])])


def build_risk_profile(patient_input: PatientInput, findings: StructuredFindings) -> RiskProfile:
    """Build a typed risk profile from objective vitals and LLM-provided typed context."""
    effective_vitals = _merge_vitals(patient_input, findings)
    vital_alerts = _compute_vital_alerts(effective_vitals)
    demographic_alerts: list[str] = []
    ctx = _ctx_map(findings)
    hazard_markers = _time_sensitive_hazard_markers(patient_input, findings)
    unsafe_plan_markers = _unsafe_self_treatment_markers(findings)

    escalation_reasons = _unique(
        [
            *findings.red_flags,
            *vital_alerts,
            *demographic_alerts,
            *hazard_markers,
            *ctx.get("hazard", []),
            *ctx.get("device_reliability", []),
            *ctx.get("causal_loop", []),
            *ctx.get("masquerade_risk", []),
            *ctx.get("contradiction", []),
        ]
    )
    blocked_actions = _unique(
        [
            *ctx.get("blocked_order", []),
            *unsafe_plan_markers,
        ]
    )
    workflow_guards = _unique([*ctx.get("workflow_guard", [])])
    required_actions = _base_required_actions(ctx)

    urgency = UrgencyTier.ROUTINE
    if (
        _objective_instability(vital_alerts, findings)
        or "time_sensitive_permanent_harm_risk" in hazard_markers
        or (unsafe_plan_markers and len(hazard_markers) >= 3)
        or ctx.get("hazard")
        or ctx.get("causal_loop")
        or ctx.get("device_reliability")
    ):
        urgency = UrgencyTier.EMERGENCY
    elif escalation_reasons or ctx.get("pathway_fit"):
        urgency = UrgencyTier.URGENT

    if urgency == UrgencyTier.EMERGENCY:
        workflow_guards = _unique([*workflow_guards, "unsupervised_discharge", "delay_objective_workup"])
        required_actions = _unique(
            [
                *required_actions,
                "urgent_clinician_review",
                "repeat_and_verify_vitals",
                "obtain_objective_testing",
            ]
        )
    elif urgency == UrgencyTier.URGENT:
        workflow_guards = _unique([*workflow_guards, "low_information_finalization"])
        required_actions = _unique(
            [
                *required_actions,
                "same_day_clinician_assessment",
                "targeted_objective_workup",
            ]
        )
    else:
        required_actions = _unique([*required_actions, "collect_missing_context_if_symptoms_persist"])

    manual_review_required = urgency != UrgencyTier.ROUTINE or bool(blocked_actions) or bool(workflow_guards)
    score = 0.1
    score += min(0.45, len(escalation_reasons) * 0.1)
    score += min(0.2, len(required_actions) * 0.03)
    score += 0.12 if urgency == UrgencyTier.EMERGENCY else (0.06 if urgency == UrgencyTier.URGENT else 0.0)
    score = max(0.0, min(1.0, round(score, 2)))

    return RiskProfile(
        urgency=urgency,
        score=score,
        escalation_reasons=escalation_reasons[:10],
        blocked_actions=blocked_actions,
        workflow_guards=workflow_guards,
        required_actions=required_actions,
        vital_alerts=vital_alerts,
        manual_review_required=manual_review_required,
    )
