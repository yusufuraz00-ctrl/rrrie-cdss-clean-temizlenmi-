"""Phase-1 extractor agent for the rebuilt DDx + Safety runtime."""

from __future__ import annotations

import os
import re
from typing import Any

from src.cdss.clinical.phenotype import build_context_lanes, compile_phenotype_atoms, compile_phenotype_fingerprint
from src.cdss.contracts.models import DecisionTrace, PatientInput, StructuredFindings
from src.cdss.core.state import StatePatch
from src.cdss.runtime.llm_bridge import LocalLlmClinicalExtractor
from src.cdss.runtime.policy import load_runtime_policy
from src.cdss.text_normalization import ascii_fold, normalize_clinical_text


def _normalize_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _split_surface_segments(text: str) -> list[str]:
    normalized = _normalize_text(text)
    raw = re.split(r"(?<=[\.\?!;])\s+|\n+", normalized)
    segments: list[str] = []
    seen: set[str] = set()
    for item in raw:
        cleaned = _normalize_text(item)
        if len(cleaned) > 140 and "," in cleaned:
            for clause in cleaned.split(","):
                clause_clean = _normalize_text(clause)
                key = clause_clean.lower()
                if len(clause_clean) < 3 or key in seen:
                    continue
                seen.add(key)
                segments.append(clause_clean)
                if len(segments) >= 16:
                    return segments
        key = cleaned.lower()
        if len(cleaned) < 3 or key in seen:
            continue
        seen.add(key)
        segments.append(cleaned)
    return segments[:16]


def _extract_timeline(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    number_token = r"(?:\d+(?:\.\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    patterns = [
        rf"\b{number_token}\s*(?:minutes?|hours?|days?|weeks?|months?)\s*(?:ago|earlier|prior|before)\b",
        rf"\bfor\s+(?:the\s+last\s+)?{number_token}\s*(?:minutes?|hours?|days?|weeks?|months?)\b",
        r"\b(?:today|yesterday|tonight|this morning|this afternoon|this evening|overnight)\b",
        r"\b(?:after|following|since|during|before)\s+[^.;]{3,80}\b",
        r"\b(?:sudden onset|suddenly|acute onset|abruptly|gradually|progressively)\b",
    ]
    results: list[str] = []
    for segment in _split_surface_segments(normalized):
        lowered = segment.lower()
        if any(re.search(pattern, lowered, flags=re.IGNORECASE) for pattern in patterns):
            results.append(segment[:120])
    return _merge_unique(results, [], 8)


def _extract_uncertainty_markers(text: str) -> list[str]:
    normalized = _normalize_text(text).lower()
    markers: list[str] = []
    cue_map = {
        "possible_or_maybe_language": r"\b(?:maybe|might|could|possibly|perhaps|seems|appears)\b",
        "patient_expressed_fear_or_speculation": r"\b(?:worried|afraid|fear|concerned|i think|i am not sure|not sure|could be|may be)\b",
        "conditional_plan_language": r"\bif\b",
        "colloquial_distress_intensity": r"[\"'].*?[\"']|!{1,}",
    }
    for label, pattern in cue_map.items():
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            markers.append(label)
    return _merge_unique(markers, [], 6)





def _merge_unique(base: list[str], extra: list[str], limit: int) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for source in (base, extra):
        for item in source:
            cleaned = _normalize_text(item)
            key = cleaned.lower()
            if not cleaned or key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
            if len(merged) >= limit:
                return merged
    return merged


def _llm_primary_merge(
    llm_items: list[str],
    heuristic_items: list[str],
    limit: int,
    *,
    llm_primary: bool,
) -> list[str]:
    llm_clean = _merge_unique(list(llm_items or []), [], limit)
    if llm_primary and llm_clean:
        return llm_clean
    return _merge_unique(llm_clean, list(heuristic_items or []), limit)


def _coerce_number(value: str) -> float | int | None:
    try:
        numeric = float(str(value).strip())
    except (TypeError, ValueError, AttributeError):
        return None
    return int(numeric) if numeric.is_integer() else numeric


def _extract_vitals_from_text(text: str) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    source = ascii_fold(str(text or ""))

    bp_match = re.search(
        r"(?:blood pressure|bp|tansiyon)\s*[:=]?\s*(\d{2,3})\s*/\s*(\d{2,3})",
        source,
        re.IGNORECASE,
    )
    if bp_match:
        derived["sbp"] = int(bp_match.group(1))
        derived["dbp"] = int(bp_match.group(2))

    hr_match = re.search(
        r"(?:heart rate|hr|pulse|nabiz)\s*[:=]?\s*(\d{2,3})",
        source,
        re.IGNORECASE,
    )
    if hr_match:
        derived["heart_rate"] = int(hr_match.group(1))

    temp_match = re.search(
        r"(?:temperature|temp|ates)\s*[:=]?\s*(\d{2}(?:\.\d)?)",
        source,
        re.IGNORECASE,
    )
    if temp_match:
        derived["temperature_c"] = float(temp_match.group(1))

    spo2_match = re.search(
        r"(?:spo2|o2 sat|oxygen saturation)\s*[:=]?\s*%?\s*(\d{2,3})",
        source,
        re.IGNORECASE,
    )
    if spo2_match:
        derived["spo2"] = int(spo2_match.group(1))

    rr_match = re.search(
        r"(?:respiratory rate|rr|solunum)\s*[:=]?\s*(\d{1,2})",
        source,
        re.IGNORECASE,
    )
    if rr_match:
        derived["respiratory_rate"] = int(rr_match.group(1))

    return derived


def _derive_vital_findings(derived_vitals: dict[str, Any]) -> list[str]:
    positives: list[str] = []
    hr = _coerce_number(derived_vitals.get("heart_rate"))
    sbp = _coerce_number(derived_vitals.get("sbp"))
    dbp = _coerce_number(derived_vitals.get("dbp"))
    temp = _coerce_number(derived_vitals.get("temperature_c"))
    rr = _coerce_number(derived_vitals.get("respiratory_rate"))

    # System-calculated Facts (10x Anti-Hallucination)
    if hr is not None and hr <= 60:
        positives.append("bradycardia")
    if hr is not None and hr >= 100:
        positives.append("tachycardia")
    if sbp is not None and sbp <= 100:
        positives.append("borderline hypotension")
    if sbp is not None and sbp <= 90:
        positives.append("hypotension_emergency")

    # Shock Index Calculation
    if hr and sbp:
        shock_index = hr / sbp
        if shock_index >= 0.9:
            positives.append(f"SYSTEM_ALERT: SHOCK_INDEX_CRITICAL({shock_index:.2f})")

    # MAP Calculation
    if sbp and dbp:
        map_val = dbp + (sbp - dbp) / 3
        if map_val < 65:
            positives.append(f"SYSTEM_ALERT: LOW_MAP_PRESSURE({map_val:.1f})")

    # SIRS Criteria Detection
    sirs_count = 0
    if temp is not None and (temp > 38.0 or temp < 36.0): sirs_count += 1
    if hr is not None and hr > 90: sirs_count += 1
    if rr is not None and rr > 20: sirs_count += 1
    if sirs_count >= 2:
        positives.append(f"SYSTEM_ALERT: SIRS_CRITERIA_MET({sirs_count}/3)")

    return positives


def _extract_suspected_conditions(text: str) -> list[str]:
    """Lightweight regex extraction — augments LLM output only, not primary source."""
    results: list[str] = []
    patterns = [
        r"(?:history of|known to have|diagnosed with|past medical history of|treated for)\s+([^.;]{3,60})",
        r"(?:concern for|suspicion for|possible|probable|consistent with)\s+([^.;]{3,60})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = _normalize_text(str(match.group(1) or "").strip(" .,:;"))
            if phrase and 3 < len(phrase.split()) <= 6:
                results.append(phrase)
    return _merge_unique(results, [], 6)


def _normalize_condition_candidates(values: list[str], *, limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in values:
        cleaned = re.sub(r"[^a-z0-9_ -]+", " ", ascii_fold(_normalize_text(item)))
        cleaned = "_".join(part for part in cleaned.replace("-", " ").split() if part)
        if not cleaned or cleaned in seen:
            continue
        if len(cleaned) < 4 or len(cleaned.split("_")) > 8:
            continue
        seen.add(cleaned)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def _extract_planned_interventions(text: str) -> list[str]:
    """Lightweight regex extraction — augments LLM output only, not primary source."""
    results: list[str] = []
    patterns = [
        r"(?:plan to|planned to|will|ordered|start|give|send for|obtain|perform|check|measure)\s+([^.;]{3,80})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            phrase = _normalize_text(str(match.group(1) or "").strip(" .,:;"))
            if phrase:
                results.append(phrase)
    return _merge_unique(results, [], 6)


def _merge_vitals(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for source in (primary, secondary):
        for key, value in (source or {}).items():
            if value in (None, ""):
                continue
            merged[str(key)] = value
    return merged


def _extract_contextual_exposures(text: str) -> list[str]:
    normalized = _normalize_text(text)
    exposures: list[str] = []
    patterns = [
        r"\b(?:after|following|since|during|before)\s+([^.;]{3,80})",
        r"\b(?:from|in)\s+the same\s+([^.;]{3,40})",
        r"\b(?:recently started|started|stopped|changed)\s+([^.;]{3,60})",
        r"\b(?:brought by|with|along with)\s+([^.;]{3,60})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.IGNORECASE):
            cleaned = _normalize_text(str(match.group(0) or "").strip(" .,:;"))
            if cleaned:
                exposures.append(cleaned)
    return _merge_unique(exposures, [], 8)


def _extract_negative_findings(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    findings: list[str] = []
    for segment in _split_surface_segments(normalized):
        lowered = segment.lower()
        if re.search(r"\b(?:no|without|denies?|denied|not)\b", lowered):
            findings.append(segment)
            continue
        if re.search(r"\bnon[- ]\w+", lowered):
            findings.append(segment)
            continue
        if "abdomen is soft" in lowered or "abdomen soft" in lowered:
            findings.append(segment)
    return _merge_unique(findings, [], 10)


def _derive_exam_contradictions(
    *,
    patient_text: str,
    positive_findings: list[str],
    negative_findings: list[str],
) -> list[str]:
    corpus = " ".join(
        [
            _normalize_text(patient_text).lower(),
            *[str(item or "").lower() for item in positive_findings[:12]],
            *[str(item or "").lower() for item in negative_findings[:10]],
        ]
    )
    severe_distress = bool(
        re.search(
            r"\b(?:\d{1,2}/10|12 out of 10|severe pain|agonizing|writhing|cannot sit still|can't sit still|kivraniyor|oluyorum)\b",
            corpus,
            flags=re.IGNORECASE,
        )
    )
    bland_exam = bool(
        re.search(
            r"\b(?:non[- ]tender|no guarding|no rebound|soft abdomen|abdomen is soft|completely non[- ]tender)\b",
            corpus,
            flags=re.IGNORECASE,
        )
    )
    contradictions: list[str] = []
    if severe_distress and bland_exam:
        contradictions.append("contradiction: severe reported distress with unexpectedly benign exam reactivity")
    return _merge_unique(contradictions, [], 4)


def _merge_input_context(llm_context: list[str], patient_text: str, *, llm_primary: bool) -> list[str]:
    """Merge LLM-produced context with lightweight heuristic story-frame tags.

    When the LLM is primary and has produced content, LLM output is used as-is.
    Falls back to lightweight structural tags derived from presence of key story
    elements (timeline, conditions, etc.) without any regex content matching.
    """
    if llm_primary and llm_context:
        return _merge_unique(llm_context, [], 14)
    # Lightweight story-frame tags from structural signals only
    story_context: list[str] = []
    timeline = _extract_timeline(patient_text)
    exposures = _extract_contextual_exposures(patient_text)
    conditions = _extract_suspected_conditions(patient_text)
    interventions = _extract_planned_interventions(patient_text)
    uncertainty = _extract_uncertainty_markers(patient_text)
    if timeline:
        story_context.append("story_frame:temporal_progression_present")
    if exposures:
        story_context.append("story_frame:event_or_background_context_present")
    if conditions:
        story_context.append("story_frame:diagnostic_anchor_present")
    if interventions:
        story_context.append("story_frame:clinician_plan_present")
    if uncertainty:
        story_context.append("story_frame:epistemic_uncertainty_present")
    return _merge_unique(llm_context, story_context, 14)


def _objective_red_flags(positive_findings: list[str]) -> list[str]:
    return [item for item in positive_findings if str(item or "").strip().lower().startswith("system_alert:")]


def _red_flag_grounded_in_narrative(red_flag: str, narrative: str) -> bool:
    """Return True if the red flag's informative tokens appear in the patient
    narrative. Conservative filter against LLM-fabricated alerts: a flag like
    "system_alert:cardiac_tamponade" is dropped when none of its content tokens
    appear in the source text. Short-acronym flags (PE, MI, ...) require an
    exact substring match. The check ignores ASCII case and joiner characters.
    """
    if not red_flag or not narrative:
        return False
    rf = str(red_flag).strip().lower()
    if rf.startswith("system_alert:"):
        rf = rf.split(":", 1)[1]
    rf_clean = re.sub(r"[^a-z0-9 ]+", " ", rf).strip()
    if len(rf_clean) < 2:
        return False
    narrative_lower = narrative.lower()
    informative_tokens = [token for token in rf_clean.split() if len(token) >= 4]
    if not informative_tokens:
        return rf_clean in narrative_lower
    return any(token in narrative_lower for token in informative_tokens)



class ExtractorAgent:
    """Narrative-to-findings bootstrap extractor."""

    def __init__(self, llm_extractor: LocalLlmClinicalExtractor | None = None) -> None:
        self._policy = load_runtime_policy()
        self._llm_extractor = llm_extractor or LocalLlmClinicalExtractor(self._policy)

    async def run(self, patient_input: PatientInput) -> StatePatch:
        normalization_payload: dict[str, Any] = {}
        if hasattr(self._llm_extractor, "normalize_semantic_text"):
            normalization_payload = await self._llm_extractor.normalize_semantic_text(
                patient_input.patient_text,
                patient_input.language,
            )
        normalized_text = str(normalization_payload.get("normalized_text", "") or "")
        patient_text = _normalize_text(normalized_text or normalize_clinical_text(patient_input.patient_text))
        segments = _split_surface_segments(patient_text)
        llm_payload = await self._llm_extractor.extract(patient_input.model_copy(update={"patient_text": patient_text}))
        llm_error = dict((llm_payload.get("_meta", {}) or {}).get("error", {}) or {})
        normalization_error = dict((normalization_payload.get("_meta", {}) or {}).get("error", {}) or {})
        literal_mappings = list(normalization_payload.get("literal_mappings", []) or [])
        llm_structured = any(
            bool(llm_payload.get(field))
            for field in (
                "summary",
                "positive_findings",
                "negative_findings",
                "timeline",
                "exposures",
                "medications",
                "suspected_conditions",
                "planned_interventions",
                "input_context",
                "red_flags",
                "derived_vitals",
                "constitutional_findings",
            )
        )
        llm_primary = self._policy.llm_primary_reasoning_enabled
        heuristic_timeline = _extract_timeline(patient_text)
        uncertainty_markers = _extract_uncertainty_markers(patient_text)
        heuristic_conditions = _extract_suspected_conditions(patient_text)
        heuristic_interventions = _extract_planned_interventions(patient_text)
        heuristic_exposures = _extract_contextual_exposures(patient_text)
        positive_findings: list[str] = []
        negative_findings: list[str] = []

        derived_vitals = _merge_vitals(
            _extract_vitals_from_text(patient_text),
            dict(llm_payload.get("derived_vitals", {}) or {}),
        )
        llm_positive = list(llm_payload.get("positive_findings", []) or [])
        llm_suspected_conditions = list(llm_payload.get("suspected_conditions", []) or [])
        positive_findings = _merge_unique(
            llm_positive,
            _derive_vital_findings(derived_vitals) + ([] if llm_positive else segments),
            12,
        )
        negative_findings = _merge_unique(
            list(llm_payload.get("negative_findings", []) or []),
            _extract_negative_findings(patient_text),
            8,
        )
        suspected_conditions = _llm_primary_merge(
            llm_suspected_conditions,
            heuristic_conditions,
            8,
            llm_primary=llm_primary,
        )
        suspected_conditions = _normalize_condition_candidates(suspected_conditions, limit=8)
        condition_validation_error: dict[str, Any] = {}
        condition_rejections = 0
        if suspected_conditions and hasattr(self._llm_extractor, "validate_diagnostic_labels"):
            validation_payload = await self._llm_extractor.validate_diagnostic_labels(
                suspected_conditions,
                context=patient_text,
            )
            condition_validation_error = dict((validation_payload.get("_meta", {}) or {}).get("error", {}) or {})
            accepted_items = list(validation_payload.get("accepted", []) or [])
            condition_rejections = len(list(validation_payload.get("rejected", []) or []))
            if not condition_validation_error:
                accepted_labels: list[str] = []
                for item in accepted_items:
                    canonical = _normalize_text(str(item.get("canonical_label", "") or "")).replace(" ", "_").lower()
                    if canonical:
                        accepted_labels.append(canonical)
                if accepted_labels or condition_rejections:
                    suspected_conditions = _merge_unique(accepted_labels, [], 8)
        planned_interventions = _llm_primary_merge(
            list(llm_payload.get("planned_interventions", []) or []),
            heuristic_interventions,
            8,
            llm_primary=llm_primary,
        )
        input_context = _merge_input_context(
            list(llm_payload.get("input_context", []) or []),
            patient_text,
            llm_primary=llm_primary,
        )
        input_context = _merge_unique(
            input_context,
            _derive_exam_contradictions(
                patient_text=patient_text,
                positive_findings=positive_findings,
                negative_findings=negative_findings,
            ),
            14,
        )
        exposures = _llm_primary_merge(
            list(llm_payload.get("exposures", []) or []),
            heuristic_exposures,
            8,
            llm_primary=llm_primary,
        )
        medications = _merge_unique(
            list(llm_payload.get("medications", []) or []),
            patient_input.medications[:10],
            10,
        )
        red_flags = _merge_unique(
            list(llm_payload.get("red_flags", []) or []),
            _objective_red_flags(positive_findings),
            8,
        )
        # 3c: drop LLM-fabricated red flags whose informative tokens are absent
        # from the patient narrative. Operator can disable via env if a clinical
        # workflow needs to surface speculative flags downstream.
        if os.environ.get("CDSS_RED_FLAG_NARRATIVE_GROUND", "1").strip().lower() not in {"0", "false", "off"}:
            grounded_red_flags = [
                flag for flag in red_flags
                if _red_flag_grounded_in_narrative(flag, patient_text)
            ]
            red_flags = grounded_red_flags
        mapped_segments = [
            f"{str(item.get('literal', '')).strip()} => {str(item.get('canonical', '')).strip()}"
            for item in literal_mappings
            if str(item.get("literal", "")).strip() and str(item.get("canonical", "")).strip()
        ]
        findings = StructuredFindings(
            summary=str(llm_payload.get("summary", "") or (segments[0] if segments else patient_text[:140])),
            language=patient_input.language or "",
            positive_findings=positive_findings[:12],
            negative_findings=negative_findings[:8],
            timeline=_llm_primary_merge(
                list(llm_payload.get("timeline", []) or []),
                heuristic_timeline,
                8,
                llm_primary=llm_primary,
            ),
            exposures=exposures,
            medications=medications,
            demographics=patient_input.demographics,
            derived_vitals=derived_vitals,
            suspected_conditions=suspected_conditions,
            planned_interventions=planned_interventions,
            input_context=input_context,
            context_lanes={},
            phenotype_atoms=[],
            red_flags=red_flags[:8],
            uncertainty_markers=uncertainty_markers,
            raw_segments=_merge_unique(segments, mapped_segments, 20),
            constitutional_findings=_merge_unique(
                list(llm_payload.get("constitutional_findings", []) or []),
                [],
                6,
            ),
        )
        findings = findings.model_copy(
            update={
                "context_lanes": build_context_lanes(findings),
            }
        )
        findings = findings.model_copy(
            update={
                "phenotype_fingerprint": compile_phenotype_fingerprint(findings),
            }
        )
        findings = findings.model_copy(
            update={
                "phenotype_atoms": compile_phenotype_atoms(patient_text, findings),
            }
        )
        return StatePatch(
            status="findings_ready",
            active_stage="extraction",
            findings=findings,
            trace=[
                DecisionTrace(
                    stage="extractor",
                    message="Structured findings extracted from patient narrative.",
                    payload={
                        "positive_count": len(findings.positive_findings),
                        "negative_count": len(findings.negative_findings),
                        "red_flag_count": len(findings.red_flags),
                        "llm_structured": llm_structured,
                        "llm_primary_reasoning": llm_primary,
                        "llm_error": llm_error,
                        "normalization_error": normalization_error,
                        "llm_normalization_used": bool(normalized_text),
                        "llm_literal_mappings": len(literal_mappings),
                        "suspected_condition_count": len(findings.suspected_conditions),
                        "condition_validation_rejections": condition_rejections,
                        "condition_validation_error": condition_validation_error,
                        "planned_intervention_count": len(findings.planned_interventions),
                        "input_context_count": len(findings.input_context),
                        "metrics": dict((llm_payload.get("_meta", {}) or {}).get("metrics", {}) or {}),
                    },
                )
            ],
        )
