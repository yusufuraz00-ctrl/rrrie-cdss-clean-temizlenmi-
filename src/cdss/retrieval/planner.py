"""Evidence planning helpers for the rebuilt DDx + Safety runtime."""

from __future__ import annotations

from dataclasses import dataclass

from src.cdss.clinical.explanation import derive_dangerous_treatment_assumptions, derive_state_frames
from src.cdss.clinical.phenotype import phenotype_query_terms
from src.cdss.contracts.models import (
    ContradictionCluster,
    EvidenceNeed,
    DifferentialSet,
    EvidenceBundle,
    EvidenceItem,
    FactGraph,
    GeneratedCandidateProfile,
    HypothesisFrontier,
    InterventionSet,
    PatientInput,
    RetrievalIntent,
    RiskProfile,
    StructuredFindings,
)
from src.cdss.core import thresholds as clinical_thresholds
from src.cdss.knowledge.registry import load_syndrome_registry
from src.cdss.runtime.policy import load_runtime_policy
from src.cdss.text_normalization import sanitize_query_text

import json as _json
import pathlib as _pathlib


def _load_medical_synonyms() -> dict[str, list[str]]:
    """Load medical abbreviation/synonym map from config. Returns empty dict on failure."""
    path = _pathlib.Path("config/medical_synonyms.json")
    if path.exists():
        try:
            return _json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


_MEDICAL_SYNONYMS: dict[str, list[str]] = _load_medical_synonyms()


_AXIS_TEMPLATES_PATH = _pathlib.Path(__file__).resolve().parents[3] / "data" / "cdss" / "retrieval" / "axis_templates.json"


def _load_axis_templates() -> dict:
    """Load Plan Item 6 axis-fallback retrieval templates. Empty on missing file."""
    try:
        if _AXIS_TEMPLATES_PATH.exists():
            data = _json.loads(_AXIS_TEMPLATES_PATH.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
    except Exception:
        return {}
    return {}


_AXIS_TEMPLATES: dict = _load_axis_templates()


def _axis_template_intents(
    findings: StructuredFindings,
    *,
    limit: int = 4,
) -> list[RetrievalIntent]:
    """Synthesize retrieval intents from the dominant phenotype axes when the
    gap-driven planner produced too few intents (frontier was thin or every
    candidate query degenerated). Templates live in
    ``data/cdss/retrieval/axis_templates.json`` — pure data, no code changes
    needed to extend.

    See Plan Item 6 (cross-case engineering improvements 2026-04-26).
    """

    if not _AXIS_TEMPLATES:
        return []
    fp = getattr(findings, "phenotype_fingerprint", None)
    if fp is None or not getattr(fp, "axis_weights", None):
        return []
    axes = sorted(
        ((str(a), float(w or 0.0)) for a, w in fp.axis_weights.items() if w),
        key=lambda kv: kv[1],
        reverse=True,
    )
    if not axes:
        return []
    top1 = axes[0][0]
    top2 = axes[1][0] if len(axes) >= 2 else ""

    queries: list[str] = []
    if top2:
        pair_key = f"{top1}|{top2}"
        rev_key = f"{top2}|{top1}"
        pair_map = _AXIS_TEMPLATES.get("pair", {}) or {}
        queries.extend(pair_map.get(pair_key, []) or pair_map.get(rev_key, []))
    solo_map = _AXIS_TEMPLATES.get("solo", {}) or {}
    queries.extend(solo_map.get(top1, []))
    if top2 and len(queries) < limit:
        queries.extend(solo_map.get(top2, []))

    intents: list[RetrievalIntent] = []
    seen: set[str] = set()
    for q in queries:
        q = str(q or "").strip()
        if not q or q in seen:
            continue
        if _is_degenerate_query(q):
            continue
        seen.add(q)
        intents.append(
            RetrievalIntent(
                objective="axis_template_fallback",
                decision_target="explore",
                query_hint=q,
                target_candidate="",
                priority=0.35,
            )
        )
        if len(intents) >= limit:
            break
    return intents


def _expand_query_synonyms(query: str) -> str:
    """Expand medical abbreviations in a retrieval query to improve recall.

    For each lowercase token that matches a known abbreviation, appends up to
    2 synonym terms. Only adds unique terms not already in the query.
    Capped to avoid query bloat (max +4 terms total).
    """
    if not _MEDICAL_SYNONYMS or not query:
        return query
    tokens = query.lower().split()
    query_token_set = set(tokens)
    additions: list[str] = []
    for token in tokens:
        syns = _MEDICAL_SYNONYMS.get(token, [])
        for syn in syns[:2]:
            if syn.lower() not in query_token_set and syn.lower() not in additions:
                additions.append(syn)
        if len(additions) >= 4:
            break
    if not additions:
        return query
    return query + " " + " ".join(additions)


GENERIC_BUCKET_LABELS = {
    "cardiorespiratory_process",
    "infectious_inflammatory_process",
    "metabolic_or_abdominal_process",
    "neurologic_process",
    "undifferentiated_high_variance_process",
}

_MULTILINGUAL_QUERY_EXPANSIONS: dict[str, dict[str, str]] = {
    "tr": {
        "nefes darligi": "dyspnea",
        "hava acligi": "air hunger",
        "gogus agrisi": "chest pain",
        "karin agrisi": "abdominal pain",
        "kulak cinlamasi": "tinnitus",
        "ates": "fever",
        "bayilma": "syncope",
        "nobet": "seizure",
        "bilinc degisikligi": "altered mental status",
    }
}


def _is_generic_label(label: str) -> bool:
    return label in GENERIC_BUCKET_LABELS or str(label or "").endswith("_process")


def _is_placeholder_fragment(text: str) -> bool:
    normalized = sanitize_query_text(text or "", max_terms=6)
    if not normalized:
        return True
    placeholder_tokens = {"true", "false", "unknown", "unspecified", "not", "specified"}
    tokens = set(normalized.split())
    return bool(tokens) and tokens.issubset(placeholder_tokens)


def _safe_query_fragment(text: str, *, fallback: str = "", expand_synonyms: bool = True) -> str:
    normalized = sanitize_query_text(text or "", max_terms=8)
    if not normalized or _is_placeholder_fragment(normalized):
        return fallback
    result = normalized.replace("_", " ")
    if expand_synonyms:
        result = _expand_query_synonyms(result)
    return result


def _available_signals(patient_input: PatientInput, findings: StructuredFindings) -> set[str]:
    signals: set[str] = set()
    if patient_input.vitals or findings.derived_vitals:
        signals.update({"objective_vitals", "serial_vitals", "pulse_oximetry"})
    if patient_input.labs:
        signals.update({"initial_laboratory_or_imaging_workup", "cbc_and_inflammatory_markers", "point_of_care_glucose"})
    if findings.timeline:
        signals.update({"symptom_timeline", "mental_status_timeline", "focused_history", "infection_focused_history"})
    if findings.exposures:
        signals.update({"exposure_timeline", "substance_identification"})
    if findings.medications or patient_input.medications:
        signals.update({"medication_history"})
    if findings.planned_interventions:
        signals.update({"planned_interventions"})
    return signals


@dataclass(frozen=True)
class _SignalRule:
    keywords: tuple[str, ...]
    signal: str = ""
    requires_labs: bool = False


_SIGNAL_COVERAGE_RULES: tuple[_SignalRule, ...] = (
    _SignalRule(keywords=("vital",), signal="objective_vitals"),
    _SignalRule(keywords=("history",), signal="focused_history"),
    _SignalRule(keywords=("glucose",), signal="point_of_care_glucose"),
    _SignalRule(keywords=("timeline",), signal="symptom_timeline"),
    _SignalRule(keywords=("lab", "cbc", "panel"), requires_labs=True),
)


def _coverage_score(evidence_needs: list[str], available_signals: set[str], has_labs: bool) -> float:
    if not evidence_needs:
        return 0.0
    hits = 0
    for need in evidence_needs:
        need_text = need.lower()
        if need in available_signals:
            hits += 1
            continue
        for rule in _SIGNAL_COVERAGE_RULES:
            if not any(kw in need_text for kw in rule.keywords):
                continue
            if rule.requires_labs and has_labs:
                hits += 1
                break
            if rule.signal and rule.signal in available_signals:
                hits += 1
                break
    return round(hits / len(evidence_needs), 2)


def _candidate_intents(profile: object, priority: float, query_hint: str, *, generic_label: bool = False) -> list[RetrievalIntent]:
    base_relevance = 0.82 if profile.must_not_miss else 0.64 if generic_label else 0.58
    intents = [
        RetrievalIntent(
            objective="support_or_refute_candidate",
            decision_target="confirm",
            query_hint=query_hint,
            target_candidate=profile.id,
            priority=priority,
            decision_relevance=round(base_relevance, 2),
            expected_value=round((priority * 0.6) + (base_relevance * 0.4), 2),
        )
    ]
    if generic_label:
        intents.extend(
            [
                RetrievalIntent(
                    objective="specificity_resolution",
                    decision_target="discriminate",
                    query_hint=sanitize_query_text(f"{query_hint} specific causes", max_terms=12),
                    target_candidate=profile.id,
                    priority=max(0.1, round(priority + 0.04, 2)),
                    decision_relevance=0.72,
                    expected_value=round(max(0.2, priority + 0.08), 2),
                ),
                RetrievalIntent(
                    objective="must_not_miss_exclusion",
                    decision_target="exclude",
                    query_hint=sanitize_query_text(f"{query_hint} dangerous causes", max_terms=12),
                    target_candidate=profile.id,
                    priority=max(0.1, round(priority + 0.08, 2)),
                    decision_relevance=0.88,
                    expected_value=round(max(0.24, priority + 0.14), 2),
                ),
            ]
        )
    else:
        for query in profile.challenge_queries[:1]:
            intents.append(
                RetrievalIntent(
                    objective="challenge_candidate",
                    decision_target="discriminate",
                    query_hint=query,
                    target_candidate=profile.id,
                    priority=max(0.1, round(priority - 0.05, 2)),
                    decision_relevance=0.58,
                    expected_value=round(max(0.16, priority - 0.02), 2),
                )
            )
    return intents


def _salient_fact_terms(fact_graph: FactGraph | None, findings: StructuredFindings, limit: int = 4) -> list[str]:
    candidates: list[tuple[int, str]] = []
    for item in findings.suspected_conditions[:4]:
        cleaned = sanitize_query_text(item.replace("_", " "), max_terms=5)
        if cleaned and not _is_generic_label(item):
            candidates.append((6, cleaned))
    for item in findings.positive_findings[:6]:
        cleaned = sanitize_query_text(item, max_terms=5)
        if cleaned:
            candidates.append((5, cleaned))
    for node in (fact_graph.nodes if fact_graph else []):
        if node.category in {"authority_claim", "timeline"}:
            continue
        if not node.label or len(node.label) < 4:
            continue
        cleaned = sanitize_query_text(node.label, max_terms=4)
        if not cleaned:
            continue
        weight = 0
        if node.category in {"semantic_pattern", "risk_marker"}:
            weight += 4
        if node.category in {"exposure", "laterality_marker", "course_marker"}:
            weight += 3
        if node.category in {"vital", "diagnostic_context"}:
            weight += 2
        if node.category == "finding" and len(cleaned.split()) > 4:
            continue
        candidates.append((weight, cleaned.replace("_", " ")))
    for item in findings.red_flags[:3]:
        cleaned = sanitize_query_text(item, max_terms=4)
        if cleaned:
            candidates.append((4, cleaned))
    for item in findings.exposures[:3]:
        cleaned = sanitize_query_text(item.replace("_", " "), max_terms=4)
        if cleaned:
            candidates.append((3, cleaned))
    ordered = [text for _, text in sorted(candidates, key=lambda item: item[0], reverse=True)]
    output: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _generic_query_hint(label: str, fact_graph: FactGraph | None, findings: StructuredFindings) -> str:
    phenotype_terms = phenotype_query_terms(findings, limit=4)
    if phenotype_terms:
        base = " ".join(phenotype_terms[:3])
        label_hint = "" if _is_generic_label(label) else _safe_query_fragment(label.replace("_", " "), fallback="")
        return " ".join(part for part in [base, label_hint] if part).strip()
    salient = _salient_fact_terms(fact_graph, findings, limit=3)
    label_hint = "" if _is_generic_label(label) else _safe_query_fragment(label.replace("_", " "), fallback="")
    if salient:
        base = " ".join(salient)
        if len(base.split()) < 3 and label_hint:
            return " ".join(part for part in [base, label_hint] if part).strip()
        return base
    if not salient:
        fallback = sanitize_query_text(
            " ".join(findings.suspected_conditions[:2] + findings.positive_findings[:4] + findings.red_flags[:2]),
            max_terms=6,
        )
        return label_hint or fallback
    return label_hint


def _critical_findings(findings: StructuredFindings) -> list[str]:
    output: list[str] = []
    for item in [*findings.red_flags, *findings.positive_findings]:
        cleaned = str(item or "").strip()
        if not cleaned or cleaned in output:
            continue
        output.append(cleaned)
        if len(output) >= 5:
            break
    return output


def _desired_discriminator(objective: str) -> str:
    mapping = {
        "support_primary_hypothesis": "confirming features",
        "challenge_competing_hypothesis": "key mimics",
        "specificity_resolution": "specific causes",
        "must_not_miss_exclusion": "exclusion clues",
        "wrong_treatment_harm_review": "treatment contraindications",
        "intervention_safety_review": "intervention contraindications",
        "timeline_gap_resolution": "time course",
        "open_world_hypothesis_generation": "candidate etiologies",
        "objective_confirmation_priority": "objective confirmation tests",
        "high_risk_exclusion_search": "must-not-miss causes",
        "pairwise_discriminator": "best discriminator",
        "prevalence_prior_query": "population prevalence base rate",
    }
    return mapping.get(objective, "diagnostic clues")


def _decision_target(objective: str) -> str:
    mapping = {
        "support_primary_hypothesis": "confirm",
        "challenge_competing_hypothesis": "discriminate",
        "specificity_resolution": "discriminate",
        "must_not_miss_exclusion": "exclude",
        "wrong_treatment_harm_review": "contraindication",
        "intervention_safety_review": "contraindication",
        "timeline_gap_resolution": "close",
        "open_world_hypothesis_generation": "discriminate",
        "objective_confirmation_priority": "close",
        "high_risk_exclusion_search": "exclude",
        "pairwise_discriminator": "discriminate",
        "prevalence_prior_query": "calibrate",
    }
    return mapping.get(objective, "confirm")


def _state_descriptor(active_state: str) -> str:
    return sanitize_query_text(str(active_state or "").replace("_", " "), max_terms=7).replace("_", " ")


def _compact_signal(text: str, *, fallback: str = "") -> str:
    normalized = sanitize_query_text(text or "", max_terms=6)
    if not normalized:
        return fallback
    if normalized.startswith("objective_discriminator_for") or normalized.startswith("label_quality"):
        return fallback
    if normalized in {"benign outpatient closure", "closure", "competing mechanism", "dangerous alternative", "best discriminator"}:
        return fallback
    return normalized.replace("_", " ")


def _compile_request_query(
    *,
    objective: str,
    decision_target: str,
    active_state: str,
    unresolved_critical_finding: str,
    rival_mechanism: str,
    action_hazard: str,
    desired_discriminator: str,
    fallback: str,
) -> str:
    focus = _compact_signal(desired_discriminator, fallback="")
    parts = [_state_descriptor(active_state) if active_state else "", _compact_signal(unresolved_critical_finding)]
    rival = _compact_signal(rival_mechanism.replace("_", " "), fallback="")
    hazard = _compact_signal(action_hazard.replace("_", " "), fallback="") if action_hazard else ""

    if decision_target == "discriminate" and rival:
        parts.append(rival)
    if decision_target in {"contraindication", "exclude"} and hazard:
        parts.append(hazard)
    if focus:
        parts.append(focus)
    compiled = sanitize_query_text(" ".join(part for part in parts if part), max_terms=7)
    if compiled:
        return compiled
    return sanitize_query_text(fallback, max_terms=7)


def _allow_rival_mechanism(active_state: str, rival_mechanism: str) -> bool:
    del active_state
    return bool(str(rival_mechanism or "").strip())


def _meta_query_tokens() -> set[str]:
    return {
        "candidate",
        "criteria",
        "impact",
        "mimic",
        "mimics",
        "exclusion",
        "exclude",
        "contraindication",
        "contraindications",
        "adverse",
        "effect",
        "effects",
        "specific",
        "diagnostic",
        "diagnosis",
        "differential",
        "data",
        "text",
        "review",
        "detected",
        "type",
        "workup",
    }


def _is_degenerate_query(query: str) -> bool:
    normalized = sanitize_query_text(query, max_terms=12)
    if not normalized:
        return True
    tokens = normalized.split()
    meta_tokens = _meta_query_tokens()
    informative = [token for token in tokens if token not in meta_tokens]
    if not informative:
        return True
    if len(tokens) <= 2 and len(informative) <= 1:
        return True
    return False


def _semantic_tokens(query: str) -> set[str]:
    normalized = sanitize_query_text(query, max_terms=16)
    if not normalized:
        return set()
    stop_tokens = {
        "diagnosis",
        "differential",
        "review",
        "objective",
        "confirm",
        "exclude",
        "contraindications",
    }
    return {token for token in normalized.split() if token not in stop_tokens}


def _token_jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def _expand_query_multilingual(query_hint: str, language: str, *, context_text: str = "") -> str:
    language_key = str(language or "").strip().lower()
    expansions = _MULTILINGUAL_QUERY_EXPANSIONS.get(language_key)
    if not expansions:
        return query_hint
    normalized = sanitize_query_text(query_hint, max_terms=14)
    if not normalized:
        return query_hint
    context_normalized = sanitize_query_text(context_text, max_terms=10)
    source_tokens = set(sanitize_query_text(f"{normalized} {context_normalized}", max_terms=20).split())
    expanded_terms: list[str] = []
    for source, target in expansions.items():
        required_tokens = set(sanitize_query_text(source, max_terms=4).split())
        if required_tokens and required_tokens.issubset(source_tokens):
            expanded_terms.append(target)
    if not expanded_terms:
        return normalized
    merged = sanitize_query_text(f"{normalized} {' '.join(expanded_terms)}", max_terms=14)
    return merged or normalized


def _query_hygiene(query: str, *, decision_target: str, action_hazard: str, rival_mechanism: str) -> tuple[float, bool]:
    normalized = sanitize_query_text(query, max_terms=12)
    if not normalized:
        return 0.0, False
    tokens = normalized.split()
    score = 0.96
    leakage = False
    if len(tokens) > 7:
        score -= 0.08
    generic_meta_tokens = _meta_query_tokens()
    if decision_target not in {"contraindication", "exclude"} and any(token in generic_meta_tokens for token in tokens):
        score -= 0.1
    if _is_degenerate_query(normalized):
        score -= 0.34
    hazard_tokens = set(sanitize_query_text(action_hazard, max_terms=5).split()) if action_hazard else set()
    rival_tokens = set(sanitize_query_text(rival_mechanism, max_terms=5).split()) if rival_mechanism else set()
    if decision_target not in {"contraindication", "exclude"} and hazard_tokens and (hazard_tokens & set(tokens)):
        leakage = True
        score -= 0.2
    if decision_target != "discriminate" and rival_tokens and (rival_tokens & set(tokens)):
        leakage = True
        score -= 0.14
    if any(token in {"placeholder", "unknown", "none", "null"} for token in tokens):
        score -= 0.18
    return round(max(0.0, min(1.0, score)), 2), leakage


def _fallback_query_from_need(need: EvidenceNeed) -> str:
    fallback = _safe_query_fragment(need.hypothesis_label.replace("_", " "), fallback="")
    if not fallback and need.rival_mechanism and need.decision_target == "discriminate":
        fallback = _safe_query_fragment(need.rival_mechanism.replace("_", " "), fallback="")
    if not fallback and need.action_hazard and need.decision_target in {"contraindication", "exclude"}:
        fallback = _safe_query_fragment(need.action_hazard.replace("_", " "), fallback="")
    if not fallback:
        fallback = _safe_query_fragment(need.unresolved_critical_finding, fallback=_state_descriptor(need.active_state) if need.active_state else "")
    if not fallback:
        fallback = need.desired_discriminator
    return _compile_request_query(
        objective=need.objective,
        decision_target=need.decision_target,
        active_state=need.active_state,
        unresolved_critical_finding=need.unresolved_critical_finding,
        rival_mechanism=need.rival_mechanism if need.decision_target == "discriminate" else "",
        action_hazard=need.action_hazard if need.decision_target in {"contraindication", "exclude"} else "",
        desired_discriminator=need.desired_discriminator,
        fallback=fallback,
    )


@dataclass(frozen=True)
class _NeedScoreSpec:
    relevance: float
    info_gain_delta: float
    relevance_non_generic: float | None = None


_NEED_SCORE_TABLE: dict[str, _NeedScoreSpec] = {
    "must_not_miss_exclusion":          _NeedScoreSpec(0.9,  0.16),
    "wrong_treatment_harm_review":      _NeedScoreSpec(0.9,  0.16),
    "intervention_safety_review":       _NeedScoreSpec(0.9,  0.16),
    "high_risk_exclusion_search":       _NeedScoreSpec(0.88, 0.18),
    "open_world_hypothesis_generation": _NeedScoreSpec(0.78, 0.18),
    "objective_confirmation_priority":  _NeedScoreSpec(0.7,  0.12),
    "pairwise_discriminator":           _NeedScoreSpec(0.84, 0.2),
    "specificity_resolution":           _NeedScoreSpec(0.72, 0.12, relevance_non_generic=0.64),
    "challenge_competing_hypothesis":   _NeedScoreSpec(0.72, 0.12, relevance_non_generic=0.64),
    "support_primary_hypothesis":       _NeedScoreSpec(0.6,  0.06, relevance_non_generic=0.54),
    "timeline_gap_resolution":          _NeedScoreSpec(0.46, 0.04),
    "prevalence_prior_query":           _NeedScoreSpec(0.62, 0.08),
}
_DEFAULT_NEED_SCORE = _NeedScoreSpec(0.5, 0.0)
_DANGEROUS_BONUS = 0.08


def _need_scores(objective: str, base_priority: float, *, generic_label: bool = False, dangerous: bool = False) -> tuple[float, float]:
    spec = _NEED_SCORE_TABLE.get(objective, _DEFAULT_NEED_SCORE)
    relevance = spec.relevance if (generic_label or spec.relevance_non_generic is None) else spec.relevance_non_generic
    info_gain = base_priority + spec.info_gain_delta
    if dangerous:
        relevance += _DANGEROUS_BONUS
        info_gain += _DANGEROUS_BONUS
    return round(min(0.99, relevance), 2), round(min(0.99, max(0.1, info_gain)), 2)


def _normalize_label(value: str) -> str:
    return str(value or "").strip().lower()


def _cluster_priority_boost(cluster: ContradictionCluster) -> float:
    severity = _normalize_label(cluster.severity)
    base_by_severity = {
        "minor": 0.03,
        "major": 0.08,
        "critical": 0.14,
    }
    base = base_by_severity.get(severity, 0.04)
    base += min(0.08, max(0, int(cluster.resolution_cost or 1) - 1) * 0.02)
    if not cluster.remediable:
        base += 0.02
    return round(min(0.22, max(0.0, base)), 2)


def _cluster_applies_to_need(need: EvidenceNeed, cluster: ContradictionCluster) -> bool:
    cluster_labels = {_normalize_label(label) for label in cluster.hypothesis_labels if _normalize_label(label)}
    need_labels = {_normalize_label(need.hypothesis_label), _normalize_label(need.rival_mechanism)}
    need_labels.discard("")
    if cluster_labels and (cluster_labels & need_labels):
        return True
    cluster_is_high = _normalize_label(cluster.severity) in {"major", "critical"}
    if cluster_is_high and need.objective in {"open_world_hypothesis_generation", "high_risk_exclusion_search"}:
        return True
    return False


def _apply_contradiction_cluster_priority_boost(
    needs: list[EvidenceNeed],
    contradiction_clusters: list[ContradictionCluster] | None,
) -> list[EvidenceNeed]:
    if not needs or not contradiction_clusters:
        return needs
    boosted_needs: list[EvidenceNeed] = []
    for need in needs:
        boost = 0.0
        for cluster in contradiction_clusters:
            if _cluster_applies_to_need(need, cluster):
                boost = max(boost, _cluster_priority_boost(cluster))
        if boost <= 0.0:
            boosted_needs.append(need)
            continue
        boosted_needs.append(
            need.model_copy(
                update={
                    "priority": round(min(0.99, need.priority + boost), 2),
                    "decision_relevance": round(min(0.99, need.decision_relevance + (boost * 0.8)), 2),
                    "expected_information_gain": round(min(0.99, need.expected_information_gain + boost), 2),
                }
            )
        )
    return boosted_needs


def _extract_demographic_hint(findings: StructuredFindings) -> str:
    """Return compact demographic string for prevalence queries (e.g. 'adult male')."""
    try:
        demo = findings.demographics or {}
        parts: list[str] = []
        age = demo.get("age") or demo.get("age_years")
        if age:
            try:
                age_int = int(float(str(age)))
                if age_int < 18:
                    parts.append("pediatric")
                elif age_int < 40:
                    parts.append("young adult")
                elif age_int < 65:
                    parts.append("adult")
                else:
                    parts.append("elderly")
            except (ValueError, TypeError):
                parts.append(str(age))
        sex = demo.get("sex") or demo.get("gender")
        if sex:
            parts.append(str(sex).lower())
        return " ".join(parts) if parts else ""
    except Exception:
        return ""


def _gap_driven_evidence_needs(
    findings: StructuredFindings,
    risk_profile: RiskProfile,
    frontier: HypothesisFrontier | None,
    interventions: InterventionSet | None,
    fact_graph: FactGraph | None,
    contradiction_clusters: list[ContradictionCluster] | None = None,
) -> list[EvidenceNeed]:
    needs: list[EvidenceNeed] = []
    policy = load_runtime_policy()
    hypotheses = list((frontier.hypotheses if frontier else []) or [])
    state_frames = derive_state_frames(findings, fact_graph or FactGraph(), interventions)
    dangerous_assumptions = derive_dangerous_treatment_assumptions(findings, fact_graph or FactGraph(), interventions or InterventionSet(), state_frames)
    critical_findings = _critical_findings(findings)
    if not hypotheses:
        active_state = state_frames[0] if state_frames else ""
        unresolved_critical = critical_findings[0] if critical_findings else findings.summary
        signature_query = _generic_query_hint("", fact_graph, findings) or sanitize_query_text(
            " ".join([findings.summary, *findings.positive_findings[:4], *findings.red_flags[:2]]),
            max_terms=7,
        )
        needs.append(
            EvidenceNeed(
                objective="open_world_hypothesis_generation",
                decision_target=_decision_target("open_world_hypothesis_generation"),
                rationale="No ranked disease hypotheses are stable yet; expand disease-level candidate search from the current syndrome signature.",
                priority=0.84 if risk_profile.urgency.value != "routine" else 0.68,
                query_hint=_compile_request_query(
                    objective="open_world_hypothesis_generation",
                    decision_target=_decision_target("open_world_hypothesis_generation"),
                    active_state=active_state,
                    unresolved_critical_finding=unresolved_critical,
                    rival_mechanism="",
                    action_hazard="",
                    desired_discriminator=_desired_discriminator("open_world_hypothesis_generation"),
                    fallback=f"{signature_query} differential diagnosis",
                ),
                active_state=active_state,
                unresolved_critical_finding=unresolved_critical,
                desired_discriminator=_desired_discriminator("open_world_hypothesis_generation"),
                decision_relevance=_need_scores("open_world_hypothesis_generation", 0.74)[0],
                expected_information_gain=_need_scores("open_world_hypothesis_generation", 0.74)[1],
            )
        )
        needs.append(
            EvidenceNeed(
                objective="objective_confirmation_priority",
                decision_target=_decision_target("objective_confirmation_priority"),
                rationale="The current story needs objective confirmation targets before the differential can stabilize.",
                priority=0.72 if risk_profile.urgency.value != "routine" else 0.58,
                query_hint=_compile_request_query(
                    objective="objective_confirmation_priority",
                    decision_target=_decision_target("objective_confirmation_priority"),
                    active_state=active_state,
                    unresolved_critical_finding=unresolved_critical,
                    rival_mechanism="",
                    action_hazard="",
                    desired_discriminator=_desired_discriminator("objective_confirmation_priority"),
                    fallback=f"{signature_query} objective confirmation tests",
                ),
                active_state=active_state,
                unresolved_critical_finding=unresolved_critical,
                desired_discriminator=_desired_discriminator("objective_confirmation_priority"),
                decision_relevance=_need_scores("objective_confirmation_priority", 0.62)[0],
                expected_information_gain=_need_scores("objective_confirmation_priority", 0.62)[1],
            )
        )
        if risk_profile.urgency.value != "routine" or critical_findings or dangerous_assumptions:
            needs.append(
                EvidenceNeed(
                    objective="high_risk_exclusion_search",
                    decision_target=_decision_target("high_risk_exclusion_search"),
                    rationale="High-risk story remains open without a disease-level anchor; dangerous alternatives must stay in play during search.",
                    priority=0.88 if risk_profile.urgency.value == "emergency" else 0.74,
                    query_hint=_compile_request_query(
                        objective="high_risk_exclusion_search",
                        decision_target=_decision_target("high_risk_exclusion_search"),
                        active_state=active_state,
                        unresolved_critical_finding=unresolved_critical,
                        rival_mechanism="",
                        action_hazard=dangerous_assumptions[0] if dangerous_assumptions else "",
                        desired_discriminator=_desired_discriminator("high_risk_exclusion_search"),
                        fallback=f"{signature_query} dangerous causes",
                    ),
                    active_state=active_state,
                    unresolved_critical_finding=unresolved_critical,
                    action_hazard=dangerous_assumptions[0] if dangerous_assumptions else "",
                    desired_discriminator=_desired_discriminator("high_risk_exclusion_search"),
                    decision_relevance=_need_scores("high_risk_exclusion_search", 0.78)[0],
                    expected_information_gain=_need_scores("high_risk_exclusion_search", 0.78)[1],
                )
            )
    top_margin = abs(hypotheses[0].score - hypotheses[1].score) if len(hypotheses) > 1 else 0.25
    for hypothesis in hypotheses[:3]:
        generic_label = _is_generic_label(hypothesis.label)
        active_state = state_frames[0] if state_frames else ""
        unresolved_critical = critical_findings[0] if critical_findings else ""
        rival_mechanism = (
            hypotheses[1].label
            if len(hypotheses) > 1 and hypothesis.rank == 1
            else ""
        )
        if not _allow_rival_mechanism(active_state, rival_mechanism):
            rival_mechanism = ""
        action_hazard = hypothesis.dangerous_if_treated_as or (dangerous_assumptions[0] if dangerous_assumptions else "")
        if _is_placeholder_fragment(action_hazard):
            action_hazard = ""
        primary_objective = "support_primary_hypothesis" if hypothesis.rank == 1 else "challenge_competing_hypothesis"
        primary_target = _decision_target(primary_objective)
        desired_discriminator = _desired_discriminator(primary_objective)
        fallback_query = _generic_query_hint(hypothesis.label, fact_graph, findings) or hypothesis.label.replace("_", " ")
        query_hint = _compile_request_query(
            objective=primary_objective,
            decision_target=primary_target,
            active_state=active_state,
            unresolved_critical_finding=unresolved_critical,
            rival_mechanism=rival_mechanism,
            action_hazard=action_hazard,
            desired_discriminator=desired_discriminator,
            fallback=fallback_query,
        )
        margin_bonus = 0.1 if top_margin < 0.12 and hypothesis.rank <= 2 else 0.0
        critical_bonus = 0.08 if unresolved_critical else 0.0
        needs.append(
            EvidenceNeed(
                objective=primary_objective,
                decision_target=primary_target,
                rationale=f"Clarify support vs contradiction for {hypothesis.label}.",
                hypothesis_label=hypothesis.label,
                priority=max(0.1, round(hypothesis.score, 2)),
                query_hint=query_hint,
                active_state=active_state,
                unresolved_critical_finding=unresolved_critical,
                rival_mechanism=rival_mechanism,
                action_hazard=action_hazard,
                desired_discriminator=desired_discriminator,
                decision_relevance=_need_scores(
                    primary_objective,
                    max(0.1, round(hypothesis.score, 2)),
                    generic_label=generic_label,
                    dangerous=bool(hypothesis.must_not_miss or hypothesis.dangerous_if_missed),
                )[0],
                expected_information_gain=round(
                    min(
                        0.99,
                        _need_scores(
                            primary_objective,
                            max(0.1, round(hypothesis.score, 2)),
                            generic_label=generic_label,
                            dangerous=bool(hypothesis.must_not_miss or hypothesis.dangerous_if_missed),
                        )[1]
                        + margin_bonus
                        + critical_bonus,
                    ),
                    2,
                ),
            )
        )
        if generic_label:
            specificity_query = _compile_request_query(
                objective="specificity_resolution",
                decision_target="discriminate",
                active_state=active_state,
                unresolved_critical_finding=unresolved_critical,
                rival_mechanism=rival_mechanism,
                action_hazard="",
                desired_discriminator=_desired_discriminator("specificity_resolution"),
                fallback=f"{fallback_query} specific causes",
            )
            needs.append(
                EvidenceNeed(
                    objective="specificity_resolution",
                    decision_target="discriminate",
                    rationale=f"Resolve overly broad family bucket for {hypothesis.label}.",
                    hypothesis_label=hypothesis.label,
                    priority=max(0.1, round(hypothesis.score + 0.04, 2)),
                    query_hint=specificity_query,
                    active_state=active_state,
                    unresolved_critical_finding=unresolved_critical,
                    rival_mechanism=rival_mechanism,
                    desired_discriminator=_desired_discriminator("specificity_resolution"),
                    decision_relevance=0.74,
                    expected_information_gain=round(min(0.99, hypothesis.score + 0.16), 2),
                )
            )
        should_force_safety_exclusion = generic_label and hypothesis.rank == 1
        if hypothesis.must_not_miss or hypothesis.dangerous_if_missed or should_force_safety_exclusion:
            safety_query = _compile_request_query(
                objective="must_not_miss_exclusion",
                decision_target="exclude",
                active_state=active_state,
                unresolved_critical_finding=unresolved_critical,
                rival_mechanism=rival_mechanism,
                action_hazard=action_hazard,
                desired_discriminator=_desired_discriminator("must_not_miss_exclusion"),
                fallback=f"{fallback_query} dangerous causes",
            )
            needs.append(
                EvidenceNeed(
                    objective="must_not_miss_exclusion",
                    decision_target="exclude",
                    rationale=f"Exclude high-risk miss scenario for {hypothesis.label}.",
                    hypothesis_label=hypothesis.label,
                    priority=max(0.2, round(hypothesis.score + 0.08, 2)),
                    query_hint=safety_query,
                    active_state=active_state,
                    unresolved_critical_finding=unresolved_critical,
                    rival_mechanism=rival_mechanism,
                    action_hazard=action_hazard,
                    desired_discriminator=_desired_discriminator("must_not_miss_exclusion"),
                    decision_relevance=0.92,
                    expected_information_gain=round(min(0.99, hypothesis.score + 0.2), 2),
                )
            )
        if hypothesis.dangerous_if_treated_as and not _is_placeholder_fragment(hypothesis.dangerous_if_treated_as):
            mistaken_for = _safe_query_fragment(
                hypothesis.dangerous_if_treated_as.replace("_", " "),
                fallback="competing mechanism",
            )
            needs.append(
                EvidenceNeed(
                    objective="wrong_treatment_harm_review",
                    decision_target="contraindication",
                    rationale=f"Check harm if {hypothesis.label} is treated as {hypothesis.dangerous_if_treated_as}.",
                    hypothesis_label=hypothesis.label,
                    priority=max(0.2, round(hypothesis.score + 0.12, 2)),
                    query_hint=_compile_request_query(
                        objective="wrong_treatment_harm_review",
                        decision_target="contraindication",
                        active_state=active_state,
                        unresolved_critical_finding=unresolved_critical,
                        rival_mechanism=mistaken_for,
                        action_hazard=action_hazard,
                        desired_discriminator=_desired_discriminator("wrong_treatment_harm_review"),
                        fallback=f"{fallback_query} {mistaken_for} contraindications",
                    ),
                    active_state=active_state,
                    unresolved_critical_finding=unresolved_critical,
                    rival_mechanism=mistaken_for,
                    action_hazard=action_hazard,
                    desired_discriminator=_desired_discriminator("wrong_treatment_harm_review"),
                    decision_relevance=0.94,
                    expected_information_gain=round(min(0.99, hypothesis.score + 0.22), 2),
                )
            )
    if policy.retrieval_pair_discriminator_enabled and len(hypotheses) >= 2 and top_margin < 0.14:
        lead = hypotheses[0]
        rival = hypotheses[1]
        active_state = state_frames[0] if state_frames else ""
        unresolved_critical = critical_findings[0] if critical_findings else ""
        lead_label = _safe_query_fragment(lead.label.replace("_", " "), fallback="")
        rival_label = _safe_query_fragment(rival.label.replace("_", " "), fallback="")
        pair_fallback = sanitize_query_text(
            f"{lead_label} versus {rival_label} key discriminators",
            max_terms=10,
        )
        base_priority = max(0.1, round(max(float(lead.score or 0.0), float(rival.score or 0.0)) + 0.1, 2))
        needs.append(
            EvidenceNeed(
                objective="pairwise_discriminator",
                decision_target="discriminate",
                rationale=f"Frontier top pair {lead.label} vs {rival.label} remains close and needs direct discriminator evidence.",
                hypothesis_label=lead.label,
                priority=base_priority,
                query_hint=_compile_request_query(
                    objective="pairwise_discriminator",
                    decision_target="discriminate",
                    active_state=active_state,
                    unresolved_critical_finding=unresolved_critical,
                    rival_mechanism=rival.label,
                    action_hazard="",
                    desired_discriminator=_desired_discriminator("pairwise_discriminator"),
                    fallback=pair_fallback,
                ),
                active_state=active_state,
                unresolved_critical_finding=unresolved_critical,
                rival_mechanism=rival.label,
                desired_discriminator=_desired_discriminator("pairwise_discriminator"),
                decision_relevance=_need_scores("pairwise_discriminator", base_priority)[0],
                expected_information_gain=_need_scores("pairwise_discriminator", base_priority)[1],
            )
        )
    for intervention in (interventions.items if interventions else []):
        intervention_hint = _safe_query_fragment(
            intervention.label.replace("_", " "),
            fallback=_safe_query_fragment(intervention.class_hint or intervention.source_text, fallback=""),
        )
        if not intervention_hint or intervention_hint in {"contraindications adverse effects", "adverse effects", "contraindications"}:
            continue
        needs.append(
            EvidenceNeed(
                objective="intervention_safety_review",
                decision_target="contraindication",
                rationale=f"Validate safety and contraindications for planned intervention {intervention.label}.",
                hypothesis_label="",
                priority=0.92 if risk_profile.manual_review_required else 0.72,
                query_hint=_compile_request_query(
                    objective="intervention_safety_review",
                    decision_target="contraindication",
                    active_state=state_frames[0] if state_frames else "",
                    unresolved_critical_finding=critical_findings[0] if critical_findings else "",
                    rival_mechanism="",
                    action_hazard=intervention_hint,
                    desired_discriminator=_desired_discriminator("intervention_safety_review"),
                    fallback=f"{intervention_hint} contraindications",
                ),
                active_state=state_frames[0] if state_frames else "",
                unresolved_critical_finding=critical_findings[0] if critical_findings else "",
                action_hazard=intervention_hint,
                desired_discriminator=_desired_discriminator("intervention_safety_review"),
                decision_relevance=0.9 if risk_profile.manual_review_required else 0.72,
                expected_information_gain=0.9 if risk_profile.manual_review_required else 0.74,
            )
        )
    if not findings.timeline:
        needs.append(
            EvidenceNeed(
                objective="timeline_gap_resolution",
                decision_target="close",
                rationale="Timeline is missing and weakens diagnostic sequencing.",
                priority=0.75,
                query_hint=_compile_request_query(
                    objective="timeline_gap_resolution",
                    decision_target="close",
                    active_state=state_frames[0] if state_frames else "",
                    unresolved_critical_finding=critical_findings[0] if critical_findings else "",
                    rival_mechanism="",
                    action_hazard="",
                    desired_discriminator=_desired_discriminator("timeline_gap_resolution"),
                    fallback="symptom timeline diagnostic value",
                ),
                active_state=state_frames[0] if state_frames else "",
                unresolved_critical_finding=critical_findings[0] if critical_findings else "",
                desired_discriminator=_desired_discriminator("timeline_gap_resolution"),
                decision_relevance=0.46,
                expected_information_gain=0.5,
            )
        )
    has_explicit_safety_need = any(
        item.objective in {"must_not_miss_exclusion", "wrong_treatment_harm_review", "intervention_safety_review"}
        for item in needs
    )
    if not has_explicit_safety_need and hypotheses:
        anchor = hypotheses[0]
        generic_label = _is_generic_label(anchor.label)
        query_hint = _generic_query_hint(anchor.label, fact_graph, findings) if generic_label else (_generic_query_hint(anchor.label, fact_graph, findings) or anchor.label.replace("_", " "))
        benign_anchor = bool(generic_label and anchor.score < 0.45 and not anchor.must_not_miss and not anchor.dangerous_if_missed)
        needs.append(
            EvidenceNeed(
                objective="wrong_treatment_harm_review",
                decision_target="contraindication",
                rationale=f"Review harmful treatment assumptions while the anchor remains {anchor.label}.",
                hypothesis_label=anchor.label,
                priority=max(0.12 if benign_anchor else 0.2, round(anchor.score + (0.02 if benign_anchor else 0.08), 2)),
                query_hint=_compile_request_query(
                    objective="wrong_treatment_harm_review",
                    decision_target="contraindication",
                    active_state=state_frames[0] if state_frames else "",
                    unresolved_critical_finding=critical_findings[0] if critical_findings else "",
                    rival_mechanism=(hypotheses[1].label if len(hypotheses) > 1 else ""),
                    action_hazard=anchor.dangerous_if_treated_as or (dangerous_assumptions[0] if dangerous_assumptions else ""),
                    desired_discriminator=_desired_discriminator("wrong_treatment_harm_review"),
                    fallback=f"{query_hint} contraindications",
                ),
                active_state=state_frames[0] if state_frames else "",
                unresolved_critical_finding=critical_findings[0] if critical_findings else "",
                rival_mechanism=(hypotheses[1].label if len(hypotheses) > 1 else ""),
                action_hazard=anchor.dangerous_if_treated_as or (dangerous_assumptions[0] if dangerous_assumptions else ""),
                desired_discriminator=_desired_discriminator("wrong_treatment_harm_review"),
                decision_relevance=0.62 if benign_anchor else (0.9 if risk_profile.manual_review_required else 0.78),
                expected_information_gain=round(
                    min(
                        0.99,
                        anchor.score
                        + (
                            0.04
                            if benign_anchor
                            else (0.18 if risk_profile.manual_review_required else 0.12)
                        ),
                    ),
                    2,
                ),
            )
        )
    # Prevalence prior query: fired when policy enables PubMed grounding AND
    # the top-2 margin is narrow (calibration most valuable when differential is uncertain).
    if getattr(policy, "epi_prior_pubmed_enabled", False) and len(hypotheses) >= 2:
        top_margin_check = abs(hypotheses[0].score - hypotheses[1].score) if len(hypotheses) > 1 else 0.25
        if top_margin_check < 0.18:
            lead = hypotheses[0]
            demographic_hint = _extract_demographic_hint(findings)
            prevalence_query = f"{lead.label.replace('_', ' ')} prevalence incidence {demographic_hint}".strip()
            needs.append(
                EvidenceNeed(
                    objective="prevalence_prior_query",
                    decision_target=_decision_target("prevalence_prior_query"),
                    rationale=(
                        f"Top-2 margin is narrow ({top_margin_check:.2f}); calibrate base-rate prior "
                        f"for {lead.label} vs {hypotheses[1].label} to resolve epidemiological uncertainty."
                    ),
                    hypothesis_label=lead.label,
                    priority=round(max(0.1, lead.score - 0.05), 2),
                    query_hint=prevalence_query,
                    desired_discriminator=_desired_discriminator("prevalence_prior_query"),
                    decision_relevance=_need_scores("prevalence_prior_query", lead.score)[0],
                    expected_information_gain=_need_scores("prevalence_prior_query", lead.score)[1],
                )
            )

    needs = _apply_contradiction_cluster_priority_boost(needs, contradiction_clusters)
    needs.sort(key=lambda item: (item.expected_information_gain, item.decision_relevance, item.priority), reverse=True)
    return needs[:6]


def _needs_to_intents(needs: list[EvidenceNeed], *, language: str = "") -> list[RetrievalIntent]:
    intents: list[RetrievalIntent] = []
    seen: set[tuple[str, str]] = set()
    for need in needs:
        hygiene_score, leakage = _query_hygiene(
            need.query_hint,
            decision_target=need.decision_target,
            action_hazard=need.action_hazard,
            rival_mechanism=need.rival_mechanism,
        )
        query_hint = need.query_hint
        if _is_degenerate_query(query_hint):
            query_hint = _fallback_query_from_need(need)
            hygiene_score, leakage = _query_hygiene(
                query_hint,
                decision_target=need.decision_target,
                action_hazard=need.action_hazard,
                rival_mechanism=need.rival_mechanism,
            )
        if leakage and need.decision_target not in {"contraindication", "exclude"}:
            query_hint = _fallback_query_from_need(
                need.model_copy(update={"action_hazard": "", "rival_mechanism": need.rival_mechanism if need.decision_target == "discriminate" else ""})
            )
            hygiene_score, leakage = _query_hygiene(
                query_hint,
                decision_target=need.decision_target,
                action_hazard=need.action_hazard,
                rival_mechanism=need.rival_mechanism,
            )
        query_hint = _expand_query_multilingual(
            query_hint,
            language,
            context_text=f"{need.unresolved_critical_finding} {need.active_state}",
        )
        key = (need.objective, query_hint)
        if key in seen or not query_hint:
            continue
        candidate_tokens = _semantic_tokens(query_hint)
        semantic_duplicate_index: int | None = None
        for index, existing in enumerate(intents):
            if existing.decision_target != need.decision_target:
                continue
            similarity = _token_jaccard(candidate_tokens, _semantic_tokens(existing.query_hint))
            if similarity >= 0.74:
                semantic_duplicate_index = index
                break
        candidate_expected_value = round(min(0.99, need.expected_information_gain * max(0.45, hygiene_score)), 2)
        if semantic_duplicate_index is not None:
            existing = intents[semantic_duplicate_index]
            if candidate_expected_value <= float(existing.expected_value or 0.0):
                continue
            intents[semantic_duplicate_index] = existing.model_copy(
                update={
                    "query_hint": query_hint,
                    "target_candidate": need.hypothesis_label,
                    "active_state": need.active_state,
                    "unresolved_critical_finding": need.unresolved_critical_finding,
                    "rival_mechanism": need.rival_mechanism,
                    "action_hazard": need.action_hazard,
                    "desired_discriminator": need.desired_discriminator,
                    "priority": need.priority,
                    "decision_relevance": need.decision_relevance,
                    "expected_value": candidate_expected_value,
                }
            )
            continue
        seen.add(key)
        intents.append(
            RetrievalIntent(
                objective=need.objective,
                decision_target=need.decision_target,
                query_hint=query_hint,
                target_candidate=need.hypothesis_label,
                active_state=need.active_state,
                unresolved_critical_finding=need.unresolved_critical_finding,
                rival_mechanism=need.rival_mechanism,
                action_hazard=need.action_hazard,
                desired_discriminator=need.desired_discriminator,
                priority=need.priority,
                decision_relevance=need.decision_relevance,
                expected_value=candidate_expected_value,
            )
        )
    intents.sort(key=lambda item: (item.expected_value, item.priority, item.decision_relevance), reverse=True)
    return intents[:6]


def summarize_query_quality(intents: list[RetrievalIntent]) -> dict[str, float | bool]:
    policy = load_runtime_policy()
    if not intents:
        return {
            "query_hygiene_score": 0.0,
            "hazard_leakage_rate": 0.0,
            "hazard_leakage_detected": False,
            "novelty_gain": 0.0,
            "repeated_query_ratio": 0.0,
            "semantic_dedup_ratio": 0.0,
            "pairwise_discriminator_coverage": 0.0,
            "marginal_utility_score": 0.0,
        }
    scores: list[float] = []
    leakage_hits = 0
    normalized_queries: list[str] = []
    query_tokens: list[set[str]] = []
    pairwise_discriminator_hits = 0
    for intent in intents:
        hygiene_score, leakage = _query_hygiene(
            intent.query_hint,
            decision_target=intent.decision_target,
            action_hazard=intent.action_hazard,
            rival_mechanism=intent.rival_mechanism,
        )
        scores.append(hygiene_score)
        normalized = sanitize_query_text(intent.query_hint, max_terms=14)
        if normalized:
            normalized_queries.append(normalized)
            query_tokens.append(set(normalized.split()))
        if leakage:
            leakage_hits += 1
        if intent.decision_target == "discriminate" and str(intent.rival_mechanism or "").strip():
            pairwise_discriminator_hits += 1

    unique_query_count = len(set(normalized_queries)) if normalized_queries else 0
    semantic_cluster_count = 0
    if query_tokens:
        clusters: list[set[str]] = []
        for tokens in query_tokens:
            matched_cluster = False
            for cluster_tokens in clusters:
                if _token_jaccard(tokens, cluster_tokens) >= 0.72:
                    matched_cluster = True
                    break
            if not matched_cluster:
                clusters.append(tokens)
        semantic_cluster_count = len(clusters)
    repeated_query_ratio = 0.0
    semantic_dedup_ratio = 0.0
    novelty_gain = 0.0
    if normalized_queries:
        repeated_query_ratio = round(max(0.0, 1.0 - (unique_query_count / len(normalized_queries))), 2)
        semantic_dedup_ratio = round(max(0.0, 1.0 - (semantic_cluster_count / len(normalized_queries))), 2)
    if policy.retrieval_query_novelty_metrics_enabled and len(query_tokens) >= 2:
        novelty_scores: list[float] = []
        for index, current_tokens in enumerate(query_tokens):
            others = [tokens for idx, tokens in enumerate(query_tokens) if idx != index]
            if not current_tokens or not others:
                continue
            best_overlap = 0.0
            for other_tokens in others:
                union = current_tokens | other_tokens
                if not union:
                    continue
                overlap = len(current_tokens & other_tokens) / len(union)
                best_overlap = max(best_overlap, overlap)
            novelty_scores.append(max(0.0, 1.0 - best_overlap))
        novelty_gain = round(sum(novelty_scores) / max(1, len(novelty_scores)), 2) if novelty_scores else 0.0
    query_hygiene_score = round(sum(scores) / max(1, len(scores)), 2)
    hazard_leakage_rate = round(leakage_hits / max(1, len(intents)), 2)
    pairwise_discriminator_coverage = round(pairwise_discriminator_hits / max(1, len(intents)), 2)
    marginal_utility_score = round(
        max(
            0.0,
            min(
                1.0,
                (query_hygiene_score * 0.34)
                + (pairwise_discriminator_coverage * 0.32)
                + (novelty_gain * 0.24)
                + ((1.0 - repeated_query_ratio) * 0.06)
                + ((1.0 - semantic_dedup_ratio) * 0.04)
                - (hazard_leakage_rate * 0.2),
            ),
        ),
        2,
    )
    return {
        "query_hygiene_score": query_hygiene_score,
        "hazard_leakage_rate": hazard_leakage_rate,
        "hazard_leakage_detected": bool(leakage_hits),
        "novelty_gain": novelty_gain,
        "repeated_query_ratio": repeated_query_ratio,
        "semantic_dedup_ratio": semantic_dedup_ratio,
        "pairwise_discriminator_coverage": pairwise_discriminator_coverage,
        "marginal_utility_score": marginal_utility_score,
    }




def build_evidence_bundle(
    patient_input: PatientInput,
    findings: StructuredFindings,
    risk_profile: RiskProfile,
    differential: DifferentialSet,
    hypothesis_frontier: HypothesisFrontier | None = None,
    interventions: InterventionSet | None = None,
    fact_graph: FactGraph | None = None,
    generated_candidate_profiles: list[GeneratedCandidateProfile] | None = None,
    contradiction_clusters: list[ContradictionCluster] | None = None,
) -> EvidenceBundle:
    """Build a provisional evidence bundle from typed frontier state."""
    policy = load_runtime_policy()
    registry = load_syndrome_registry()
    profile_by_label = {profile.label: profile for profile in (generated_candidate_profiles or [])}
    available_signals = _available_signals(patient_input, findings)
    intents: list[RetrievalIntent] = []
    items: list[EvidenceItem] = []
    coverage_inputs: list[float] = []
    evidence_needs = _gap_driven_evidence_needs(
        findings,
        risk_profile,
        hypothesis_frontier,
        interventions,
        fact_graph,
        contradiction_clusters=contradiction_clusters,
    )
    intents.extend(_needs_to_intents(evidence_needs, language=findings.language))

    # Plan Item 6 — axis-template fallback. When the gap-driven planner
    # produces fewer than `min_retrieval_intents` informative intents (thin
    # frontier or every query degenerated), synthesize axis-pair queries from
    # phenotype_fingerprint.axis_weights so retrieval is not silently empty.
    _min_intents = int(getattr(policy, "min_retrieval_intents", 2))
    _axis_fallback_on = bool(getattr(policy, "retrieval_axis_fallback_enabled", True))
    if _axis_fallback_on and len(intents) < _min_intents:
        fallback = _axis_template_intents(findings, limit=max(2, _min_intents - len(intents) + 1))
        if fallback:
            intents.extend(fallback)

    for candidate in differential.candidates[:3]:
        profile = profile_by_label.get(candidate.label) or (registry.by_id(candidate.label) if registry else None)
        query_hint = _generic_query_hint(candidate.label, fact_graph, findings) if _is_generic_label(candidate.label) else (_generic_query_hint(candidate.label, fact_graph, findings) or candidate.label.replace("_", " "))
        if not profile:
            items.append(
                EvidenceItem(
                    source="frontier_hint",
                    title=candidate.label,
                    citation=f"frontier://{candidate.label}",
                    excerpt="Unmapped but specific frontier hypothesis; retrieval should preserve specificity.",
                    trust_score=0.12,
                    linked_hypotheses=[candidate.label],
                    relation_type="neutral",
                    verification_status="unverified",
                )
            )
            intents.append(
                RetrievalIntent(
                    objective="support_or_refute_candidate",
                    decision_target="confirm",
                    query_hint=query_hint,
                    target_candidate=candidate.label,
                    priority=max(0.1, round(candidate.score, 2)),
                )
            )
            coverage_inputs.append(
                _coverage_score(
                    [need.query_hint for need in evidence_needs if need.hypothesis_label == candidate.label and need.query_hint],
                    available_signals,
                    bool(patient_input.labs),
                )
            )
            continue
        intents.extend(_candidate_intents(profile, candidate.score, query_hint, generic_label=_is_generic_label(candidate.label)))
        coverage_inputs.append(_coverage_score(profile.evidence_needs, available_signals, bool(patient_input.labs)))
        items.append(
            EvidenceItem(
                source="syndrome_registry",
                title=profile.label,
                citation=f"seed://syndrome_profiles/{profile.id}",
                excerpt=profile.summary,
                trust_score=0.36,
                linked_hypotheses=[candidate.label],
                relation_type="neutral",
                verification_status="unverified",
            )
        )
        if profile.challenge_queries:
            items.append(
                EvidenceItem(
                    source="retrieval_plan",
                    title=f"Challenge query for {profile.label}",
                    citation=f"plan://challenge/{profile.id}",
                    excerpt=profile.challenge_queries[0],
                    trust_score=0.22,
                    linked_hypotheses=[candidate.label],
                    relation_type="neutral",
                    verification_status="unverified",
                )
            )

    profile_coverage = round(sum(coverage_inputs) / len(coverage_inputs), 2) if coverage_inputs else 0.0
    intent_coverage = round(len(intents) / max(1, len(evidence_needs)), 2) if evidence_needs else 0.0
    coverage = round((intent_coverage * 0.6) + (profile_coverage * 0.4), 2)
    contradiction_mass = 0.0
    if len(differential.candidates) > 1:
        gap = abs(differential.candidates[0].score - differential.candidates[1].score)
        contradiction_mass = round(max(0.0, 0.18 - gap), 2)
    if risk_profile.manual_review_required:
        contradiction_mass = min(0.45, round(contradiction_mass + 0.08, 2))

    return EvidenceBundle(
        items=items,
        coverage=coverage,
        contradiction_mass=contradiction_mass,
        retrieval_intents=intents,
        evidence_needs=evidence_needs,
    )


# ---------------------------------------------------------------------------
# Retrieval reranker (merged from reranker.py)
# ---------------------------------------------------------------------------

import re as _re


def _tokenize_evidence(value: str) -> set[str]:
    return {token for token in _re.findall(r"[a-z0-9]{3,}", str(value or "").lower()) if token}


_GENERIC_BUCKET_LABELS = {
    "cardiorespiratory_process", "infectious_inflammatory_process",
    "metabolic_or_abdominal_process", "neurologic_process",
    "undifferentiated_high_variance_process",
}


def _is_generic_label(label: str) -> bool:
    return str(label or "") in _GENERIC_BUCKET_LABELS or str(label or "").endswith("_process")


class RetrievalReranker:
    """Heuristic fallback for MedCPT-style reranking."""

    def rerank(
        self,
        *,
        query_hints: list[str],
        items: list["EvidenceItem"],
        findings: "StructuredFindings",
        differential: "DifferentialSet",
    ) -> "tuple[list[EvidenceItem], RetrievalRankingStats]":
        from src.cdss.contracts.models import RetrievalRankingStats
        if not items:
            # Total retrieval failure: every non-generic candidate is starved.
            starved = [
                c.label for c in (differential.candidates or [])[:8]
                if not _is_generic_label(c.label)
            ]
            return items, RetrievalRankingStats(
                query_encoder_used="heuristic_fallback",
                cross_encoder_used="heuristic_fallback",
                evidence_starvation_flag=bool(starved),
                starved_candidates=starved,
                coverage_per_candidate={label: 0.0 for label in starved},
            )

        query_tokens: set[str] = set()
        unique_hints: list[str] = []
        for hint in query_hints[:8]:
            query_tokens.update(_tokenize_evidence(hint))
            if hint and hint not in unique_hints:
                unique_hints.append(hint)
        summary_tokens = _tokenize_evidence(findings.summary)
        candidate_tokens = {
            token for candidate in differential.candidates[:4]
            for token in _tokenize_evidence(candidate.label.replace("_", " "))
            if not _is_generic_label(candidate.label)
        }
        rescored: list[tuple[float, "EvidenceItem"]] = []
        specific_hits = 0
        alignment_total = 0.0
        specificity_bonus_w = clinical_thresholds.get_float("retrieval_reranker.specificity_bonus", 0.08)
        query_overlap_w = clinical_thresholds.get_float("retrieval_reranker.query_overlap_weight", 0.03)
        summary_overlap_w = clinical_thresholds.get_float("retrieval_reranker.summary_overlap_weight", 0.01)
        novelty_specificity_w = clinical_thresholds.get_float("retrieval_reranker.novelty_specificity_weight", 0.45)
        novelty_alignment_w = clinical_thresholds.get_float("retrieval_reranker.novelty_alignment_weight", 0.35)
        novelty_diversity_w = clinical_thresholds.get_float("retrieval_reranker.novelty_diversity_weight", 0.20)
        for item in items:
            title_tokens = _tokenize_evidence(item.title)
            excerpt_tokens = _tokenize_evidence(item.excerpt)
            overlap = len((title_tokens | excerpt_tokens) & query_tokens)
            candidate_overlap = len((title_tokens | excerpt_tokens) & candidate_tokens)
            summary_overlap = len((title_tokens | excerpt_tokens) & summary_tokens)
            specificity_bonus = specificity_bonus_w if candidate_overlap > 0 else 0.0
            score = round(item.trust_score + (overlap * query_overlap_w) + (summary_overlap * summary_overlap_w) + specificity_bonus, 3)
            if candidate_overlap > 0:
                specific_hits += 1
            alignment_total += min(1.0, (overlap * 0.08) + (candidate_overlap * 0.14))
            rescored.append((score, item.model_copy(update={"trust_score": min(0.99, score)})))
        rescored.sort(key=lambda pair: pair[0], reverse=True)
        reranked_items = [item for _, item in rescored]
        specificity_gain = round(min(1.0, specific_hits / max(1, len(reranked_items))), 2)
        citation_alignment = round(min(1.0, alignment_total / max(1, len(reranked_items))), 2)
        repeated_query_ratio = round(max(0.0, 1.0 - (len(unique_hints) / max(1, len(query_hints[:8])))), 2) if query_hints else 0.0
        novelty_gain = round(max(0.0, min(1.0, (specificity_gain * novelty_specificity_w) + (citation_alignment * novelty_alignment_w) + ((1.0 - repeated_query_ratio) * novelty_diversity_w))), 2)

        # Per-candidate evidence coverage: count items whose tokens overlap the
        # candidate's label tokens, normalized by item count. Starvation flag
        # fires when any non-generic candidate has zero overlapping items.
        coverage_per_candidate: dict[str, float] = {}
        starved_candidates: list[str] = []
        for candidate in (differential.candidates or [])[:8]:
            if _is_generic_label(candidate.label):
                continue
            cand_tokens = _tokenize_evidence(candidate.label.replace("_", " "))
            if not cand_tokens:
                continue
            hits = 0
            for item in reranked_items:
                tokens = _tokenize_evidence(item.title) | _tokenize_evidence(item.excerpt)
                if tokens & cand_tokens:
                    hits += 1
            ratio = round(hits / max(1, len(reranked_items)), 3)
            coverage_per_candidate[candidate.label] = ratio
            if hits == 0:
                starved_candidates.append(candidate.label)
        starvation_flag = bool(starved_candidates)

        return reranked_items, RetrievalRankingStats(
            query_encoder_used="heuristic_fallback", cross_encoder_used="heuristic_fallback",
            retrieved_count=len(items), reranked_count=len(reranked_items),
            specificity_gain=specificity_gain, citation_alignment=citation_alignment,
            novelty_gain=novelty_gain, repeated_query_ratio=repeated_query_ratio,
            coverage_per_candidate=coverage_per_candidate,
            evidence_starvation_flag=starvation_flag,
            starved_candidates=starved_candidates,
        )
