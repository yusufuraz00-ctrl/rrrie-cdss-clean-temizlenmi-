"""Adaptive phenotype fingerprints, prototype memory, and contrastive helpers."""

from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import logging

from src.cdss.contracts.models import (
    PhenotypeFingerprint,
    PrototypeMatch,
    PrototypeMemoryRecord,
    PrototypeUpdateSuggestion,
    StructuredFindings,
)

logger = logging.getLogger(__name__)

from src.cdss.knowledge.registry import (
    ChainedTrigger,
    SyndromeCombo,
    SyndromeProfile,
    load_syndrome_registry,
)
from src.cdss.text_normalization import ascii_fold

_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "bir",
    "bu",
    "by",
    "da",
    "de",
    "for",
    "gibi",
    "her",
    "hocam",
    "i",
    "ile",
    "in",
    "like",
    "mi",
    "my",
    "of",
    "or",
    "that",
    "the",
    "to",
    "ve",
    "was",
    "with",
}

_SLOT_WEIGHTS: dict[str, float] = {
    "tempo": 1.1,
    "trigger": 1.1,
    "trajectory": 1.35,
    "cluster": 1.2,
    "hemodynamics": 1.2,
    "physiology": 0.95,
    "localization": 0.9,
    "context": 1.05,
    "exposure": 1.25,
    "physique": 1.15,
}


def _runtime_substrate_mode() -> str:
    import os

    raw = str(os.getenv("CDSS_RUNTIME_SUBSTRATE") or "").strip().lower()
    if raw in {"adaptive_only", "adaptive_first", "compatibility"}:
        return raw
    return "adaptive_first"


def _runtime_slot_axis_map() -> dict[str, tuple[tuple[str, float], ...]]:
    """Return slot→axis weight map.

    In adaptive_first mode (default) no static axis priors are applied —
    the LLM phenotype atoms drive axis weights. In compatibility mode the
    hardcoded map is used as a fallback for registry-seeded prototypes.
    """
    import os

    mode = _runtime_substrate_mode()
    if mode == "compatibility":
        # Minimal axis map — just expose/hemodynamic axes which are objective
        return {
            "exposure": (("toxic_exposure", 1.0),),
            "hemodynamics": (("hemodynamic", 1.0), ("hemorrhagic", 0.38)),
        }
    # adaptive_first: no static priors — rely on LLM phenotype atoms
    raw = str(os.getenv("CDSS_DISABLE_LEGACY_SLOT_PRIORS") or "").strip().lower()
    if raw in {"0", "false", "no", "off"}:
        return {
            "exposure": (("toxic_exposure", 1.0),),
            "hemodynamics": (("hemodynamic", 1.0), ("hemorrhagic", 0.38)),
        }
    return {}


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ascii_fold(str(value or "")).strip().lower()).strip("_")


def _tokenize(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-z0-9]{3,}", ascii_fold(str(value or "")).lower()) if token not in _STOPWORDS]


def _specific_tokens(tokens: set[str]) -> set[str]:
    return {token for token in tokens if len(token) >= 5 and not token.isdigit()}


def _dedupe(values: list[str], *, limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value or "").split()).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if limit and len(output) >= limit:
            break
    return output


def _normalized_value(value: str, *, max_terms: int = 6) -> str:
    tokens = _tokenize(value)
    return "_".join(tokens[:max_terms]) if tokens else _slug(value) or "unspecified"


def _source_slots(source_hint: str) -> tuple[str, ...]:
    """Map a text source type to phenotype slots.

    Fast heuristic mapping — the LLM phenotype atoms are the primary slot
    source. This function is used only when building registry prototypes
    from profile text, where LLM atoms are not available.
    """
    _QUICK_MAP: dict[str, tuple[str, ...]] = {
        "timeline": ("tempo",),
        "exposure": ("exposure", "context", "trigger"),
        "red_flag": ("hemodynamics", "physiology"),
        "summary":  ("context",),
        "positive": ("physiology", "localization"),
        "negative": ("context", "localization"),
        "raw":      ("context",),
        "narrative": ("context", "trajectory"),
        "cue":      ("physiology", "context"),
        "need":     ("trigger", "context"),
        "challenge": ("trajectory", "context"),
        "risk":     ("hemodynamics", "context"),
        "constitutional": ("physique",),
    }
    return _QUICK_MAP.get(source_hint, ("context",))


_NEGATIVE_SOURCE_HINTS: frozenset[str] = frozenset({"negative"})


def _candidate_sources(findings: StructuredFindings) -> list[tuple[str, str]]:
    """Backward-compatible flat source list (positives + negatives mixed).

    Deprecated for fingerprint construction; use ``_split_sources_by_polarity``
    so positive and negative findings stay in parallel slot dicts. Retained
    only for any external caller that wants the raw merged stream.
    """

    positives, negatives = _split_sources_by_polarity(findings)
    return positives + negatives


def _split_sources_by_polarity(
    findings: StructuredFindings,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return (positive_sources, negative_sources).

    Negative findings (explicitly-denied symptoms) MUST NOT contaminate the
    positive slot dict — earlier versions merged them, which surfaced as
    atoms appearing simultaneously in NEGATIVES and PHENOTYPE_ATOMS in the
    IE verifier output.
    """

    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []
    if findings.summary:
        positives.append(("summary", findings.summary))
    positives.extend(("positive", item) for item in findings.positive_findings[:14])
    negatives.extend(("negative", item) for item in findings.negative_findings[:10])
    positives.extend(("timeline", item) for item in findings.timeline[:10])
    positives.extend(("exposure", item) for item in findings.exposures[:10])
    positives.extend(("red_flag", item) for item in findings.red_flags[:8])
    positives.extend(("raw", item) for item in findings.raw_segments[:18])
    positives.extend(("narrative", item) for item in (findings.context_lanes or {}).get("patient_narrative", [])[:10])
    positives.extend(("constitutional", item) for item in (getattr(findings, "constitutional_findings", None) or [])[:6])
    positives = [(kind, value) for kind, value in positives if str(value or "").strip()]
    negatives = [(kind, value) for kind, value in negatives if str(value or "").strip()]
    return positives, negatives


def derive_slot_values_from_texts(texts: list[tuple[str, str]]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    slot_values: dict[str, list[str]] = {}
    slot_evidence: dict[str, list[str]] = {}
    for source_hint, text in texts:
        normalized = _normalized_value(text)
        for slot in _source_slots(source_hint):
            slot_values.setdefault(slot, []).append(normalized)
            slot_evidence.setdefault(slot, []).append(text)
    return (
        {key: _dedupe(values, limit=8) for key, values in slot_values.items()},
        {key: _dedupe(values, limit=8) for key, values in slot_evidence.items()},
    )


def _axis_weights_from_slots(slot_values: dict[str, list[str]], slot_evidence: dict[str, list[str]]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for slot, axis_targets in _runtime_slot_axis_map().items():
        value_count = len(slot_values.get(slot, []))
        evidence_count = len(slot_evidence.get(slot, []))
        if value_count <= 0 and evidence_count <= 0:
            continue
        slot_strength = min(1.0, 0.22 + (value_count * 0.08) + (evidence_count * 0.04))
        for axis, multiplier in axis_targets:
            weights[axis] = weights.get(axis, 0.0) + (slot_strength * multiplier)

    if slot_values.get("exposure") and slot_values.get("hemodynamics"):
        weights["toxic_exposure"] = weights.get("toxic_exposure", 0.0) + 0.08
        weights["hemodynamic"] = weights.get("hemodynamic", 0.0) + 0.08

    if slot_values.get("tempo") and slot_values.get("physiology"):
        weights["inflammatory_infectious"] = weights.get("inflammatory_infectious", 0.0) + 0.06

    if weights:
        weights = {axis: round(min(1.0, score), 2) for axis, score in weights.items() if score > 0}
    if not weights:
        weights["undifferentiated"] = 0.35
    return weights


def _embedding_terms(slot_values: dict[str, list[str]], slot_evidence: dict[str, list[str]]) -> list[str]:
    tokens: list[str] = []
    for values in slot_values.values():
        for value in values[:4]:
            tokens.extend(_tokenize(value))
    for values in slot_evidence.values():
        for value in values[:2]:
            tokens.extend(_tokenize(value))
    return _dedupe(tokens, limit=36)


def _negative_terms(findings: StructuredFindings) -> list[str]:
    tokens: list[str] = []
    for item in findings.negative_findings[:12]:
        tokens.extend(_tokenize(item))
    return _dedupe(tokens, limit=24)


def _contradiction_terms(findings: StructuredFindings) -> list[str]:
    tokens: list[str] = []
    for raw in findings.input_context[:12]:
        text = str(raw or "").strip()
        if ":" not in text:
            continue
        prefix, value = text.split(":", 1)
        if _slug(prefix) not in {"contradiction", "masquerade_risk", "hazard"}:
            continue
        tokens.extend(_tokenize(value))
    return _dedupe(tokens, limit=24)


_SEX_ALIASES: dict[str, str] = {
    "f": "female", "woman": "female", "kadin": "female",
    "m": "male",   "man": "male",    "erkek": "male",
}


def _compute_clinical_context(findings: StructuredFindings) -> dict:
    """Extract objective, structured clinical context from findings.

    Returns a dict suitable for storage in PhenotypeFingerprint.clinical_context
    and consumed by _demographic_boost_score() in match_prototypes().
    """
    ctx: dict = {}

    # ── Demographics ──────────────────────────────────────────────────────────
    demographics = dict(findings.demographics or {})
    sex_raw = str(
        demographics.get("sex") or demographics.get("gender") or ""
    ).strip().lower()
    sex = _SEX_ALIASES.get(sex_raw, sex_raw if sex_raw in ("female", "male") else "")
    if sex:
        ctx["sex"] = sex

    age_raw = demographics.get("age") or demographics.get("age_years") or demographics.get("yaş") or demographics.get("yas")
    try:
        ctx["age"] = float(str(age_raw).strip())
    except (TypeError, ValueError):
        pass

    # ── Hemodynamics: shock index ─────────────────────────────────────────────
    vitals = dict(findings.derived_vitals or {})

    def _vfloat(*keys: str) -> float | None:
        for k in keys:
            val = vitals.get(k)
            if val is None:
                continue
            raw = str(val).strip()
            if "/" in raw:
                raw = raw.split("/")[0]
            try:
                return float(raw)
            except ValueError:
                continue
        return None

    hr = _vfloat("hr", "heart_rate", "pulse", "nabız")
    sbp = _vfloat("sbp", "systolic_bp", "bp_systolic", "systolic")
    if sbp is None:
        bp_str = str(vitals.get("bp") or vitals.get("blood_pressure") or "")
        if "/" in bp_str:
            try:
                sbp = float(bp_str.split("/")[0].strip())
            except ValueError:
                pass
    if hr and sbp and sbp > 0:
        ctx["shock_index"] = round(hr / sbp, 3)

    # ── Complaint domains (keyword scan over complaint text) ──────────────────
    complaint_text = " ".join([
        str(findings.summary or ""),
        *[str(p) for p in findings.positive_findings[:14]],
        *[str(p) for p in findings.negative_findings[:6]],
        *[str(p) for p in findings.raw_segments[:10]],
    ]).lower()
    domains: list[str] = []
    if re.search(r"\b(abdomen|abdominal|belly|stomach|pelvic|pelvis|cramp)\b", complaint_text):
        domains.append("abdominal")
    if re.search(r"\b(pelvis|pelvic|gynecolog|ovarian|uterine|vaginal|menstrual|period)\b", complaint_text):
        domains.append("pelvic")
    if re.search(r"\b(chest|thorac|cardiac|palpitat|precordial)\b", complaint_text):
        domains.append("chest")
    if re.search(r"\b(head|skull|cranial|neuro|cerebral|seizure|syncope|faint)\b", complaint_text):
        domains.append("neuro")
    if domains:
        ctx["complaint_domains"] = domains

    # Constitutional phenotype — LLM-extracted, zero Python thresholds
    const_findings = list(getattr(findings, "constitutional_findings", None) or [])
    if const_findings:
        ctx["constitutional_findings"] = const_findings

    return ctx


def _demographic_boost_score(demographic_risk_profile: dict, clinical_context: dict, confidence: float = 0.5) -> float:
    """Compute a score boost [0.0, cap] from a learned demographic_risk_profile.

    - Reads the LLM-synthesised profile from the learned store (never hardcoded).
    - Returns 0.0 if no profile, no match, or confidence too low.
    - Max boost scales with profile confidence (refined by case outcomes).
    - A must_not_miss full-match at max confidence contributes ~+0.22 to similarity.
    """
    if not demographic_risk_profile:
        return 0.0

    score = 0.0
    total = 0.0

    # Sex
    expected_sex = str(demographic_risk_profile.get("sex") or "any").lower()
    if expected_sex not in ("any", ""):
        total += 1.0
        patient_sex = str(clinical_context.get("sex") or "").lower()
        if patient_sex == expected_sex:
            score += 1.0
        elif not patient_sex:
            # Unknown sex — partial credit to avoid full suppression
            score += 0.3

    # Age range
    age_min = demographic_risk_profile.get("age_min")
    age_max = demographic_risk_profile.get("age_max")
    if age_min is not None or age_max is not None:
        total += 1.0
        patient_age = clinical_context.get("age")
        if patient_age is not None:
            lo = age_min if age_min is not None else 0
            hi = age_max if age_max is not None else 999
            if lo <= float(patient_age) <= hi:
                score += 1.0
        else:
            score += 0.3  # age unknown — partial credit

    # Hemodynamic flags
    hemo_flags: list[str] = demographic_risk_profile.get("hemodynamic_flags") or []
    if hemo_flags:
        total += 1.0
        si = clinical_context.get("shock_index")
        if si is not None:
            si = float(si)
            from src.cdss.core import thresholds as _ct
            si_critical = _ct.get_float("shock_index.critical_gte", 1.2)
            si_high = _ct.get_float("shock_index.high_gte", 0.9)
            if "shock_index_critical" in hemo_flags and si >= si_critical:
                score += 1.0
            elif "shock_index_high" in hemo_flags and si >= si_high:
                score += 1.0
            elif "hypotension" in hemo_flags and si >= 0.8:
                score += 0.7
            elif "tachycardia" in hemo_flags and si >= 0.6:
                score += 0.5
        else:
            score += 0.2  # vitals unknown — small partial credit

    # Complaint domains (at least one intersection required)
    req_domains: list[str] = demographic_risk_profile.get("complaint_domains") or []
    if req_domains and "any" not in req_domains:
        total += 1.0
        patient_domains = set(clinical_context.get("complaint_domains") or [])
        req_set = set(req_domains)
        if req_set & patient_domains:
            score += 1.0
        elif not patient_domains:
            score += 0.2  # unknown complaint — minimal credit

    # Constitutional phenotype (learned per-syndrome, token-overlap against free-form LLM output)
    req_constitutional: list[str] = demographic_risk_profile.get("constitutional_flags") or []
    if req_constitutional and "any" not in req_constitutional:
        total += 1.0
        patient_const_text = " ".join(clinical_context.get("constitutional_findings") or []).lower()
        if patient_const_text and any(
            flag.replace("_", " ") in patient_const_text or flag in patient_const_text
            for flag in req_constitutional
        ):
            score += 1.0
        elif not patient_const_text:
            score += 0.2

    if total <= 0.0:
        return 0.0

    match_ratio = score / total

    # Priority cap: must_not_miss gets largest boost
    priority = str(demographic_risk_profile.get("priority") or "moderate").lower()
    base_cap = {"must_not_miss": 0.22, "high": 0.12, "moderate": 0.06, "low": 0.03}.get(priority, 0.06)

    # Scale by confidence (LLM prior 0.5, grows toward 1.0 from confirmed cases)
    effective_cap = base_cap * min(1.0, confidence * 1.6)

    return round(match_ratio * effective_cap, 4)


_LOCATION_PAIN_RE = re.compile(
    r"\b(right\s+shoulder|left\s+shoulder|shoulder|neck|scapular?|back)\b.{0,12}\b(tip\s+)?pain\b",
    re.IGNORECASE,
)
_LOCAL_TENDERNESS_RE = re.compile(
    r"\b(local|point|direct|site)\s+(tenderness|tender)\b|\bno\s+tenderness\b",
    re.IGNORECASE,
)
_TRAUMA_RE = re.compile(
    r"\b(injur|trauma|heavy\s+lift|sprain|strain|fall|hit|blow|fractur)\b",
    re.IGNORECASE,
)


def _detect_referred_pain_signals(findings: StructuredFindings) -> list[tuple[str, str]]:
    """Derive referred-pain red-flags from physical exam contradictions.

    Pattern: [location] pain reported + no local tenderness on exam + no trauma history
    → anatomical-illusion referred pain is likely; add as red_flag so it feeds the
    hemorrhagic/localization axis and boosts ruptured_ectopic_pregnancy prototype score.
    """
    all_positive = " ".join(str(s) for s in findings.positive_findings[:16] + findings.raw_segments[:12])
    all_negative = " ".join(str(s) for s in findings.negative_findings[:12])
    all_input = " ".join(str(s) for s in findings.input_context[:12])
    combined = f"{all_positive} {all_input}"

    pain_match = _LOCATION_PAIN_RE.search(combined)
    if not pain_match:
        return []

    location = pain_match.group(1).lower().strip()

    # Local tenderness absent?
    has_local_tenderness = bool(_LOCAL_TENDERNESS_RE.search(all_positive))
    explicit_no_tenderness = bool(_LOCAL_TENDERNESS_RE.search(all_negative)) or (
        re.search(rf"\b{re.escape(location)}\b.{{0,30}}\bno\b.{{0,15}}\btenderness\b", combined, re.IGNORECASE) is not None
    ) or (
        re.search(r"\bno\s+local\s+tenderness\b|\bpainless\s+range\b|\bfull.{0,10}range.{0,10}motion\b", combined, re.IGNORECASE) is not None
    )

    if has_local_tenderness and not explicit_no_tenderness:
        return []

    # No traumatic cause?
    trauma_present = bool(_TRAUMA_RE.search(combined))
    if trauma_present:
        return []

    signals = [
        ("red_flag", f"referred pain suspected at {location} no local pathology"),
        ("red_flag", "diaphragm irritation suspected intraperitoneal source"),
        ("positive", f"{location} pain without local tenderness anatomical mismatch"),
    ]
    return signals


def build_phenotype_fingerprint(findings: StructuredFindings) -> PhenotypeFingerprint:
    referred_pain_signals = _detect_referred_pain_signals(findings)
    positives, negatives = _split_sources_by_polarity(findings)
    if referred_pain_signals:
        positives = positives + referred_pain_signals
    slot_values, slot_evidence = derive_slot_values_from_texts(positives)
    neg_slot_values, neg_slot_evidence = derive_slot_values_from_texts(negatives)
    temporal_signature = _dedupe([*slot_values.get("tempo", []), *slot_values.get("trigger", [])], limit=6)
    evidence_spans = _dedupe([value for _, value in positives + negatives], limit=24)
    return PhenotypeFingerprint(
        slots=slot_values,
        slot_evidence=slot_evidence,
        negative_slots=neg_slot_values,
        negative_slot_evidence=neg_slot_evidence,
        axis_weights=_axis_weights_from_slots(slot_values, slot_evidence),
        temporal_signature=temporal_signature,
        embedding_terms=_embedding_terms(slot_values, slot_evidence),
        evidence_spans=evidence_spans,
        negative_terms=_negative_terms(findings),
        contradiction_terms=_contradiction_terms(findings),
        clinical_context=_compute_clinical_context(findings),
    )


def dominant_fingerprint_terms(fingerprint: PhenotypeFingerprint, *, limit: int = 10) -> list[str]:
    terms: list[str] = []
    for slot in ("tempo", "trigger", "trajectory", "physiology", "exposure", "hemodynamics", "cluster", "localization", "context"):
        values = fingerprint.slots.get(slot, [])
        terms.extend(value.replace("_", " ") for value in values[:2])
    if not terms:
        terms.extend(term.replace("_", " ") for term in fingerprint.embedding_terms[:limit])
    return _dedupe(terms, limit=limit)


def prototype_state_frames(fingerprint: PhenotypeFingerprint, *, limit: int = 6) -> list[str]:
    frames: list[str] = []
    top_axes = [axis for axis, _ in sorted(fingerprint.axis_weights.items(), key=lambda item: item[1], reverse=True)[:3]]
    dominant_slots = dominant_fingerprint_terms(fingerprint, limit=4)
    for axis in top_axes:
        if axis == "undifferentiated":
            frames.append("adaptive_state__undifferentiated_pattern")
            continue
        focus = dominant_slots[0] if dominant_slots else axis
        frames.append(_slug(f"adaptive_state {axis} {focus}"))
    return _dedupe(frames, limit=limit)


def _profile_texts(profile: SyndromeProfile) -> list[tuple[str, str]]:
    texts: list[tuple[str, str]] = [("summary", profile.label.replace("_", " "))]
    if profile.summary:
        texts.append(("summary", profile.summary))
    texts.extend(("cue", item) for item in profile.cue_lexicon[:12])
    texts.extend(("need", item) for item in profile.evidence_needs[:8])
    texts.extend(("challenge", item) for item in profile.challenge_queries[:8])
    texts.extend(("risk", item) for item in profile.unsafe_interventions[:6])
    if profile.dangerous_if_treated_as:
        texts.append(("risk", profile.dangerous_if_treated_as))
    return [(kind, text) for kind, text in texts if str(text or "").strip()]


def _profile_to_prototype(profile: SyndromeProfile) -> PrototypeMemoryRecord:
    slot_values, slot_evidence = derive_slot_values_from_texts(_profile_texts(profile))
    mechanism_signature = dominant_fingerprint_terms(
        PhenotypeFingerprint(
            slots=slot_values,
            slot_evidence=slot_evidence,
            axis_weights=_axis_weights_from_slots(slot_values, slot_evidence),
            embedding_terms=_embedding_terms(slot_values, slot_evidence),
        ),
        limit=6,
    )
    return PrototypeMemoryRecord(
        id=profile.id,
        label=profile.label or profile.id,
        family_label=profile.id if profile.id.endswith("_process") else "",
        source="registry",
        slots=slot_values,
        slot_evidence=slot_evidence,
        axis_weights=_axis_weights_from_slots(slot_values, slot_evidence),
        mechanism_signature=mechanism_signature,
        discriminator_set=_dedupe([*profile.evidence_needs[:4], *profile.challenge_queries[:4]], limit=6),
        wrong_treatment_risks=_dedupe([*profile.unsafe_interventions[:4], profile.dangerous_if_treated_as], limit=4),
        embedding_terms=_embedding_terms(slot_values, slot_evidence),
        canonical_examples=_dedupe([profile.summary, *profile.cue_lexicon[:4]], limit=6),
    )


@lru_cache(maxsize=1)
def load_registry_prototypes() -> tuple[PrototypeMemoryRecord, ...]:
    registry = load_syndrome_registry()
    return tuple(_profile_to_prototype(profile) for profile in registry.profiles)


def _load_shadow_prototypes() -> list[PrototypeMemoryRecord]:
    path = Path("output") / "learning" / "prototype_memory.jsonl"
    if not path.exists():
        return []
    records: list[PrototypeMemoryRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        try:
            records.append(PrototypeMemoryRecord.model_validate(json.loads(raw)))
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue
    return records


def load_prototype_memory() -> list[PrototypeMemoryRecord]:
    return [*list(load_registry_prototypes()), *_load_shadow_prototypes()]


def _slot_overlap(case_values: list[str], proto_values: list[str]) -> float:
    if not case_values or not proto_values:
        return 0.0
    case_tokens = {token for value in case_values for token in _tokenize(value)}
    proto_tokens = {token for value in proto_values for token in _tokenize(value)}
    if not case_tokens or not proto_tokens:
        return 0.0
    case_specific = _specific_tokens(case_tokens)
    proto_specific = _specific_tokens(proto_tokens)
    shared_specific = case_specific & proto_specific
    if not shared_specific:
        return 0.0
    return len(shared_specific) / max(1, len(case_specific | proto_specific))


def _axis_overlap(case_axes: dict[str, float], proto_axes: dict[str, float]) -> float:
    numer = 0.0
    denom_a = 0.0
    denom_b = 0.0
    for axis in set(case_axes) | set(proto_axes):
        a = float(case_axes.get(axis, 0.0))
        b = float(proto_axes.get(axis, 0.0))
        numer += a * b
        denom_a += a * a
        denom_b += b * b
    if denom_a <= 0 or denom_b <= 0:
        return 0.0
    return numer / math.sqrt(denom_a * denom_b)


def _token_overlap(case_terms: list[str], proto_terms: list[str]) -> float:
    a = set(case_terms)
    b = set(proto_terms)
    if not a or not b:
        return 0.0
    a_specific = _specific_tokens(a)
    b_specific = _specific_tokens(b)
    if not a_specific or not b_specific:
        return 0.0
    return len(a_specific & b_specific) / max(1, len(a_specific | b_specific))


def _negative_overlap(fingerprint: PhenotypeFingerprint, prototype: PrototypeMemoryRecord) -> float:
    proto_terms = [
        *prototype.embedding_terms,
        *prototype.discriminator_set,
        *prototype.canonical_examples,
        *prototype.wrong_treatment_risks,
    ]
    term_overlap = _token_overlap(fingerprint.negative_terms, proto_terms) if fingerprint.negative_terms else 0.0

    # Slot-aware overlap: if a prototype's defining slot value (e.g.
    # trigger="dental_procedure") matches a case's negative slot value, the
    # patient explicitly denied a feature that the prototype requires. That is
    # a stronger signal than generic embedding-token overlap with negatives.
    slot_overlap = 0.0
    if fingerprint.negative_slots:
        slot_hits = 0
        slot_total = 0
        for slot, proto_values in (prototype.slots or {}).items():
            case_negatives = fingerprint.negative_slots.get(slot, [])
            if not proto_values:
                continue
            slot_total += 1
            if case_negatives and (set(proto_values) & set(case_negatives)):
                slot_hits += 1
        if slot_total:
            slot_overlap = slot_hits / slot_total

    return max(term_overlap, slot_overlap)


def _discriminator_absence_penalty(
    fingerprint: PhenotypeFingerprint,
    prototype: PrototypeMemoryRecord,
) -> float:
    """Penalise prototype when its key discriminating features are explicitly absent.

    Rationale: a prototype that requires chest pain + dyspnoea (e.g. PE) should
    be scored down when those features are in the negative-findings list or are
    simply absent from all positive slots.  Classic symptom-overlap scoring only
    checks what IS present; this function checks what SHOULD BE present but is not.

    Returns a penalty in [0.0, 1.0] that is multiplied by a weight at the call site.
    """
    discriminator_tokens = _prototype_discriminator_terms(prototype)
    if not discriminator_tokens:
        return 0.0

    discriminator_set = _specific_tokens(set(discriminator_tokens))
    if not discriminator_set:
        return 0.0

    positive_set = _specific_tokens(set(fingerprint.embedding_terms))
    negative_set = _specific_tokens(set(fingerprint.negative_terms))

    explicitly_absent = discriminator_set & negative_set
    simply_missing = discriminator_set - positive_set - negative_set

    # Explicit negation (e.g. "no chest pain") counts 3× compared to mere absence.
    weighted_absent = len(explicitly_absent) * 3 + len(simply_missing)
    max_possible = len(discriminator_set) * 3

    if max_possible <= 0:
        return 0.0

    return round(min(1.0, weighted_absent / max_possible), 2)


def _prototype_discriminator_terms(prototype: PrototypeMemoryRecord) -> list[str]:
    tokens: list[str] = []
    for item in [*prototype.discriminator_set[:6], *prototype.wrong_treatment_risks[:4], *prototype.canonical_examples[:4]]:
        tokens.extend(_tokenize(item))
    return _dedupe(tokens, limit=24)


# ---------------------------------------------------------------------------
# Symbolic combo / chain promotion (Item 2 — pathognomonic conjunctions)
# ---------------------------------------------------------------------------

_COMBO_PROMOTION_SIMILARITY = 0.92
_COMBO_PRIORITY_BUMP: dict[str, float] = {"must_not_miss": 0.05, "high": 0.02}


def _atom_tokens(atom: str) -> set[str]:
    return {token for token in _tokenize(atom.replace("_", " ")) if token}


def _fingerprint_positive_token_set(fingerprint: PhenotypeFingerprint) -> set[str]:
    tokens: set[str] = set()
    for value in fingerprint.embedding_terms:
        tokens.update(_atom_tokens(value))
    for values in fingerprint.slots.values():
        for value in values:
            tokens.update(_atom_tokens(value))
    return tokens


def _fingerprint_negative_token_set(fingerprint: PhenotypeFingerprint) -> set[str]:
    tokens: set[str] = set(fingerprint.negative_terms or [])
    for values in (fingerprint.negative_slots or {}).values():
        for value in values:
            tokens.update(_atom_tokens(value))
    return tokens


def _atom_present(atom: str, positive_tokens: set[str], negative_tokens: set[str]) -> bool:
    needed = _atom_tokens(atom)
    if not needed:
        return False
    if needed & negative_tokens:
        return False
    # Multi-token atoms ("acral_purple_lesions") require ALL their tokens
    # to appear in the positive set; single-token atoms behave as expected.
    return needed.issubset(positive_tokens)


def _context_constraints_met(
    constraint: dict, fingerprint: PhenotypeFingerprint
) -> bool:
    if not constraint:
        return True
    ctx = fingerprint.clinical_context or {}
    age = ctx.get("age")
    if "min_age" in constraint:
        if age is None or age < float(constraint["min_age"]):
            return False
    if "max_age" in constraint:
        if age is None or age > float(constraint["max_age"]):
            return False
    if "sex" in constraint:
        target = str(constraint["sex"] or "").strip().lower()
        observed = str(ctx.get("sex") or "").strip().lower()
        if target and observed and target != observed:
            return False
    return True


def _combo_match(
    combo: SyndromeCombo,
    positive_tokens: set[str],
    negative_tokens: set[str],
    fingerprint: PhenotypeFingerprint,
) -> tuple[bool, list[str]]:
    """Return (matched, matched_atom_labels). Matched iff ALL required_atoms
    present, no forbidden_atom present, context constraints satisfied."""

    if not combo.required_atoms:
        return False, []
    if not _context_constraints_met(combo.trigger_context or {}, fingerprint):
        return False, []
    matched_required: list[str] = []
    for atom in combo.required_atoms:
        if not _atom_present(atom, positive_tokens, negative_tokens):
            return False, []
        matched_required.append(atom)
    for atom in combo.forbidden_atoms or []:
        if _atom_present(atom, positive_tokens, negative_tokens):
            return False, []
    matched_optional = [
        atom
        for atom in (combo.optional_atoms or [])
        if _atom_present(atom, positive_tokens, negative_tokens)
    ]
    return True, matched_required + matched_optional


def _chain_match(
    chain: ChainedTrigger,
    positive_tokens: set[str],
    negative_tokens: set[str],
) -> tuple[bool, list[str]]:
    if not chain.trigger_atom or not chain.consequent_atoms:
        return False, []
    if not _atom_present(chain.trigger_atom, positive_tokens, negative_tokens):
        return False, []
    matched: list[str] = [chain.trigger_atom]
    for atom in chain.consequent_atoms:
        if not _atom_present(atom, positive_tokens, negative_tokens):
            return False, []
        matched.append(atom)
    matched_optional = [
        atom
        for atom in (chain.optional_atoms or [])
        if _atom_present(atom, positive_tokens, negative_tokens)
    ]
    return True, matched + matched_optional


def _combo_promotion_match(
    profile: SyndromeProfile,
    matched_atoms: list[str],
    *,
    promotion_kind: str,
    promotion_name: str,
    rationale: str,
    priority: str,
) -> PrototypeMatch:
    bump = _COMBO_PRIORITY_BUMP.get(priority, 0.0)
    similarity = round(min(0.99, _COMBO_PROMOTION_SIMILARITY + bump), 2)
    matched_slots: dict[str, list[str]] = {
        f"combo:{promotion_kind}": [promotion_name]
    }
    if matched_atoms:
        matched_slots["combo:atoms"] = list(matched_atoms[:6])
    mechanism_paths: list[str] = []
    if rationale:
        mechanism_paths.append(rationale)
    return PrototypeMatch(
        prototype_id=profile.id,
        label=profile.id,
        family_label=profile.label or profile.id,
        similarity=similarity,
        slot_overlap=similarity,
        axis_overlap=similarity,
        token_overlap=similarity,
        matched_slots=matched_slots,
        missing_slots={},
        evidence_spans=list(matched_atoms[:6]),
        mechanism_paths=mechanism_paths,
        discriminator_set=list(profile.cue_lexicon[:4]),
        wrong_treatment_risks=list(profile.unsafe_interventions[:4]),
        source=f"combo_promotion:{promotion_kind}",
    )


def detect_combo_promotions(
    fingerprint: PhenotypeFingerprint,
    *,
    registry=None,
) -> list[PrototypeMatch]:
    """Symbolic conjunction detector — items 2 of the cross-case plan.

    Scans every ``SyndromeProfile.pathognomonic_combos`` and
    ``chained_triggers`` against the fingerprint's atom set. When all required
    atoms are present (and forbidden ones absent), emits a high-similarity
    ``PrototypeMatch`` with source ``combo_promotion:<kind>``. The match
    bypasses bag-of-tokens scoring; downstream consumers treat it like any
    other match but it floats to the top of the similarity sort.

    No hardcoded clinical knowledge lives here — combos are loaded from the
    JSON overlay at ``data/cdss/knowledge/syndrome_combos.json``.
    """

    reg = registry if registry is not None else load_syndrome_registry()
    if not reg or not reg.profiles:
        return []
    positive_tokens = _fingerprint_positive_token_set(fingerprint)
    negative_tokens = _fingerprint_negative_token_set(fingerprint)
    if not positive_tokens:
        return []

    promotions: list[PrototypeMatch] = []
    for profile in reg.profiles:
        for combo in profile.pathognomonic_combos or []:
            ok, matched = _combo_match(combo, positive_tokens, negative_tokens, fingerprint)
            if not ok:
                continue
            promotions.append(
                _combo_promotion_match(
                    profile,
                    matched,
                    promotion_kind="combo",
                    promotion_name=combo.name or profile.id,
                    rationale=combo.rationale,
                    priority=combo.promotes_to_priority,
                )
            )
        for chain in profile.chained_triggers or []:
            ok, matched = _chain_match(chain, positive_tokens, negative_tokens)
            if not ok:
                continue
            promotions.append(
                _combo_promotion_match(
                    profile,
                    matched,
                    promotion_kind="chain",
                    promotion_name=chain.name or profile.id,
                    rationale=chain.rationale,
                    priority=chain.promotes_to_priority,
                )
            )
    # Deduplicate by prototype_id, keep highest similarity.
    best: dict[str, PrototypeMatch] = {}
    for promo in promotions:
        existing = best.get(promo.prototype_id)
        if not existing or promo.similarity > existing.similarity:
            best[promo.prototype_id] = promo
    return sorted(best.values(), key=lambda m: m.similarity, reverse=True)


def match_prototypes(
    fingerprint: PhenotypeFingerprint,
    *,
    prototypes: list[PrototypeMemoryRecord] | None = None,
    limit: int = 8,
) -> list[PrototypeMatch]:
    # Lazy import avoids circular dependency; demographic_profiler reads learned SQLite store.
    try:
        from src.cdss.learning.demographic_profiler import get_demographic_profile as _get_demo_profile
    except Exception:
        _get_demo_profile = lambda _: {}  # noqa: E731

    prototype_pool = prototypes or load_prototype_memory()
    matches: list[PrototypeMatch] = []
    for prototype in prototype_pool:
        weighted_slot_score = 0.0
        weighted_slot_total = 0.0
        matched_slots: dict[str, list[str]] = {}
        missing_slots: dict[str, list[str]] = {}
        for slot, case_values in fingerprint.slots.items():
            score = _slot_overlap(case_values, prototype.slots.get(slot, []))
            if score >= 0.12:
                matched_slots[slot] = case_values[:3]
                weight = float(_SLOT_WEIGHTS.get(slot, 1.0))
                weighted_slot_score += score * weight
                weighted_slot_total += weight
        for slot, proto_values in prototype.slots.items():
            if slot not in matched_slots and proto_values:
                missing_slots[slot] = proto_values[:2]
        slot_overlap = (weighted_slot_score / weighted_slot_total) if weighted_slot_total > 0 else 0.0
        axis_overlap = _axis_overlap(fingerprint.axis_weights, prototype.axis_weights)
        token_overlap = _token_overlap(fingerprint.embedding_terms, prototype.embedding_terms)
        discriminator_overlap = _token_overlap(fingerprint.embedding_terms, _prototype_discriminator_terms(prototype))
        negative_overlap = _negative_overlap(fingerprint, prototype)
        contradiction_overlap = _token_overlap(fingerprint.contradiction_terms, _prototype_discriminator_terms(prototype))
        prototype_slot_count = max(1, len(prototype.slots))
        missing_slot_ratio = min(1.0, len(missing_slots) / prototype_slot_count)
        evidence_hits = [
            span
            for span in fingerprint.evidence_spans[:12]
            if _token_overlap(_tokenize(span), prototype.embedding_terms) >= 0.1
        ]
        similarity = round(
            min(
                0.99,
                (slot_overlap * 0.38)
                + (axis_overlap * 0.24)
                + (token_overlap * 0.18)
                + (discriminator_overlap * 0.1)
                + (min(0.1, len(evidence_hits) * 0.03)),
            ),
            2,
        )
        absence_penalty = _discriminator_absence_penalty(fingerprint, prototype)
        # Demographic boost from LLM-synthesised learned store (never hardcoded).
        demo_profile = _get_demo_profile(prototype.id)
        demo_confidence = float(demo_profile.get("_confidence", 0.5)) if demo_profile else 0.5
        demo_boost = _demographic_boost_score(demo_profile, fingerprint.clinical_context, demo_confidence)
        similarity = round(
            min(
                0.99,
                max(
                    0.0,
                    similarity
                    - (negative_overlap * 0.24)
                    - (missing_slot_ratio * 0.14)
                    - (max(0.0, contradiction_overlap - discriminator_overlap) * 0.08)
                    - (absence_penalty * 0.18)
                    + demo_boost,
                ),
            ),
            2,
        )
        if prototype.id.endswith("_process"):
            similarity = round(max(0.0, similarity - 0.08), 2)
        if similarity < 0.12:
            continue
        matches.append(
            PrototypeMatch(
                prototype_id=prototype.id,
                label=prototype.id,
                family_label=prototype.family_label,
                similarity=similarity,
                slot_overlap=round(slot_overlap, 2),
                axis_overlap=round(axis_overlap, 2),
                token_overlap=round(max(token_overlap, discriminator_overlap), 2),
                matched_slots=matched_slots,
                missing_slots=missing_slots,
                evidence_spans=_dedupe(evidence_hits, limit=6),
                mechanism_paths=prototype.mechanism_signature[:4],
                discriminator_set=prototype.discriminator_set[:4],
                wrong_treatment_risks=prototype.wrong_treatment_risks[:4],
                source=prototype.source,
            )
        )
    promotions = detect_combo_promotions(fingerprint)
    if promotions:
        promo_ids = {p.prototype_id for p in promotions}
        # Drop scored matches that a combo-promotion already covers — the
        # symbolic match is strictly stronger evidence for that prototype.
        matches = [m for m in matches if m.prototype_id not in promo_ids]
        matches = list(promotions) + matches
    return sorted(matches, key=lambda item: (item.similarity, item.axis_overlap, item.slot_overlap), reverse=True)[:limit]


def summarize_match(match: PrototypeMatch) -> list[str]:
    phrases: list[str] = []
    for slot, values in match.matched_slots.items():
        if values:
            phrases.append(f"{slot}: {values[0].replace('_', ' ')}")
    phrases.extend(item.replace("_", " ") for item in match.mechanism_paths[:2])
    return _dedupe(phrases, limit=5)


def mechanism_summaries_from_matches(
    fingerprint: PhenotypeFingerprint,
    matches: list[PrototypeMatch],
    *,
    limit: int = 3,
) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    top_axes = [axis for axis, _ in sorted(fingerprint.axis_weights.items(), key=lambda item: item[1], reverse=True)]
    for match in matches[:limit]:
        axis = next((item for item in top_axes if item in match.label or item in " ".join(match.mechanism_paths)), top_axes[0] if top_axes else "undifferentiated")
        focus_terms = dominant_fingerprint_terms(fingerprint, limit=4)
        primary_terms = [item.replace(" ", "_") for item in summarize_match(match)[:2] if item]
        primary_mechanism = _slug(" ".join([axis, *primary_terms])) or _slug(axis)
        secondary_terms = [item.replace(" ", "_") for item in focus_terms[:2] if item]
        summaries.append(
            {
                "active_state": _slug(f"adaptive_state {axis} {' '.join(focus_terms[:1])}"),
                "organ_system": axis,
                "primary_mechanism": primary_mechanism,
                "secondary_mechanism": _slug(" ".join(secondary_terms)),
                "critical_findings": _dedupe([*match.evidence_spans[:3], *focus_terms[:2]], limit=4),
                "confidence": round(match.similarity, 2),
            }
        )
    if not summaries:
        fallback_focus = dominant_fingerprint_terms(fingerprint, limit=3)
        summaries.append(
            {
                "active_state": "adaptive_state_undifferentiated_pattern",
                "organ_system": next(iter(top_axes), "multisystem"),
                "primary_mechanism": _slug(" ".join(fallback_focus[:2])) or "undifferentiated_pattern",
                "secondary_mechanism": _slug(" ".join(fallback_focus[2:4])),
                "critical_findings": fallback_focus[:4],
                "confidence": 0.42,
            }
        )
    return summaries[:limit]


def build_prototype_update_suggestions(
    findings: StructuredFindings,
    *,
    top_label: str,
    prototype_matches: list[PrototypeMatch] | None = None,
) -> list[PrototypeUpdateSuggestion]:
    if not top_label:
        return []
    matches = prototype_matches or match_prototypes(findings.phenotype_fingerprint)
    top_match = matches[0] if matches else None
    fingerprint_terms = dominant_fingerprint_terms(findings.phenotype_fingerprint, limit=6)
    if top_match and top_match.label == top_label and top_match.similarity >= 0.78:
        action = "reweight"
        rationale = "Validated case aligns closely with an existing prototype and should update its centroid or weights."
        target_id = top_match.prototype_id
    elif top_match and top_match.label == top_label:
        action = "branch"
        rationale = "Validated case matches the current prototype family but adds enough variance to justify a shadow variant."
        target_id = top_match.prototype_id
    else:
        action = "propose_new"
        rationale = "Validated case expresses a clinically distinct fingerprint that is not well-covered by current prototypes."
        target_id = ""
    return [
        PrototypeUpdateSuggestion(
            action=action,
            target_prototype_id=target_id,
            candidate_label=top_label,
            similarity=round(float(top_match.similarity if top_match else 0.0), 2),
            rationale=rationale,
            fingerprint_summary=fingerprint_terms,
            requires_review=True,
        )
    ]
