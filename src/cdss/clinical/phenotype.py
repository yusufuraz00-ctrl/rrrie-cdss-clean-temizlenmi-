"""Phenotype compilation and context-lane helpers for narrative-first reasoning."""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.cdss.clinical.prototype_memory import build_phenotype_fingerprint, prototype_state_frames
from src.cdss.contracts.models import PhenotypeAtom, PhenotypeFingerprint, StructuredFindings
from src.cdss.text_normalization import ascii_fold, sanitize_query_text

# Lane assignment is determined by the semantic prefix of each context entry.
# Instead of explicit whitelists, we use prefix-token pattern matching so that
# any LLM-generated prefix routes to the correct lane adaptively.
_CLINICIAN_PREFIXES = re.compile(
    r"^(working_diagnosis|clinician_anchor|expert_opinion|attending_impression)",
    re.IGNORECASE,
)
_OPERATIONAL_PREFIXES = re.compile(
    r"^(blocked_order|causal_loop|contradiction|data_request|device_reliability"
    r"|hazard|masquerade_risk|pathway_fit|required_action|workflow|safety|alert|critical_gap)",
    re.IGNORECASE,
)
_EXTERNAL_PREFIXES = re.compile(
    r"^(external_evidence|literature|verified_evidence|pubmed|research|guideline)",
    re.IGNORECASE,
)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", ascii_fold(str(value or "")).strip()).strip("_")


def _dedupe(values: Iterable[str], *, limit: int | None = None) -> list[str]:
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


def _ctx_rows(findings: StructuredFindings) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in findings.input_context:
        text = str(raw or "").strip()
        if not text:
            continue
        if ":" not in text:
            rows.append(("signal", text))
            continue
        head, tail = text.split(":", 1)
        rows.append((_slug(head) or "signal", " ".join(tail.split()).strip() or " ".join(head.split()).strip()))
    return rows


def _classify_lane(prefix: str) -> str:
    """Return lane name for a context-entry prefix."""
    if _CLINICIAN_PREFIXES.match(prefix):
        return "clinician_anchor"
    if _OPERATIONAL_PREFIXES.match(prefix):
        return "operational_safety"
    if _EXTERNAL_PREFIXES.match(prefix):
        return "external_evidence"
    return "patient_narrative"


def build_context_lanes(findings: StructuredFindings) -> dict[str, list[str]]:
    lanes: dict[str, list[str]] = {
        "patient_narrative": _dedupe(
            [
                findings.summary,
                *findings.positive_findings[:12],
                *findings.negative_findings[:8],
                *findings.timeline[:8],
                *findings.exposures[:8],
                *findings.raw_segments[:16],
            ],
            limit=24,
        ),
        "clinician_anchor": [],
        "operational_safety": [],
        "external_evidence": [],
    }
    for prefix, value in _ctx_rows(findings):
        lane = _classify_lane(prefix)
        lanes[lane].append(value)
    return {key: _dedupe(values, limit=24) for key, values in lanes.items()}


def compile_phenotype_fingerprint(findings: StructuredFindings) -> PhenotypeFingerprint:
    return build_phenotype_fingerprint(findings)


def compile_phenotype_atoms(patient_text: str, findings: StructuredFindings) -> list[PhenotypeAtom]:
    del patient_text
    fingerprint = findings.phenotype_fingerprint or build_phenotype_fingerprint(findings)
    atoms: list[PhenotypeAtom] = []
    for slot, values in fingerprint.slots.items():
        evidence = list((fingerprint.slot_evidence or {}).get(slot, []))
        for index, normalized_value in enumerate(values[:6], start=1):
            confidence = round(min(0.94, 0.36 + (0.08 * len(evidence[:2])) + (0.05 * max(0, 3 - index))), 2)
            atoms.append(
                PhenotypeAtom(
                    label=f"{slot}__{normalized_value}",
                    category=slot,
                    slot=slot,
                    value=evidence[index - 1] if index - 1 < len(evidence) else normalized_value.replace("_", " "),
                    normalized_value=normalized_value,
                    confidence=confidence,
                    evidence=evidence[:2] or [normalized_value.replace("_", " ")],
                    provenance="adaptive_slot_compiler",
                    polarity="positive",
                )
            )
    for slot, values in (fingerprint.negative_slots or {}).items():
        evidence = list((fingerprint.negative_slot_evidence or {}).get(slot, []))
        for index, normalized_value in enumerate(values[:6], start=1):
            confidence = round(min(0.94, 0.36 + (0.08 * len(evidence[:2])) + (0.05 * max(0, 3 - index))), 2)
            atoms.append(
                PhenotypeAtom(
                    label=f"NOT__{slot}__{normalized_value}",
                    category=slot,
                    slot=slot,
                    value=evidence[index - 1] if index - 1 < len(evidence) else normalized_value.replace("_", " "),
                    normalized_value=normalized_value,
                    confidence=confidence,
                    evidence=evidence[:2] or [normalized_value.replace("_", " ")],
                    provenance="adaptive_slot_compiler_negative",
                    polarity="negative",
                )
            )
    if not atoms:
        fallback = _dedupe(
            [
                findings.summary,
                *findings.positive_findings[:4],
                *findings.timeline[:2],
                *findings.exposures[:2],
            ],
            limit=4,
        )
        for item in fallback:
            normalized = sanitize_query_text(item, max_terms=6).replace(" ", "_") or _slug(item) or "unspecified"
            atoms.append(
                PhenotypeAtom(
                    label=f"story__{normalized}",
                    category="story",
                    slot="story",
                    value=item,
                    normalized_value=normalized,
                    confidence=0.32,
                    evidence=[item],
                    provenance="adaptive_story_fallback",
                )
            )
    return sorted(atoms, key=lambda item: (item.confidence, item.slot != "story"), reverse=True)


def phenotype_labels(findings: StructuredFindings, *, limit: int | None = None) -> list[str]:
    labels = [atom.label for atom in findings.phenotype_atoms if str(atom.label or "").strip()]
    return _dedupe(labels, limit=limit)


def phenotype_query_terms(findings: StructuredFindings, *, limit: int = 6) -> list[str]:
    phrases: list[str] = []
    fingerprint = findings.phenotype_fingerprint or build_phenotype_fingerprint(findings)
    for atom in findings.phenotype_atoms:
        phrases.append(atom.value or atom.normalized_value.replace("_", " "))
        phrases.extend(atom.evidence[:1])
    for term in fingerprint.temporal_signature[:3]:
        phrases.append(term.replace("_", " "))
    for term in fingerprint.embedding_terms[:6]:
        phrases.append(term.replace("_", " "))
    if len(phrases) < limit:
        phrases.extend(findings.exposures[:3])
        phrases.extend(findings.timeline[:3])
        phrases.extend(findings.positive_findings[:4])
    output: list[str] = []
    seen: set[str] = set()
    for item in phrases:
        cleaned = sanitize_query_text(str(item or ""), max_terms=6).replace("_", " ").strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        output.append(cleaned)
        if len(output) >= limit:
            break
    return output


def phenotype_state_frames(findings: StructuredFindings) -> list[str]:
    fingerprint = findings.phenotype_fingerprint or build_phenotype_fingerprint(findings)
    return prototype_state_frames(fingerprint)


def has_dominant_phenotype_signature(findings: StructuredFindings) -> bool:
    fingerprint = findings.phenotype_fingerprint or build_phenotype_fingerprint(findings)
    confident_atoms = [atom for atom in findings.phenotype_atoms if float(atom.confidence or 0.0) >= 0.72]
    dominant_slots = [slot for slot, values in (fingerprint.slots or {}).items() if values]
    return bool(
        len(confident_atoms) >= 2
        and (
            len(dominant_slots) >= 3
            or len(fingerprint.temporal_signature) >= 2
            or len(fingerprint.axis_weights) >= 2
        )
    )
