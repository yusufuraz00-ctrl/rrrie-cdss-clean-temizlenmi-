from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


ClinicalRole = Literal[
    "source_disease",
    "manifestation",
    "lab_abnormality",
    "syndrome",
    "mechanism",
    "test",
    "intervention_hazard",
    "unknown",
]

ResearchStatus = Literal["pending", "running", "complete", "blocked"]


def _clean_text(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    text = " ".join(text.replace("_", " ").split())
    lowered = text.lower()
    if lowered in {"none", "none specified", "unknown", "n/a", "na"}:
        return ""
    if "semi colon" in lowered or lowered.endswith(" semi colon"):
        return ""
    if lowered.startswith("exclusions ") and len(lowered.split()) <= 3:
        return ""
    return text


def sanitize_string_list(values: Any) -> list[str]:
    cleaned: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        text = _clean_text(value)
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


def coerce_research_status(value: Any) -> ResearchStatus:
    text = str(value or "").strip().lower()
    if text in {"running", "complete", "blocked"}:
        return text  # type: ignore[return-value]
    return "pending"


@dataclass
class SourceCandidate:
    diagnosis: str
    confidence: float = 0.0
    rationale: str = ""


@dataclass
class ResearchEvidenceItem:
    claim: str
    citation: str = ""
    source_class: str = ""
    trust_tier: str = ""
    provenance: str = ""
    linked_to: str = ""


@dataclass
class CanonicalDiagnosticState:
    phenotype_candidates: list[dict[str, Any]] = field(default_factory=list)
    source_disease_candidates: list[SourceCandidate] = field(default_factory=list)
    resolved_anchor: str = ""
    anchor_role: ClinicalRole = "unknown"
    source_disease_resolved: bool = False
    research_status: ResearchStatus = "pending"
    linked_evidence: list[ResearchEvidenceItem] = field(default_factory=list)
    required_data: list[str] = field(default_factory=list)
    objective_discriminators: list[str] = field(default_factory=list)
    must_not_miss: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    urgency: str = ""
    resolution_rationale: str = ""


def build_canonical_state(
    *,
    anchor: Any,
    anchor_role: Any,
    source_disease_resolved: Any,
    phenotype_candidates: Any,
    source_disease_candidates: Any,
    linked_evidence: Any,
    required_data: Any,
    objective_discriminators: Any,
    must_not_miss: Any,
    contraindications: Any,
    urgency: Any,
    resolution_rationale: Any = "",
) -> CanonicalDiagnosticState:
    source_candidates: list[SourceCandidate] = []
    for item in source_disease_candidates or []:
        if isinstance(item, SourceCandidate):
            if item.diagnosis:
                source_candidates.append(item)
            continue
        if isinstance(item, dict):
            diagnosis = _clean_text(item.get("diagnosis") or item.get("name") or item.get("candidate"))
            if not diagnosis:
                continue
            source_candidates.append(
                SourceCandidate(
                    diagnosis=diagnosis,
                    confidence=float(item.get("confidence") or 0.0),
                    rationale=_clean_text(item.get("rationale")),
                )
            )
            continue
        text = _clean_text(item)
        if text:
            source_candidates.append(SourceCandidate(diagnosis=text))

    evidence_items: list[ResearchEvidenceItem] = []
    for item in linked_evidence or []:
        if isinstance(item, ResearchEvidenceItem):
            evidence_items.append(item)
            continue
        if isinstance(item, dict):
            claim = _clean_text(item.get("claim") or item.get("title") or item.get("summary") or item.get("query"))
            if not claim:
                continue
            evidence_items.append(
                ResearchEvidenceItem(
                    claim=claim,
                    citation=_clean_text(item.get("citation") or item.get("url") or item.get("pmid")),
                    source_class=_clean_text(item.get("source_class") or item.get("source")),
                    trust_tier=_clean_text(item.get("trust_tier")),
                    provenance=_clean_text(item.get("provenance") or item.get("rationale")),
                    linked_to=_clean_text(item.get("linked_to") or item.get("diagnosis")),
                )
            )

    state = CanonicalDiagnosticState(
        phenotype_candidates=list(phenotype_candidates or []),
        source_disease_candidates=source_candidates,
        resolved_anchor=_clean_text(anchor),
        anchor_role=(str(anchor_role or "unknown").strip().lower() or "unknown"),  # type: ignore[arg-type]
        source_disease_resolved=bool(source_disease_resolved),
        research_status=coerce_research_status("complete" if evidence_items else "blocked"),
        linked_evidence=evidence_items,
        required_data=sanitize_string_list(required_data),
        objective_discriminators=sanitize_string_list(objective_discriminators),
        must_not_miss=sanitize_string_list(must_not_miss),
        contraindications=sanitize_string_list(contraindications),
        urgency=_clean_text(urgency),
        resolution_rationale=_clean_text(resolution_rationale),
    )

    if state.anchor_role != "source_disease":
        state.source_disease_resolved = False
    if state.source_disease_candidates and not state.source_disease_resolved:
        state.resolved_anchor = state.source_disease_candidates[0].diagnosis
        state.anchor_role = "source_disease"
    if not state.source_disease_resolved and "underlying_cause_resolution" not in state.required_data:
        state.required_data.append("underlying_cause_resolution")
    if state.objective_discriminators:
        for item in state.objective_discriminators:
            if item not in state.required_data:
                state.required_data.append(item)
    return state
