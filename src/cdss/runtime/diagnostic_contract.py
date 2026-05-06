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
]

ResearchStatus = Literal["pending", "running", "complete", "blocked"]


@dataclass(slots=True)
class ResearchQueryPlan:
    intent: str
    query: str
    source_classes: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass(slots=True)
class ResearchEvidenceItem:
    claim: str
    citation: str
    source_class: str
    trust_tier: str
    provenance: dict[str, Any] = field(default_factory=dict)
    linked_discriminator: str | None = None


@dataclass(slots=True)
class DiagnosticContract:
    anchor: str | None = None
    anchor_role: ClinicalRole | None = None
    source_disease_resolved: bool | None = None
    candidates: list[str] = field(default_factory=list)
    required_data: list[str] = field(default_factory=list)
    objective_discriminators: list[str] = field(default_factory=list)
    must_not_miss: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    urgency: str | None = None
    research_status: ResearchStatus = "pending"

    def to_dict(self) -> dict[str, Any]:
        return {
            "anchor": self.anchor,
            "anchor_role": self.anchor_role,
            "source_disease_resolved": self.source_disease_resolved,
            "candidates": list(self.candidates),
            "required_data": list(self.required_data),
            "objective_discriminators": list(self.objective_discriminators),
            "must_not_miss": list(self.must_not_miss),
            "contraindications": list(self.contraindications),
            "urgency": self.urgency,
            "research_status": self.research_status,
        }


def build_contract_from_profile(anchor: str | None) -> DiagnosticContract:
    contract = DiagnosticContract(anchor=anchor or None)
    if not anchor:
        return contract
    try:
        # Bug-fix: was `from cdss.knowledge.registry` (wrong root —
        # silently raised ImportError) AND `registry.profiles.get(anchor)`
        # (`profiles` is a list, not a dict — would raise AttributeError).
        # Both errors were swallowed by the bare `except`, so this
        # function NEVER successfully populated a contract from the
        # registry. The whole source-disease / must-not-miss /
        # contraindication contract feature was effectively dead code.
        from src.cdss.knowledge.registry import load_syndrome_registry

        registry = load_syndrome_registry()
        profile = registry.by_id(anchor) if registry is not None else None
    except Exception as exc:  # noqa: BLE001
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "build_contract_from_profile: registry lookup failed for anchor=%r (%s)",
            anchor, exc,
        )
        profile = None
    if profile is None:
        return contract

    contract.anchor_role = getattr(profile, "clinical_role", None)
    contract.source_disease_resolved = getattr(profile, "source_disease_resolved", None)
    if contract.anchor_role and contract.anchor_role != "source_disease" and contract.source_disease_resolved is None:
        contract.source_disease_resolved = False
    contract.required_data = list(getattr(profile, "evidence_needs", []) or [])
    contract.objective_discriminators = list(getattr(profile, "objective_discriminators", []) or [])
    if not contract.objective_discriminators:
        contract.objective_discriminators = list(getattr(profile, "evidence_needs", []) or [])
    if contract.source_disease_resolved is False and "underlying_cause_resolution" not in contract.required_data:
        contract.required_data.append("underlying_cause_resolution")
    contract.must_not_miss = list(getattr(profile, "must_not_miss", []) or [])
    contract.contraindications = list(getattr(profile, "contraindications", []) or [])
    if not contract.contraindications:
        contract.contraindications = list(getattr(profile, "unsafe_interventions", []) or [])
    if contract.objective_discriminators and contract.research_status == "pending":
        contract.research_status = "running"
    return contract


def contract_requires_research(contract: DiagnosticContract) -> bool:
    if contract.research_status == "blocked":
        return False
    if contract.source_disease_resolved is False:
        return True
    if contract.must_not_miss:
        return True
    if contract.objective_discriminators:
        return True
    return False
