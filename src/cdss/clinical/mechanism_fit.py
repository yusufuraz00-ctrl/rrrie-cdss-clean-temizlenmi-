"""Mechanism completeness scoring for candidate closure."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.cdss.contracts.models import DifferentialCandidate, RiskProfile, StructuredFindings, UrgencyTier
from src.cdss.text_normalization import ascii_fold


_TOKEN_RE = re.compile(r"[a-z0-9/+.-]{3,}", re.IGNORECASE)

_AXIS_TERMS = {
    "timing": {"sudden", "acute", "rapid", "progressive", "hour", "hours", "today", "onset", "worsened"},
    "anatomy": {
        "abdomen", "abdominal", "chest", "back", "leg", "foot", "eye", "vision",
        "face", "thigh", "knee", "joint", "neck", "urine", "skin", "neurologic",
    },
    "risk": {"age", "diabetes", "pregnant", "hypertension", "atrial", "cancer", "immun", "elderly"},
    "exposure_medication": {"medication", "drug", "toxin", "exposure", "heater", "alcohol", "injury", "scratch"},
    "severity": {"severe", "cannot", "unable", "weakness", "confusion", "fever", "shock", "dizzy", "10/10"},
    "confirmation": {"test", "exam", "imaging", "lab", "labs", "scan", "ultrasound", "urgent", "objective"},
}


@dataclass(frozen=True)
class MechanismCompletenessScore:
    label: str
    score: float = 0.0
    applicable_axes: list[str] = field(default_factory=list)
    satisfied_axes: list[str] = field(default_factory=list)
    missing_axes: list[str] = field(default_factory=list)
    closure_blockers: list[str] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {match.group(0).lower() for match in _TOKEN_RE.finditer(ascii_fold(text))}


def _case_text(findings: StructuredFindings) -> str:
    return " ".join(
        [
            findings.summary or "",
            *list(findings.positive_findings or []),
            *list(findings.red_flags or []),
            *list(findings.timeline or []),
            *list(findings.exposures or []),
            *list(findings.medications or []),
            " ".join(f"{k} {v}" for k, v in (findings.demographics or {}).items()),
            " ".join(f"{k} {v}" for k, v in (findings.derived_vitals or {}).items()),
        ]
    )


def _candidate_text(candidate: DifferentialCandidate) -> str:
    return " ".join(
        [
            str(candidate.label or "").replace("_", " "),
            *list(candidate.rationale or []),
            *list(candidate.evidence_needed or []),
        ]
    )


def score_mechanism_completeness(
    candidate: DifferentialCandidate,
    findings: StructuredFindings,
    risk_profile: RiskProfile,
) -> MechanismCompletenessScore:
    case_tokens = _tokens(_case_text(findings))
    candidate_tokens = _tokens(_candidate_text(candidate))

    applicable: list[str] = []
    satisfied: list[str] = []
    missing: list[str] = []
    for axis, terms in _AXIS_TERMS.items():
        axis_applicable = bool(case_tokens & terms)
        if axis == "confirmation" and risk_profile.urgency != UrgencyTier.ROUTINE:
            axis_applicable = True
        if not axis_applicable:
            continue
        applicable.append(axis)
        if candidate_tokens & terms:
            satisfied.append(axis)
        else:
            missing.append(axis)

    if not applicable:
        applicable = ["confirmation"]
        missing = ["confirmation"]

    score = round(len(satisfied) / max(1, len(applicable)), 2)
    blockers: list[str] = []
    if risk_profile.urgency != UrgencyTier.ROUTINE and score < 0.65:
        blockers.append("high_risk_mechanism_incomplete")
    for required_axis in ("timing", "severity"):
        if required_axis in applicable and required_axis not in satisfied and risk_profile.urgency == UrgencyTier.EMERGENCY:
            blockers.append(f"missing_{required_axis}_mechanism")
    if "confirmation" in applicable and "confirmation" not in satisfied:
        blockers.append("missing_objective_confirmation_path")

    return MechanismCompletenessScore(
        label=candidate.label,
        score=score,
        applicable_axes=applicable,
        satisfied_axes=satisfied,
        missing_axes=missing,
        closure_blockers=list(dict.fromkeys(blockers)),
    )
