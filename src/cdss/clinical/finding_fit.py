"""Evidence-to-diagnosis finding-fit scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.cdss.contracts.models import DifferentialCandidate, StructuredFindings
from src.cdss.text_normalization import ascii_fold


_TOKEN_RE = re.compile(r"[a-z0-9/+.-]{3,}", re.IGNORECASE)
_CRITICAL_MARKERS = {
    "sudden",
    "severe",
    "worst",
    "rapid",
    "progressive",
    "cannot",
    "unable",
    "weakness",
    "paralysis",
    "confusion",
    "syncope",
    "faint",
    "dizzy",
    "shock",
    "bleeding",
    "cold",
    "pale",
    "blue",
    "purple",
    "vision",
    "blind",
    "fever",
    "altered",
    "lethargic",
    "dyspnea",
    "chest",
    "neurologic",
    "new",
}
_STOPWORDS = {"the", "and", "with", "from", "that", "this", "have", "has", "had", "after"}


@dataclass(frozen=True)
class CandidateFindingFit:
    label: str
    evidence_fit_score: float = 0.0
    orphan_penalty: float = 0.0
    contradiction_penalty: float = 0.0
    explained_findings: list[str] = field(default_factory=list)
    partially_explained_findings: list[str] = field(default_factory=list)
    unexplained_findings: list[str] = field(default_factory=list)
    critical_unexplained: list[str] = field(default_factory=list)


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(ascii_fold(text)))
        if token not in _STOPWORDS
    }


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, min(len(left), len(right)))


def _is_critical(finding: str, findings: StructuredFindings) -> bool:
    folded = ascii_fold(finding)
    if any(ascii_fold(flag) == folded for flag in findings.red_flags or []):
        return True
    toks = _tokens(finding)
    return bool(toks & _CRITICAL_MARKERS)


def score_candidate_finding_fit(candidate: DifferentialCandidate, findings: StructuredFindings) -> CandidateFindingFit:
    candidate_text = " ".join(
        [
            str(candidate.label or "").replace("_", " "),
            *list(candidate.rationale or []),
            *list(candidate.evidence_needed or []),
        ]
    )
    candidate_tokens = _tokens(candidate_text)
    all_findings = list(dict.fromkeys([*list(findings.positive_findings or []), *list(findings.red_flags or [])]))
    if not all_findings:
        return CandidateFindingFit(label=candidate.label, evidence_fit_score=0.0, orphan_penalty=0.0)

    explained: list[str] = []
    partial: list[str] = []
    unexplained: list[str] = []
    critical_unexplained: list[str] = []
    weighted_total = 0.0
    weighted_support = 0.0
    weighted_orphan = 0.0

    for finding in all_findings:
        ft = _tokens(finding)
        critical = _is_critical(finding, findings)
        weight = 1.75 if critical else 1.0
        weighted_total += weight
        score = _overlap(ft, candidate_tokens)
        if score >= 0.42:
            explained.append(finding)
            weighted_support += weight
        elif score >= 0.22:
            partial.append(finding)
            weighted_support += weight * 0.5
        else:
            unexplained.append(finding)
            if critical:
                critical_unexplained.append(finding)
                weighted_orphan += weight

    evidence_fit = round(weighted_support / max(1.0, weighted_total), 2)
    orphan_penalty = round(weighted_orphan / max(1.0, weighted_total), 2)
    return CandidateFindingFit(
        label=candidate.label,
        evidence_fit_score=evidence_fit,
        orphan_penalty=orphan_penalty,
        contradiction_penalty=0.0,
        explained_findings=explained,
        partially_explained_findings=partial,
        unexplained_findings=unexplained,
        critical_unexplained=critical_unexplained,
    )
