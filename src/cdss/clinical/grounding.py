"""Narrative-grounding checks for structured clinical facts.

This module does not decide diagnoses. It validates whether extracted positives,
negatives, exposures, and medication facts are traceable to the patient's story.
Unknown facts stay unknown; they must not become denied findings.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from src.cdss.contracts.models import StructuredFindings
from src.cdss.text_normalization import ascii_fold, normalize_clinical_text


_TOKEN_RE = re.compile(r"[a-z0-9/+.-]{3,}", re.IGNORECASE)
_NEGATION_MARKERS = {
    "no",
    "not",
    "none",
    "without",
    "denies",
    "denied",
    "negative",
    "yok",
    "degil",
}
_NEGATIVE_PREFIX_RE = re.compile(r"^(?:no|not|without|denies?|negative for|yok|degil)\b", re.IGNORECASE)
_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "have",
    "has",
    "had",
    "feel",
    "felt",
    "pain",
    "symptom",
}


@dataclass(frozen=True)
class GroundingIssue:
    field: str
    item: str
    reason: str
    source_text: str = ""


@dataclass(frozen=True)
class GroundingReport:
    issues: list[GroundingIssue] = field(default_factory=list)
    supported_count: int = 0
    checked_count: int = 0
    hallucination_risk_score: float = 0.0

    @property
    def unsupported_count(self) -> int:
        return len(self.issues)


def _tokens(text: str) -> set[str]:
    folded = ascii_fold(text)
    return {
        token
        for token in (match.group(0).lower() for match in _TOKEN_RE.finditer(folded))
        if token not in _STOPWORDS
    }


def _dedupe(items: Iterable[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = normalize_clinical_text(str(item or ""))
        key = ascii_fold(text)
        if not text or key in seen:
            continue
        seen.add(key)
        output.append(text)
    return output


def _narrative_segments(findings: StructuredFindings) -> list[str]:
    lanes = findings.context_lanes or {}
    return _dedupe(
        [
            *list(findings.raw_segments or []),
            *list(lanes.get("patient_narrative", []) or []),
            findings.summary,
        ]
    )


def _supporting_segment(item: str, segments: list[str], *, minimum_overlap: float = 0.45) -> str:
    item_tokens = _tokens(item)
    if not item_tokens:
        return ""
    item_folded = ascii_fold(item)
    for segment in segments:
        segment_folded = ascii_fold(segment)
        if item_folded and item_folded in segment_folded:
            return segment
        segment_tokens = _tokens(segment)
        if not segment_tokens:
            continue
        overlap = len(item_tokens & segment_tokens) / max(1, len(item_tokens))
        if overlap >= minimum_overlap:
            return segment
    return ""


def _negative_body(item: str) -> str:
    body = _NEGATIVE_PREFIX_RE.sub("", normalize_clinical_text(item)).strip(" .,:;-")
    return body or item


def _is_explicitly_negated(body: str, segment: str) -> bool:
    folded = ascii_fold(segment)
    if not any(marker in folded.split() for marker in _NEGATION_MARKERS):
        return False
    body_tokens = _tokens(body)
    if not body_tokens:
        return False
    segment_tokens = _tokens(segment)
    return len(body_tokens & segment_tokens) / max(1, len(body_tokens)) >= 0.45


def validate_narrative_grounding(findings: StructuredFindings) -> GroundingReport:
    segments = _narrative_segments(findings)
    issues: list[GroundingIssue] = []
    checked = 0
    supported = 0

    positive_fields = {
        "positive_findings": list(findings.positive_findings or []),
        "exposures": list(findings.exposures or []),
        "medications": list(findings.medications or []),
        "timeline": list(findings.timeline or []),
    }
    for field, items in positive_fields.items():
        for item in _dedupe(items):
            checked += 1
            support = _supporting_segment(item, segments)
            if support:
                supported += 1
                continue
            issues.append(
                GroundingIssue(
                    field=field,
                    item=item,
                    reason="not_traceable_to_patient_narrative",
                )
            )

    for item in _dedupe(findings.negative_findings or []):
        checked += 1
        body = _negative_body(item)
        support = next((segment for segment in segments if _is_explicitly_negated(body, segment)), "")
        if support:
            supported += 1
            continue
        issues.append(
            GroundingIssue(
                field="negative_findings",
                item=item,
                reason="negative_not_explicitly_denied",
            )
        )

    risk = round(len(issues) / max(1, checked), 2)
    return GroundingReport(
        issues=issues,
        supported_count=supported,
        checked_count=checked,
        hallucination_risk_score=risk,
    )
