"""Ontology-constrained diagnosis label resolution helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from src.cdss.contracts.models import FactGraph, StructuredFindings
from src.cdss.clinical.phenotype import phenotype_labels
from src.cdss.text_normalization import ascii_fold, sanitize_query_text
from src.tools.pubmed_tool import search_pubmed
from src.utils.medical_codes import search_icd11_who

_GENERIC_SUFFIXES = (
    "_process",
    "_state",
    "_syndrome",
    "_condition",
    "_disorder",
)
_MIN_SINGLE_TOKEN_LABEL_LEN = 5
# Clinical acronyms whose 2-3 char single-token form must be preserved.
# Without this whitelist, "PE" / "MI" / "DKA" etc. are silently rejected as
# implausible labels — high-acuity must-not-miss diagnoses score zero.
_ALLOWED_SHORT_ACRONYMS: frozenset[str] = frozenset({
    "pe", "mi", "ami", "tia", "cva", "dvt", "gi", "cns", "dka",
    "hhs", "uti", "pid", "sah", "ich", "stemi", "nstemi", "copd",
    "ards", "dic", "afib", "vfib", "vt", "svt", "mr", "ms", "as",
    "ai", "tr", "ts", "vsd", "asd", "pda", "tof",
})


@dataclass(frozen=True)
class OntologyDecision:
    accepted: bool
    canonical_label: str
    reason: str
    externally_verified: bool = False


@lru_cache(maxsize=2048)
def _normalize_candidate_label_cached(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", ascii_fold(value or "")).strip("_")
    return normalized


def normalize_candidate_label(value: str) -> str:
    # 4c: cache hot label normalization. Each case feeds the same labels through
    # this helper dozens of times across hierarchy folding, anchor detection,
    # and candidate filling; the result is purely a function of the input.
    return _normalize_candidate_label_cached(str(value or ""))


@lru_cache(maxsize=2048)
def is_plausible_diagnosis_label(label: str) -> bool:
    normalized = normalize_candidate_label(label)
    if not normalized:
        return False
    tokens = [token for token in normalized.split("_") if token]
    if len(tokens) < 1:
        return False
    if len(tokens) == 1 and len(tokens[0]) < _MIN_SINGLE_TOKEN_LABEL_LEN:
        if tokens[0] not in _ALLOWED_SHORT_ACRONYMS:
            return False
    if len(tokens) > 15: # increased limit for descriptive complex syndromes
        return False
    if normalized.startswith(("http_", "https_", "www_")):
        return False
    if all(token.isdigit() for token in tokens):
        return False
    return True


class DiagnosisCandidateResolver:
    """Validate candidate labels against local ontology + external literature."""

    def __init__(self) -> None:
        self._pubmed_cache: dict[str, tuple[bool, bool]] = {}
        self._icd_cache: dict[str, tuple[bool, bool]] = {}

    @staticmethod
    def _working_diagnosis_context(findings: StructuredFindings) -> list[str]:
        values: list[str] = []
        for raw in findings.input_context:
            text = str(raw or "").strip()
            if ":" not in text:
                continue
            prefix, value = text.split(":", 1)
            if normalize_candidate_label(prefix) != "working_diagnosis":
                continue
            cleaned = str(value or "").strip()
            if cleaned:
                values.append(cleaned)
        return values

    @staticmethod
    def _fact_context_labels(fact_graph: FactGraph | None) -> list[str]:
        if not fact_graph:
            return []
        allowed_categories = {"finding", "risk_marker", "diagnostic_context", "exposure", "medication", "vital"}
        return [node.label for node in fact_graph.nodes if node.category in allowed_categories][:24]

    @staticmethod
    def _non_diagnostic_context_labels(findings: StructuredFindings, fact_graph: FactGraph | None) -> set[str]:
        labels: set[str] = set()
        for raw in findings.input_context:
            text = str(raw or "").strip()
            if not text:
                continue
            if ":" not in text:
                labels.add(normalize_candidate_label(text))
                continue
            prefix, value = text.split(":", 1)
            if normalize_candidate_label(prefix) == "working_diagnosis":
                continue
            labels.add(normalize_candidate_label(value or prefix))
        if fact_graph:
            for node in fact_graph.nodes:
                if node.category == "context_frame":
                    labels.add(normalize_candidate_label(node.label))
        return {item for item in labels if item}

    @staticmethod
    def _context_tokens(findings: StructuredFindings, fact_graph: FactGraph | None, rationale: list[str]) -> set[str]:
        patient_lane = list((findings.context_lanes or {}).get("patient_narrative", []))
        external_lane = list((findings.context_lanes or {}).get("external_evidence", []))
        bag = " ".join(
            [
                findings.summary,
                *findings.positive_findings[:10],
                *findings.timeline[:6],
                *findings.exposures[:6],
                *findings.red_flags[:8],
                *findings.suspected_conditions[:8],
                *phenotype_labels(findings, limit=8),
                *findings.raw_segments[:10],
                *patient_lane[:10],
                *external_lane[:6],
                *rationale[:4],
                *DiagnosisCandidateResolver._fact_context_labels(fact_graph),
            ]
        )
        return {token for token in re.findall(r"[a-z0-9]{3,}", ascii_fold(bag)) if token}

    @staticmethod
    def _case_tokens(findings: StructuredFindings, fact_graph: FactGraph | None) -> set[str]:
        patient_lane = list((findings.context_lanes or {}).get("patient_narrative", []))
        external_lane = list((findings.context_lanes or {}).get("external_evidence", []))
        bag = " ".join(
            [
                findings.summary,
                *findings.positive_findings[:10],
                *findings.timeline[:6],
                *findings.exposures[:6],
                *findings.red_flags[:8],
                *findings.suspected_conditions[:8],
                *phenotype_labels(findings, limit=8),
                *findings.raw_segments[:10],
                *patient_lane[:10],
                *external_lane[:6],
                *DiagnosisCandidateResolver._fact_context_labels(fact_graph),
            ]
        )
        return {token for token in re.findall(r"[a-z0-9]{3,}", ascii_fold(bag)) if token}

    @staticmethod
    def _rationale_tokens(rationale: list[str]) -> set[str]:
        bag = " ".join(rationale[:6])
        return {token for token in re.findall(r"[a-z0-9]{3,}", ascii_fold(bag)) if token}

    @staticmethod
    def _narrative_grounding_score(
        findings: StructuredFindings,
        fact_graph: FactGraph | None,
        rationale: list[str],
    ) -> float:
        positive_density = min(1.0, len(findings.positive_findings[:10]) / 6.0)
        red_flag_density = min(1.0, len(findings.red_flags[:8]) / 3.0)
        timeline_density = min(1.0, len(findings.timeline[:6]) / 3.0)
        exposure_density = min(1.0, len(findings.exposures[:6]) / 3.0)
        segment_density = min(1.0, len(findings.raw_segments[:10]) / 6.0)
        fact_density = min(1.0, len((fact_graph.nodes if fact_graph else [])[:18]) / 10.0)
        rationale_density = min(1.0, len([item for item in rationale[:4] if str(item or "").strip()]) / 3.0)
        return round(
            min(
                1.0,
                (positive_density * 0.28)
                + (red_flag_density * 0.18)
                + (timeline_density * 0.14)
                + (exposure_density * 0.1)
                + (segment_density * 0.12)
                + (fact_density * 0.1)
                + (rationale_density * 0.08),
            ),
            2,
        )

    async def _verify_with_pubmed(self, canonical_label: str) -> tuple[bool, bool]:
        cached = self._pubmed_cache.get(canonical_label)
        if cached is not None:
            return cached
        query = sanitize_query_text(f"{canonical_label.replace('_', ' ')} diagnosis", max_terms=8).replace("_", " ")
        if not query:
            self._pubmed_cache[canonical_label] = (False, False)
            return False, False
        try:
            result = await search_pubmed(query=query, max_results=1, article_types=None)
            if result.get("error"):
                verified, available = False, False
            else:
                verified, available = int(result.get("total_found", 0) or 0) > 0, True
        except Exception:
            verified, available = False, False
        self._pubmed_cache[canonical_label] = (verified, available)
        return verified, available

    async def _verify_with_icd(self, canonical_label: str) -> tuple[bool, bool]:
        cached = self._icd_cache.get(canonical_label)
        if cached is not None:
            return cached

        normalized = normalize_candidate_label(canonical_label)
        if not normalized:
            self._icd_cache[canonical_label] = (False, False)
            return False, False

        try:
            matches = await search_icd11_who(normalized.replace("_", " "), max_results=3)
            verified = bool(matches)
            available = True
        except Exception:
            verified, available = False, False
        self._icd_cache[canonical_label] = (verified, available)
        return verified, available

    async def _verify_external_ontology(self, canonical_label: str) -> tuple[bool, bool]:
        pubmed_verified, pubmed_available = await self._verify_with_pubmed(canonical_label)
        if pubmed_verified:
            return True, True
        icd_verified, icd_available = await self._verify_with_icd(canonical_label)
        return icd_verified, (pubmed_available or icd_available)

    @staticmethod
    def _guardrail_score(
        *,
        label_tokens: set[str],
        context_tokens: set[str],
        case_tokens: set[str],
        rationale_tokens: set[str],
        grounding_score: float,
        validated_by_llm: bool,
    ) -> tuple[float, dict[str, float]]:
        label_overlap = len(label_tokens & context_tokens)
        rationale_case_overlap = len(rationale_tokens & case_tokens)

        label_strength = min(1.0, label_overlap / max(1, len(label_tokens)))
        rationale_strength = min(1.0, rationale_case_overlap / max(1, len(rationale_tokens) or 1))
        llm_bonus = 0.14 if validated_by_llm else 0.0

        score = min(
            1.0,
            (grounding_score * 0.52)
            + (label_strength * 0.24)
            + (rationale_strength * 0.1)
            + llm_bonus,
        )
        signals = {
            "label_overlap": float(label_overlap),
            "rationale_case_overlap": float(rationale_case_overlap),
            "grounding_score": grounding_score,
            "score": round(score, 3),
        }
        return round(score, 3), signals

    async def validate(
        self,
        *,
        label: str,
        findings: StructuredFindings,
        rationale: list[str],
        fact_graph: FactGraph | None = None,
        require_external_verification: bool = True,
        validated_by_llm: bool = False,
    ) -> OntologyDecision:
        canonical = normalize_candidate_label(label)
        if not canonical:
            return OntologyDecision(False, "", "empty_label")
        if not is_plausible_diagnosis_label(canonical):
            return OntologyDecision(False, canonical, "failed_structural_gate")
        if canonical in self._non_diagnostic_context_labels(findings, fact_graph):
            return OntologyDecision(False, canonical, "non_diagnostic_context_label")

        label_tokens = {token for token in canonical.split("_") if token}
        context_tokens = self._context_tokens(findings, fact_graph, rationale)
        case_tokens = self._case_tokens(findings, fact_graph)
        rationale_tokens = self._rationale_tokens(rationale)
        grounding_score = self._narrative_grounding_score(findings, fact_graph, rationale)

        guardrail_score, signals = self._guardrail_score(
            label_tokens=label_tokens,
            context_tokens=context_tokens,
            case_tokens=case_tokens,
            rationale_tokens=rationale_tokens,
            grounding_score=grounding_score,
            validated_by_llm=validated_by_llm,
        )

        llm_factor = 1.0 if validated_by_llm else 0.0
        # Threshold INCREASES as grounding decreases: low-grounding cases need stricter validation.
        # Previous formula was inverted (high grounding → high threshold, low grounding → low threshold).
        adaptive_context_threshold = max(
            0.22,
            min(0.5, 0.32 + (0.16 * (1.0 - grounding_score)) - (llm_factor * 0.1)),
        )
        if guardrail_score < adaptive_context_threshold and not validated_by_llm:
            return OntologyDecision(False, canonical, "no_case_context_overlap")

        if not require_external_verification or validated_by_llm:
            return OntologyDecision(
                True,
                canonical,
                "validated_structural_case_context_gate" if validated_by_llm else "structural_case_context_gate",
                externally_verified=True if validated_by_llm else False
            )

        verified, available = await self._verify_external_ontology(canonical)
        if verified:
            return OntologyDecision(True, canonical, "external_ontology_verified", externally_verified=True)

        if not available:
            fallback_threshold = min(0.78, max(0.3, adaptive_context_threshold + 0.12 - (llm_factor * 0.04)))
            if guardrail_score >= fallback_threshold:
                return OntologyDecision(
                    True,
                    canonical,
                    "external_unavailable_validated_gate" if validated_by_llm else "external_unavailable_structural_gate",
                    externally_verified=False,
                )
            return OntologyDecision(False, canonical, "external_unavailable_low_grounding")

        verified_threshold = min(0.84, max(0.42, adaptive_context_threshold + 0.22 - (llm_factor * 0.04)))
        if guardrail_score >= verified_threshold:
            return OntologyDecision(True, canonical, "high_grounding_without_external_confirmation", externally_verified=False)
        if signals.get("grounding_score", 0.0) >= 0.86 and validated_by_llm:
            return OntologyDecision(True, canonical, "llm_high_grounding_override", externally_verified=False)
        return OntologyDecision(False, canonical, "external_ontology_verification_failed")
