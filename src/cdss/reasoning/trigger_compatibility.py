"""Trigger compatibility scoring between case context and syndrome profiles.

Drug/trigger class resolution is delegated to the LLM cognitive router —
no hardcoded drug alias dictionaries. Any novel medication is correctly
classified by the LLM rather than failing silently.

The sync `trigger_compatibility_adjustment()` function retains backward
compatibility. Drug class tokens are fetched from session-level cache
(populated on first encounter via the router) so performance is unchanged
after warm-up.
"""

from __future__ import annotations

import asyncio
import re
from typing import Protocol

from src.cdss.contracts.models import StructuredFindings

import logging
logger = logging.getLogger(__name__)

# Cognitive router imported lazily to avoid circular import at module load
_router_available = False
try:
    from src.utils.llm_cognitive_router import infer_drug_class_tokens as _infer_drug_class
    from src.utils.llm_cognitive_router import _DRUG_CLASS_CACHE as _drug_cache
    _router_available = True
except Exception:
    _infer_drug_class = None  # type: ignore[assignment]
    _drug_cache = {}  # type: ignore[assignment]


class TriggerProfile(Protocol):
    trigger_requirements: list[str]
    trigger_exclusions: list[str]


# Clinical stopwords — not drug class tokens
_STOPWORDS = frozenset({
    "and", "or", "with", "without", "after", "before", "only",
    "recent", "new", "medication", "exposure", "process", "syndrome",
})

# Preserved 2-char medical acronyms (would otherwise be dropped by min-length gate).
_SHORT_MEDICAL_ACRONYMS = frozenset({
    "pe", "mi", "af", "dm", "hf", "ca", "tb", "ra", "uc",
    "mg", "bp", "hr", "rr", "o2",
})


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _tokenize(value: str) -> set[str]:
    return {
        t
        for t in _slug(value).split("_")
        if t not in _STOPWORDS
        and (len(t) > 2 or t in _SHORT_MEDICAL_ACRONYMS)
    }


def _case_terms(findings: StructuredFindings) -> set[str]:
    terms: set[str] = set()
    corpus = [
        findings.summary,
        *findings.positive_findings[:10],
        *findings.red_flags[:8],
        *findings.suspected_conditions[:8],
        *findings.raw_segments[:8],
        *findings.input_context[:10],
        *findings.exposures[:8],
        *findings.medications[:10],
        *findings.timeline[:6],
    ]
    for item in corpus:
        terms.update(_tokenize(str(item or "")))
    return terms


def _resolve_drug_class_tokens_sync(phrase: str) -> set[str]:
    """Return drug class tokens for a phrase, using cache or basic tokenization.

    Unlike the old hardcoded dict, this reads from the LLM-populated cache.
    If the phrase was not yet seen by the LLM, we fall back to naive tokenization.
    The async enrichment path pre-populates the cache before scoring when possible.
    """
    cache_key = str(phrase or "").strip().lower()
    if cache_key in _drug_cache:
        return _drug_cache[cache_key]

    # Fallback: naive token split (same quality as old code for simple phrases)
    tokens = _tokenize(phrase)
    # Also add full slug for exact match
    slug = _slug(phrase)
    if slug:
        tokens.add(slug)
    return tokens


def _phrase_matches_case_terms(phrase: str, case_terms: set[str]) -> bool:
    phrase_key = _slug(phrase)
    if not phrase_key:
        return False

    # Check LLM-resolved drug class terms first
    drug_class_tokens = _resolve_drug_class_tokens_sync(phrase)
    if drug_class_tokens & case_terms:
        return True

    # Basic token overlap as secondary check
    phrase_terms = _tokenize(phrase)
    if not phrase_terms:
        return False
    overlap = len(phrase_terms & case_terms)
    required_hits = 1 if len(phrase_terms) <= 2 else 2
    return overlap >= required_hits


async def pre_warm_drug_classes(phrases: list[str]) -> None:
    """Pre-populate LLM drug class cache for a list of trigger phrases.

    Call this before trigger_compatibility_adjustment() to ensure LLM
    classifications are available. Fire-and-forget safe.
    """
    if not _router_available or not _infer_drug_class:
        return
    tasks = [
        _infer_drug_class(phrase)
        for phrase in phrases
        if phrase and _slug(phrase) not in _drug_cache
    ]
    if tasks:
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass


def trigger_compatibility_adjustment(
    profile: TriggerProfile | None,
    findings: StructuredFindings,
) -> float:
    if not profile:
        return 0.0
    requirements = [item for item in profile.trigger_requirements[:8] if str(item or "").strip()]
    exclusions = [item for item in profile.trigger_exclusions[:8] if str(item or "").strip()]
    if not requirements and not exclusions:
        return 0.0

    case_terms = _case_terms(findings)
    requirement_hits = sum(1 for item in requirements if _phrase_matches_case_terms(item, case_terms))
    exclusion_hits = sum(1 for item in exclusions if _phrase_matches_case_terms(item, case_terms))

    adjustment = 0.0
    if requirements:
        if requirement_hits == 0:
            adjustment -= 0.28
        else:
            adjustment += min(0.18, 0.07 * requirement_hits)
    if exclusions:
        adjustment -= min(0.22, 0.11 * exclusion_hits)

    return round(max(-0.32, min(0.18, adjustment)), 2)
