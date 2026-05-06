"""Clinical scoring: epidemiological priors and candidate adjudication.

Merged from epi_prior.py and state_adjudication.py.
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from typing import Any


import logging
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Epidemiological prior types
# ---------------------------------------------------------------------------

_TIER_WEIGHTS: dict[str, float] = {
    "common": 1.0,
    "uncommon": 0.6,
    "rare": 0.28,
    "exceptional": 0.10,
}


@dataclass
class EpiPriorTier:
    label: str
    tier: str = "uncommon"
    tier_rationale: str = ""
    prior_weight: float = 1.0


@dataclass
class EpiPriorResult:
    tiers: list[EpiPriorTier] = field(default_factory=list)
    context_used: str = ""
    pubmed_grounded: bool = False
    raw_evidence: list[str] = field(default_factory=list)

    def weight_for(self, label: str) -> float:
        label = str(label or "").strip().lower()
        for tier in self.tiers:
            if tier.label == label:
                return tier.prior_weight
        return 1.0

    def tier_for(self, label: str) -> str:
        label = str(label or "").strip().lower()
        for tier in self.tiers:
            if tier.label == label:
                return tier.tier
        return "uncommon"


# ---------------------------------------------------------------------------
# Epidemiological prior — LLM prompt & parsing
# ---------------------------------------------------------------------------

_PRIOR_LINE_RE = re.compile(r"^PRIOR\|([^|]+)\|([^|]+)\|(.+)$", re.IGNORECASE)


def _demographics_context(findings: Any) -> str:
    try:
        demo = findings.demographics or {}
        parts: list[str] = []
        age = demo.get("age") or demo.get("age_years")
        if age:
            parts.append(f"age {age}")
        sex = demo.get("sex") or demo.get("gender")
        if sex:
            parts.append(str(sex))
        setting = demo.get("setting") or demo.get("clinical_setting")
        if setting:
            parts.append(str(setting))
        return ", ".join(parts) if parts else "not specified"
    except Exception:
        return "not specified"


def _urgency_context(risk_profile: Any) -> str:
    try:
        urgency = str(risk_profile.urgency.value if hasattr(risk_profile.urgency, "value") else risk_profile.urgency)
        return urgency.lower()
    except Exception:
        return "outpatient"


_TROPICAL_DISEASE_TOKENS: frozenset[str] = frozenset({
    "malaria", "dengue", "typhoid", "plasmodium", "arboviral", "leptospira",
    "chikungunya", "yellow_fever", "leishmaniasis", "schistosomiasis",
})

_SUBSTANCE_RELATED_TOKENS: frozenset[str] = frozenset({
    "drug_reaction", "drug_induced", "serum_sickness", "medication_hypersensitivity",
    "drug_hypersensitivity", "withdrawal", "detox", "alcohol", "benzodiazepine",
    "opioid", "intoxication", "substance",
})


def _build_epi_prompt(
    candidates: list[str],
    demographics: str,
    urgency: str,
    positive_findings: list[str],
    pubmed_snippets: list[str],
    exposures: list[str] | None = None,
) -> str:
    cand_text = "\n".join(f"  - {c}" for c in candidates)
    pos_text = "; ".join(positive_findings[:8]) if positive_findings else "not specified"
    evidence_block = ""
    if pubmed_snippets:
        lines = "\n".join(f"EVIDENCE: {s[:200]}" for s in pubmed_snippets[:4])
        evidence_block = f"\nPUBMED EVIDENCE (use to calibrate tiers):\n{lines}\n"
    exposure_block = ""
    if exposures:
        exp_text = "\n".join(f"  {e}" for e in exposures[:6])
        exposure_block = (
            f"\nTRAVEL / EXPOSURE CONTEXT\n{exp_text}\n"
            "If the above exposures reference a geographic region, country, or endemic area, "
            "adjust tier assignments to reflect local disease prevalence for that region.\n"
        )

    # Context-aware calibration blocks: generated from candidate and exposure analysis
    context_calibration_block = ""
    candidate_token_set: set[str] = set()
    for c in candidates:
        candidate_token_set.update(c.lower().split("_"))

    # No substance/medication context → drug reactions and withdrawal are exceptional
    has_substance_exposure = bool(exposures) and any(
        any(tok in e.lower() for tok in ("drug", "medication", "alcohol", "substance", "chemical", "toxin"))
        for e in exposures
    )
    has_substance_candidates = bool(candidate_token_set & _SUBSTANCE_RELATED_TOKENS)
    if has_substance_candidates and not has_substance_exposure:
        context_calibration_block += (
            "\nSUBSTANCE CONTEXT: No medications or substance exposures are documented for this patient. "
            "Classify drug_reaction, drug_induced_*, withdrawal_syndrome, and substance-related diagnoses "
            "as 'exceptional' tier — they require a documented causative agent. "
            "Do not assign 'common' or 'uncommon' to substance diagnoses without confirmed exposure.\n"
        )

    # Tropical disease candidates present but no geographic exposure documented
    has_tropical_candidates = bool(candidate_token_set & _TROPICAL_DISEASE_TOKENS)
    if has_tropical_candidates and not exposures:
        context_calibration_block += (
            "\nTROPICAL DISEASE CONTEXT: Tropical disease candidates are present but no travel or "
            "geographic exposure is documented for this patient. Without confirmed endemic area exposure, "
            "classify tropical infections (malaria, dengue, typhoid, etc.) as 'uncommon' to 'rare' tier. "
            "They belong in the differential but should not rank as 'common' without geographic context.\n"
        )

    return (
        "You are an expert clinical epidemiologist.\n\n"
        "PATIENT CONTEXT\n"
        f"Demographics: {demographics}\n"
        f"Clinical setting: {urgency}\n"
        f"Key positive findings: {pos_text}\n"
        f"{exposure_block}"
        f"{evidence_block}"
        f"{context_calibration_block}\n"
        "CANDIDATE DIAGNOSES\n"
        f"{cand_text}\n\n"
        "TASK\n"
        "For each candidate, classify its population-level epidemiological frequency "
        "for a patient matching the above demographics and clinical setting.\n\n"
        "Tiers:\n"
        "  common      — very frequently seen in this demographic/setting (>1:100)\n"
        "  uncommon    — occasionally seen (1:100 to 1:1000)\n"
        "  rare        — infrequently seen (<1:1000)\n"
        "  exceptional — very rare or highly atypical for this presentation (<1:10000)\n\n"
        "OUTPUT FORMAT — one line per candidate, no extra text:\n"
        "PRIOR|<label>|<tier>|<one-sentence rationale with prevalence reasoning>\n\n"
        "Now classify each candidate:\n"
    )


def _parse_prior_lines(raw: str, candidates: list[str]) -> list[EpiPriorTier]:
    tiers: list[EpiPriorTier] = []
    seen: set[str] = set()
    for line in raw.splitlines():
        m = _PRIOR_LINE_RE.match(line.strip())
        if not m:
            continue
        label = m.group(1).strip().lower().replace(" ", "_")
        tier = m.group(2).strip().lower()
        if tier not in _TIER_WEIGHTS:
            tier = "uncommon"
        if label in seen:
            continue
        seen.add(label)
        tiers.append(EpiPriorTier(label=label, tier=tier, tier_rationale=m.group(3).strip(), prior_weight=_TIER_WEIGHTS[tier]))
    for c in candidates:
        if c not in seen:
            tiers.append(EpiPriorTier(label=c, tier="uncommon", prior_weight=1.0))
    return tiers


def _neutral_result(candidates: list[str], context: str = "") -> EpiPriorResult:
    return EpiPriorResult(
        tiers=[EpiPriorTier(label=c, tier="uncommon", prior_weight=1.0) for c in candidates],
        context_used=context,
    )


async def assess_epidemiological_priors(
    candidates: list[str],
    *,
    findings: Any,
    risk_profile: Any,
    pubmed_snippets: list[str] | None = None,
    llm_client: Any,
    policy: Any,
) -> EpiPriorResult:
    if not candidates:
        return EpiPriorResult()
    if not getattr(policy, "epi_prior_enabled", True):
        return _neutral_result(candidates)

    candidates = [str(c or "").strip().lower() for c in candidates[:5] if str(c or "").strip()]
    if not candidates:
        return EpiPriorResult()

    demographics = _demographics_context(findings)
    urgency = _urgency_context(risk_profile)
    context_used = f"{demographics} | {urgency}"

    positive_findings: list[str] = []
    exposures: list[str] = []
    try:
        positive_findings = list(findings.positive_findings or [])[:8]
        if findings.summary:
            positive_findings = [findings.summary] + positive_findings
        exposures = [str(e) for e in (findings.exposures or []) if str(e or "").strip()][:6]
    except Exception:  # noqa: BLE001
        logger.debug("swallowed exception", exc_info=True)
        pass

    snippets = list(pubmed_snippets or [])[:4]
    prompt = _build_epi_prompt(candidates, demographics, urgency, positive_findings, snippets, exposures=exposures or None)

    raw_output = ""
    try:
        if hasattr(llm_client, "complete"):
            raw_output = await llm_client.complete(prompt, max_tokens=300, temperature=0.0)
        elif hasattr(llm_client, "complete_sync"):
            raw_output = await asyncio.get_event_loop().run_in_executor(None, lambda: llm_client.complete_sync(prompt, max_tokens=300, temperature=0.0))
        elif hasattr(llm_client, "call_sync"):
            raw_output = await asyncio.get_event_loop().run_in_executor(None, lambda: llm_client.call_sync(prompt, max_tokens=300))
        elif callable(llm_client):
            raw_output = await asyncio.get_event_loop().run_in_executor(None, lambda: llm_client(prompt))
        else:
            return _neutral_result(candidates, context_used)
    except Exception:
        return _neutral_result(candidates, context_used)

    if not raw_output or not isinstance(raw_output, str):
        return _neutral_result(candidates, context_used)

    return EpiPriorResult(
        tiers=_parse_prior_lines(raw_output, candidates),
        context_used=context_used,
        pubmed_grounded=bool(snippets),
        raw_evidence=snippets,
    )


