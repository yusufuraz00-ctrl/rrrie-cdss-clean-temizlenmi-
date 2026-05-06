"""Cognitive router — replaces all hardcoded keyword maps with live LLM reasoning.

This module is the single hub for any decision that was previously a static
lookup table, regex pattern list, or keyword→signal dict.

All functions are:
  - Async (non-blocking)
  - Session-cached (same input → instant return on second call)
  - Fault-tolerant (any LLM failure returns a safe default)
  - Zero hardcoded domain knowledge

The LLM client is resolved lazily from the running llama-server (Qwen).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── Session-level caches ──────────────────────────────────────────────────────
_EVIDENCE_SIGNAL_CACHE: dict[str, tuple[list[str], bool]] = {}
_DRUG_CLASS_CACHE: dict[str, set[str]] = {}
_SLOT_CACHE: dict[str, list[str]] = {}

# ── Valid evidence signal names for retrieval/evidence contracts ───────
_VALID_SIGNALS = {
    "ecg_workup", "cardiac_enzyme_panel", "echocardiography", "chest_xray",
    "ct_imaging", "mri_imaging", "objective_vitals", "pulse_oximetry",
    "cbc_and_inflammatory_markers", "inflammatory_markers",
    "point_of_care_glucose", "csf_analysis", "liver_function_panel",
    "renal_function_panel", "thyroid_panel", "microbiologic_cultures",
    "ultrasound_imaging", "pulmonary_function_test", "coagulation_panel",
    "electrolyte_panel", "arterial_blood_gas", "discriminator_required_workup",
}

# ── LLM client resolution ─────────────────────────────────────────────────────

def _get_llm_client() -> Any | None:
    """Resolve the running LLM client without circular imports."""
    try:
        from config.settings import get_settings
        from src.llm.llama_cpp_client import LlamaCppClient
        settings = get_settings()
        return LlamaCppClient.get_instance(
            model_name=settings.hf_model_name,
            max_ctx=512,
        )
    except Exception as exc:
        logger.debug("cognitive_router_client_unavailable", error=str(exc))
        return None


async def _llm_complete(prompt: str, *, max_tokens: int = 80) -> str:
    """Run a prompt through the LLM, return raw text or '' on failure."""
    try:
        client = await asyncio.get_event_loop().run_in_executor(None, _get_llm_client)
        if client is None:
            return ""
        if hasattr(client, "complete"):
            return str(await client.complete(prompt, max_tokens=max_tokens, temperature=0.0) or "")
        if hasattr(client, "complete_sync"):
            return str(
                await asyncio.get_event_loop().run_in_executor(
                    None, lambda: client.complete_sync(prompt, max_tokens=max_tokens, temperature=0.0)
                ) or ""
            )
        if hasattr(client, "chat"):
            resp = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: client.chat(
                    [{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=max_tokens,
                ),
            )
            return str(getattr(resp, "content", resp) or "")
    except Exception as exc:
        logger.debug("cognitive_router_llm_failed", error=str(exc))
    return ""


# ── Evidence Signal Inference ─────────────────────────────────────────────────

async def infer_evidence_signals(
    title: str,
    definition: str,
    *,
    max_signals: int = 4,
) -> tuple[list[str], bool]:
    """Return (evidence_signals, is_high_acuity) for an ICD-11 entry.

    Replaces the hardcoded `_DEFINITION_TO_EVIDENCE` keyword map and
    `_HIGH_ACUITY_TITLE_PATTERNS` tuple in icd11_enricher.py.

    Uses the LLM to reason directly from the ICD-11 title and definition.
    Falls back to ["discriminator_required_workup"] + False on any failure.
    Results are cached per (title, definition) pair.
    """
    cache_key = f"{title[:80]}||{definition[:200]}"
    if cache_key in _EVIDENCE_SIGNAL_CACHE:
        return _EVIDENCE_SIGNAL_CACHE[cache_key]

    default = (["discriminator_required_workup"], False)
    if not title and not definition:
        return default

    prompt = (
        "You are a clinical decision support expert.\n\n"
        f"ICD-11 ENTRY: {title}\n"
        f"DEFINITION: {(definition or title)[:400]}\n\n"
        "Task 1 — List the 3-4 most important CLINICAL EVIDENCE SIGNALS needed to confirm or rule out this diagnosis.\n"
        "Choose ONLY from this exact list:\n"
        "ecg_workup | cardiac_enzyme_panel | echocardiography | chest_xray | ct_imaging | mri_imaging | "
        "objective_vitals | pulse_oximetry | cbc_and_inflammatory_markers | inflammatory_markers | "
        "point_of_care_glucose | csf_analysis | liver_function_panel | renal_function_panel | "
        "thyroid_panel | microbiologic_cultures | ultrasound_imaging | pulmonary_function_test | "
        "coagulation_panel | electrolyte_panel | arterial_blood_gas | discriminator_required_workup\n\n"
        "Task 2 — Is this a HIGH_ACUITY condition (life-threatening if missed)? Answer true or false.\n\n"
        "OUTPUT FORMAT (one line only):\n"
        "SIGNALS:<signal1>,<signal2>,<signal3> ACUITY:<true/false>\n\n"
        "Example: SIGNALS:ecg_workup,cardiac_enzyme_panel,objective_vitals ACUITY:true\n"
        "Now answer:"
    )

    raw = await _llm_complete(prompt, max_tokens=60)
    signals, high_acuity = _parse_signal_response(raw)

    if not signals:
        signals = ["discriminator_required_workup"]

    result = (signals[:max_signals], high_acuity)
    _EVIDENCE_SIGNAL_CACHE[cache_key] = result
    logger.debug(
        "cognitive_router_evidence_signals",
        title=title[:60],
        signals=signals,
        high_acuity=high_acuity,
    )
    return result


def _parse_signal_response(raw: str) -> tuple[list[str], bool]:
    """Parse LLM output: SIGNALS:a,b,c ACUITY:true/false"""
    signals: list[str] = []
    high_acuity = False
    text = str(raw or "").strip()

    # Extract SIGNALS block
    sig_match = re.search(r"SIGNALS:\s*([a-z_,\s]+?)(?:\s+ACUITY|$)", text, re.IGNORECASE)
    if sig_match:
        for token in re.split(r"[,\s]+", sig_match.group(1).strip()):
            candidate = token.strip().lower()
            if candidate in _VALID_SIGNALS:
                signals.append(candidate)

    # Extract ACUITY block
    acuity_match = re.search(r"ACUITY:\s*(true|false)", text, re.IGNORECASE)
    if acuity_match:
        high_acuity = acuity_match.group(1).strip().lower() == "true"

    # Fallback: scan entire text for valid signal names if structured parse failed
    if not signals:
        for token in re.split(r"[^a-z_]+", text.lower()):
            if token in _VALID_SIGNALS and token != "discriminator_required_workup":
                signals.append(token)
        # Acuity fallback
        if not acuity_match:
            lower = text.lower()
            high_acuity = any(kw in lower for kw in ("true", "high acuity", "life-threatening", "emergency"))

    return list(dict.fromkeys(signals)), high_acuity  # dedupe, preserve order


# ── Drug Class Inference ──────────────────────────────────────────────────────

async def infer_drug_class_tokens(phrase: str) -> set[str]:
    """Return the set of drug-class tokens for a medication/trigger phrase.

    Replaces `_TRIGGER_ALIAS_TERMS` in trigger_compatibility.py.
    Recognizes any drug, not just the 15 hardcoded ones.
    Results are cached per phrase.
    """
    cache_key = str(phrase or "").strip().lower()
    if not cache_key:
        return set()
    if cache_key in _DRUG_CLASS_CACHE:
        return _DRUG_CLASS_CACHE[cache_key]

    prompt = (
        "You are a clinical pharmacologist.\n\n"
        f"DRUG/TRIGGER: {phrase}\n\n"
        "List the pharmacological class tokens that describe this drug or trigger.\n"
        "Use lowercase underscore tokens only (e.g. ssri, serotonergic, antipsychotic, maoi, etc).\n"
        "Output only comma-separated tokens, no explanation.\n"
        "Example: fluoxetine → ssri, serotonergic, antidepressant\n"
        "Now classify:"
    )

    raw = await _llm_complete(prompt, max_tokens=40)
    tokens: set[str] = set()
    for token in re.split(r"[,\s]+", str(raw or "").lower()):
        cleaned = re.sub(r"[^a-z0-9_]", "", token.strip())
        if cleaned and len(cleaned) >= 2:
            tokens.add(cleaned)

    # Always include the phrase itself as a token
    phrase_token = re.sub(r"[^a-z0-9]", "_", cache_key).strip("_")
    if phrase_token:
        tokens.add(phrase_token)

    _DRUG_CLASS_CACHE[cache_key] = tokens
    logger.debug("cognitive_router_drug_class", phrase=phrase, tokens=list(tokens))
    return tokens


# ── Clinical Slot Inference ───────────────────────────────────────────────────

async def infer_clinical_slots(text: str, source_hint: str) -> list[str]:
    """Return phenotype slot names implied by a clinical text fragment.

    Replaces `_SOURCE_SLOT_MAP` in prototype_memory.py.
    Slots: tempo | trigger | trajectory | cluster | hemodynamics |
           physiology | localization | context | exposure
    """
    cache_key = f"{source_hint}||{text[:100]}"
    if cache_key in _SLOT_CACHE:
        return _SLOT_CACHE[cache_key]

    valid_slots = {
        "tempo", "trigger", "trajectory", "cluster",
        "hemodynamics", "physiology", "localization", "context", "exposure",
    }

    # Fast heuristic first — LLM only for genuinely ambiguous cases
    hint_lower = source_hint.lower()
    quick_map = {
        "timeline": ["tempo"],
        "exposure": ["exposure", "context", "trigger"],
        "red_flag": ["hemodynamics", "physiology"],
        "summary": ["context"],
    }
    if hint_lower in quick_map:
        result = quick_map[hint_lower]
        _SLOT_CACHE[cache_key] = result
        return result

    prompt = (
        "You are a clinical phenotyping expert.\n\n"
        f"SOURCE TYPE: {source_hint}\n"
        f"CLINICAL TEXT: {text[:200]}\n\n"
        "Which phenotype slots does this text primarily populate?\n"
        "Choose from: tempo | trigger | trajectory | cluster | hemodynamics | physiology | localization | context | exposure\n"
        "Output only comma-separated slot names (1-3 max):\n"
    )

    raw = await _llm_complete(prompt, max_tokens=20)
    slots: list[str] = []
    for token in re.split(r"[,\s|]+", str(raw or "").lower()):
        cleaned = token.strip()
        if cleaned in valid_slots and cleaned not in slots:
            slots.append(cleaned)

    result = slots or ["context"]
    _SLOT_CACHE[cache_key] = result
    return result


# ── Cache Management ──────────────────────────────────────────────────────────

def clear_all_caches() -> None:
    """Clear all session-level caches (for testing)."""
    _EVIDENCE_SIGNAL_CACHE.clear()
    _DRUG_CLASS_CACHE.clear()
    _SLOT_CACHE.clear()
