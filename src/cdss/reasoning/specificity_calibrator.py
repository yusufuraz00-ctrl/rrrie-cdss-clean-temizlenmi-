"""LLM-driven specificity calibration: determines if a candidate is a root disease
or a complication/subtype, and whether the case evidence actually supports that
specificity level.  Replaces the mechanical token-count formula in specificity.py
with an evidence-grounded LLM judgment.

Output format from LLM (one line per candidate):
    SPEC|<label>|<role>|<parent_label_or_none>|<evidence_supports: true/false>|<score 0-1>|<rationale>

Roles:
    root_disease  — the primary underlying pathology (preferred when evidence is ambiguous)
    complication  — a downstream consequence of a root disease
    subtype       — a specific variant where generic parent also fits
    syndrome      — a symptom constellation, not a mechanistic diagnosis

Design principles:
    - No hardcoded disease lists; the LLM applies its own medical training.
    - Falls back to token-count formula on any parse failure.
    - Safe: exceptions never propagate to the caller.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


import logging
logger = logging.getLogger(__name__)

@dataclass
class SpecificityJudgment:
    label: str
    role: str = "root_disease"                    # root_disease | complication | subtype | syndrome
    parent_label: str | None = None               # e.g. "diabetes" for "diabetic_peripheral_neuropathy"
    evidence_supports_specificity: bool = True    # False → prefer parent disease
    calibrated_specificity_score: float = 0.5    # replaces token-count output
    rationale: str = ""


_ROLE_DEFAULT_SCORE: dict[str, float] = {
    "root_disease": 0.72,
    "subtype": 0.60,
    "syndrome": 0.50,
    "complication": 0.34,
}

_SPEC_LINE_RE = re.compile(
    r"^SPEC\|([^|]+)\|([^|]+)\|([^|]*)\|(true|false)\|([\d.]+)\|(.+)$",
    re.IGNORECASE,
)


def _build_prompt(
    candidates: list[str],
    positive_findings: list[str],
    negative_findings: list[str],
    demographics_summary: str,
) -> str:
    pos_text = "; ".join(positive_findings[:12]) if positive_findings else "not specified"
    neg_text = "; ".join(negative_findings[:6]) if negative_findings else "none documented"
    cand_text = "\n".join(f"  - {c}" for c in candidates)
    return (
        "You are a senior clinical diagnostician performing specificity calibration.\n\n"
        f"PATIENT CONTEXT\n"
        f"Demographics: {demographics_summary or 'not specified'}\n"
        f"Positive findings: {pos_text}\n"
        f"Absent/negative findings: {neg_text}\n\n"
        "CANDIDATE DIAGNOSES\n"
        f"{cand_text}\n\n"
        "TASK\n"
        "For each candidate, classify its diagnostic role and assess whether the case "
        "evidence specifically supports that level of specificity.\n\n"
        "Use these roles:\n"
        "  root_disease  — primary underlying pathology\n"
        "  complication  — downstream consequence of a root disease\n"
        "  subtype       — specific variant where a more general parent also fits\n"
        "  syndrome      — symptom constellation, not a mechanistic diagnosis\n\n"
        "For 'complication' and 'subtype': name the parent disease in the parent field.\n"
        "evidence_supports should be 'true' only when the case has specific findings that "
        "confirm the complication/subtype (not just the parent disease).\n"
        "calibrated_score should reflect how well the evidence anchors this specific label "
        "(0.0–1.0). Root diseases with direct evidence → 0.65–0.85. Complications without "
        "confirming specific findings → 0.25–0.40.\n\n"
        "OUTPUT FORMAT — one line per candidate, no extra text:\n"
        "SPEC|<label>|<role>|<parent_or_none>|<true/false>|<score>|<one-sentence rationale>\n\n"
        "Examples:\n"
        "SPEC|diabetic_peripheral_neuropathy|complication|diabetes|false|0.32|"
        "No neuropathy exam findings documented; only hyperglycemia present.\n"
        "SPEC|diabetes|root_disease|none|true|0.74|"
        "Hyperglycemia, polyuria, polydipsia all present — root disease confirmed.\n"
        "SPEC|acute_hepatocellular_necrosis|complication|liver_disease|false|0.30|"
        "Jaundice documented but no specific hepatocyte necrosis markers.\n"
        "SPEC|jaundice|syndrome|none|true|0.55|"
        "Presenting symptom only; requires workup for underlying etiology.\n"
        "SPEC|common_cold|root_disease||true|0.78|"
        "Cough, rhinorrhea, sore throat — generic URI symptoms without localizing discriminators; "
        "use root label not subspecialty ICD variant.\n"
        "SPEC|acute_viral_upper_respiratory_infection|subtype|common_cold|false|0.42|"
        "No findings discriminate from common_cold (no viral culture, no systemic involvement "
        "requiring subtyping); parent label preferred.\n"
        "SPEC|urinary_tract_infection|root_disease||true|0.80|"
        "Dysuria + frequency + suprapubic pain — root diagnosis fully supported; do not specify "
        "cystitis vs pyelonephritis without anatomical localization evidence.\n"
        "SPEC|acute_cystitis|subtype|urinary_tract_infection|false|0.38|"
        "Bladder-specific findings not documented; only generic UTI symptoms — "
        "evidence does not support subtype specificity.\n\n"
        "Now classify each candidate:\n"
    )


def _adaptive_unsupported_cap(evidence_density: float) -> float:
    """Adaptive cap on unsupported subtype/complication score.

    Base cap 0.45. In evidence-rich cases (density > 0.6) raise cap up to 0.55 so
    subtypes with strong surrounding context still surface. Density <= 0.6 stays 0.45.
    """
    try:
        d = max(0.0, min(1.0, float(evidence_density or 0.0)))
    except Exception:
        d = 0.0
    if d <= 0.6:
        return 0.45
    return round(0.45 + 0.25 * (d - 0.6), 3)  # 0.45 at d=0.6 → 0.55 at d=1.0


def _parse_spec_lines(
    raw: str,
    candidates: list[str],
    *,
    evidence_density: float = 0.0,
) -> dict[str, SpecificityJudgment]:
    judgments: dict[str, SpecificityJudgment] = {}
    unsupported_cap = _adaptive_unsupported_cap(evidence_density)
    for line in raw.splitlines():
        line = line.strip()
        m = _SPEC_LINE_RE.match(line)
        if not m:
            continue
        label = m.group(1).strip().lower().replace(" ", "_")
        role = m.group(2).strip().lower()
        if role not in {"root_disease", "complication", "subtype", "syndrome"}:
            role = "root_disease"
        parent_raw = m.group(3).strip().lower().replace(" ", "_")
        parent = parent_raw if parent_raw and parent_raw not in {"none", "n/a", "-", ""} else None
        evidence_supports = m.group(4).strip().lower() == "true"
        try:
            score = max(0.0, min(1.0, float(m.group(5))))
        except ValueError:
            score = _ROLE_DEFAULT_SCORE.get(role, 0.5)
        # If complication/subtype without supporting evidence, apply adaptive cap
        # (0.45 default, scales to 0.55 when evidence density is high — lets subtypes surface
        # in evidence-rich cases instead of being crushed by fixed floor).
        if role in {"complication", "subtype"} and not evidence_supports:
            score = min(score, unsupported_cap)
        rationale = m.group(6).strip()
        judgments[label] = SpecificityJudgment(
            label=label,
            role=role,
            parent_label=parent,
            evidence_supports_specificity=evidence_supports,
            calibrated_specificity_score=round(score, 2),
            rationale=rationale,
        )
    return judgments


def _fallback_judgments(candidates: list[str]) -> dict[str, SpecificityJudgment]:
    """Return neutral judgments (will cause token-count formula to be used instead)."""
    return {
        c: SpecificityJudgment(
            label=c,
            role="root_disease",
            evidence_supports_specificity=True,
            calibrated_specificity_score=-1.0,  # sentinel: use token-count fallback
        )
        for c in candidates
    }


def _demographics_summary(findings: Any) -> str:
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


def calibrate_specificity(
    candidates: list[str],
    *,
    findings: Any,
    fact_graph: Any,
    llm_client: Any,
    policy: Any,
) -> dict[str, SpecificityJudgment]:
    """Run LLM-driven specificity calibration for up to 5 candidates.

    Returns a dict mapping label → SpecificityJudgment.
    On any failure, returns fallback judgments (calibrated_specificity_score=-1.0)
    which tells specificity.py to use the token-count formula instead.
    """
    if not candidates:
        return {}
    if not getattr(policy, "specificity_calibrator_enabled", True):
        return _fallback_judgments(candidates)

    candidates = [str(c or "").strip().lower() for c in candidates[:5] if str(c or "").strip()]
    if not candidates:
        return {}

    try:
        positive_findings: list[str] = []
        negative_findings: list[str] = []
        try:
            positive_findings = list(findings.positive_findings or [])[:12]
            negative_findings = list(findings.negative_findings or [])[:6]
            if findings.summary:
                positive_findings = [findings.summary] + positive_findings
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

        demo_summary = _demographics_summary(findings)
        prompt = _build_prompt(candidates, positive_findings, negative_findings, demo_summary)

        # Evidence density proxy: positive_findings + red_flags count, saturates at 8+.
        try:
            _dense_count = len(positive_findings or []) + len(list(getattr(findings, "red_flags", None) or [])[:8])
            _evidence_density = min(1.0, _dense_count / 8.0)
        except Exception:
            _evidence_density = 0.0

        # Use the LLM client synchronously (specificity resolver is called sync)
        raw_output: str = ""
        try:
            # Try the common sync interface patterns
            if hasattr(llm_client, "complete_sync"):
                raw_output = llm_client.complete_sync(prompt, max_tokens=256, temperature=0.0)
            elif hasattr(llm_client, "complete"):
                import asyncio
                raw_output = asyncio.get_event_loop().run_until_complete(
                    llm_client.complete(prompt, max_tokens=256, temperature=0.0)
                )
            elif hasattr(llm_client, "call_sync"):
                raw_output = llm_client.call_sync(prompt, max_tokens=256)
            elif callable(llm_client):
                raw_output = llm_client(prompt)
            else:
                return _fallback_judgments(candidates)
        except Exception:
            return _fallback_judgments(candidates)

        if not raw_output or not isinstance(raw_output, str):
            return _fallback_judgments(candidates)

        judgments = _parse_spec_lines(raw_output, candidates, evidence_density=_evidence_density)

        # Fill in any candidates the LLM skipped
        for c in candidates:
            if c not in judgments:
                judgments[c] = SpecificityJudgment(
                    label=c,
                    calibrated_specificity_score=-1.0,  # use token-count fallback
                )

        return judgments

    except Exception:
        return _fallback_judgments(candidates)


async def calibrate_specificity_async(
    candidates: list[str],
    *,
    findings: Any,
    fact_graph: Any,
    llm_client: Any,
    policy: Any,
) -> dict[str, SpecificityJudgment]:
    """Async entrypoint — calls llm_client.complete() directly, no executor wrapping.

    Eliminates the double-nesting risk of run_in_executor(lambda: run_until_complete(...)).
    Falls back to a single run_in_executor only when the client exposes no async interface.
    """
    import asyncio as _asyncio

    if not candidates:
        return {}
    if not getattr(policy, "specificity_calibrator_enabled", True):
        return _fallback_judgments(candidates)

    candidates = [str(c or "").strip().lower() for c in candidates[:5] if str(c or "").strip()]
    if not candidates:
        return {}

    try:
        positive_findings: list[str] = []
        negative_findings: list[str] = []
        try:
            positive_findings = list(findings.positive_findings or [])[:12]
            negative_findings = list(findings.negative_findings or [])[:6]
            if findings.summary:
                positive_findings = [findings.summary] + positive_findings
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass

        demo_summary = _demographics_summary(findings)
        prompt = _build_prompt(candidates, positive_findings, negative_findings, demo_summary)

        # Evidence density proxy: positive_findings + red_flags count, saturates at 8+.
        try:
            _dense_count = len(positive_findings or []) + len(list(getattr(findings, "red_flags", None) or [])[:8])
            _evidence_density = min(1.0, _dense_count / 8.0)
        except Exception:
            _evidence_density = 0.0

        raw_output: str = ""
        try:
            if hasattr(llm_client, "complete"):
                raw_output = await llm_client.complete(prompt, max_tokens=256, temperature=0.0)
            elif hasattr(llm_client, "complete_sync"):
                raw_output = await _asyncio.get_event_loop().run_in_executor(
                    None, lambda: llm_client.complete_sync(prompt, max_tokens=256, temperature=0.0)
                )
            elif hasattr(llm_client, "call_sync"):
                raw_output = await _asyncio.get_event_loop().run_in_executor(
                    None, lambda: llm_client.call_sync(prompt, max_tokens=256)
                )
            elif callable(llm_client):
                raw_output = await _asyncio.get_event_loop().run_in_executor(
                    None, lambda: llm_client(prompt)
                )
            else:
                return _fallback_judgments(candidates)
        except Exception:
            return _fallback_judgments(candidates)

        if not raw_output or not isinstance(raw_output, str):
            return _fallback_judgments(candidates)

        judgments = _parse_spec_lines(raw_output, candidates, evidence_density=_evidence_density)
        for c in candidates:
            if c not in judgments:
                judgments[c] = SpecificityJudgment(label=c, calibrated_specificity_score=-1.0)
        return judgments

    except Exception:
        return _fallback_judgments(candidates)


def extract_parent_injections(
    judgments: dict[str, SpecificityJudgment],
    *,
    only_unsupported: bool = True,
) -> list[str]:
    """Return parent_label values for subtypes/complications whose specificity is not
    evidence-supported.  These parents should be injected as competing candidates
    before final arbitration so root diseases can be scored against their subtypes.

    Capped at 2 injections per pass to avoid differential bloat.
    """
    parents: list[str] = []
    seen: set[str] = set()
    for label, j in judgments.items():
        if j.parent_label is None:
            continue
        if only_unsupported and j.evidence_supports_specificity:
            continue
        if j.role not in {"subtype", "complication"}:
            continue
        p = j.parent_label.strip().lower().replace(" ", "_")
        if p and p not in seen and p != label:
            seen.add(p)
            parents.append(p)
        if len(parents) >= 2:
            break
    return parents
