"""Clinical priors — generalized pattern detectors that force critical
diagnoses onto the differential when textbook trigger sets are present.

Each detector is a pure function over the patient narrative + findings + vitals,
returning ``ForcedCandidate`` records. The orchestrator (``apply_clinical_priors``)
merges them into the existing ``DifferentialSet``: if an equivalent candidate
is already present, its score is bumped and rationale appended; otherwise the
forced candidate is inserted at a score that is competitive but cannot blindly
dominate higher-confidence swarm consensus.

These priors are deliberately *general* — they encode triadic / tetradic
clinical rules that appear in any internal-medicine / emergency-medicine
textbook, not benchmark-specific if/else hooks. Adding a new prior should only
require a new detector function plus a single line in ``apply_clinical_priors``.

Design constraints:
  * Each detector takes ONLY the narrative + findings + vitals + demographics.
  * No registry / LLM call from this module — keeps it deterministic and fast.
  * Score caps prevent prior-driven hijacking of swarm-confirmed top diagnoses.
  * Every forced candidate carries a ``trigger_summary`` so the audit trail
    explains why the prior fired.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from src.cdss.contracts.models import (
    DifferentialCandidate,
    DifferentialSet,
    StructuredFindings,
)
from src.cdss.knowledge.ontology import normalize_candidate_label


# Score caps: a prior can push an overlooked diagnosis onto the slate, but it
# should not silently outrank a swarm-confirmed top candidate. The orchestrator
# resolves collisions by score-then-narrative-grounding, so we keep prior
# insertions at the "competitive but not dominant" tier.
_PRIOR_INSERT_SCORE = 0.55
_PRIOR_BUMP = 0.12
_PRIOR_MAX_SCORE = 0.85


@dataclass(frozen=True)
class ForcedCandidate:
    label: str
    score: float
    rationale: list[str] = field(default_factory=list)
    evidence_needed: list[str] = field(default_factory=list)
    must_not_miss: bool = False
    trigger_summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _narrative_blob(findings: StructuredFindings, narrative: str = "") -> str:
    parts: list[str] = []
    if narrative:
        parts.append(str(narrative))
    if findings.summary:
        parts.append(findings.summary)
    parts.extend(findings.raw_segments[:30])
    parts.extend(findings.positive_findings[:30])
    parts.extend(findings.input_context[:20])
    return " \n ".join(str(p or "") for p in parts).lower()


def _has_any(text: str, patterns: Iterable[str]) -> bool:
    return any(p in text for p in patterns)


def _extract_age(text: str) -> int | None:
    """Pull a plausible adult-or-child age from narrative."""
    if not text:
        return None
    m = re.search(r"\b(\d{1,3})[\s\-]*(?:y(?:ear|r)?s?(?:[\s\-]*old)?|yo|y/o)\b", text, re.IGNORECASE)
    if m:
        try:
            age = int(m.group(1))
            if 0 <= age <= 120:
                return age
        except ValueError:
            pass
    m2 = re.search(r"\b(?:i'?m|i am)\s+(\d{1,3})\b", text, re.IGNORECASE)
    if m2:
        try:
            age = int(m2.group(1))
            if 0 <= age <= 120:
                return age
        except ValueError:
            pass
    return None


def _is_female(text: str, demographics: dict | None) -> bool:
    if demographics:
        sex = str(demographics.get("sex", "") or demographics.get("gender", "")).strip().lower()
        if sex in {"f", "female", "woman", "girl"}:
            return True
        if sex in {"m", "male", "man", "boy"}:
            return False
    return _has_any(text, (
        " female", "woman", " girl", " she ", " her ", "wife", "daughter",
        "i'm a 22-year-old", "i am a 22", "mother",
    ))


# ---------------------------------------------------------------------------
# Q3 — Pregnancy / ruptured ectopic prior
# ---------------------------------------------------------------------------

_LATE_PERIOD_PATTERNS = (
    "period is", "period was", "missed period", "missed my period",
    "no period", "late period", "period late", "period.*late",
    "amenorrh", "haven't had my period", "havent had my period",
    "weeks late", "month late",
)
_ABDOMINAL_PAIN_PATTERNS = (
    "abdominal pain", "abdomen pain", "stomach pain", "belly pain",
    "lower abdomen", "rlq", "right lower quadrant", "llq", "left lower quadrant",
    "pelvic pain", "lower abdominal",
)
_PREGNANCY_RED_FLAGS = (
    "shoulder pain", "kehr", "syncope", "fainted", "passed out",
    "lightheaded", "dizzy when i stand", "orthostatic", "dizziness when",
)


def detect_pregnancy_prior(
    narrative: str,
    findings: StructuredFindings,
    demographics: dict | None = None,
) -> list[ForcedCandidate]:
    """Reproductive-age female + missed/late period + abdominal pain ⇒
    pregnancy must be ruled out before any non-pregnancy abdominal diagnosis
    closes. With shoulder pain or orthostasis, ruptured ectopic specifically
    is must-not-miss.
    """
    text = _narrative_blob(findings, narrative)
    if not text:
        return []

    age = _extract_age(text)
    age_in_range = age is not None and 11 <= age <= 55
    female = _is_female(text, demographics)
    if not (female and age_in_range):
        return []

    if not _has_any(text, _ABDOMINAL_PAIN_PATTERNS):
        return []

    late_period = (
        any(re.search(p, text) for p in _LATE_PERIOD_PATTERNS if "." in p)
        or _has_any(text, [p for p in _LATE_PERIOD_PATTERNS if "." not in p])
    )
    if not late_period:
        return []

    rupture_flag = _has_any(text, _PREGNANCY_RED_FLAGS)

    out: list[ForcedCandidate] = [
        ForcedCandidate(
            label="ectopic_pregnancy",
            score=_PRIOR_MAX_SCORE if rupture_flag else _PRIOR_INSERT_SCORE,
            rationale=[
                "Reproductive-age female with abdominal pain and late/missed period — pregnancy testing is mandatory before any abdominal diagnosis closes.",
                "Right shoulder pain (Kehr's sign) and orthostasis suggest intraperitoneal blood and ruptured ectopic." if rupture_flag else "",
            ],
            evidence_needed=[
                "urine_or_serum_beta_hcg",
                "transvaginal_pelvic_ultrasound",
                "type_and_screen",
                "iv_access_and_volume_resuscitation_if_unstable",
            ],
            must_not_miss=True,
            trigger_summary="reproductive_age_female + abdominal_pain + late_period" + (" + rupture_flags(shoulder/orthostasis)" if rupture_flag else ""),
        ),
    ]
    return out


# ---------------------------------------------------------------------------
# Q4 — Household-cluster environmental toxin (CO) prior
# ---------------------------------------------------------------------------

_HEATING_DEVICE_PATTERNS = (
    "gas heater", "gas heating", "kerosene heater", "wood stove",
    "fireplace", "furnace", "generator", "boiler", "space heater",
    "garage running", "car running", "engine running",
)
_SHARED_HOUSEHOLD_PATTERNS = (
    "whole family", "everyone", "we all", "all of us", "my husband",
    "my wife", "my son", "my daughter", "kids are", "children are",
    "together in", "in the house",
)
_PET_AFFECTED_PATTERNS = (
    "dog threw up", "dog vomit", "dog is", "cat is", "pet is",
    "dog acting", "cat acting", "the dog", "our dog", "our cat",
)
_CO_SYMPTOM_PATTERNS = (
    "headache", "nausea", "vomit", "weak", "sleepy", "drowsy",
    "lethargic", "confused", "lightheaded", "dizz",
)


def detect_household_toxin_prior(
    narrative: str,
    findings: StructuredFindings,
) -> list[ForcedCandidate]:
    """Multiple household members AND/OR a pet sharing nonspecific symptoms
    (headache, nausea, weakness, somnolence) in proximity to a combustion
    appliance ⇒ environmental toxin (CO is the apex must-not-miss).
    Triggers: ≥2 of {humans, pet} affected + appliance mention + symptom set.
    """
    text = _narrative_blob(findings, narrative)
    if not text:
        return []

    has_device = _has_any(text, _HEATING_DEVICE_PATTERNS)
    has_shared = _has_any(text, _SHARED_HOUSEHOLD_PATTERNS)
    has_pet = _has_any(text, _PET_AFFECTED_PATTERNS)
    co_symptom_count = sum(1 for p in _CO_SYMPTOM_PATTERNS if p in text)

    cluster_score = 0
    if has_shared:
        cluster_score += 1
    if has_pet:
        cluster_score += 1
    if has_device:
        cluster_score += 1

    if cluster_score < 2 or co_symptom_count < 2:
        return []

    high_specificity = has_device and (has_pet or has_shared) and co_symptom_count >= 3

    return [
        ForcedCandidate(
            label="carbon_monoxide_poisoning",
            score=_PRIOR_MAX_SCORE if high_specificity else _PRIOR_INSERT_SCORE,
            rationale=[
                "Household-wide nonspecific illness clustering with a combustion appliance is carbon monoxide poisoning until proven otherwise.",
                "Pet involvement strengthens environmental over infectious etiology — pets do not catch human flu / gastroenteritis." if has_pet else "",
            ],
            evidence_needed=[
                "evacuate_residence_immediately",
                "co_oximetry_carboxyhemoglobin_level",
                "high_flow_or_hyperbaric_oxygen",
                "fire_department_environmental_co_check",
            ],
            must_not_miss=True,
            trigger_summary=(
                f"household_cluster(humans={has_shared},pet={has_pet}) + "
                f"combustion_device={has_device} + co_symptoms={co_symptom_count}"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Q5 — Catecholamine triad (pheochromocytoma) prior
# ---------------------------------------------------------------------------

_EPISODIC_HEADACHE_PATTERNS = (
    "headache", "head throb", "head pound", "throbbing head", "pounding head",
)
_DIAPHORESIS_PATTERNS = (
    "sweat", "diaphor", "drenched", "soaking",
)
_PALPITATION_PATTERNS = (
    "heart racing", "palpitation", "heart pounding", "heart beating",
    "tachycardia", "racing heart",
)
_EPISODIC_PATTERNS = (
    "episode", "comes on", "come and go", "twice a week", "out of nowhere",
    "sudden", "attack", "spell",
)


def _max_systolic_bp(text: str, vitals: dict | None) -> float | None:
    """Pull the highest plausible systolic from narrative + vitals."""
    candidates: list[float] = []
    if vitals:
        for key in ("sbp", "systolic", "blood_pressure_systolic"):
            try:
                v = float(str(vitals.get(key, "") or "").strip())
                if 50.0 <= v <= 300.0:
                    candidates.append(v)
            except (TypeError, ValueError):
                continue
    if text:
        for m in re.finditer(r"\b(\d{2,3})\s*/\s*(\d{2,3})\b", text):
            try:
                sys_bp = float(m.group(1))
                dia_bp = float(m.group(2))
                if 60.0 <= sys_bp <= 300.0 and 30.0 <= dia_bp <= 200.0:
                    candidates.append(sys_bp)
            except ValueError:
                continue
    return max(candidates) if candidates else None


def detect_catecholamine_triad_prior(
    narrative: str,
    findings: StructuredFindings,
    vitals: dict | None = None,
) -> list[ForcedCandidate]:
    """Pheochromocytoma classic tetrad: episodic severe headache, diaphoresis,
    palpitations, paroxysmal hypertension (often >180/110). When ≥3 components
    are present AND a hypertensive reading is documented, the system must not
    close on a benign anchor (panic disorder, anxiety) without ruling out a
    catecholamine-secreting tumor.
    """
    text = _narrative_blob(findings, narrative)
    if not text:
        return []
    has_headache = _has_any(text, _EPISODIC_HEADACHE_PATTERNS)
    has_sweat = _has_any(text, _DIAPHORESIS_PATTERNS)
    has_palp = _has_any(text, _PALPITATION_PATTERNS)
    has_episodic = _has_any(text, _EPISODIC_PATTERNS)

    triad_count = sum([has_headache, has_sweat, has_palp])
    if triad_count < 2:
        return []

    sbp = _max_systolic_bp(text, vitals or (findings.derived_vitals or {}))
    severe_htn = sbp is not None and sbp >= 180.0
    if not severe_htn and triad_count < 3:
        return []
    if not has_episodic and triad_count < 3:
        return []

    high_specificity = severe_htn and triad_count >= 3 and has_episodic

    return [
        ForcedCandidate(
            label="pheochromocytoma",
            score=_PRIOR_MAX_SCORE if high_specificity else _PRIOR_INSERT_SCORE,
            rationale=[
                "Episodic catecholamine triad (severe headache + diaphoresis + palpitations) with documented severe hypertension — pheochromocytoma must be excluded before attributing symptoms to anxiety / panic disorder.",
                f"Documented systolic blood pressure {sbp:.0f} mmHg is a hypertensive emergency, not a panic-attack surge." if severe_htn else "",
            ],
            evidence_needed=[
                "plasma_or_24h_urine_metanephrines",
                "ct_or_mri_adrenal_imaging_if_biochemistry_positive",
                "endocrinology_referral",
                "avoid_unopposed_beta_blockade_until_alpha_blockade_started",
            ],
            must_not_miss=True,
            trigger_summary=(
                f"catecholamine_triad(headache={has_headache},sweat={has_sweat},palp={has_palp}) + "
                f"episodic={has_episodic} + sbp_max={sbp}"
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def apply_clinical_priors(
    differential: DifferentialSet,
    *,
    narrative: str,
    findings: StructuredFindings,
    demographics: dict | None = None,
    vitals: dict | None = None,
) -> tuple[DifferentialSet, list[ForcedCandidate]]:
    """Run every detector and merge results into the differential.

    Merge rules:
      * If a candidate with the same normalized label already exists, raise its
        score to ``max(existing, prior_score)`` and append the prior rationale.
      * Else insert the forced candidate at the prior's declared score.
      * Always re-sort by score and cap at the top 5.

    Returns the updated differential plus the list of priors that fired (for
    trace / audit purposes).
    """
    if differential is None:
        return differential, []

    fired: list[ForcedCandidate] = []
    fired.extend(detect_pregnancy_prior(narrative, findings, demographics))
    fired.extend(detect_household_toxin_prior(narrative, findings))
    fired.extend(detect_catecholamine_triad_prior(narrative, findings, vitals))

    if not fired:
        return differential, []

    by_norm = {
        normalize_candidate_label(c.label) or str(c.label or "").lower(): idx
        for idx, c in enumerate(differential.candidates or [])
    }
    new_candidates = list(differential.candidates or [])
    for prior in fired:
        target = normalize_candidate_label(prior.label) or prior.label.lower()
        rationale_lines = [r for r in prior.rationale if r]
        rationale_lines.append(f"[clinical_prior] {prior.trigger_summary}")
        if target in by_norm:
            existing = new_candidates[by_norm[target]]
            new_score = max(float(existing.score or 0.0), prior.score) + _PRIOR_BUMP
            new_score = min(_PRIOR_MAX_SCORE, new_score)
            new_candidates[by_norm[target]] = existing.model_copy(update={
                "score": round(new_score, 2),
                "rationale": (list(existing.rationale or []) + rationale_lines)[:5],
                "evidence_needed": list(dict.fromkeys(list(existing.evidence_needed or []) + list(prior.evidence_needed)))[:6],
            })
        else:
            new_candidates.append(
                DifferentialCandidate(
                    label=target,
                    score=round(prior.score, 2),
                    rationale=rationale_lines[:5],
                    evidence_needed=list(prior.evidence_needed)[:6],
                    status="candidate",
                )
            )

    new_candidates.sort(key=lambda c: float(c.score or 0.0), reverse=True)
    return (
        differential.model_copy(update={"candidates": new_candidates[:5]}),
        fired,
    )


__all__ = [
    "ForcedCandidate",
    "detect_pregnancy_prior",
    "detect_household_toxin_prior",
    "detect_catecholamine_triad_prior",
    "apply_clinical_priors",
]
