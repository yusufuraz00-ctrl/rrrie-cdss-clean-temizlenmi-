"""Inline candidate-level grounding gate.

Runs after the swarm dedup step in the differential stage. For each
``DifferentialCandidate`` it checks that the *factual* parts of the
rationale (the parts that cite findings the patient actually reported)
are traceable to the patient narrative. Conceptual / interpretive
fragments (mechanism names, diagnostic-pattern labels, planned tests)
are deliberately excluded from grounding because, by definition, they
are clinical interpretation rather than narrative content.

This separation is the fix for the "Batch-2 inline grounding regression"
where the original gate over-penalized specific must-not-miss
diagnoses (epidural_hematoma, pulmonary_embolism, STEMI ...) whose
rationales naturally use technical vocabulary not present in lay
patient speech ("lucid interval", "post-traumatic intracranial bleed",
"CT head"). Generic umbrella labels like ``traumatic_brain_injury``
escaped the old gate because their rationales reused common-language
tokens already present in the narrative — a precision-sacrificing bias
this module now removes.

Verdict policy (thresholds from ``config/clinical_thresholds.json``,
section ``grounding``):

  - ``pass``   : factual-claim score >= ``inline_demote_score_below``
                 OR no factual claims to check (cannot fail what we
                 cannot evaluate)
  - ``demote`` : ``inline_drop_score_below`` <= score <
                 ``inline_demote_score_below``
  - ``drop``   : factual-claim score < ``inline_drop_score_below``
                 AND at least ``min_factual_claims_for_drop`` factual
                 claims were checked (don't drop on tiny samples)

Demote multiplies the candidate score by ``max(0.5, score)`` to limit
the worst-case demotion to a 2× penalty (was an unbounded multiplication
in the old gate, which collapsed scores to ~0.10 for partially-grounded
specific diagnoses).
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

from src.cdss.contracts.models import (
    DifferentialCandidate,
    DifferentialSet,
    GroundingVerdict,
    StructuredFindings,
)
from src.cdss.core import thresholds as clinical_thresholds
from src.cdss.clinical.grounding import _narrative_segments, _supporting_segment, _tokens
from src.cdss.text_normalization import ascii_fold

_LOG = logging.getLogger(__name__)


# Conceptual / interpretive markers. A claim fragment containing any of
# these tokens is an *interpretation* (mechanism name, diagnostic
# pattern, planned test) rather than a factual claim about the patient,
# and must NOT be checked against the narrative — checking it
# systematically penalizes the most-specific (and clinically most
# valuable) diagnoses.
_CONCEPTUAL_MARKERS: frozenset[str] = frozenset(
    {
        # diagnosis-name suffixes / pathology terms
        "hematoma", "hemorrhage", "haemorrhage", "infarct", "infarction",
        "embolism", "embolus", "ischemia", "ischaemia", "ischemic",
        "dissection", "tamponade", "thrombosis", "thrombus", "edema",
        "oedema", "axonal", "encephalopathy", "encephalitis", "meningitis",
        "myocarditis", "pericarditis", "endocarditis", "vasculitis",
        "myopathy", "neuropathy", "stenosis", "occlusion", "rupture",
        "perforation", "obstruction", "necrosis", "abscess", "sepsis",
        "shock", "syndrome", "pneumonitis", "pyelonephritis", "gastritis",
        "colitis", "otitis", "sinusitis", "mastoiditis", "dermatitis",
        "tumor", "tumour", "neoplasm", "carcinoma", "lymphoma", "leukemia",
        # reasoning / interpretation cues
        "pattern", "consistent", "suggests", "suggest", "suggestive",
        "consider", "diagnosis", "differential", "etiology", "aetiology",
        "mechanism", "pathophysiology", "classic", "hallmark", "characteristic",
        "lucid", "prodrome", "ddx", "rule-out", "rule_out", "ruleout",
        "must-not-miss", "must_not_miss",
        # imaging / test names that belong in evidence_needed, not in
        # narrative-grounding
        "ct", "mri", "ultrasound", "echocardiogram", "echocardiography",
        "angiogram", "angiography", "scan", "neuroimaging", "neurosurgical",
        "inr", "ptt", "cbc", "cmp", "bmp", "troponin", "lactate", "lipase",
        "amylase", "electrolytes", "ldh", "esr", "crp", "d-dimer", "ddimer",
        "pcr", "tsh", "bnp", "abg", "vbg", "creatinine", "ekg", "ecg",
        "x-ray", "xray", "fluoroscopy",
    }
)


_SENTENCE_SPLIT = re.compile(r"[.;\n]+")
_TOKEN_FOR_CONCEPT = re.compile(r"[a-z][a-z0-9-]{2,}")


# Lay→formal synonym fold. Maps the common-language form a patient
# uses to the formal token a clinician's rationale tends to use. Both
# directions are added so token-overlap matching catches either side.
# Intentionally small and high-precision; do not over-extend.
_LAY_FORMAL_PAIRS: tuple[tuple[str, str], ...] = (
    ("knocked", "unconscious"),
    ("knocked", "loss"),
    ("knocked", "consciousness"),
    ("blacked", "unconscious"),
    ("blacked", "consciousness"),
    ("passed", "unconscious"),
    ("hit", "trauma"),
    ("hit", "impact"),
    ("slipped", "fall"),
    ("fell", "fall"),
    ("vomited", "vomiting"),
    ("vomit", "vomiting"),
    ("threw", "vomiting"),
    ("nausea", "nausea"),
    ("queasy", "nausea"),
    ("weak", "weakness"),
    ("weakness", "weakness"),
    ("numb", "numbness"),
    ("dizzy", "dizziness"),
    ("dizziness", "dizziness"),
    ("splitting", "severe"),
    ("worst", "severe"),
    ("pounding", "severe"),
    ("blurry", "blurred"),
    ("vision", "visual"),
    ("breath", "breathing"),
    ("breathing", "breathing"),
    ("short", "dyspnea"),
    ("racing", "tachycardia"),
    ("pounding", "palpitation"),
    ("crushing", "pressure"),
    ("squeezing", "pressure"),
    ("tearing", "tearing"),
    ("ripping", "tearing"),
    ("sweating", "diaphoresis"),
    ("sweaty", "diaphoresis"),
    ("clammy", "diaphoresis"),
    ("fever", "febrile"),
    ("hot", "febrile"),
    ("freezing", "chills"),
    ("shivering", "chills"),
    ("confused", "confusion"),
    ("foggy", "confusion"),
    ("seizure", "convulsion"),
    ("seizing", "convulsion"),
    ("limping", "gait"),
    ("walking", "gait"),
)


_SUFFIX_STEM_RULES: tuple[tuple[str, str], ...] = (
    ("ying", "y"),       # vomiting (n-form)
    ("ies", "y"),        # injuries
    ("ed", ""),          # vomited → vomit
    ("ing", ""),         # vomiting → vomit
    ("ness", ""),        # weakness → weak
    ("ously", ""),       # consciously
    ("ous", ""),         # conscious
    ("ity", ""),         # severity
    ("al", ""),          # focal → foc (rough but cheap)
    ("ic", ""),          # ischemic → ischem
    ("s", ""),           # plural
)


def _stem(token: str) -> str:
    """Cheap suffix-stripping stem. Token assumed lowercase."""
    if len(token) <= 4:
        return token
    for suffix, replacement in _SUFFIX_STEM_RULES:
        if token.endswith(suffix) and len(token) - len(suffix) + len(replacement) >= 3:
            return token[: -len(suffix)] + replacement
    return token


def _expand_with_synonyms(tokens: set[str]) -> set[str]:
    """Add lay/formal synonyms and stems to a token set."""
    out = set(tokens)
    for token in tokens:
        out.add(_stem(token))
    for lay, formal in _LAY_FORMAL_PAIRS:
        if lay in out:
            out.add(formal)
        if formal in out:
            out.add(lay)
    return out


def _is_conceptual_fragment(text: str) -> bool:
    """A fragment is conceptual if it contains any conceptual marker."""
    if not text:
        return False
    folded = ascii_fold(text).lower()
    tokens = set(_TOKEN_FOR_CONCEPT.findall(folded))
    if not tokens:
        return False
    return bool(tokens & _CONCEPTUAL_MARKERS)


def _factual_subclaims(rationale_fragment: str) -> list[str]:
    """Split a rationale string into sub-claims and return only factual ones.

    Splits on sentence-like delimiters (``.``, ``;``, newline). Each
    sub-claim is classified by ``_is_conceptual_fragment``; conceptual
    sub-claims are dropped. Falls back to the whole fragment if it is
    too short to split.
    """
    text = (rationale_fragment or "").strip()
    if not text:
        return []
    subs = [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]
    if not subs:
        subs = [text]
    return [s for s in subs if not _is_conceptual_fragment(s)]


def _candidate_claim_strings(candidate: DifferentialCandidate) -> list[str]:
    """Return only the *factual* claim strings to ground-check.

    Excludes:
    - the candidate label itself (a canonicalized diagnosis name);
    - all ``evidence_needed`` items (these are tests/exams/imaging that,
      by definition, have not happened yet — checking them against the
      patient narrative is a category error);
    - any rationale sub-claim flagged conceptual (mechanism names,
      diagnostic pattern labels, planned-test references).

    What remains are the *narrative-citing* fragments — the symptoms,
    timing, exposures, and risk-factor mentions that the rationale
    derived directly from what the patient said. Those are the only
    fragments where "is this in the narrative?" is a meaningful test.
    """
    out: list[str] = []
    for item in (candidate.rationale or []):
        out.extend(_factual_subclaims(str(item or "")))
    return out


def _verdict_for_score(score: float, factual_claim_count: int) -> str:
    demote_below = clinical_thresholds.get_float("grounding.inline_demote_score_below", 0.60)
    drop_below = clinical_thresholds.get_float("grounding.inline_drop_score_below", 0.40)
    min_for_drop = int(clinical_thresholds.get_float("grounding.min_factual_claims_for_drop", 3))

    # Cannot drop a candidate on the basis of fewer than ``min_for_drop``
    # factual claims — too small a sample, too high a false-negative risk.
    if score < drop_below and factual_claim_count >= max(1, min_for_drop):
        return "drop"
    if score < demote_below:
        return "demote"
    return "pass"


def _supporting_segment_lenient(claim: str, segments: list[str], *, minimum_overlap: float = 0.45) -> str:
    """Token-overlap supporter with lay/formal synonym fold + cheap stemming.

    Stricter base check is delegated to grounding._supporting_segment. If
    that fails, retry with synonym-and-stem-expanded token sets so
    common phrasings like "knocked out for a minute" / "loss of
    consciousness" map onto the same expanded set.
    """
    direct = _supporting_segment(claim, segments, minimum_overlap=minimum_overlap)
    if direct:
        return direct

    claim_tokens = _expand_with_synonyms(_tokens(claim))
    if not claim_tokens:
        return ""
    for segment in segments:
        seg_tokens = _expand_with_synonyms(_tokens(segment))
        if not seg_tokens:
            continue
        overlap = len(claim_tokens & seg_tokens) / max(1, len(claim_tokens))
        if overlap >= minimum_overlap:
            return segment
    return ""


def _ground_one(candidate: DifferentialCandidate, segments: list[str]) -> GroundingVerdict:
    claims = _candidate_claim_strings(candidate)
    if not claims:
        # No factual claims to check → cannot fail; pass with neutral score.
        # This is the central fix for the Batch-2 regression: candidates
        # whose rationale was entirely conceptual (e.g. "epidural hematoma
        # — classic post-traumatic pattern") used to be marked unsupported
        # because the conceptual tokens were absent from the narrative.
        # Now those candidates correctly bypass the gate.
        return GroundingVerdict(verdict="pass", score=1.0, unsupported_claims=[], checked_claims=0)

    unsupported: list[str] = []
    supported = 0
    for claim in claims:
        try:
            support = _supporting_segment_lenient(claim, segments)
        except Exception:  # noqa: BLE001 — never let grounding crash the pipeline
            _LOG.debug("inline grounding error on claim %r; treating as supported", claim)
            supported += 1
            continue
        if support:
            supported += 1
        else:
            unsupported.append(claim)

    score = round(supported / max(1, len(claims)), 3)
    return GroundingVerdict(
        verdict=_verdict_for_score(score, len(claims)),
        score=score,
        unsupported_claims=unsupported[:8],
        checked_claims=len(claims),
    )


def gate_candidate_grounding(
    candidate: DifferentialCandidate,
    findings: StructuredFindings,
) -> GroundingVerdict:
    """Public entry: ground-check a single candidate against the patient narrative."""
    try:
        segments = _narrative_segments(findings)
    except Exception:  # noqa: BLE001
        _LOG.debug("could not derive narrative segments; passing candidate")
        return GroundingVerdict(verdict="pass", score=1.0, unsupported_claims=[], checked_claims=0)
    return _ground_one(candidate, segments)


def gate_differential(
    differential: DifferentialSet,
    findings: StructuredFindings,
) -> tuple[DifferentialSet, dict[str, GroundingVerdict], float]:
    """Apply inline grounding to every candidate in a ``DifferentialSet``.

    Returns
    -------
    updated_differential
        New ``DifferentialSet`` with ``drop`` candidates removed and
        ``demote`` candidates' scores reduced (capped at 2× penalty so
        a partially-grounded specific diagnosis cannot collapse below
        a fully-grounded generic diagnosis purely on grounding-score
        differences).
    verdicts
        Map of candidate label → ``GroundingVerdict`` for the *original*
        differential (before drops). Useful for trace / eval.
    pass_rate
        Fraction of candidates whose verdict was ``pass``.
    """
    try:
        segments = _narrative_segments(findings)
    except Exception:  # noqa: BLE001
        return differential, {}, 1.0

    verdicts: dict[str, GroundingVerdict] = {}
    surviving: list[DifferentialCandidate] = []
    pass_count = 0
    # Cap the demote penalty: a candidate whose factual claims are 50 %
    # grounded should not be punished worse than 2× (was unbounded). This
    # is what allowed a specific dx with strong but partially-grounded
    # rationale to collapse below a generic dx that simply reused
    # common-language tokens.
    demote_cap = float(
        clinical_thresholds.get_float("grounding.inline_demote_floor", 0.50)
    )

    for candidate in (differential.candidates or []):
        verdict = _ground_one(candidate, segments)
        verdicts[candidate.label] = verdict
        if verdict.verdict == "pass":
            pass_count += 1
            surviving.append(
                candidate.model_copy(
                    update={
                        "grounding_score": verdict.score,
                        "grounding_verdict": "pass",
                        "grounding_unsupported": list(verdict.unsupported_claims),
                    }
                )
            )
        elif verdict.verdict == "demote":
            multiplier = max(demote_cap, verdict.score)
            new_score = round(float(candidate.score or 0.0) * multiplier, 4)
            surviving.append(
                candidate.model_copy(
                    update={
                        "score": new_score,
                        "grounding_score": verdict.score,
                        "grounding_verdict": "demote",
                        "grounding_unsupported": list(verdict.unsupported_claims),
                    }
                )
            )
        # drop: candidate omitted from surviving list

    surviving.sort(key=lambda c: float(c.score or 0.0), reverse=True)
    pass_rate = round(pass_count / max(1, len(differential.candidates or [])), 3)
    updated = differential.model_copy(update={"candidates": surviving})
    return updated, verdicts, pass_rate


def aggregate_pass_rate(verdicts: Iterable[GroundingVerdict]) -> float:
    items = list(verdicts)
    if not items:
        return 1.0
    passes = sum(1 for v in items if v.verdict == "pass")
    return round(passes / len(items), 3)


__all__ = [
    "gate_candidate_grounding",
    "gate_differential",
    "aggregate_pass_rate",
]
