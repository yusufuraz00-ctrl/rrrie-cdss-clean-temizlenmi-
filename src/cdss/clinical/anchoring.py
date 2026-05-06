"""Anchoring detector — Plan Item 7 (cross-case engineering improvements).

Detects when the top-1 fused candidate matches a phrase the patient used in
their own free-text narrative (a self-diagnosis or "is it X?" hypothesis).
When anchoring is detected the gateway downweights the anchored candidate by
a fixed penalty so the rank reflects objective findings rather than the
patient's own framing.

Mechanism mirrors inter-rater anchor disagreement (Cohen's kappa style) — the
detector is a similarity score between two token sets, not a free-text
classifier. One threshold (cosine ≥ ``anchor_cosine_threshold``) and one
penalty (multiplicative ``anchor_penalty``) — both live in policy.py.
"""

from __future__ import annotations

import re

from src.cdss.contracts.models import AnchoringReport, StructuredFindings

__all__ = ["detect_anchoring"]


_TOKEN_RE = re.compile(r"[a-z0-9çğıöşü]{3,}", re.IGNORECASE)

# Patient-question markers — language-agnostic union. Hitting any of these in
# the matched phrase boosts the score because the patient is *proposing* a
# diagnosis, not reporting a symptom.
_QUESTION_MARKERS = {
    # Turkish question particles + diagnostic phrasings
    "mı", "mi", "mu", "mü",
    "olabilir",  # "could it be"
    "midir",
    # English diagnostic-question phrasings (kept as substrings; cheap match)
    "is it",
    "could it be",
    "is this",
    "do i have",
    "could this be",
    "might it be",
}

_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "for", "in", "on", "with",
    "is", "it", "that", "this", "be", "as", "at", "by", "from", "i",
    "ben", "bir", "bu", "ile", "için", "ya", "ve", "de", "da",
    "syndrome", "disease", "disorder", "acute", "chronic",
})


def _tokenize(text: str) -> set[str]:
    if not text:
        return set()
    return {
        token
        for token in (m.group(0).lower() for m in _TOKEN_RE.finditer(text))
        if token not in _STOPWORDS
    }


def _label_tokens(label: str) -> set[str]:
    if not label:
        return set()
    flat = re.sub(r"[_\-]+", " ", str(label).strip().lower())
    return _tokenize(flat)


def _stem_match(a: str, b: str, *, prefix_len: int = 5) -> bool:
    """True iff two tokens share a non-trivial stem.

    Cross-language anchoring exists for stems that differ only by
    inflectional suffix (English ``gastritis`` vs Turkish ``gastrit``); pure
    set-intersection misses these. The detector treats two tokens as a
    near-match when their first ``prefix_len`` characters agree AND the
    shorter of the two is at least ``prefix_len`` long.
    """

    if not a or not b:
        return False
    if a == b:
        return True
    n = min(len(a), len(b))
    if n < prefix_len:
        return False
    return a[:prefix_len] == b[:prefix_len]


def _cosine_like(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = 0
    for token in a:
        if token in b or any(_stem_match(token, other) for other in b):
            inter += 1
    denom = (len(a) * len(b)) ** 0.5
    return inter / denom if denom else 0.0


def _has_question_marker(phrase: str) -> bool:
    if not phrase:
        return False
    lowered = phrase.lower()
    for marker in _QUESTION_MARKERS:
        if marker in lowered:
            return True
    # Trailing-particle Turkish patterns ("gastrit mi?", "alerji mi?")
    if re.search(r"\b\w+\s+m[iıuü]\??\b", lowered):
        return True
    return False


def detect_anchoring(
    findings: StructuredFindings,
    top_label: str,
    *,
    cosine_threshold: float = 0.78,
    question_marker_boost: float = 0.15,
) -> AnchoringReport:
    """Return ``AnchoringReport`` for the (findings, top_label) pair.

    Heuristic: tokenise the patient-narrative lane sentence-by-sentence,
    compute cosine-style overlap with the top-label tokens, and pick the best
    sentence. If a sentence contains a diagnostic-question marker, add
    ``question_marker_boost`` (default 0.15). Anchoring fires when the final
    score crosses ``cosine_threshold``.
    """

    if not top_label:
        return AnchoringReport(top_label="")
    label_set = _label_tokens(top_label)
    if not label_set:
        return AnchoringReport(top_label=str(top_label))

    lane_sentences: list[str] = []
    if findings.summary:
        lane_sentences.append(findings.summary)
    lane_sentences.extend(findings.raw_segments[:18])
    lane = (findings.context_lanes or {}).get("patient_narrative", []) if findings.context_lanes else []
    lane_sentences.extend(lane[:24])
    # Split each sentence-block on common terminators so a long paragraph
    # doesn't dominate the cosine via sheer length.
    pieces: list[str] = []
    for blk in lane_sentences:
        blk = str(blk or "").strip()
        if not blk:
            continue
        for piece in re.split(r"[.?!\n]+", blk):
            piece = piece.strip()
            if piece:
                pieces.append(piece)

    best_score = 0.0
    best_phrase = ""
    for piece in pieces:
        toks = _tokenize(piece)
        if not toks:
            continue
        score = _cosine_like(label_set, toks)
        if score <= 0.0:
            continue
        if _has_question_marker(piece):
            score = min(1.0, score + question_marker_boost)
        if score > best_score:
            best_score = score
            best_phrase = piece

    is_anchored = best_score >= float(cosine_threshold)
    rationale = ""
    if is_anchored:
        rationale = (
            f"Top candidate '{top_label}' overlaps a patient-narrative phrase "
            f"(cosine={best_score:.2f}) — likely anchoring on self-diagnosis."
        )
    return AnchoringReport(
        is_anchored=is_anchored,
        score=round(best_score, 3),
        matched_phrase=best_phrase[:200],
        top_label=str(top_label),
        rationale=rationale,
    )
