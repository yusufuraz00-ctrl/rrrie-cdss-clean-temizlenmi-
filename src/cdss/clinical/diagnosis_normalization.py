"""Generic differential label normalization and sibling consolidation."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.cdss.contracts.models import DifferentialCandidate, DifferentialSet
from src.cdss.text_normalization import ascii_fold


@dataclass(frozen=True)
class DifferentialNormalizationResult:
    differential: DifferentialSet
    absorbed: dict[str, list[str]] = field(default_factory=dict)


def canonicalize_label(label: str) -> str:
    text = ascii_fold(str(label or ""))
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _tokens(label: str) -> set[str]:
    return {token for token in canonicalize_label(label).split("_") if len(token) >= 4}


def _similarity(left: str, right: str) -> float:
    l_norm = canonicalize_label(left)
    r_norm = canonicalize_label(right)
    if not l_norm or not r_norm:
        return 0.0
    char_score = SequenceMatcher(None, l_norm, r_norm).ratio()
    lt, rt = _tokens(l_norm), _tokens(r_norm)
    token_score = (len(lt & rt) / max(1, min(len(lt), len(rt)))) if lt and rt else 0.0
    return max(char_score, token_score)


def _merge_candidate(primary: DifferentialCandidate, absorbed: DifferentialCandidate) -> DifferentialCandidate:
    score = round(min(0.99, max(primary.score, absorbed.score) + min(0.04, absorbed.score * 0.04)), 2)
    rationale = list(dict.fromkeys([*primary.rationale, *absorbed.rationale]))[:6]
    evidence_needed = list(dict.fromkeys([*primary.evidence_needed, *absorbed.evidence_needed]))[:6]
    return primary.model_copy(
        update={
            "score": score,
            "rationale": rationale,
            "evidence_needed": evidence_needed,
        }
    )


def normalize_differential(
    differential: DifferentialSet,
    *,
    similarity_threshold: float = 0.86,
) -> DifferentialNormalizationResult:
    candidates = [
        candidate.model_copy(update={"label": canonicalize_label(candidate.label) or candidate.label})
        for candidate in list(differential.candidates or [])
        if str(candidate.label or "").strip()
    ]
    candidates.sort(key=lambda item: float(item.score or 0.0), reverse=True)

    survivors: list[DifferentialCandidate] = []
    absorbed_map: dict[str, list[str]] = {}
    for candidate in candidates:
        matched_index: int | None = None
        for idx, survivor in enumerate(survivors):
            if candidate.label == survivor.label or _similarity(candidate.label, survivor.label) >= similarity_threshold:
                matched_index = idx
                break
        if matched_index is None:
            survivors.append(candidate)
            continue

        survivor = survivors[matched_index]
        if candidate.score > survivor.score:
            winner, loser = candidate, survivor
        else:
            winner, loser = survivor, candidate
        merged = _merge_candidate(winner, loser)
        survivors[matched_index] = merged
        absorbed_map.setdefault(merged.label, [])
        if loser.label != merged.label and loser.label not in absorbed_map[merged.label]:
            absorbed_map[merged.label].append(loser.label)

    survivors.sort(key=lambda item: float(item.score or 0.0), reverse=True)
    normalized = differential.model_copy(update={"candidates": survivors})
    return DifferentialNormalizationResult(differential=normalized, absorbed=absorbed_map)
