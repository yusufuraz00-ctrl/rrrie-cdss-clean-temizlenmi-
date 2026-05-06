"""Shared clinical text normalization helpers for multilingual intake."""

from __future__ import annotations

import re
import unicodedata

_MOJIBAKE_REPLACEMENTS = {
    "Ã§": "c",
    "Ã‡": "C",
    "ÄŸ": "g",
    "Ä": "G",
    "Ä±": "i",
    "Ä°": "I",
    "Ã¶": "o",
    "Ã–": "O",
    "ÅŸ": "s",
    "Å": "S",
    "Ã¼": "u",
    "Ãœ": "U",
    "Â°": "°",
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€": '"',
    "â€“": "-",
    "â€”": "-",
}
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\xa0": " ",
    }
)
_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "with",
    "within",
    "without",
    "ve",
    "veya",
    "ve",
    "ile",
    "icin",
    "gibi",
    "bir",
    "bu",
    "da",
    "de",
    "mi",
    "midir",
    "veya",
}
_QUERY_NOISE_TOKENS = {
    "agent",
    "agents",
    "associated",
    "candidate",
    "criteria",
    "disease",
    "directly",
    "event",
    "history",
    "impact",
    "must",
    "present",
    "review",
    "specific",
    "target",
    "targets",
    "workup",
}


def repair_mojibake(text: str) -> str:
    value = str(text or "").translate(_QUOTE_TRANSLATION)
    for broken, fixed in _MOJIBAKE_REPLACEMENTS.items():
        if broken in value:
            value = value.replace(broken, fixed)
    return value


def ascii_fold(text: str) -> str:
    repaired = repair_mojibake(text)
    normalized = unicodedata.normalize("NFKD", repaired)
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).lower()


def normalize_clinical_text(text: str) -> str:
    return " ".join(repair_mojibake(text).split()).strip()


def sanitize_query_text(text: str, *, max_terms: int = 12) -> str:
    folded = ascii_fold(text)
    folded = re.sub(r"[^a-z0-9/%+_ -]+", " ", folded)
    folded = re.sub(r"\s+", " ", folded).strip()
    tokens = [token for token in folded.split() if token]
    if not tokens:
        return ""

    frequencies: dict[str, int] = {}
    for token in tokens:
        frequencies[token] = frequencies.get(token, 0) + 1

    repeat_cutoff = max(2, len(tokens) // 3)
    informative_token_count = len(
        [
            token
            for token in tokens
            if len(token) >= 3 and token not in _QUERY_STOPWORDS and token not in _QUERY_NOISE_TOKENS
        ]
    )
    ranked: list[tuple[float, str]] = []
    for idx, token in enumerate(tokens):
        if token.isdigit() or len(token) < 3:
            continue
        if token in _QUERY_STOPWORDS:
            continue
        if informative_token_count >= 3 and token in _QUERY_NOISE_TOKENS:
            continue
        score = min(1.5, len(token) / 12.0)
        if re.search(r"[0-9]|[%/+]", token):
            score += 0.35
        score += max(0.0, 0.3 - (idx * 0.01))
        if frequencies.get(token, 0) >= repeat_cutoff:
            score -= 0.45
        ranked.append((score, token))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected: list[str] = []
    seen: set[str] = set()
    for _, token in ranked:
        if token in seen:
            continue
        seen.add(token)
        selected.append(token)
        if len(selected) >= max_terms:
            break
    return " ".join(selected)
