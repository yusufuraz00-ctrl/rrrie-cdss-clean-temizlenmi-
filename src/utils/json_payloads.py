"""Lightweight JSON salvage helpers shared outside the legacy pipeline."""

from __future__ import annotations

import json
import re
from typing import Any


THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def _strip_think_artifacts(text: str) -> str:
    cleaned = str(text or "")
    cleaned = THINK_RE.sub("", cleaned)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def sanitize_json_text(text: str) -> str:
    """Normalize common malformed JSON patterns emitted by LLMs."""

    cleaned = _strip_think_artifacts(text).strip()
    if not cleaned:
        return ""

    if "```" in cleaned:
        cleaned = "\n".join(
            line for line in cleaned.splitlines() if not line.strip().startswith("```")
        ).strip()

    cleaned = cleaned.replace("\ufeff", "").replace("\r\n", "\n")
    cleaned = cleaned.replace("“", '"').replace("”", '"').replace("’", "'")
    cleaned = re.sub(r'(?<=[,\[])\s*\\+"', '"', cleaned)
    cleaned = re.sub(r'\\(?="\s*[,}\]])', "", cleaned)
    cleaned = re.sub(r'"\s*\n\s*"(?=[^:\n])', '",\n"', cleaned)
    cleaned = re.sub(r'([\]}])\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*:', r'\1,\n"\2":', cleaned)
    cleaned = re.sub(r':\s*([0-9]+/[0-9]+)([,}\n\s])', r': "\1"\2', cleaned)
    cleaned = re.sub(r'"([^"]+)"\s*->\s*([^,\]\n]+)', r'"\1 -> \2"', cleaned)
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)
    return cleaned.strip()


def _repair_truncated_json(candidate: str) -> dict[str, Any] | None:
    stack: list[str] = []
    in_string = False
    escape = False
    output: list[str] = []

    for char in candidate:
        output.append(char)
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            stack.append("}")
        elif char == "[":
            stack.append("]")
        elif char in {"}", "]"}:
            if not stack or stack[-1] != char:
                return None
            stack.pop()

    if in_string:
        output.append('"')
    while stack:
        output.append(stack.pop())

    try:
        parsed = json.loads("".join(output))
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_json_from_response(text: str, *, strict: bool = False) -> dict[str, Any]:
    """Best-effort JSON parse for structured model payloads.

    Returns an empty dict when parsing fails, matching the legacy helper's
    failure semantics used by case generation and repair paths.
    """

    cleaned = sanitize_json_text(text)
    if not cleaned:
        return {}

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass

    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        candidate = sanitize_json_text(cleaned[first_brace : last_brace + 1])
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            if strict:
                return {}
            repaired = _repair_truncated_json(candidate)
            return repaired or {}

    if strict:
        return {}

    candidate = sanitize_json_text(cleaned[first_brace:]) if first_brace != -1 else cleaned
    repaired = _repair_truncated_json(candidate)
    return repaired or {}
