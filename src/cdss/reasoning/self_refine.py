"""Self-refine pass (W3 Module I.3).

After the cognitive loop terminates, fire one LLM call asking a senior-clinician
archetype: "Given the current belief state, is there anything a seasoned
attending would flag that the system missed?" Output parsed into `CTX|...` atoms
that are appended to `StructuredFindings.input_context` so they flow into
downstream learning logs AND into the next iteration's LLM context.

Response format (prompt-enforced):

    CTX|<slot>|<text>

Each line: one observation. `slot` is a short tag (e.g. `missed_dx`, `risk`,
`atypia`, `drug_interaction`). Unknown slots default to `note`.

The module is a pure I/O helper. Caller passes an `llm_client` compatible with
`likelihood_ensemble._one_call` duck-typing (has `complete` or `complete_sync`).

Gated by `meta_controller_enabled` at the call site.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

_log = logging.getLogger("rrrie-cdss")
_CTX_LINE_RE = re.compile(r"^\s*CTX\|([a-z_]+)\|(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
_KNOWN_SLOTS = {
    "missed_dx",
    "risk",
    "atypia",
    "drug_interaction",
    "red_flag",
    "test",
    "note",
    "history",
    "follow_up",
}


def _build_prompt(
    *,
    chief_complaint: str,
    top_species: list[tuple[str, float]],
    top_family: list[tuple[str, float]],
    red_flags: list[str],
    evidence_summary: str = "",
) -> str:
    top_s = ", ".join(f"{h}={p:.2f}" for h, p in top_species[:5]) or "(none)"
    top_f = ", ".join(f"{h}={p:.2f}" for h, p in top_family[:5]) or "(none)"
    rf = ", ".join(red_flags[:8]) or "(none reported)"
    return (
        "You are a senior attending physician doing a final sanity check on a case the system just closed. "
        "Look ONLY for gaps — things a junior clinician would miss but a seasoned attending would catch. "
        "Do NOT re-justify the current top diagnosis. Examples of what to flag:\n"
        "  - an atypical presentation of a dangerous diagnosis that fits the pattern\n"
        "  - a drug-drug interaction or withdrawal syndrome hiding in plain sight\n"
        "  - missing history that would change management (travel, occupation, family)\n"
        "  - a red flag that warrants one more test before disposition\n"
        "Return at most 5 lines, each EXACTLY in the format:\n"
        "  CTX|<slot>|<one-line observation>\n"
        "Allowed slots: missed_dx, risk, atypia, drug_interaction, red_flag, test, history, follow_up, note.\n"
        "If you genuinely see nothing worth flagging, return exactly one line: CTX|note|no_additional_concerns\n"
        "\n"
        f"CHIEF_COMPLAINT: {chief_complaint}\n"
        f"TOP_FAMILY: {top_f}\n"
        f"TOP_SPECIES: {top_s}\n"
        f"RED_FLAGS: {rf}\n"
        f"EVIDENCE_SUMMARY: {evidence_summary[:800]}\n"
        "\nRespond only with CTX lines:\n"
    )


def _parse_ctx_lines(raw: str) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    if not isinstance(raw, str):
        return out
    for m in _CTX_LINE_RE.finditer(raw):
        slot = m.group(1).strip().lower()
        text = m.group(2).strip()
        if not text:
            continue
        if slot not in _KNOWN_SLOTS:
            slot = "note"
        out.append((slot, text[:240]))
        if len(out) >= 5:
            break
    return out


async def self_refine(
    *,
    llm_client: Any,
    chief_complaint: str,
    top_species: list[tuple[str, float]],
    top_family: list[tuple[str, float]],
    red_flags: list[str] | None = None,
    evidence_summary: str = "",
    max_tokens: int = 220,
    temperature: float = 0.2,
) -> list[tuple[str, str]]:
    """Fire one LLM call; return parsed list of (slot, text) pairs.

    On LLM failure returns empty list — caller treats as "no gaps noted".
    """
    prompt = _build_prompt(
        chief_complaint=str(chief_complaint or ""),
        top_species=list(top_species or []),
        top_family=list(top_family or []),
        red_flags=list(red_flags or []),
        evidence_summary=str(evidence_summary or ""),
    )

    async def _call() -> str | None:
        try:
            if hasattr(llm_client, "complete"):
                return await llm_client.complete(prompt, max_tokens=max_tokens, temperature=temperature)
            if hasattr(llm_client, "complete_sync"):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    None, lambda: llm_client.complete_sync(prompt, max_tokens=max_tokens, temperature=temperature)
                )
            if callable(llm_client):
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, lambda: llm_client(prompt))
        except Exception as exc:
            _log.warning("[SELF_REFINE] LLM call failed: %s", exc)
        return None

    raw = await _call()
    if not raw:
        return []
    return _parse_ctx_lines(str(raw))


def atoms_from_ctx_lines(
    ctx_lines: list[tuple[str, str]],
) -> list[str]:
    """Serialize parsed CTX output back to `input_context` atoms.

    Each atom: `self_refine:<slot>:<text>`. Length-bounded.
    """
    out: list[str] = []
    for slot, text in (ctx_lines or [])[:5]:
        token = f"self_refine:{slot}:{text}"
        out.append(token[:280])
    return out


__all__ = [
    "self_refine",
    "atoms_from_ctx_lines",
]
