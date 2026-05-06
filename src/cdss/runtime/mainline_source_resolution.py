from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import requests


DEFAULT_LLAMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


@dataclass
class SourceResolutionResult:
    resolved_anchor: str = ""
    anchor_role: str = "unknown"
    resolved: bool = False
    candidates: list[dict[str, Any]] | None = None
    rationale: str = ""
    required_data: list[str] | None = None
    must_not_miss: list[str] | None = None
    contraindications: list[str] | None = None


def _strip_json(text: str) -> str:
    text = (text or "").strip()
    if "```" in text:
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _safe_list(value: Any) -> list[str]:
    items: list[str] = []
    seen: set[str] = set()
    for item in value or []:
        text = " ".join(str(item or "").strip().replace("_", " ").split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        items.append(text)
    return items


def resolve_source_disease_mainline(
    *,
    patient_text: str,
    phenotype_anchor: str,
    phenotype_candidates: list[dict[str, Any]] | None,
    safety_issue_details: list[str] | None,
    research_evidence: list[dict[str, Any]] | None,
    llama_url: str = DEFAULT_LLAMA_URL,
    model_name: str = DEFAULT_MODEL,
    timeout_s: float = 12.0,
) -> SourceResolutionResult:
    patient_text = (patient_text or "").strip()
    phenotype_anchor = str(phenotype_anchor or "").strip()
    if not patient_text and not phenotype_anchor:
        return SourceResolutionResult()

    candidate_lines = []
    for item in phenotype_candidates or []:
        if not isinstance(item, dict):
            continue
        diagnosis = str(item.get("diagnosis") or "").strip()
        confidence = item.get("confidence")
        if diagnosis:
            candidate_lines.append(f"- {diagnosis} ({confidence})")

    evidence_lines = []
    for item in (research_evidence or [])[:5]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("claim") or item.get("summary") or item.get("query") or "").strip()
        source = str(item.get("source") or item.get("source_class") or "").strip()
        if title:
            evidence_lines.append(f"- {title} [{source}]")

    issue_lines = [f"- {detail}" for detail in _safe_list(safety_issue_details)]

    system_prompt = (
        "You are a clinical source-resolution engine.\n"
        "Your task is to convert a surface manifestation or syndrome anchor into the most likely upstream source-disease space.\n"
        "Do not return the same manifestation label unless there is truly no upstream disease/process candidate.\n"
        "If the case is unresolved, mark resolved=false and still provide the best upstream source-disease candidates.\n"
        "Return JSON only with this schema:\n"
        "{"
        "\"resolved\": false,"
        "\"resolved_anchor\": \"\","
        "\"anchor_role\": \"source_disease|manifestation|unknown\","
        "\"candidates\": [{\"diagnosis\": \"\", \"confidence\": 0.0, \"rationale\": \"\"}],"
        "\"rationale\": \"\","
        "\"required_data\": [\"\"],"
        "\"must_not_miss\": [\"\"],"
        "\"contraindications\": [\"\"]"
        "}"
    )
    user_prompt = (
        f"PATIENT TEXT:\n{patient_text[:2400]}\n\n"
        f"CURRENT PHENOTYPE ANCHOR:\n{phenotype_anchor or 'unknown'}\n\n"
        f"PHENOTYPE CANDIDATES:\n" + ("\n".join(candidate_lines) if candidate_lines else "- none") + "\n\n"
        "SAFETY ISSUES:\n" + ("\n".join(issue_lines) if issue_lines else "- none") + "\n\n"
        "TRUSTED RESEARCH EVIDENCE:\n" + ("\n".join(evidence_lines) if evidence_lines else "- none") + "\n\n"
        "Promote upstream causes when the current anchor is only a manifestation, laboratory state, or syndrome.\n"
        "Use adaptive reasoning from the case details. Do not use placeholders."
    )

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.1,
        "top_p": 0.9,
        "max_tokens": 256,
        "stream": False,
        "response_format": {"type": "json_object"},
    }

    try:
        response = requests.post(llama_url, json=payload, timeout=timeout_s)
        response.raise_for_status()
        content = (
            response.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        parsed = json.loads(_strip_json(content))
    except Exception:
        return SourceResolutionResult()

    candidates = parsed.get("candidates") if isinstance(parsed, dict) else None
    if not isinstance(candidates, list):
        candidates = []
    cleaned_candidates: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        diagnosis = " ".join(str(item.get("diagnosis") or "").strip().replace("_", " ").split())
        if not diagnosis:
            continue
        cleaned_candidates.append(
            {
                "diagnosis": diagnosis,
                "confidence": float(item.get("confidence") or 0.0),
                "rationale": " ".join(str(item.get("rationale") or "").strip().split()),
            }
        )

    return SourceResolutionResult(
        resolved_anchor=" ".join(str(parsed.get("resolved_anchor") or "").strip().replace("_", " ").split()),
        anchor_role=str(parsed.get("anchor_role") or "unknown").strip().lower() or "unknown",
        resolved=bool(parsed.get("resolved")),
        candidates=cleaned_candidates,
        rationale=" ".join(str(parsed.get("rationale") or "").strip().split()),
        required_data=_safe_list(parsed.get("required_data")),
        must_not_miss=_safe_list(parsed.get("must_not_miss")),
        contraindications=_safe_list(parsed.get("contraindications")),
    )
