"""Local LLM bridge for the rebuilt CDSS extractor/ranker/verifier paths."""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from config.settings import get_settings
from src.cdss.clinical.explanation import derive_dangerous_treatment_assumptions, derive_state_frames
from src.cdss.contracts.models import DifferentialSet, EvidenceBundle, FactGraph, HypothesisFrontier, InterventionSet, PatientInput, RiskProfile, StructuredFindings
from src.cdss.runtime.policy import CdssRuntimePolicy, load_runtime_policy
from src.llm.llama_cpp_client import LlamaCppClient

import logging
logger = logging.getLogger(__name__)

try:
    from src.llm.gemini_client import GeminiClient
except Exception:  # pragma: no cover - optional cloud dependency
    GeminiClient = None  # type: ignore[assignment]


def _unique(values: list[str], limit: int) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = str(value or "").strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if len(output) >= limit:
            break
    return output


def _format_evidence_for_prompt(evidence: "EvidenceBundle | None", top_k: int = 3) -> str:
    """Format top-K high-trust evidence items for injection into verify/challenge prompts.

    Injects retrieved evidence so verification and challenge are evidence-grounded
    rather than relying solely on LLM intrinsic knowledge.
    """
    if not evidence or not evidence.items:
        return ""
    try:
        sorted_items = sorted(
            [e for e in (evidence.items or []) if hasattr(e, "trust_score") and e.trust_score is not None],
            key=lambda e: float(e.trust_score or 0.0),
            reverse=True,
        )[:top_k]
    except Exception:
        return ""
    if not sorted_items:
        return ""
    lines = ["\nRETRIEVED EVIDENCE (use to ground verification):"]
    for i, ev in enumerate(sorted_items, 1):
        title = str(getattr(ev, "title", "") or "").strip()
        excerpt = str(getattr(ev, "excerpt", "") or "").strip()
        trust = float(getattr(ev, "trust_score", 0.0) or 0.0)
        if not (title or excerpt):
            continue
        lines.append(f"  [{i}] {title}: {excerpt[:280]} (trust={trust:.2f})")
    return "\n".join(lines) if len(lines) > 1 else ""


def _evidence_digest(evidence: "EvidenceBundle | None", *, limit: int = 6) -> str:
    if not evidence or not evidence.items:
        return ""
    ranked = sorted(
        list(evidence.items),
        key=lambda item: (
            str(item.verification_status or "").strip().lower() == "verified",
            str(item.relation_type or "").strip().lower() == "supports",
            float(item.trust_score or 0.0),
        ),
        reverse=True,
    )
    lines: list[str] = []
    for item in ranked[:limit]:
        title = str(item.title or "").strip()
        if not title:
            continue
        linked = ", ".join(item.linked_hypotheses[:3])
        excerpt = " ".join(str(item.excerpt or "").split())[:180]
        parts = [
            str(item.source or "").strip(),
            title,
            f"linked={linked}" if linked else "",
            excerpt,
        ]
        line = " | ".join(part for part in parts if part)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _retrieval_digest(evidence: EvidenceBundle | None, *, limit: int = 6) -> str:
    if not evidence or not evidence.retrieval_intents:
        return ""
    lines: list[str] = []
    for item in evidence.retrieval_intents[:limit]:
        parts = [
            str(item.target_candidate or "").strip(),
            str(item.decision_target or "").strip(),
            str(item.query_hint or "").strip(),
        ]
        line = " | ".join(part for part in parts if part)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _phenotype_digest(findings: StructuredFindings, *, limit: int = 8) -> str:
    lines: list[str] = []
    for atom in findings.phenotype_atoms[:limit]:
        if not str(atom.label or "").strip():
            continue
        evidence = "; ".join(atom.evidence[:2])
        lines.append(
            " | ".join(
                part
                for part in [
                    str(atom.slot or atom.category or "").strip(),
                    str(atom.normalized_value or atom.label or "").strip(),
                    f"confidence={float(atom.confidence or 0.0):.2f}",
                    evidence,
                ]
                if part
            )
        )
    if not lines and findings.phenotype_fingerprint.embedding_terms:
        return "\n".join(findings.phenotype_fingerprint.embedding_terms[:limit])
    return "\n".join(lines)


def _context_lane_digest(findings: StructuredFindings, *, limit: int = 4) -> str:
    if not findings.context_lanes:
        return ""
    lines: list[str] = []
    for lane_name in ("patient_narrative", "clinician_anchor", "operational_safety", "external_evidence"):
        values = list((findings.context_lanes or {}).get(lane_name, [])[:limit])
        if values:
            lines.append(f"{lane_name}: {', '.join(values)}")
    return "\n".join(lines)


def _coerce_number(value: str) -> float | int | str:
    text = str(value or "").strip()
    if not text:
        return ""
    try:
        numeric = float(text)
    except ValueError:
        return text
    return int(numeric) if numeric.is_integer() else numeric


def _split_compound_context_entries(value: str) -> list[str]:
    normalized = " ".join(str(value or "").split()).strip()
    if not normalized or ":" not in normalized:
        return [normalized] if normalized else []
    prefix_matches = list(re.finditer(r"(^|[\s,;])([a-z][a-z0-9_]{2,32}):", normalized, flags=re.IGNORECASE))
    if len(prefix_matches) <= 1:
        return [normalized]
    chunks: list[str] = []
    starts: list[int] = []
    for match in prefix_matches:
        starts.append(match.start(2))
    starts.append(len(normalized))
    for index in range(len(starts) - 1):
        chunk = normalized[starts[index] : starts[index + 1]].strip(" ,;")
        if chunk:
            chunks.append(chunk)
    return chunks or [normalized]


def _parse_vital_content(payload: dict[str, Any], content: str) -> None:
    if "=" not in content:
        return
    key, raw_number = content.split("=", 1)
    vital_key = key.strip().lower().replace(" ", "_")
    number = raw_number.strip()
    if vital_key == "bp" and "/" in number:
        systolic, diastolic = [part.strip() for part in number.split("/", 1)]
        payload["derived_vitals"]["sbp"] = _coerce_number(systolic)
        payload["derived_vitals"]["dbp"] = _coerce_number(diastolic)
    else:
        payload["derived_vitals"][vital_key] = _coerce_number(number)


def _parse_protocol_lines(raw: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "summary": "",
        "positive_findings": [],
        "negative_findings": [],
        "timeline": [],
        "exposures": [],
        "medications": [],
        "suspected_conditions": [],
        "planned_interventions": [],
        "input_context": [],
        "red_flags": [],
        "derived_vitals": {},
        "constitutional_findings": [],
    }
    _TAG_DISPATCH: dict[str, Any] = {
        "SUMMARY": lambda c: payload.update({"summary": c[:220]}),
        "POS": lambda c: payload["positive_findings"].append(c),
        "NEG": lambda c: payload["negative_findings"].append(c),
        "TIME": lambda c: payload["timeline"].append(c),
        "EXP": lambda c: payload["exposures"].append(c),
        "MED": lambda c: payload["medications"].append(c),
        "DX": lambda c: payload["suspected_conditions"].append(c),
        "TX": lambda c: payload["planned_interventions"].append(c),
        "CTX": lambda c: payload["input_context"].extend(_split_compound_context_entries(c)),
        "FLAG": lambda c: payload["red_flags"].append(c),
        "VITAL": lambda c: _parse_vital_content(payload, c),
        "PHYSIQUE": lambda c: payload["constitutional_findings"].append(c),
    }
    for line in str(raw or "").splitlines():
        if "|" not in line:
            continue
        prefix, value = line.split("|", 1)
        tag = prefix.strip().upper()
        content = value.strip()
        if not content:
            continue
        handler = _TAG_DISPATCH.get(tag)
        if handler:
            handler(content)

    payload["positive_findings"] = _unique(payload["positive_findings"], 12)
    payload["negative_findings"] = _unique(payload["negative_findings"], 8)
    payload["timeline"] = _unique(payload["timeline"], 6)
    payload["exposures"] = _unique(payload["exposures"], 8)
    payload["medications"] = _unique(payload["medications"], 8)
    payload["suspected_conditions"] = _unique(payload["suspected_conditions"], 8)
    payload["planned_interventions"] = _unique(payload["planned_interventions"], 8)
    payload["input_context"] = _unique(payload["input_context"], 14)
    payload["red_flags"] = _unique(payload["red_flags"], 8)
    payload["constitutional_findings"] = _unique(payload["constitutional_findings"], 6)
    return payload


def _extract_metrics(response: Any, *, elapsed: float = 0.0) -> dict[str, float]:
    prompt_tokens = int(getattr(response, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(response, "completion_tokens", 0) or 0)
    total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "time_s": round(float(elapsed or 0.0), 2),
    }


_CRITICAL_LLM_STAGES = {
    "extract",
    "generate_hypotheses",
    "induce_diagnoses",
    "generate_mechanism_frames",
    "generate_retrieval_queries",
    "verify",
    "challenge",
}


def _error_code_for(stage: str, exc: Exception) -> str:
    stage_key = str(stage or "").strip().lower() or "unknown"
    exc_text = str(exc or "").strip().lower()
    if "no structured output" in exc_text or "empty structured output" in exc_text:
        return f"cdss_llm_{stage_key}_empty_output"
    if "cloud_gemini mode selected" in exc_text or "gemini" in exc_text and "not available" in exc_text:
        return f"cdss_llm_{stage_key}_gemini_unavailable"
    if "timeout" in exc_text:
        return f"cdss_llm_{stage_key}_timeout"
    if "rate limit" in exc_text or "429" in exc_text:
        return f"cdss_llm_{stage_key}_rate_limited"
    if isinstance(exc, ValueError):
        return f"cdss_llm_{stage_key}_invalid_input"
    if isinstance(exc, RuntimeError):
        return f"cdss_llm_{stage_key}_runtime_error"
    return f"cdss_llm_{stage_key}_failure"


def _error_payload(stage: str, exc: Exception, *, metrics: dict[str, Any] | None = None) -> dict[str, Any]:
    stage_key = str(stage or "").strip().lower()
    critical_stage = stage_key in _CRITICAL_LLM_STAGES
    return {
        "_meta": {
            "metrics": dict(metrics or {}),
            "error": {
                "stage": stage,
                "type": exc.__class__.__name__,
                "message": str(exc)[:240],
                "code": _error_code_for(stage_key, exc),
                "critical_stage": critical_stage,
            },
        }
    }


def _structured_output_guard(
    *,
    stage: str,
    response: Any,
    started: float,
    hard_fail: bool,
) -> dict[str, Any] | None:
    metrics = _extract_metrics(response, elapsed=time.time() - started)
    content = str(getattr(response, "content", "") or "").strip()
    if content or not hard_fail:
        return None
    return _error_payload(
        stage,
        RuntimeError(f"{stage} returned no structured output."),
        metrics=metrics,
    )


class _BridgeChatResponse:
    def __init__(self, content: str, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.content = content
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _GeminiCompatClient:
    """Sync-compatible adapter so existing stage code can call .chat()."""

    def __init__(self, client: Any, lane: str) -> None:
        self._client = client
        self._lane = lane

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        structured_output: bool = False,
        stage: str = "",
    ) -> _BridgeChatResponse:
        del structured_output
        use_pro = False  # pro model disabled; all cloud stages use flash
        result = asyncio.run(
            self._client.chat_complete(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                use_pro=use_pro,
                lane=self._lane,
            )
        )
        return _BridgeChatResponse(
            content=str(result.get("content", "") or ""),
            prompt_tokens=int(result.get("prompt_tokens", 0) or 0),
            completion_tokens=int(result.get("completion_tokens", 0) or 0),
        )


def _parse_normalization_protocol(raw: str) -> dict[str, Any]:
    normalized_text = ""
    mappings: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        if "|" not in line:
            continue
        tag, rest = line.split("|", 1)
        marker = tag.strip().upper()
        content = rest.strip()
        if not content:
            continue
        if marker == "NORM":
            normalized_text = content
            continue
        if marker != "TERM":
            continue
        parts = content.split("|")
        if len(parts) < 2:
            continue
        literal = str(parts[0]).strip()
        canonical = str(parts[1]).strip()
        confidence = 0.0
        if len(parts) >= 3:
            try:
                confidence = float(str(parts[2]).strip())
            except ValueError:
                confidence = 0.0
        mappings.append(
            {
                "literal": literal,
                "canonical": canonical,
                "confidence": max(0.0, min(1.0, confidence)),
            }
        )
    return {"normalized_text": normalized_text, "literal_mappings": mappings[:10]}


def _parse_label_validation_protocol(raw: str) -> dict[str, Any]:
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        if "LBL|" in cleaned.upper():
            cleaned = cleaned[cleaned.upper().index("LBL|") :]
        if not cleaned.upper().startswith("LBL|"):
            continue
        parts = cleaned.split("|", 5)
        if len(parts) < 6:
            continue
        _, raw_label, canonical_label, raw_is_dx, raw_confidence, reason = parts
        try:
            confidence = float(raw_confidence.strip())
        except ValueError:
            confidence = 0.0
        payload = {
            "raw_label": raw_label.strip(),
            "canonical_label": canonical_label.strip().lower().replace(" ", "_"),
            "confidence": max(0.0, min(1.0, confidence)),
            "reason": reason.strip(),
        }
        is_diagnosis = raw_is_dx.strip().lower() in {"1", "true", "yes", "diagnosis"}
        if is_diagnosis and payload["canonical_label"]:
            accepted.append(payload)
        else:
            rejected.append(payload)
    return {"accepted": accepted[:24], "rejected": rejected[:24]}


# W6.1 J.6 — Inference-time label validation. Drops LLM-hallucinated
# symptom-concat ids and projects near-matches onto the registry. Pure
# Jaccard token overlap with no LLM round-trip; safe to call inline.

def _label_tokens(label: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", str(label or "").lower()) if t and len(t) > 1}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _validate_rank_candidates(
    candidates: list[dict[str, Any]],
    *,
    hierarchy_level: int | None = None,
    candidate_set: list[str] | None = None,
    enabled: bool = False,
    jaccard_min: float = 0.6,
) -> tuple[list[dict[str, Any]], int]:
    """Validate parsed candidate labels against the syndrome registry.

    Returns (kept_candidates, dropped_count). When `enabled=False`, identity
    pass-through. Otherwise: each label is normalized; if registry contains
    a same-id entry at the requested level (or any level if `hierarchy_level`
    is None), keep verbatim. Else attempt token-Jaccard ≥ `jaccard_min` match
    against `candidate_set` (preferred) or `registry.by_level(level)`. Drop on
    no match — these are the symptom-concat hallucinations the plan targets.
    """
    if not enabled or not candidates:
        return list(candidates or []), 0
    try:
        from src.cdss.knowledge.ontology import normalize_candidate_label
        from src.cdss.knowledge.registry import load_syndrome_registry
    except Exception:
        return list(candidates or []), 0
    try:
        reg = load_syndrome_registry()
    except Exception:
        return list(candidates or []), 0

    pool_ids: list[str] = []
    if candidate_set:
        pool_ids = [normalize_candidate_label(str(x)) for x in candidate_set if x]
    elif hierarchy_level is not None:
        try:
            pool_ids = [normalize_candidate_label(str(p.id)) for p in reg.by_level(int(hierarchy_level)) if p.id]
        except Exception:
            pool_ids = []

    pool_tokens = {pid: _label_tokens(pid) for pid in pool_ids if pid}

    kept: list[dict[str, Any]] = []
    dropped = 0
    for cand in candidates:
        raw_label = str(cand.get("label") or "").strip()
        if not raw_label:
            dropped += 1
            continue
        nid = normalize_candidate_label(raw_label)
        if not nid:
            dropped += 1
            continue
        # Direct registry hit at requested level (or any level if unspecified).
        try:
            entry = reg.by_id(nid)
        except Exception:
            entry = None
        accept_direct = False
        if entry is not None:
            if hierarchy_level is None:
                accept_direct = True
            else:
                try:
                    accept_direct = int(getattr(entry, "level", -1)) == int(hierarchy_level)
                except Exception:
                    accept_direct = True  # missing level → don't reject on that alone.
        if accept_direct:
            cand["label"] = nid
            kept.append(cand)
            continue
        # Jaccard match against pool.
        toks = _label_tokens(nid)
        best_id, best_j = "", 0.0
        for pid, ptoks in pool_tokens.items():
            j = _jaccard(toks, ptoks)
            if j > best_j:
                best_j, best_id = j, pid
        if best_id and best_j >= float(jaccard_min):
            cand["label"] = best_id
            cand["validation_match_jaccard"] = round(best_j, 3)
            kept.append(cand)
        else:
            dropped += 1
    return kept, dropped


def _parse_rank_protocol(raw: str) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        if "CAND|" in cleaned.upper():
            cleaned = cleaned[cleaned.upper().index("CAND|") :]
        if not cleaned.upper().startswith("CAND|"):
            continue
        parts = cleaned.split("|", 3)
        if len(parts) < 4:
            continue
        _, label, raw_score, rationale = parts
        try:
            score = float(raw_score.strip())
        except ValueError:
            continue
        candidates.append(
            {
                "label": label.strip(),
                "score": max(0.0, min(0.99, score)),
                "rationale": rationale.strip(),
            }
        )
    if candidates:
        return {"candidates": candidates[:4]}

    fallback_pattern = re.compile(r"^\s*(?:[-*]|\d+[.)])?\s*([A-Za-z][A-Za-z0-9_ /-]{2,80})\s*[:|\-]\s*(0(?:\.\d+)?|0?\.\d+|1(?:\.0+)?)\s*[:|\-]\s*(.+)$")
    for line in str(raw or "").splitlines():
        match = fallback_pattern.match(line.strip())
        if not match:
            continue
        label, raw_score, rationale = match.groups()
        try:
            score = float(raw_score.strip())
        except ValueError:
            continue
        candidates.append(
            {
                "label": label.strip(),
                "score": max(0.0, min(0.99, score)),
                "rationale": rationale.strip(),
            }
        )
    return {"candidates": candidates[:4]}


def _parse_hypothesis_protocol(raw: str) -> dict[str, Any]:
    hypotheses: list[dict[str, Any]] = []

    # 1. Pipeline Format (HYP|...)
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        if "HYP|" in cleaned.upper():
            try:
                cleaned = cleaned[cleaned.upper().index("HYP|") :]
                parts = cleaned.split("|", 6)
                if len(parts) >= 7:
                    _, label, raw_score, raw_mnm, raw_danger, dangerous_as, rationale = parts
                    score = float(raw_score.strip())
                    hypotheses.append(
                        {
                            "label": label.strip(),
                            "score": max(0.0, min(0.99, score)),
                            "must_not_miss": raw_mnm.strip().lower() in {"1", "true", "yes"},
                            "dangerous_if_missed": raw_danger.strip().lower() in {"1", "true", "yes"},
                            "dangerous_if_treated_as": dangerous_as.strip(),
                            "rationale": rationale.strip(),
                        }
                    )
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass

    if len(hypotheses) >= 1:
        return {"hypotheses": hypotheses[:6]}

    # 2. Resilient SOTA Markdown Format
    # Matches: "- acute_subarachnoid_hemorrhage (0.85): Rationale | parent: disease_category"
    # Matches: "1. [acute_mi] 0.9 Rationale"
    resilient_pattern = re.compile(
        r"(?:[-*]|\d+[.)])?\s*(?:\[)?([a-zA-Z][a-zA-Z0-9_ /-]{2,90})(?:\])?\s*"
        r"(?:\(\s*([01](?:\.\d+)?)\s*\)|[:|\-]\s*([01](?:\.\d+)?))\s*[:|\-]?\s*(.+)",
        re.IGNORECASE
    )
    _parent_re = re.compile(r"\|\s*parent\s*:\s*([a-zA-Z][a-zA-Z0-9_ /-]{1,60})", re.IGNORECASE)
    for line in str(raw or "").splitlines():
        match = resilient_pattern.match(line.strip())
        if not match:
            continue
        label, paren_score, inline_score, rationale = match.groups()
        raw_score = paren_score or inline_score or ""
        try:
            score = float(raw_score.strip())
            parent_match = _parent_re.search(rationale or "")
            parent_category = parent_match.group(1).strip().lower().replace(" ", "_") if parent_match else ""
            clean_rationale = _parent_re.sub("", rationale or "").strip(" |")
            hypothesis = {
                "label": label.strip(),
                "score": max(0.0, min(0.99, score)),
                "must_not_miss": "must not miss" in rationale.lower() or "critical" in rationale.lower(),
                "dangerous_if_missed": "dangerous if missed" in rationale.lower(),
                "dangerous_if_treated_as": "",
                "rationale": clean_rationale,
                "parent_category": parent_category,
            }
            if not any(h["label"].lower() == hypothesis["label"].lower() for h in hypotheses):
                hypotheses.append(hypothesis)
        except ValueError:
            continue

    needs_search: list[str] = []
    needs_clarification: list[str] = []
    for line in str(raw or "").splitlines():
        ls = line.strip()
        if ls.upper().startswith("NEEDS_SEARCH:"):
            query = ls[len("NEEDS_SEARCH:"):].strip()
            if query:
                needs_search.append(query)
        elif ls.upper().startswith("NEEDS_CLARIFICATION:"):
            question = ls[len("NEEDS_CLARIFICATION:"):].strip()
            if question:
                needs_clarification.append(question)

    result: dict[str, Any] = {"hypotheses": hypotheses[:6]}
    if needs_search:
        result["needs_search"] = needs_search[:3]
    if needs_clarification:
        result["needs_clarification"] = needs_clarification[:2]
    return result


def _parse_verify_protocol(raw: str) -> dict[str, Any]:
    issues: list[dict[str, str]] = []
    reliability_delta = 0.0
    decision_hint = ""
    for line in str(raw or "").splitlines():
        parts = line.split("|")
        tag = parts[0].strip().upper() if parts else ""
        if tag == "ISSUE" and len(parts) >= 4:
            severity = parts[1].strip().lower()
            issue_type = parts[2].strip().lower()
            detail = "|".join(parts[3:]).strip()
            if detail:
                issues.append({"severity": severity or "major", "issue_type": issue_type or "llm_issue", "detail": detail})
        elif tag == "DELTA" and len(parts) >= 2:
            try:
                reliability_delta = float(parts[1].strip())
            except ValueError:
                pass
        elif tag == "DECISION" and len(parts) >= 2:
            decision_hint = parts[1].strip().lower()
    return {
        "issues": issues[:4],
        "reliability_delta": max(-0.4, min(0.1, reliability_delta)),
        "decision_hint": decision_hint,
    }


def _parse_self_verify_protocol(raw: str, fallback_diagnosis: str) -> dict[str, Any]:
    """Parse VERIFY|signal|detail line from LLM self-check."""
    for line in str(raw or "").splitlines():
        parts = line.split("|")
        if len(parts) >= 3 and parts[0].strip().upper() == "VERIFY":
            signal = parts[1].strip().lower()
            detail = "|".join(parts[2:]).strip()
            if signal in {"confident", "revise", "escalate"}:
                return {"signal": signal, "diagnosis": detail or fallback_diagnosis}
    return {"signal": "confident", "diagnosis": fallback_diagnosis}


def _parse_challenge_protocol(raw: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "issues": [],
        "alt_hypotheses": [],
        "anchor": "",
        "anchor_delta": 0.0,
        "loop_action": "",
        "loop_reason": "",
    }
    for line in str(raw or "").splitlines():
        parts = line.split("|")
        tag = parts[0].strip().upper() if parts else ""
        if tag == "ISSUE" and len(parts) >= 4:
            payload["issues"].append(
                {
                    "severity": parts[1].strip().lower() or "major",
                    "issue_type": parts[2].strip().lower() or "challenge_issue",
                    "detail": "|".join(parts[3:]).strip(),
                }
            )
        elif tag == "ALT" and len(parts) >= 4:
            try:
                score = float(parts[2].strip())
            except ValueError:
                continue
            payload["alt_hypotheses"].append(
                {
                    "label": parts[1].strip(),
                    "score": max(0.0, min(0.99, score)),
                    "rationale": "|".join(parts[3:]).strip(),
                }
            )
        elif tag == "ANCHOR" and len(parts) >= 3:
            payload["anchor"] = parts[1].strip()
            try:
                payload["anchor_delta"] = max(-0.35, min(0.15, float(parts[2].strip())))
            except ValueError:
                payload["anchor_delta"] = -0.15
        elif tag == "LOOP" and len(parts) >= 3:
            payload["loop_action"] = parts[1].strip()
            payload["loop_reason"] = "|".join(parts[2:]).strip()
    payload["issues"] = payload["issues"][:5]
    payload["alt_hypotheses"] = payload["alt_hypotheses"][:4]
    return payload


def _parse_steelman_protocol(raw: str) -> dict[str, Any]:
    """Parse STEELMAN|<label>|<argument_score>|<rationale> rows.

    Returns the highest-scoring single row (the strongest steel-man defense).
    """
    best: dict[str, Any] = {"label": "", "argument_score": 0.0, "rationale": ""}
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        if not cleaned.upper().startswith("STEELMAN|"):
            continue
        parts = cleaned.split("|", 3)
        if len(parts) < 4:
            continue
        try:
            score = float(parts[2].strip())
        except ValueError:
            continue
        score = max(0.0, min(1.0, score))
        if score >= float(best.get("argument_score", 0.0) or 0.0):
            best = {
                "label": parts[1].strip(),
                "argument_score": score,
                "rationale": parts[3].strip(),
            }
    return best


def _parse_pairwise_judge_protocol(raw: str) -> dict[str, Any]:
    """Parse WIN|<winner>|<prob:0..1>|<rationale> rows.

    Returns the most recent valid row.
    """
    payload: dict[str, Any] = {"winner": "", "prob": 0.5, "rationale": ""}
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        if not cleaned.upper().startswith("WIN|"):
            continue
        parts = cleaned.split("|", 3)
        if len(parts) < 3:
            continue
        try:
            prob = float(parts[2].strip())
        except ValueError:
            continue
        prob = max(0.0, min(1.0, prob))
        payload = {
            "winner": parts[1].strip(),
            "prob": prob,
            "rationale": parts[3].strip() if len(parts) >= 4 else "",
        }
    return payload


def _parse_conflict_resolution_protocol(raw: str) -> dict[str, Any]:
    """Parse premise-conflict resolver output.

    Format:
        ARTIFACT|<premise_id>|<rationale>
        DISCRIMINATOR|<test_or_finding>|<expected_resolution>
        DECISION|<keep|escalate>|<reason>
    """
    payload: dict[str, Any] = {
        "artifact_premise": "",
        "artifact_rationale": "",
        "discriminator": "",
        "discriminator_expected": "",
        "decision": "",
        "decision_reason": "",
    }
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        parts = cleaned.split("|")
        tag = parts[0].strip().upper() if parts else ""
        if tag == "ARTIFACT" and len(parts) >= 3:
            payload["artifact_premise"] = parts[1].strip()
            payload["artifact_rationale"] = "|".join(parts[2:]).strip()
        elif tag == "DISCRIMINATOR" and len(parts) >= 3:
            payload["discriminator"] = parts[1].strip()
            payload["discriminator_expected"] = "|".join(parts[2:]).strip()
        elif tag == "DECISION" and len(parts) >= 3:
            payload["decision"] = parts[1].strip().lower()
            payload["decision_reason"] = "|".join(parts[2:]).strip()
    return payload


def _parse_query_protocol(raw: str) -> dict[int, str]:
    queries: dict[int, str] = {}
    for line in str(raw or "").splitlines():
        parts = line.split("|", 2)
        tag = parts[0].strip().upper() if parts else ""
        if tag != "QUERY" or len(parts) < 3:
            continue
        try:
            index = int(parts[1].strip())
        except ValueError:
            continue
        query = str(parts[2] or "").strip()
        if query:
            queries[index] = query
    return queries


def _parse_research_protocol(raw: str) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "active_hypotheses": [],
        "discriminators": [],
        "queries": {},
        "expected_impact": {},
    }
    seen_hypotheses: set[str] = set()
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        parts = cleaned.split("|")
        tag = parts[0].strip().upper() if parts else ""
        if tag == "HYP" and len(parts) >= 2:
            label = str(parts[1] or "").strip()
            key = label.lower()
            if label and key not in seen_hypotheses:
                payload["active_hypotheses"].append(label)
                seen_hypotheses.add(key)
        elif tag == "DISC" and len(parts) >= 3:
            label = str(parts[1] or "").strip()
            discriminator = str(parts[2] or "").strip()
            if discriminator:
                payload["discriminators"].append({"hypothesis": label, "discriminator": discriminator})
        elif tag == "QUERY" and len(parts) >= 3:
            try:
                index = int(str(parts[1] or "").strip())
            except ValueError:
                continue
            query = str(parts[2] or "").strip()
            if query:
                payload["queries"][index] = query
            if len(parts) >= 4:
                impact = str(parts[3] or "").strip()
                if impact:
                    payload["expected_impact"][index] = impact
    payload["active_hypotheses"] = payload["active_hypotheses"][:8]
    payload["discriminators"] = payload["discriminators"][:12]
    return payload


def _parse_mechanism_protocol(raw: str) -> dict[str, Any]:
    frames: list[dict[str, Any]] = []
    for line in str(raw or "").splitlines():
        cleaned = line.strip()
        if "MECH|" in cleaned.upper():
            cleaned = cleaned[cleaned.upper().index("MECH|") :]
        if not cleaned.upper().startswith("MECH|"):
            continue
        parts = cleaned.split("|", 8)
        if len(parts) < 9:
            continue
        _, active_state, organ_system, primary, secondary, criticals, exclusions, hazards, raw_conf = parts
        try:
            confidence = float(raw_conf.strip())
        except ValueError:
            confidence = 0.0
        frames.append(
            {
                "active_state": active_state.strip(),
                "organ_system": organ_system.strip(),
                "primary_mechanism": primary.strip(),
                "secondary_mechanism": secondary.strip(),
                "critical_findings": _unique([item.strip() for item in criticals.split(";")], 5),
                "exclusions": _unique([item.strip() for item in exclusions.split(";")], 4),
                "hazard_context": _unique([item.strip() for item in hazards.split(";")], 4),
                "confidence": max(0.0, min(0.99, confidence)),
                "provenance": "llm",
            }
        )
    return {"mechanism_frames": frames[:3]}


class LocalLlmClinicalExtractor:
    """Best-effort structured extraction over the local llama-server."""

    def __init__(self, policy: CdssRuntimePolicy | None = None) -> None:
        self.policy = policy or load_runtime_policy()
        self._runtime_mode = self.policy.runtime_mode
        self._forced_engine_mode: str | None = None
        self._gemini_client: Any = None
        self._gemini_enabled = False
        if self.policy.gemini_enabled_in_cloud_mode and GeminiClient is not None:
            settings = get_settings()
            self._gemini_client = GeminiClient(
                api_key=settings.google_api_key,
                stable_flash_model=settings.gemini_stable_flash_model,
                stable_pro_model=settings.gemini_stable_pro_model,
                experimental_flash_model=settings.gemini_experimental_flash_model,
                experimental_pro_model=settings.gemini_experimental_pro_model,
                default_lane=settings.gemini_lane,
            )
            self._gemini_enabled = self._gemini_client.is_available

    @property
    def cloud_available(self) -> bool:
        return bool(self._gemini_enabled and self._gemini_client is not None)

    def lock_engine_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized in {"local_qwen", "cloud_gemini"}:
            self._forced_engine_mode = normalized

    def unlock_engine_mode(self) -> None:
        self._forced_engine_mode = None

    @property
    def locked_engine_mode(self) -> str:
        return str(self._forced_engine_mode or "").strip().lower()

    def _resolve_active_mode(self) -> str:
        if self._forced_engine_mode == "cloud_gemini":
            return "cloud"
        if self._forced_engine_mode == "local_qwen":
            return "local"
        if self._runtime_mode == "cloud" and self._gemini_enabled:
            return "cloud"
        return "local"

    @property
    def active_mode(self) -> str:
        return self._resolve_active_mode()

    @property
    def active_engine_model(self) -> str:
        if self.active_mode == "cloud":
            if self._gemini_client:
                return self._gemini_client.flash_model
            settings = get_settings()
            return getattr(settings, "gemini_pro_model", "") or getattr(settings, "gemini_stable_pro_model", "")
        settings = get_settings()
        return getattr(settings, "hf_model_name", "")

    def _local_disabled(self, flag: bool) -> bool:
        """True when running in local mode and the capability flag is off."""
        return self.active_mode == "local" and not flag

    def _client(self) -> Any:
        if self.active_mode == "cloud":
            if not self.cloud_available:
                raise RuntimeError("cloud_gemini mode selected but Gemini is not available.")
            settings = get_settings()
            return _GeminiCompatClient(self._gemini_client, lane=settings.gemini_lane)
        settings = get_settings()
        return LlamaCppClient.get_instance(
            model_name=settings.hf_model_name,
            max_ctx=2048,
        )

    async def extract(self, patient_input: PatientInput) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_extraction_enabled):
            return {}
        patient_text = str(patient_input.patient_text or "").strip()
        if not patient_text:
            return {}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._extract_sync, patient_input), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("extract", exc)

    async def normalize_semantic_text(self, text: str, language: str = "") -> dict[str, Any]:
        source = str(text or "").strip()
        if not source:
            return {"normalized_text": "", "literal_mappings": []}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._normalize_semantic_text_sync, source, language), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("normalize_semantic_text", exc)

    async def validate_diagnostic_labels(
        self,
        labels: list[str],
        *,
        context: str = "",
    ) -> dict[str, Any]:
        candidates = [str(item or "").strip() for item in labels if str(item or "").strip()]
        if not candidates:
            return {"accepted": [], "rejected": []}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._validate_diagnostic_labels_sync, candidates, context), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("validate_diagnostic_labels", exc)

    def _normalize_semantic_text_sync(self, text: str, language: str = "") -> dict[str, Any]:
        client = self._client()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a multilingual clinical semantic normalizer. "
                    "Return ONLY protocol lines. "
                    "Line 1 must be: NORM|<canonical clinical narrative in concise English>. "
                    "Optional lines: TERM|<literal phrase>|<canonical medical meaning>|<confidence 0-1>. "
                    "Preserve urgency, chronology, and clinician-order semantics. "
                    "Translate colloquial distress or idiomatic language into severity/intensity context, not literal organ rupture or death, unless the text explicitly states objective evidence. "
                    "Preserve the full patient story: background disease, triggering event, symptom evolution, clinician anchor, and staged plan. "
                    "Do not invent diagnoses or treatments."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"LANGUAGE_HINT: {language or 'unknown'}\n"
                    f"TEXT:\n{text}\n"
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=self.policy.local_llm_max_tokens,
            structured_output=True,
            stage="R0",
        )
        guard = _structured_output_guard(
            stage="normalize_semantic_text",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_normalization_protocol(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    def _validate_diagnostic_labels_sync(self, labels: list[str], context: str = "") -> dict[str, Any]:
        client = self._client()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical diagnosis label validator. "
                    "For each input label, classify whether it is a diagnosis/hypothesis (not a tool/object/action/noise). "
                    "Return ONLY rows as: LBL|raw_label|canonical_snake_case_label|is_diagnosis(1/0)|confidence_0_to_1|reason. "
                    "If label is noise, object, intervention, workflow label, time-window label, context tag, or malformed phrase, set is_diagnosis=0. "
                    "Rule-out targets, staged prerequisites, and future complications are not diagnoses unless they are explicitly present in the current case state. "
                    "Do not invent labels not grounded in the input; only normalize formatting and clinical wording."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"CONTEXT: {context[:300]}\n"
                    "LABELS:\n"
                    + "\n".join(f"- {item}" for item in labels[:24])
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=self.policy.local_llm_max_tokens,
            structured_output=True,
            stage="R1",
        )
        guard = _structured_output_guard(
            stage="validate_diagnostic_labels",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_label_validation_protocol(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    def _extract_sync(self, patient_input: PatientInput) -> dict[str, Any]:
        client = self._client()
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical structure extractor. "
                    "Return ONLY line-protocol rows in this exact format: "
                    "SUMMARY|..., POS|..., NEG|..., TIME|..., EXP|..., MED|..., DX|..., TX|..., CTX|..., FLAG|..., VITAL|name=value, PHYSIQUE|<constitutional finding>. "
                    "PHYSIQUE rows: Use when the patient's body habitus, height, weight, or physical build is clinically significant to their presentation. Use standard medical terminology. "
                    "Do not output PHYSIQUE for unremarkable or clinically irrelevant physique. "
                    "Use concise canonical English labels. "
                    "Normalize colloquial, multilingual, or folk-clinical terms into the nearest standard medical concept instead of literal translation. "
                    "CRITICAL STRATEGIC RULE: If the input text is in a non-English language (e.g., Turkish), you MUST mentally translate EVERY symptom, history, and observation to standard English medical terminology (e.g., SNOMED-CT equivalents) BEFORE extracting them into the line protocol. Never output non-English words. "
                    "Interpret the case in story order before extraction: baseline conditions, triggering event/exposure, symptom progression, patient concern, clinician anchor, and staged management plan. "
                    "Do not flatten the narrative into isolated symptoms if the story implies causality, chronology, witness context, or competing explanations. "
                    "Capture diagnoses mentioned by the patient OR clinician, planned treatments/tests, and vitals embedded in text. "
                    "DX rows must contain disease-level diagnoses only. Never place workflows, time-window labels, device caveats, or action items into DX. "
                    "Do not place conditional rule-out targets, future treatment complications, or post-treatment hazards into DX unless the current case directly supports them as present disease states. "
                    "Use CTX rows for typed contextual interpretation only, in `prefix:value` form. Preferred prefixes are working_diagnosis, pathway_fit, time_window, hazard, workflow, blocked_order, required_action, data_request, device_reliability, causal_loop, masquerade_risk, contradiction, story_frame, baseline_context, event_context, clinician_anchor, patient_concern, alternative_explanation. "
                    "CRITICAL DE-BIASING RULE: Strictly separate Objective_Symptoms (what the patient is physically experiencing, e.g., vomiting, jaw ache, fatigue) from Patient_Interpretations (what the patient believes they have, e.g., 'stomach bug', 'caught a flu'). Do NOT let the patient's self-diagnosis or reasoning (e.g., 'jaw hurts from clenching') mask the objective symptom in your DX or CTX rows. If a patient vehemently denies a symptom (e.g., 'absolutely no chest pain'), document it as NEG, but do NOT ignore other objective factors (nausea, age, diabetes) that may point to atypical presentations of serious illnesses. "
                    "PATIENT SELF-LABEL RULE: Any disease name, causal explanation, or mechanistic guess from the patient is a Patient_Interpretation, NOT a clinical finding. This includes: explicit labels ('I have migraines'), causal guesses ('I think I popped a disc'), mechanism assumptions ('the hernia is pressing on the nerve'), minimizing frames ('it's just muscle strain'). For ALL such interpretations: "
                    "(1) Record ONLY as CTX|patient_concern|[patient interpretation text]. Do NOT create DX rows from patient guesses. "
                    "(2) Extract EACH underlying objective symptom as a separate POS row using canonical medical terms — the physical experience only (e.g., 'interscapular_pain', 'left_lower_extremity_paresthesia'), NOT the patient's label. "
                    "(3) DX rows MUST derive from the objective symptom pattern ALONE — as if you had never read the patient's guess. The patient's self-label is an anchoring trap that kills differential breadth. Ignore it for DX purposes. "
                    "TEST: Before writing any DX row, ask: 'Would I write this if the patient had said nothing about what they think it is?' If no, do not write it. "
                    "SYMPTOM COMPLETENESS RULE: Extract ALL symptoms the patient mentions, including seemingly minor or unrelated ones (e.g., 'can't sleep', 'body shakes'). Do not discard symptoms that don't fit the dominant pattern — they are often the key discriminator. Flag any multi-system symptom cluster (neurological + GI, or sleep + movement + headache) explicitly with CTX|symptom_cluster|[description] so downstream reasoning can find the unifying cause. "
                    "CTX rows may contain disease names only under working_diagnosis. All other CTX rows must be operational or safety context, not diagnoses. "
                    "Prefer a small number of sharp CTX rows over generic commentary. "
                    "Do not explain anything."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Extract structured clinical facts and context from this case.\n\n"
                    f"LANGUAGE_HINT: {patient_input.language or 'unknown'}\n"
                    f"DEMOGRAPHICS: {patient_input.demographics}\n"
                    f"VITALS: {patient_input.vitals}\n"
                    f"LABS: {patient_input.labs}\n"
                    f"MEDICATIONS: {patient_input.medications}\n"
                    f"CASE:\n{patient_input.patient_text}\n"
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=self.policy.local_llm_max_tokens,
            structured_output=True,
            stage="R1",
        )
        guard = _structured_output_guard(
            stage="extract",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_protocol_lines(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def rank(self, findings: StructuredFindings, risk_profile: RiskProfile, allowed_labels: list[str]) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_ranker_enabled):
            return {}
        if not allowed_labels:
            return {}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._rank_sync, findings, risk_profile, allowed_labels), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("rank", exc)

    def _rank_sync(self, findings: StructuredFindings, risk_profile: RiskProfile, allowed_labels: list[str]) -> dict[str, Any]:
        client = self._client()
        messages = [
            {
                "role": "system",
                "content": (
                    "Choose the best syndrome profiles from the allowed labels only. "
                    "Return ONLY lines formatted as CAND|profile_id|score|short rationale. "
                    "Scores must be between 0.00 and 0.99. Return at most 3 lines."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"ALLOWED_LABELS: {', '.join(allowed_labels)}\n"
                    f"SUMMARY: {findings.summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings)}\n"
                    f"CONDITIONS: {', '.join(findings.suspected_conditions)}\n"
                    f"PLANNED_INTERVENTIONS: {', '.join(findings.planned_interventions)}\n"
                    f"RED_FLAGS: {', '.join(findings.red_flags)}\n"
                    f"VITALS: {', '.join(f'{k}={v}' for k, v in (findings.derived_vitals or {}).items())}\n"
                    f"URGENCY: {risk_profile.urgency.value}\n"
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=int(self.policy.local_llm_max_tokens * 0.8),
            structured_output=True,
            stage="R3",
        )
        guard = _structured_output_guard(
            stage="rank",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_rank_protocol(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def generate_hypotheses(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        interventions: InterventionSet,
        helper_labels: list[str],
        evidence: EvidenceBundle | None = None,
    ) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_ranker_enabled):
            return {}
        try:
            result = await asyncio.wait_for(
                asyncio.to_thread(
                    self._generate_hypotheses_sync,
                    findings, risk_profile, fact_graph, interventions, helper_labels, evidence,
                ),
                timeout=self.policy.llm_sync_call_timeout_s,
            )

            # Reactive search: if LLM requested knowledge, fetch and re-run (max 2 iterations)
            max_search_iterations = getattr(self.policy, "reactive_search_max_iterations", 2)
            search_iteration = 0
            while result.get("needs_search") and search_iteration < max_search_iterations:
                queries = result["needs_search"][:3]
                search_iteration += 1
                try:
                    from src.cdss.tools.clinical_search import ClinicalSearchTool
                    searcher = ClinicalSearchTool()
                    snippets: list[str] = []
                    for query in queries:
                        found = await searcher.search(query, max_results=3)
                        snippets.extend(found)
                    extra_context = "\n".join(snippets[:9])
                    result = await asyncio.wait_for(
                        asyncio.to_thread(
                            self._generate_hypotheses_sync,
                            findings, risk_profile, fact_graph, interventions, helper_labels, evidence,
                            0.0, extra_context,
                        ),
                        timeout=self.policy.llm_sync_call_timeout_s,
                    )
                except Exception:
                    break
            # Self-consistency lite: when hypothesis scores are tightly clustered (low spread),
            # the model is uncertain. Run a second pass with diversity temperature and merge.
            hyps = result.get("hypotheses") or []
            if len(hyps) >= 2:
                scores = [float(h.get("score", 0.5)) for h in hyps]
                spread = max(scores) - min(scores)
                if spread < 0.20:
                    try:
                        result2 = await asyncio.wait_for(
                            asyncio.to_thread(
                                self._generate_hypotheses_sync,
                                findings, risk_profile, fact_graph, interventions, helper_labels, evidence,
                                0.12,  # _extra_temperature
                            ),
                            timeout=self.policy.llm_sync_call_timeout_s,
                        )
                        hyps2 = result2.get("hypotheses") or []
                        # Merge: for each unique label keep the MAX score across runs.
                        merged: dict[str, dict] = {}
                        for h in hyps + hyps2:
                            lbl = str(h.get("label", "")).strip()
                            if lbl and (lbl not in merged or h.get("score", 0) > merged[lbl].get("score", 0)):
                                merged[lbl] = h
                        result["hypotheses"] = sorted(merged.values(), key=lambda h: h.get("score", 0), reverse=True)[:6]
                    except Exception:  # noqa: BLE001
                        logger.debug("swallowed exception", exc_info=True)
                        pass
            return result
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("generate_hypotheses", exc)

    def _generate_hypotheses_sync(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        interventions: InterventionSet,
        helper_labels: list[str],
        evidence: EvidenceBundle | None = None,
        _extra_temperature: float = 0.0,
        _extra_context: str = "",
    ) -> dict[str, Any]:
        client = self._client()
        authority_claims = [node.label for node in fact_graph.nodes if node.category == "authority_claim"]
        semantic_patterns = [
            node.label
            for node in fact_graph.nodes
            if node.category in {"semantic_pattern", "pattern_marker", "laterality_marker", "course_marker"}
        ]
        state_frames = derive_state_frames(findings, fact_graph, interventions)
        dangerous_assumptions = derive_dangerous_treatment_assumptions(findings, fact_graph, interventions, state_frames)
        evidence_digest = _evidence_digest(evidence)
        retrieval_digest = _retrieval_digest(evidence)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a board-certified attending physician with extensive clinical experience across internal medicine, "
                    "infectious disease, dermatology, and emergency medicine.\n\n"
                    "CLINICAL CONSULTATION PROCESS — work through all steps before outputting diagnoses:\n\n"
                    "STEP 1 — READ THE CASE\n"
                    "  Read all findings carefully. Note what is present, what is absent, and what the timeline shows.\n"
                    "  Identify the primary organ system(s) involved and the pathological process category "
                    "(infectious, autoimmune, metabolic, structural, iatrogenic, psychiatric).\n\n"
                    "STEP 2 — UNIFIED EXPLANATION\n"
                    "  Ask: which single diagnosis explains ALL documented positive findings together, including "
                    "seemingly minor symptoms? A diagnosis that accounts for the full picture ranks higher than one "
                    "explaining only the dominant symptom.\n\n"
                    "STEP 3 — DISCRIMINATING FEATURES\n"
                    "  Which findings most sharply narrow the differential? What do positive findings rule in? "
                    "What does the absence of expected symptoms rule out?\n\n"
                    "STEP 4 — BASE RATE AND CONTEXT\n"
                    "  Weight prior probability: what is most common for this demographic, setting, and exposure history? "
                    "When EXPOSURES contain geographic signals, treat region-endemic diseases as first-class candidates. "
                    "When the patient states an existing condition ('I have asthma', 'I have diabetes'), integrate it — "
                    "symptoms consistent with that condition or its exacerbation rank above unrelated diagnoses.\n\n"
                    "STEP 5 — MUST-NOT-MISS CHECK\n"
                    "  What dangerous mimic could present identically? If you are wrong about the lead diagnosis, "
                    "what is the most likely alternative, and what finding would change your conclusion?\n\n"
                    "STEP 6 — APPLY CLINICAL CONSTRAINTS (enforce before scoring)\n"
                    "  TRIGGER CONSTRAINT: Drug reactions (drug_reaction, drug_induced_*, serum_sickness, "
                    "medication_hypersensitivity) require a documented causative agent. When MEDICATIONS is empty "
                    "AND EXPOSURES contains no drug/substance/chemical references, assign these diagnoses confidence "
                    "below 0.15 — do NOT infer a causative agent. When a rash is present without medication history, "
                    "prioritize infectious exanthems (chickenpox, measles, viral_exanthem), contact_dermatitis, or "
                    "primary dermatological diagnoses over drug reactions.\n"
                    "  BLISTERING CONSTRAINT: Autoimmune blistering conditions (bullous_pemphigoid, pemphigus_vulgaris, "
                    "autoimmune_bullous_disease) require explicitly documented vesicles, bullae, or Nikolsky sign. "
                    "Rash or skin peeling alone does not justify this diagnosis — assign confidence below 0.15.\n"
                    "  ABSTRACTION CONSTRAINT: Output diagnoses at the specificity level the evidence supports. "
                    "When fewer than 4 distinct symptom descriptors are present or symptoms are generic (cough + rhinorrhea, "
                    "rash + fever, painful urination), use root-level disease names (common_cold, chickenpox, "
                    "urinary_tract_infection, allergy) rather than subspecialty ICD labels. Evidence must actively "
                    "discriminate at subtype level before outputting a subtype — default to the root disease.\n"
                    "  PATHOGNOMONIC PATTERNS: When the presentation matches a recognized clinical signature, anchor "
                    "that diagnosis regardless of base rates: cyclic fever+chills+drenching sweats+headache = malaria; "
                    "severe joint pain the patient or clinician describes as 'breakbone' = dengue; honey-colored crusted "
                    "facial lesions = impetigo; painful urination+hematuria+urgency = urinary_tract_infection.\n\n"
                    "STEP 7 — OUTPUT DIAGNOSES\n"
                    "  Return a Markdown list. Format each line EXACTLY as:\n"
                    "  `- snake_case_diagnosis_name (confidence_0.0_to_1.0): rationale | parent: root_category`\n"
                    "  confidence = how strongly the available evidence supports this diagnosis (be calibrated and honest).\n"
                    "  parent = the root-level disease category this diagnosis belongs to (e.g., urinary_tract_infection, "
                    "allergy, infectious_disease, skin_infection). Omit parent only if diagnosis IS already the root.\n"
                    "  Never output workflow tags, CTX-style operational phrases, or future risks as diagnoses.\n"
                    "  If information is insufficient for disease-level closure, output the best diagnosis you can justify "
                    "and assign it an honestly low confidence. Do not emit an empty list.\n"
                    "  Translate non-English inputs directly into canonical English diagnoses.\n"
                    "  If psychiatric symptoms co-occur with fever, seizure, autonomic instability, or focal neurologic "
                    "findings, prioritize organic (neurologic, infectious, metabolic, toxic) etiologies first.\n"
                ),
            },
              {
                "role": "user",
                "content": (
                    f"SUMMARY: {findings.summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings)}\n"
                    f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
                    f"TIMELINE: {', '.join(findings.timeline)}\n"
                    f"EXPOSURES: {', '.join(findings.exposures)}\n"
                    f"CONDITIONS: {', '.join(findings.suspected_conditions)}\n"
                    f"PLANNED_INTERVENTIONS: {', '.join(findings.planned_interventions)}\n"
                    f"INPUT_CONTEXT: {', '.join(findings.input_context)}\n"
                    f"CONTEXT_LANES:\n{_context_lane_digest(findings) or 'none'}\n"
                    f"PHENOTYPE_ATOMS:\n{_phenotype_digest(findings) or 'none'}\n"
                    f"RED_FLAGS: {', '.join(findings.red_flags)}\n"
                    f"VITALS: {', '.join(f'{k}={v}' for k, v in (findings.derived_vitals or {}).items())}\n"
                    f"FACTS: {', '.join(node.label for node in fact_graph.nodes[:24])}\n"
                    f"AUTHORITY_CLAIMS: {', '.join(authority_claims)}\n"
                      f"PATTERNS: {', '.join(semantic_patterns)}\n"
                      f"STATE_FRAMES: {', '.join(state_frames)}\n"
                      f"DANGEROUS_TREATMENT_ASSUMPTIONS: {', '.join(dangerous_assumptions)}\n"
                      f"PRIOR_EVIDENCE:\n{evidence_digest or 'none'}\n"
                      f"PRIOR_RESEARCH_QUERIES:\n{retrieval_digest or 'none'}\n"
                      f"INTERVENTIONS: {', '.join(item.label for item in interventions.items)}\n"
                      f"URGENCY: {risk_profile.urgency.value}\n"
                      + (f"\nRETRIEVED_KNOWLEDGE:\n{_extra_context}\n" if _extra_context else "")
                  ),
              },
          ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=self.policy.local_llm_temperature + _extra_temperature,
            max_tokens=int(self.policy.local_llm_max_tokens * 1.2),
            structured_output=True,
            stage="R3",
        )
        guard = _structured_output_guard(
            stage="generate_hypotheses",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_hypothesis_protocol(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def induce_diagnoses(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        evidence: EvidenceBundle | None = None,
        focus_profile: str = "general",
        deep_thinking: bool = False,
        *,
        hierarchy_level: int | None = None,
        candidate_set: list[str] | None = None,
        shared_belief: str = "",
        override_temperature: float | None = None,
    ) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_ranker_enabled):
            return {}
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(
                    self._induce_diagnoses_sync,
                    findings,
                    risk_profile,
                    fact_graph,
                    evidence,
                    focus_profile,
                    deep_thinking,
                    hierarchy_level,
                    candidate_set,
                    shared_belief,
                    override_temperature,
                ),
                timeout=self.policy.llm_sync_call_timeout_s,
            )
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("induce_diagnoses", exc)

    async def induce_diagnoses_ensemble(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        evidence: EvidenceBundle | None = None,
        focus_profile: str = "general",
        deep_thinking: bool = False,
        *,
        hierarchy_level: int | None = None,
        candidate_set: list[str] | None = None,
        shared_belief: str = "",
        temperatures: list[float] | None = None,
    ) -> dict[str, Any]:
        """W6.2 J.2 — temperature-ensemble likelihood per worker.

        Fires `induce_diagnoses` n times in parallel (one per temperature),
        merges per-candidate scores into mean ± variance via method-of-moments.
        Returns a payload with the same shape as a single call (`candidates`,
        `_meta`) plus a per-candidate stat block under `_ensemble`:

            {"candidates": [...mean_score...],
             "_ensemble": {label: {"mean":μ, "var":σ², "alpha":α, "beta":β,
                                    "samples":[...n], "temps":[...n]}},
             "_meta": {...}}

        Cache hits (response_cache keys on prompt+temp) make repeated runs cheap.
        """
        temps = list(temperatures or [])
        if not temps:
            raw = str(getattr(self.policy, "swarm_ensemble_temps", "0.0,0.25,0.5"))
            for t in raw.split(","):
                t = t.strip()
                if not t:
                    continue
                try:
                    temps.append(max(0.0, min(2.0, float(t))))
                except ValueError:
                    continue
        if not temps:
            temps = [0.0, 0.25, 0.5]

        async def _one(t: float) -> dict[str, Any]:
            return await self.induce_diagnoses(
                findings, risk_profile, fact_graph,
                evidence=evidence,
                focus_profile=focus_profile,
                deep_thinking=deep_thinking,
                hierarchy_level=hierarchy_level,
                candidate_set=candidate_set,
                shared_belief=shared_belief,
                override_temperature=float(t),
            )

        results = await asyncio.gather(*(_one(t) for t in temps), return_exceptions=False)

        # Aggregate per-candidate samples across temperatures.
        per_label_samples: dict[str, list[float]] = {}
        per_label_temps: dict[str, list[float]] = {}
        per_label_rationale: dict[str, str] = {}
        for t, payload in zip(temps, results):
            cands = (payload or {}).get("candidates") or []
            for c in cands:
                lab = str(c.get("label") or "").strip()
                if not lab:
                    continue
                try:
                    s = float(c.get("score") or 0.0)
                except Exception:  # noqa: BLE001
                    logger.debug("swallowed exception", exc_info=True)
                    continue
                per_label_samples.setdefault(lab, []).append(max(0.0, min(0.99, s)))
                per_label_temps.setdefault(lab, []).append(float(t))
                if lab not in per_label_rationale and c.get("rationale"):
                    per_label_rationale[lab] = str(c.get("rationale"))

        # Build merged candidate list (sorted by mean score desc) + ensemble stats.
        merged: list[dict[str, Any]] = []
        ensemble_stats: dict[str, dict[str, Any]] = {}
        for lab, samples in per_label_samples.items():
            n = len(samples)
            mean = sum(samples) / n if n else 0.0
            if n >= 2:
                var = sum((x - mean) ** 2 for x in samples) / (n - 1)
            else:
                var = 0.0
            mu = max(1e-4, min(1.0 - 1e-4, mean))
            max_var = mu * (1.0 - mu)
            v = max(1e-6, min(max_var - 1e-6, var))
            if v <= 1e-6 or max_var <= 1e-6:
                alpha, beta = mu * 3.0 + 1.0, (1.0 - mu) * 3.0 + 1.0
            else:
                kappa = max(1e-3, max_var / v - 1.0)
                alpha, beta = max(1e-3, mu * kappa), max(1e-3, (1.0 - mu) * kappa)
            merged.append({
                "label": lab,
                "score": round(mean, 4),
                "rationale": per_label_rationale.get(lab, ""),
            })
            ensemble_stats[lab] = {
                "mean": round(mean, 4),
                "var": round(var, 6),
                "alpha": round(alpha, 4),
                "beta": round(beta, 4),
                "samples": [round(s, 4) for s in samples],
                "temps": list(per_label_temps[lab]),
            }
        merged.sort(key=lambda c: float(c.get("score") or 0.0), reverse=True)
        merged = merged[:5]

        # Surface metadata from the first successful sub-call.
        meta = {}
        for r in results:
            if r and r.get("_meta"):
                meta = r.get("_meta") or {}
                break

        return {
            "candidates": merged,
            "_ensemble": ensemble_stats,
            "_ensemble_temps": list(temps),
            "_meta": meta,
        }

    def _induce_diagnoses_sync(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        evidence: EvidenceBundle | None = None,
        focus_profile: str = "general",
        deep_thinking: bool = False,
        hierarchy_level: int | None = None,
        candidate_set: list[str] | None = None,
        shared_belief: str = "",
        override_temperature: float | None = None,
    ) -> dict[str, Any]:
        client = self._client()
        authority_claims = [node.label for node in fact_graph.nodes if node.category == "authority_claim"]
        semantic_patterns = [
            node.label
            for node in fact_graph.nodes
            if node.category in {"semantic_pattern", "pattern_marker", "laterality_marker", "course_marker", "risk_marker"}
        ]
        evidence_digest = _evidence_digest(evidence)
        retrieval_digest = _retrieval_digest(evidence)
        _FOCUS_PROFILE_PROMPTS: dict[str, str] = {
            "red_team_fatal": (
                "RED TEAM FATAL: Adversarial differential generator. Your sole task is to enumerate "
                "diagnoses whose miss is plausibly fatal or catastrophically morbid within 72 hours. "
                "PRODUCE AT LEAST 3 candidates. Strip the patient's own framing (self-diagnosis, "
                "their best-guess label, their preferred reassurance) before reasoning — treat those "
                "phrases as adversarial noise, not signal. Examples of the must-not-miss class: "
                "carotid/vertebral dissection masquerading as muscle strain; HSV encephalitis "
                "masquerading as substance intoxication; infective endocarditis masquerading as "
                "dermatologic or allergic rash; alcoholic / diabetic ketoacidosis masquerading as "
                "gastritis; bone-tumor / leukemia masquerading as growth pain or post-traumatic "
                "ache; massive PE masquerading as anxiety. Even when the classic symptom is denied, "
                "promote the diagnosis if its physiologic equivalent is present (e.g., Kussmaul "
                "respiration ≈ 'fast breathing'; Osler nodes ≈ 'painful purple finger bumps'). "
            ),
            "zebra_hunter": (
                "ZEBRA HUNTER: Focus on rare metabolic, endocrine, autoimmune, or toxicological causes that fit the exact pattern. "
                "Look past common illnesses to uncover hidden systemic diseases like DKA, Adrenal Insufficiency, or rare infectious processes. "
            ),
            "common_case": (
                "COMMON FIRST (Occam's Razor): You are a seasoned primary care physician. "
                "The most common diagnosis that explains the most symptoms is usually correct. "
                "Common diseases are common — rare diagnoses require rare findings. "
                "Reason like this: (1) What is the typical presentation for a patient this age/sex with these chief complaints? "
                "(2) What single diagnosis explains all or most findings with the fewest assumptions? "
                "(3) Only suggest a rare or catastrophic diagnosis if common explanations clearly and demonstrably fail. "
                "Explicitly apply base-rate epidemiology: a young adult with cough, rhinorrhea, and myalgia almost certainly has a viral URI. "
                "Do not be swayed by isolated non-specific symptoms (mild chest discomfort during a febrile illness is not ACS). "
            ),
            "sherlock": (
                "SHERLOCK (Deductive Unification — Dr. House / Holmes style): "
                "Your mission: find the single diagnosis that leaves NO objective symptom unexplained. "
                "STEP 1 — List ALL objective symptoms from the case. Ignore anything the patient says they 'have' or 'think' they have. "
                "STEP 2 — For each candidate, mentally verify: does this diagnosis account for EVERY symptom on the list? "
                "A diagnosis that explains 5/5 symptoms beats one that explains only the most prominent symptom. "
                "STEP 3 — Pay special attention to multi-system symptom clusters: "
                "  - Headache + tremor + insomnia + lightheadedness together → suspect drug withdrawal, toxin, or metabolic cause "
                "  - Dysphagia + bloating + belching + bad taste → suspect acid/peptic pathology (not structural narrowing unless progressive) "
                "  - GI burning + postprandial worsening + nocturnal pain → discriminate PUD vs GERD by timing and acid-empty pattern "
                "STEP 4 — Actively consider iatrogenic/drug/withdrawal etiologies for any multi-system presentation, EVEN if no medications are mentioned. "
                "STEP 5 — Output the diagnosis that unifies the full symptom picture. If the patient named a disease, treat that name as a red herring — re-derive the diagnosis from symptoms only. "
            ),
        }
        _DEFAULT_FOCUS_PROMPT = "GENERAL: Infer disease-level diagnoses from the full narrative. Do not anchor solely on the patient's self-diagnosis. "
        base_prompt = "You are a diagnosis induction fallback for a clinical differential engine. "
        if focus_profile.startswith("specialist:"):
            domain = focus_profile.split(":", 1)[1].upper()
            base_prompt += f"DOMAİN EXPERT ({domain}): You are a world-class {domain} specialist. Focus strictly on identifying causes, interactions, and pathophysiology relevant to {domain}. Connect findings and symptoms to {domain} diseases, toxidromes, or interactions. Look specifically for {domain} clues often ignored by generalists. "
        else:
            base_prompt += _FOCUS_PROFILE_PROMPTS.get(focus_profile, _DEFAULT_FOCUS_PROMPT)

        base_prompt += (
            "Infer disease-level diagnoses from the objective findings, timeline, exposures, and physiologic pattern. "
            "Do not output workflow states, treatment plans, safety tags, or future complications as diagnoses. "
            "Return ONLY lines formatted exactly as CAND|snake_case_diagnosis|score|rationale. "
            "Return up to 5 candidates. Scores must be between 0.10 and 0.95. "
        )

        # W2 Module C.2 — hierarchy-aware prompting.
        if hierarchy_level is not None:
            level_name = {0: "FAMILY (organ/syndrome class)", 1: "GENUS (specific disease group)", 2: "SPECIES (leaf diagnosis)"}.get(int(hierarchy_level), "SPECIES")
            base_prompt += (
                f"HIERARCHY GRANULARITY: You are reasoning at LEVEL {int(hierarchy_level)} — {level_name}. "
                "Return diagnoses at this granularity only. Do not mix granularities. "
            )
            if candidate_set:
                # Keep prompt bounded: cap at 60 ids to not blow context.
                cand_trim = [str(c) for c in list(candidate_set)[:60]]
                base_prompt += (
                    f"CANDIDATE SET (choose from these ids only, score others 0): {', '.join(cand_trim)}. "
                )

        shared_block = f"CURRENT_BELIEF:\n{shared_belief}\n" if shared_belief else ""
        messages = [
            {
                "role": "system",
                "content": base_prompt,
            },
            {
                "role": "user",
                "content": (
                    f"{shared_block}"
                    f"SUMMARY: {findings.summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings)}\n"
                    f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
                    f"TIMELINE: {', '.join(findings.timeline)}\n"
                    f"EXPOSURES: {', '.join(findings.exposures)}\n"
                    f"MEDICATIONS: {', '.join(findings.medications)}\n"
                    f"CONDITIONS: {', '.join(findings.suspected_conditions)}\n"
                    f"INPUT_CONTEXT: {', '.join(findings.input_context)}\n"
                    f"CONTEXT_LANES:\n{_context_lane_digest(findings) or 'none'}\n"
                    f"PHENOTYPE_ATOMS:\n{_phenotype_digest(findings) or 'none'}\n"
                    f"RED_FLAGS: {', '.join(findings.red_flags)}\n"
                      f"RAW_SEGMENTS: {' | '.join(findings.raw_segments[:12])}\n"
                      f"VITALS: {', '.join(f'{k}={v}' for k, v in (findings.derived_vitals or {}).items())}\n"
                      f"FACTS: {', '.join(node.label for node in fact_graph.nodes[:24])}\n"
                      f"AUTHORITY_CLAIMS: {', '.join(authority_claims)}\n"
                      f"PATTERNS: {', '.join(semantic_patterns)}\n"
                      f"PRIOR_EVIDENCE:\n{evidence_digest or 'none'}\n"
                      f"PRIOR_RESEARCH_QUERIES:\n{retrieval_digest or 'none'}\n"
                      f"URGENCY: {risk_profile.urgency.value}\n"
                  ),
              },
          ]
        started = time.time()
        stage_mode = "R3" if deep_thinking else "DEFAULT"
        # W6.2 J.2 — temperature override for ensemble runs. Floor at 0.05 only
        # when override is None (preserves legacy behavior); honor explicit 0.0.
        if override_temperature is None:
            _temp = max(0.05, self.policy.local_llm_temperature)
        else:
            _temp = max(0.0, float(override_temperature))
        response = client.chat(
            messages,
            temperature=_temp,
            max_tokens=self.policy.local_llm_max_tokens,
            structured_output=True,
            stage=stage_mode,
        )
        guard = _structured_output_guard(
            stage="induce_diagnoses",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_rank_protocol(response.content)
        # W6.1 J.6 — drop hallucinated symptom-concat ids before they pollute
        # the swarm posterior fold. No-op unless `swarm_label_validation_strict`.
        try:
            _strict = bool(getattr(self.policy, "swarm_label_validation_strict", False))
        except Exception:
            _strict = False
        if _strict and payload.get("candidates"):
            kept, dropped = _validate_rank_candidates(
                list(payload["candidates"]),
                hierarchy_level=hierarchy_level,
                candidate_set=candidate_set,
                enabled=True,
            )
            payload["candidates"] = kept
            if dropped:
                payload.setdefault("_label_validation", {})["dropped"] = dropped
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def generate_mechanism_frames(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        interventions: InterventionSet,
    ) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_ranker_enabled):
            return {}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._generate_mechanism_frames_sync, findings, risk_profile, fact_graph, interventions), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("generate_mechanism_frames", exc)

    def _generate_mechanism_frames_sync(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        interventions: InterventionSet,
    ) -> dict[str, Any]:
        client = self._client()
        state_frames = derive_state_frames(findings, fact_graph, interventions)
        dangerous_assumptions = derive_dangerous_treatment_assumptions(findings, fact_graph, interventions, state_frames)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical mechanism frame extractor. "
                    "Return ONLY lines formatted as "
                    "MECH|active_state|organ_system|primary_mechanism|secondary_mechanism|critical_findings_semi_colon|exclusions_semi_colon|hazard_context_semi_colon|confidence. "
                    "Return at most 3 lines. Prefer physiologic threat states and organ systems before disease names. "
                    "Use phenotype atoms to infer trigger, directionality, organ system, and mechanism."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"SUMMARY: {findings.summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings)}\n"
                    f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
                    f"TIMELINE: {', '.join(findings.timeline)}\n"
                    f"EXPOSURES: {', '.join(findings.exposures)}\n"
                    f"CONDITIONS: {', '.join(findings.suspected_conditions)}\n"
                    f"INPUT_CONTEXT: {', '.join(findings.input_context)}\n"
                    f"CONTEXT_LANES:\n{_context_lane_digest(findings) or 'none'}\n"
                    f"PHENOTYPE_ATOMS:\n{_phenotype_digest(findings) or 'none'}\n"
                    f"RED_FLAGS: {', '.join(findings.red_flags)}\n"
                    f"FACTS: {', '.join(node.label for node in fact_graph.nodes[:24])}\n"
                    f"INTERVENTIONS: {', '.join(item.label for item in interventions.items)}\n"
                    f"DERIVED_STATE_FRAMES: {', '.join(state_frames)}\n"
                    f"DANGEROUS_TREATMENT_ASSUMPTIONS: {', '.join(dangerous_assumptions)}\n"
                    f"URGENCY: {risk_profile.urgency.value}\n"
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=self.policy.local_llm_max_tokens,
            structured_output=True,
            stage="R3",
        )
        guard = _structured_output_guard(
            stage="generate_mechanism_frames",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_mechanism_protocol(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def verify(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        interventions: InterventionSet,
        differential: DifferentialSet,
        evidence: EvidenceBundle,
    ) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_verifier_enabled):
            return {}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._verify_sync, findings, risk_profile, fact_graph, frontier, interventions, differential, evidence), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("verify", exc)

    def _verify_sync(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        interventions: InterventionSet,
        differential: DifferentialSet,
        evidence: EvidenceBundle,
    ) -> dict[str, Any]:
        client = self._client()
        authority_claims = [node.label for node in fact_graph.nodes if node.category == "authority_claim"][:6]
        semantic_patterns = [node.label for node in fact_graph.nodes if node.category == "semantic_pattern"][:8]
        state_frames = derive_state_frames(findings, fact_graph, interventions)
        dangerous_assumptions = derive_dangerous_treatment_assumptions(findings, fact_graph, interventions, state_frames)
        candidate_lines = [
            f"{item.label}:{item.score:.2f}:{'; '.join(item.rationale[:2])}"
            for item in differential.candidates[:3]
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a clinical safety verifier. "
                    "Return ONLY line rows using ISSUE|severity|type|detail, optional DELTA|-0.10, and optional DECISION|revise or urgent_escalation or abstain. "
                    "Audit the existing structured diagnostic contract rather than manufacturing new disease state from ambiguity alone. "
                    "Prioritize unsafe closure, missed-critical risk, contraindicated intervention, unresolved objective discriminator, and only then translation/normalization gaps. "
                    "Repeated negative findings, duplicated NEG fields, or parse-normalization redundancy are not, by themselves, major translation gaps. "
                    "If the contract still shows unresolved upstream etiology or unresolved source disease, prefer ISSUE|major|missing_discriminator|... or ISSUE|major|unsafe_closure|... over translation_gap. "
                    "Focus on contraindicated interventions, missed-critical risks, unsafe closure, and translation/normalization gaps. "
                    "Check whether the active state frames are coherently explained and whether any critical findings remain under-explained. "
                    "Do not invent medication, psychiatric, or toxicology associations from ambiguous colloquial terms; emit ISSUE|major|translation_gap|... only when the ambiguity would materially change triage urgency, contraindications, or the first objective confirmation pathway. Do not demand imaging, biopsy, or laboratory findings that were never supplied unless the absence of that objective discriminator is itself the concrete blocker to safe closure. Repeated negative findings or duplicated NEG fields are not, by themselves, a major translation gap. Prefer structured-state safety failures such as missing_discriminator or unsafe_closure over translation_gap whenever the state already shows unresolved etiology or unresolved confirmation. "
                    "Do not infer a primary psychiatric or anxiety diagnosis from stress language alone when objective physiologic abnormalities are present. "
                    "Act as an audit layer over the structured diagnostic contract. Do not create a new disease anchor, do not rewrite the upstream source-disease state, and do not substitute for missing core reasoning. "
                    "Psychosis with seizure, dyskinesia, focal neurologic findings, fever, or altered consciousness should trigger concern for organic brain disease rather than primary psychiatric closure. "
                    "If the disease anchor is still open, describe the missing confirmation data rather than introducing new speculative named diagnoses in the issue detail. "
                    "If timeline, plan status, or device reliability is explicit in the case, do not treat it as missing. "
                    "Do not let clinician_anchor or working_diagnosis override stronger patient_narrative or phenotype evidence. "
                    "Differentiate true contraindications from staged prerequisites. A plan like 'CT first, then thrombolysis if hemorrhage is excluded' is a coherent conditional pathway, not a contraindicated intervention. "
                    "ORPHAN SYMPTOM RULE: For the anchor diagnosis, verify that EVERY objective symptom listed in POSITIVES is accounted for by: (a) the anchor diagnosis itself, (b) a known complication or association, or (c) an explicitly documented comorbidity. "
                    "If any objective symptom CANNOT be explained by the anchor, emit: ISSUE|major|orphan_symptom|[symptom] not explained by [anchor] — consider [alternative diagnosis that explains all symptoms including this one]. "
                    "A diagnosis that leaves objective symptoms unexplained is incomplete. The correct diagnosis has no orphans. "
                    "ANCHORING TRAP RULE: If the anchor diagnosis matches the patient's own stated self-diagnosis verbatim, actively challenge it. Ask: is there a diagnosis that explains ALL symptoms including the ones that don't fit the patient's label? If yes, emit ISSUE|major|anchoring_bias|anchor matches patient self-label; alternative [X] better explains full symptom cluster."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"SUMMARY: {findings.summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings)}\n"
                    f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
                    f"TIMELINE: {', '.join(findings.timeline)}\n"
                    f"CONDITIONS: {', '.join(findings.suspected_conditions)}\n"
                    f"PLANNED_INTERVENTIONS: {', '.join(findings.planned_interventions)}\n"
                    f"INPUT_CONTEXT: {', '.join(findings.input_context)}\n"
                    f"CONTEXT_LANES:\n{_context_lane_digest(findings) or 'none'}\n"
                    f"PHENOTYPE_ATOMS:\n{_phenotype_digest(findings) or 'none'}\n"
                    f"RAW_SEGMENTS: {' | '.join(findings.raw_segments[:12])}\n"
                    f"VITALS: {', '.join(f'{k}={v}' for k, v in (findings.derived_vitals or {}).items())}\n"
                    f"BLOCKED_ACTIONS: {', '.join(risk_profile.blocked_actions)}\n"
                    f"WORKFLOW_GUARDS: {', '.join(risk_profile.workflow_guards)}\n"
                    f"URGENCY: {risk_profile.urgency.value}\n"
                    f"FACTS: {', '.join(node.label for node in fact_graph.nodes[:24])}\n"
                    f"AUTHORITY_CLAIMS: {', '.join(authority_claims)}\n"
                    f"PATTERNS: {', '.join(semantic_patterns)}\n"
                    f"STATE_FRAMES: {', '.join(state_frames)}\n"
                    f"DANGEROUS_TREATMENT_ASSUMPTIONS: {', '.join(dangerous_assumptions)}\n"
                    f"MUST_NOT_MISS: {', '.join(frontier.must_not_miss)}\n"
                    f"DANGEROUS_IF_TREATED_AS: {', '.join(frontier.dangerous_if_treated_as)}\n"
                    f"INTERVENTIONS: {', '.join(item.label for item in interventions.items)}\n"
                    f"CANDIDATES: {' || '.join(candidate_lines)}\n"
                    f"EVIDENCE_COVERAGE: {evidence.coverage:.2f}\n"
                    f"CONTRADICTION_MASS: {evidence.contradiction_mass:.2f}\n"
                    + _format_evidence_for_prompt(evidence, top_k=3)
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=self.policy.local_llm_max_tokens,
            structured_output=True,
            stage="IE",
        )
        guard = _structured_output_guard(
            stage="verify",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_verify_protocol(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def self_verify(
        self,
        findings: StructuredFindings,
        frontier: HypothesisFrontier,
        differential: DifferentialSet,
    ) -> dict[str, Any]:
        """LLM self-check: returns VERIFY|confident|diagnosis, VERIFY|revise|reason, or VERIFY|escalate|diagnosis."""
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._self_verify_sync, findings, frontier, differential),
                timeout=self.policy.llm_sync_call_timeout_s,
            )
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("self_verify", exc)

    def _self_verify_sync(
        self,
        findings: StructuredFindings,
        frontier: HypothesisFrontier,
        differential: DifferentialSet,
    ) -> dict[str, Any]:
        client = self._client()
        top = differential.candidates[0] if differential.candidates else None
        diagnosis = top.label if top else "unknown"
        score = top.score if top else 0.0
        summary = findings.summary or " ".join(findings.positive_findings[:6])
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a board-certified physician performing a final self-check before closing a diagnosis. "
                    "Output exactly ONE line in this format:\n"
                    "VERIFY|confident|<diagnosis>   — diagnosis is supported; close the case\n"
                    "VERIFY|revise|<reason>         — one more reasoning pass needed\n"
                    "VERIFY|escalate|<diagnosis>    — urgent, cannot defer\n"
                    "No other output."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TOP DIAGNOSIS: {diagnosis} (confidence {score:.2f})\n"
                    f"PATIENT SUMMARY: {summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings[:10])}\n"
                    f"MEDICATIONS: {', '.join(findings.medications[:6])}\n"
                    f"RED FLAGS: {', '.join(findings.red_flags[:6])}\n\n"
                    "Self-check:\n"
                    "1. Are there DANGEROUS MIMICS you might be missing?\n"
                    "2. Does this diagnosis account for ALL documented findings?\n"
                    "3. Is there a SAFER (more common) alternative?\n"
                    "4. Would standard treatment harm the patient if this diagnosis is wrong?\n\n"
                    "Output one VERIFY line:"
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=80,
            structured_output=False,
            stage="self_verify",
        )
        guard = _structured_output_guard(
            stage="self_verify",
            response=response,
            started=started,
            hard_fail=False,
        )
        if guard is not None:
            return {"signal": "confident", "diagnosis": diagnosis, "_meta": {}}
        content = (response.content or "").strip()
        return _parse_self_verify_protocol(content, diagnosis)

    async def challenge(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        differential: DifferentialSet,
        evidence: EvidenceBundle,
        interventions: InterventionSet,
        deep_thinking: bool = False,
        loop_iteration: int = 0,
        tier: str = "full",
    ) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_challenger_enabled):
            return {}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._challenge_sync, findings, risk_profile, fact_graph, frontier, differential, evidence, interventions, deep_thinking, loop_iteration, tier), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("challenge", exc)

    def _challenge_sync(
        self,
        findings: StructuredFindings,
        risk_profile: RiskProfile,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        differential: DifferentialSet,
        evidence: EvidenceBundle,
        interventions: InterventionSet,
        deep_thinking: bool = False,
        loop_iteration: int = 0,
        tier: str = "full",
    ) -> dict[str, Any]:
        client = self._client()
        authority_claims = [node.label for node in fact_graph.nodes if node.category == "authority_claim"][:6]
        semantic_patterns = [node.label for node in fact_graph.nodes if node.category == "semantic_pattern"][:8]
        state_frames = derive_state_frames(findings, fact_graph, interventions)
        dangerous_assumptions = derive_dangerous_treatment_assumptions(findings, fact_graph, interventions, state_frames)
        candidate_lines = [
            f"{item.label}:{item.score:.2f}:{'; '.join(item.rationale[:2])}"
            for item in differential.candidates[:4]
        ]

        # Interventions explicitly requested (from input context) vs planned clinical interventions.
        # Blocked actions = requested-but-unsafe; intervention_request nodes = explicitly asked for.
        requested_interventions = list(risk_profile.blocked_actions[:6]) + [
            node.label for node in fact_graph.nodes if node.category == "intervention_request"
        ]

        system_content = (
            "You are the challenger module for a clinical DDx system. "
            "Return ONLY these line formats when needed: "
            "ISSUE|severity|type|detail, ALT|label|score|rationale, ANCHOR|label|delta, LOOP|action|reason. "
            "Challenge wrong anchors, surface dangerous alternatives, and request retrieval or hypothesis revision when needed. "
            "Perform bidirectional challenge — challenge anchors in both directions: "
            "(1) If the anchor is a common/benign diagnosis, ask: what dangerous condition could mimic this and what specific finding would distinguish them? "
            "(2) If the anchor is a rare or catastrophic diagnosis, ask: is there a common condition that better explains the full presentation without requiring rare pathology? Do not introduce catastrophic diagnoses that lack specific supporting findings beyond non-specific symptoms. "
            "Prefer challenging at the level of physiologic or threat-state frames before proposing disease-name alternatives. "
            "Do not hallucinate new toxicology or psychiatry anchors from ambiguous colloquial terms; if terminology is unclear, treat it as a translation gap. "
            "Do not convert psychosocial stress alone into a primary diagnosis when objective physiologic findings suggest an organic differential. "
            "Do not let clinician_anchor or working_diagnosis outrank a stronger phenotype or patient-narrative signature. "
            "If psychosis coexists with seizure, dyskinesia, altered consciousness, fever, or autonomic instability, aggressively challenge primary psychiatric anchors and surface organic neurologic, autoimmune, infectious, toxic, or metabolic alternatives. "
            "TREATMENT CASCADE ANALYSIS: Beyond wrong anchors, also analyze requested and blocked interventions. "
            "For each entry in BLOCKED_ACTIONS or REQUESTED_INTERVENTIONS: reason whether applying it to the top diagnosis "
            "(even if that diagnosis is correct) creates a second-order harm cascade through a specific physiologic mechanism. "
            "Consider rate-of-correction risks, osmotic/electrolyte shifts, rebound effects, and contraindicated protocol sequences. "
            "If a requested intervention is safe for a wrong diagnosis but lethal for the correct one, emit: "
            "ISSUE|critical|unsafe_treatment_cascade|<specific mechanism and resulting harm>. "
            "Ground every cascade claim in a concrete physiologic pathway, not general caution."
        )

        _tier = str(tier or "full").strip().lower() or "full"
        if deep_thinking and _tier != "draft":
            system_content += (
                " DEEP THINKING MODE: Play Devil's Advocate explicitly against the top candidate. "
                "Assume the top candidate is a misdirection. What else perfectly mimics it? "
                "Also, perform a forward-time simulation: if we treat for the top candidate and it's actually the 2nd/3rd candidate, what irreversible harm happens at +24h or +72h? "
                "If harm is high, output a missing_discriminator ISSUE or request a LOOP to get missing labs/images."
            )
        if _tier == "draft":
            system_content += (
                " DRAFT MODE: Be terse — emit at most 2 ISSUE lines and 2 ALT lines. "
                "Only flag severity in {high,critical} when truly catastrophic. Skip rationale longer than 12 words."
            )

        messages = [
            {
                "role": "system",
                "content": system_content,
            },
            {
                "role": "user",
                "content": (
                    f"SUMMARY: {findings.summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings)}\n"
                    f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
                    f"EXPOSURES: {', '.join(findings.exposures)}\n"
                    f"CONDITIONS: {', '.join(findings.suspected_conditions)}\n"
                    f"INTERVENTIONS: {', '.join(item.label for item in interventions.items)}\n"
                    f"REQUESTED_INTERVENTIONS: {', '.join(requested_interventions) or 'none'}\n"
                    f"CONTEXT_LANES:\n{_context_lane_digest(findings) or 'none'}\n"
                    f"PHENOTYPE_ATOMS:\n{_phenotype_digest(findings) or 'none'}\n"
                    f"BLOCKED_ACTIONS: {', '.join(risk_profile.blocked_actions)}\n"
                    f"WORKFLOW_GUARDS: {', '.join(risk_profile.workflow_guards)}\n"
                    f"FACTS: {', '.join(node.label for node in fact_graph.nodes[:24])}\n"
                    f"AUTHORITY_CLAIMS: {', '.join(authority_claims)}\n"
                    f"PATTERNS: {', '.join(semantic_patterns)}\n"
                    f"STATE_FRAMES: {', '.join(state_frames)}\n"
                    f"DANGEROUS_TREATMENT_ASSUMPTIONS: {', '.join(dangerous_assumptions)}\n"
                    f"MUST_NOT_MISS: {', '.join(frontier.must_not_miss)}\n"
                    f"DANGEROUS_IF_TREATED_AS: {', '.join(frontier.dangerous_if_treated_as)}\n"
                    f"CANDIDATES: {' || '.join(candidate_lines)}\n"
                    f"EVIDENCE_COVERAGE: {evidence.coverage:.2f}\n"
                    f"CONTRADICTION_MASS: {evidence.contradiction_mass:.2f}\n"
                    + _format_evidence_for_prompt(evidence, top_k=3)
                ),
            },
        ]
        started = time.time()

        # Adaptive Temperature: increase creativity/adversarial divergence in later loops during deep thinking
        temperature = self.policy.local_llm_temperature
        if deep_thinking and _tier != "draft" and loop_iteration > 0:
            temperature = min(0.6, temperature + (0.15 * loop_iteration))
        if _tier == "draft":
            # Force determinism on draft tier to keep cost predictable.
            temperature = 0.0

        # Token budget: full tier = 1.5x base; draft tier = ratio·base (default 0.5x).
        if _tier == "draft":
            ratio = float(getattr(self.policy, "challenger_draft_max_tokens_ratio", 0.5))
            ratio = max(0.1, min(1.0, ratio))
            max_tokens = max(64, int(self.policy.local_llm_max_tokens * ratio))
        else:
            max_tokens = int(self.policy.local_llm_max_tokens * 1.5)

        response = client.chat(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            structured_output=True,
            stage="IE",
        )
        guard = _structured_output_guard(
            stage="challenge",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        payload = _parse_challenge_protocol(response.content)

        # K.6 — schema-validated parse-with-retry. If we parsed a non-empty raw
        # but extracted zero structured rows, retry once with explicit schema
        # printed in the prompt. On second failure, emit abstain LOOP.
        retry_enabled = bool(getattr(self.policy, "challenger_schema_retry_enabled", False))
        raw_text = str(response.content or "").strip()
        empty_payload = (
            not payload.get("issues")
            and not payload.get("alt_hypotheses")
            and not str(payload.get("anchor", "")).strip()
            and not str(payload.get("loop_action", "")).strip()
        )
        if retry_enabled and raw_text and empty_payload:
            schema_msg = {
                "role": "system",
                "content": (
                    "Your previous reply contained no recognized lines. "
                    "Reply ONLY with these literal line formats — one per line, no prose: "
                    "ISSUE|severity|type|detail | ALT|label|score|rationale | "
                    "ANCHOR|label|delta | LOOP|action|reason. "
                    "If nothing applies, emit: LOOP|none|no_issues."
                ),
            }
            retry_messages = list(messages) + [
                {"role": "assistant", "content": raw_text[:400]},
                schema_msg,
            ]
            retry_response = client.chat(
                retry_messages,
                temperature=0.0,
                max_tokens=max_tokens,
                structured_output=True,
                stage="IE",
            )
            payload = _parse_challenge_protocol(retry_response.content)
            empty_payload = (
                not payload.get("issues")
                and not payload.get("alt_hypotheses")
                and not str(payload.get("anchor", "")).strip()
                and not str(payload.get("loop_action", "")).strip()
            )
            if empty_payload:
                # Force abstain so downstream knows challenger didn't run cleanly.
                payload["loop_action"] = "abstain"
                payload["loop_reason"] = "parse_failure"

        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started), "tier": _tier}
        return payload

    async def steelman_top2(
        self,
        findings: StructuredFindings,
        evidence: EvidenceBundle,
        top1_label: str,
        top2_label: str,
        top2_rationale: str = "",
    ) -> dict[str, Any]:
        """W7.2 K.2 — Construct strongest argument that #2 candidate is correct.

        Returns {label, argument_score in [0,1], rationale, _meta}. Empty dict
        when challenger LLM is disabled.
        """
        if self._local_disabled(self.policy.local_llm_challenger_enabled):
            return {}
        if not str(top1_label or "").strip() or not str(top2_label or "").strip():
            return {}
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._steelman_sync, findings, evidence, top1_label, top2_label, top2_rationale),
                timeout=self.policy.llm_sync_call_timeout_s,
            )
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("steelman", exc)

    def _steelman_sync(
        self,
        findings: StructuredFindings,
        evidence: EvidenceBundle,
        top1_label: str,
        top2_label: str,
        top2_rationale: str,
    ) -> dict[str, Any]:
        client = self._client()
        system_content = (
            "You are a steel-man advocate. The system has anchored on TOP1 as the leading diagnosis. "
            "Your job: construct the STRONGEST possible evidence-based argument that TOP2 is the correct diagnosis instead, "
            "using only the findings provided. Do NOT hallucinate findings. "
            "Score the strength of your argument honestly in [0,1] — 1.0 means the evidence genuinely fits TOP2 better than TOP1; "
            "0.3 means the case for TOP2 is weak; 0.0 means TOP2 is impossible given the findings. "
            "Return EXACTLY ONE line in this format: STEELMAN|<dx_label>|<argument_score:0..1>|<concise_rationale>"
        )
        user_content = (
            f"TOP1: {top1_label}\n"
            f"TOP2: {top2_label}\n"
            f"TOP2_RATIONALE_HINT: {str(top2_rationale or '').strip() or 'none'}\n"
            f"SUMMARY: {findings.summary}\n"
            f"POSITIVES: {', '.join(findings.positive_findings)}\n"
            f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
            f"EXPOSURES: {', '.join(findings.exposures)}\n"
            f"EVIDENCE_COVERAGE: {evidence.coverage:.2f}\n"
            f"CONTRADICTION_MASS: {evidence.contradiction_mass:.2f}\n"
            + _format_evidence_for_prompt(evidence, top_k=3)
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.2,
            max_tokens=max(64, int(self.policy.local_llm_max_tokens * 0.5)),
            structured_output=True,
            stage="IE",
        )
        guard = _structured_output_guard(
            stage="steelman",
            response=response,
            started=started,
            hard_fail=False,
        )
        if guard is not None:
            return guard
        payload = _parse_steelman_protocol(response.content)
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def pairwise_judge(
        self,
        findings: StructuredFindings,
        evidence: EvidenceBundle,
        candidate_a: str,
        candidate_b: str,
        framing: str = "bayesian",
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """W7.3 K.5 — judge LLM call for one BT pairwise comparison.

        Args:
            framing: 'bayesian' (priors), 'mechanism', or 'epi' — different
                     prompt wordings to decorrelate judges.
        Returns: {winner, prob, rationale, _meta}.
        """
        if self._local_disabled(self.policy.local_llm_challenger_enabled):
            return {}
        if not str(candidate_a or "").strip() or not str(candidate_b or "").strip():
            return {}
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._pairwise_judge_sync, findings, evidence, candidate_a, candidate_b, framing, temperature),
                timeout=self.policy.llm_sync_call_timeout_s,
            )
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("pairwise_judge", exc)

    def _pairwise_judge_sync(
        self,
        findings: StructuredFindings,
        evidence: EvidenceBundle,
        candidate_a: str,
        candidate_b: str,
        framing: str,
        temperature: float,
    ) -> dict[str, Any]:
        client = self._client()
        framing_text = {
            "bayesian": "Compare via Bayesian prior reasoning: which candidate has the higher base-rate consistent with this presentation?",
            "mechanism": "Compare via mechanism: which candidate's pathophysiology better explains the chain of findings?",
            "epi": "Compare via clinical epidemiology: which candidate fits the demographic and risk-factor profile better?",
        }.get(str(framing or "bayesian").strip().lower(), "Compare which candidate fits the case better given the evidence.")
        system_content = (
            "You are a pairwise diagnostic judge. " + framing_text + " "
            "Reply with EXACTLY ONE line: WIN|<chosen_label>|<prob_chosen_wins:0..1>|<concise_rationale>. "
            "Probability is the calibrated belief that the chosen candidate is the correct diagnosis "
            "rather than the other one (close to 0.5 means uncertain)."
        )
        user_content = (
            f"CANDIDATE_A: {candidate_a}\n"
            f"CANDIDATE_B: {candidate_b}\n"
            f"SUMMARY: {findings.summary}\n"
            f"POSITIVES: {', '.join(findings.positive_findings)}\n"
            f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
            f"EXPOSURES: {', '.join(findings.exposures)}\n"
            f"EVIDENCE_COVERAGE: {evidence.coverage:.2f}\n"
            + _format_evidence_for_prompt(evidence, top_k=3)
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=max(0.0, min(0.6, float(temperature))),
            max_tokens=max(48, int(self.policy.local_llm_max_tokens * 0.3)),
            structured_output=True,
            stage="IE",
        )
        guard = _structured_output_guard(
            stage="pairwise_judge",
            response=response,
            started=started,
            hard_fail=False,
        )
        if guard is not None:
            return guard
        payload = _parse_pairwise_judge_protocol(response.content)
        # Normalize: ensure the winner is one of the two candidates we asked about.
        winner = str(payload.get("winner", "") or "").strip()
        a_norm = str(candidate_a).strip().lower()
        b_norm = str(candidate_b).strip().lower()
        if winner.lower() not in {a_norm, b_norm}:
            # Try fuzzy match by prefix; otherwise mark uncertain.
            if winner.lower().startswith(a_norm[:6]):
                payload["winner"] = candidate_a
            elif winner.lower().startswith(b_norm[:6]):
                payload["winner"] = candidate_b
            else:
                payload["winner"] = candidate_a if payload.get("prob", 0.5) >= 0.5 else candidate_b
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started), "framing": framing}
        return payload

    async def resolve_premise_conflict(
        self,
        findings: StructuredFindings,
        evidence: EvidenceBundle,
        conflicting_premises: list[str],
        ds_conflict_k: float = 0.0,
    ) -> dict[str, Any]:
        """W7.3 K.4 — resolver LLM call when DS conflict K is high.

        Returns ARTIFACT/DISCRIMINATOR/DECISION structure.
        """
        if self._local_disabled(self.policy.local_llm_challenger_enabled):
            return {}
        if not conflicting_premises:
            return {}
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._resolve_premise_conflict_sync, findings, evidence, conflicting_premises, ds_conflict_k),
                timeout=self.policy.llm_sync_call_timeout_s,
            )
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("premise_conflict", exc)

    def _resolve_premise_conflict_sync(
        self,
        findings: StructuredFindings,
        evidence: EvidenceBundle,
        conflicting_premises: list[str],
        ds_conflict_k: float,
    ) -> dict[str, Any]:
        client = self._client()
        system_content = (
            "You are a premise-conflict resolver for a clinical reasoning system. "
            "Two or more premises in evidence are mutually inconsistent (Dempster-Shafer conflict K is high). "
            "Your job: (1) identify which premise is most likely an artifact (data error, mistranslation, "
            "transcription noise) versus a real signal; (2) propose a single concrete discriminator test "
            "or finding that, if checked, would resolve the conflict; (3) if neither premise can be ruled out "
            "from current data, decide to escalate. "
            "Return ONLY these line formats — one each: "
            "ARTIFACT|<premise_text_or_id>|<short rationale>, "
            "DISCRIMINATOR|<test_or_finding>|<expected resolution>, "
            "DECISION|<keep|escalate>|<reason>."
        )
        premise_block = "\n".join(f"- {p}" for p in conflicting_premises[:8])
        user_content = (
            f"DS_CONFLICT_K: {ds_conflict_k:.2f}\n"
            f"CONFLICTING_PREMISES:\n{premise_block}\n"
            f"SUMMARY: {findings.summary}\n"
            f"POSITIVES: {', '.join(findings.positive_findings)}\n"
            f"NEGATIVES: {', '.join(findings.negative_findings)}\n"
            f"EVIDENCE_COVERAGE: {evidence.coverage:.2f}\n"
            f"CONTRADICTION_MASS: {evidence.contradiction_mass:.2f}\n"
            + _format_evidence_for_prompt(evidence, top_k=3)
        )
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.1,
            max_tokens=max(96, int(self.policy.local_llm_max_tokens * 0.6)),
            structured_output=True,
            stage="IE",
        )
        guard = _structured_output_guard(
            stage="premise_conflict",
            response=response,
            started=started,
            hard_fail=False,
        )
        if guard is not None:
            return guard
        payload = _parse_conflict_resolution_protocol(response.content)
        # K.4 schema retry: if completely empty, default to escalate.
        if not any([payload.get("artifact_premise"), payload.get("discriminator"), payload.get("decision")]):
            payload["decision"] = "escalate"
            payload["decision_reason"] = "premise_conflict_unresolvable_parse_failure"
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    async def generate_retrieval_queries(
        self,
        findings: StructuredFindings,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        needs: list[dict[str, str]],
        *,
        iteration: int = 0,
    ) -> dict[str, Any]:
        if self._local_disabled(self.policy.local_llm_ranker_enabled):
            return {}
        if not needs:
            return {}
        try:
            return await asyncio.wait_for(asyncio.to_thread(self._generate_retrieval_queries_sync, findings, fact_graph, frontier, needs, iteration), timeout=self.policy.llm_sync_call_timeout_s)
        except (Exception, asyncio.TimeoutError) as exc:
            return _error_payload("generate_retrieval_queries", exc)

    def _generate_retrieval_queries_sync(
        self,
        findings: StructuredFindings,
        fact_graph: FactGraph,
        frontier: HypothesisFrontier,
        needs: list[dict[str, str]],
        iteration: int = 0,
    ) -> dict[str, Any]:
        client = self._client()
        authority_claims = [node.label for node in fact_graph.nodes if node.category == "authority_claim"][:6]
        semantic_patterns = [node.label for node in fact_graph.nodes if node.category == "semantic_pattern"][:8]
        state_frames = derive_state_frames(findings, fact_graph)
        need_lines = [
            f"{index}|{item.get('objective', '')}|{item.get('decision_target', '')}|{item.get('hypothesis_label', item.get('label', ''))}|{item.get('active_state', '')}|"
            f"{item.get('unresolved_critical_finding', '')}|{item.get('rival_mechanism', '')}|"
            f"{item.get('action_hazard', '')}|{item.get('desired_discriminator', '')}|{item.get('query_hint', '')}"
            for index, item in enumerate(needs[:12], start=1)
        ]
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the R2 research planner for a clinical differential diagnosis engine. "
                    "Return ONLY protocol rows: "
                    "HYP|hypothesis_label ; DISC|hypothesis_label|discriminator ; QUERY|index|query|expected_impact. "
                      "For each round produce discriminator-driven queries that narrow the active differential from broad to specific. "
                      "Use concise English medical terminology. "
                      "Build queries around unexplained findings, dangerous treatment assumptions, and decisive physiologic frames. "
                      "CRITICAL SELF-AWARENESS: You are a 4B parameter model and your internal knowledge of rare drug interactions, obscure toxidromes, and ICD-11/PubMed data is highly limited. You MUST use this step to search the web for external truth. "
                      "MANDATORY: If the patient is taking medications, you MUST autonomously generate explicit PubMed/NCBI queries checking for specific \"drug-drug interactions\", \"adverse effects\", and \"toxidromes\" matching their current symptoms (e.g. `[Drug A] AND [Drug B] interaction hyperkalemia PubMed`). Do not rely on internal memory for pharmacology; ALWAYS search it. "
                      "MANDATORY WITHDRAWAL/IATROGENIC RULE: If the case contains ANY combination of: tremor, shaking, twitching, insomnia, headache, anxiety, sweating, dizziness, restlessness, nausea, or autonomic symptoms — ALWAYS generate at least one QUERY asking about drug withdrawal, substance cessation, or medication side effects for this symptom pattern, EVEN if no medications are reported. Absence of reported medications does NOT rule out drug use or withdrawal — patients frequently omit this. Example queries: 'tremor insomnia headache benzodiazepine withdrawal syndrome', 'autonomic symptoms multi-system drug adverse effect etiology'. "
                      "DISCRIMINATOR RULE: For the top 2 differential candidates, always generate one DISC row per candidate identifying the single most decisive discriminating finding that would confirm or exclude it. Example: DISC|peptic_ulcer_disease|nocturnal_epigastric_pain_relieved_by_food vs DISC|gastroesophageal_reflux|worsening_with_lying_flat_postprandial. "
                      "When uncertainty remains high, keep expanding the research plan instead of giving up early; more queries are preferable to premature closure. "
                      "Use the decision_target field to decide whether to confirm, discriminate, exclude, or review contraindications. "
                      "Queries must look like real PubMed or guideline searches, not planner metadata. "
                      "Do not include generic planning jargon such as candidate, criteria, must-not-miss, impact, workup, or review unless they are core medical concepts in the query itself. "
                      "If the active story contains distinctive findings such as seizure, dyskinesia, focal neurologic deficits, toxin exposure, hemodynamic collapse, or unusual temporal progression, queries should explicitly include those concrete findings instead of generic syndrome labels like crisis, reaction, or confirmation. "
                      "Do not output generic placeholder queries. "
                      "Each query must have a clear expected impact on candidate elimination or promotion. "
                      "Phenotype atoms are preferred over internal rationale text when constructing signature-first queries."
                  ),
              },
            {
                "role": "user",
                "content": (
                    f"SUMMARY: {findings.summary}\n"
                    f"POSITIVES: {', '.join(findings.positive_findings)}\n"
                    f"MEDICATIONS_ACTIVE: {', '.join(findings.medications)}\n"
                    f"CONDITIONS: {', '.join(findings.suspected_conditions)}\n"
                    f"CONTEXT_LANES:\n{_context_lane_digest(findings) or 'none'}\n"
                    f"PHENOTYPE_ATOMS:\n{_phenotype_digest(findings) or 'none'}\n"
                    f"RED_FLAGS: {', '.join(findings.red_flags)}\n"
                    f"FACTS: {', '.join(node.label for node in fact_graph.nodes[:20])}\n"
                    f"AUTHORITY_CLAIMS: {', '.join(authority_claims)}\n"
                    f"PATTERNS: {', '.join(semantic_patterns)}\n"
                    f"STATE_FRAMES: {', '.join(state_frames)}\n"
                    f"MUST_NOT_MISS: {', '.join(frontier.must_not_miss)}\n"
                    f"ROUND_ITERATION: {iteration}\n"
                    "NEEDS:\n"
                    + "\n".join(need_lines)
                ),
            },
        ]
        started = time.time()
        response = client.chat(
            messages,
            temperature=0.0,
            max_tokens=self.policy.local_llm_max_tokens,
            structured_output=True,
            stage="R2",
        )
        guard = _structured_output_guard(
            stage="generate_retrieval_queries",
            response=response,
            started=started,
            hard_fail=self.policy.llm_empty_slate_hard_fail_enabled,
        )
        if guard is not None:
            return guard
        parsed = _parse_research_protocol(response.content)
        if not parsed.get("queries"):
            parsed["queries"] = _parse_query_protocol(response.content)
        payload = parsed
        payload["_meta"] = {"metrics": _extract_metrics(response, elapsed=time.time() - started)}
        return payload

    def _calibrate_differential_sync(
        self,
        patient_summary: str,
        top_candidates: list[str],
        epidemiology_context: str,
    ) -> dict:
        """Post-swarm epidemiological calibration: re-rank differential using Occam's Razor + base rates."""
        client = self._client()
        system_content = (
            "You are a clinical epidemiology consultant calibrating a differential diagnosis. "
            "Given the patient presentation and real-world prevalence/epidemiology data, "
            "re-rank the candidates from most likely to least likely. "
            "Apply Occam's Razor: the simplest diagnosis that explains the most symptoms is usually correct. "
            "Common diseases are common. Rare diagnoses require rare or specific findings. "
            "Return ONLY lines in this exact format: RANK|1|diagnosis_label|one_sentence_reasoning "
            "Do not output anything else. Do not add explanations outside the RANK lines."
        )
        candidates_text = "\n".join(f"  {i+1}. {c}" for i, c in enumerate(top_candidates))
        user_content = (
            f"PATIENT: {patient_summary}\n"
            f"CURRENT CANDIDATES:\n{candidates_text}\n"
            f"EPIDEMIOLOGY CONTEXT:\n{epidemiology_context or 'No context available.'}\n"
            "Re-rank these candidates by actual likelihood for this patient. "
            "If the current top candidate is a rare/catastrophic condition but a common diagnosis better explains the presentation, rank the common one first."
        )
        started = time.time()
        response = client.chat(
            [{"role": "system", "content": system_content}, {"role": "user", "content": user_content}],
            temperature=0.0,
            max_tokens=min(128, self.policy.local_llm_max_tokens),
            structured_output=False,
            stage="IE",
        )
        guard = _structured_output_guard(
            stage="calibrate_differential",
            response=response,
            started=started,
            hard_fail=False,
        )
        if guard is not None:
            return guard
        ranked: list[dict] = []
        for line in (response.content or "").splitlines():
            line = line.strip()
            if line.startswith("RANK|"):
                parts = line.split("|", 3)
                if len(parts) >= 3:
                    try:
                        rank_num = int(parts[1])
                    except ValueError:
                        rank_num = len(ranked) + 1
                    label = parts[2].strip().lower().replace(" ", "_")
                    reasoning = parts[3].strip() if len(parts) > 3 else ""
                    ranked.append({"rank": rank_num, "label": label, "reasoning": reasoning})
        ranked.sort(key=lambda x: x["rank"])
        return {
            "ranked": ranked,
            "_meta": {"metrics": _extract_metrics(response, elapsed=time.time() - started)},
        }

    async def calibrate_differential(
        self,
        patient_summary: str,
        top_candidates: list[str],
        epidemiology_context: str,
    ) -> dict:
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._calibrate_differential_sync,
            patient_summary,
            top_candidates,
            epidemiology_context,
        )
