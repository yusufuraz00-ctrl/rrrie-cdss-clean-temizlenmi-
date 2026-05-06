"""Online adaptive memory for retrieval and telemetry artifacts."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.cdss.contracts.models import EvidenceBundle, RetrievalRankingStats, StructuredFindings


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _tokenize(value: str) -> set[str]:
    return {token for token in re.findall(r"[a-z0-9]{4,}", str(value or "").lower())}


def _state_signature(findings: StructuredFindings) -> set[str]:
    bag = " ".join(
        [
            findings.summary,
            *findings.positive_findings[:8],
            *findings.negative_findings[:8],
            *findings.red_flags[:6],
            *findings.exposures[:6],
            *findings.suspected_conditions[:6],
            *findings.timeline[:6],
            *findings.input_context[:8],
            *findings.raw_segments[:8],
        ]
    )
    signature = _tokenize(bag)
    try:
        hr = float(str((findings.derived_vitals or {}).get("heart_rate", "")).strip())
    except (TypeError, ValueError):
        hr = 0.0
    try:
        sbp = float(str((findings.derived_vitals or {}).get("sbp", "")).strip())
    except (TypeError, ValueError):
        sbp = 0.0
    if hr > 0:
        signature.add("tachycardic_profile" if hr >= 100 else "nontachy_profile")
    if sbp > 0:
        signature.add("hypotensive_profile" if sbp <= 90 else "nonhypotensive_profile")
    return signature


class EvidenceMemoryStore:
    """Append-only memory for retrieval queries, evidence, and reranking telemetry."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.base_dir / "evidence_memory.jsonl"
        # In-memory cache: avoids O(N) JSONL scan on every recall(). The cache
        # is invalidated when the file's mtime/size changes (covers external
        # writers); record() appends both to disk and to the cache so a fresh
        # entry is visible immediately without forcing a reload.
        self._cache_rows: list[dict[str, Any]] | None = None
        self._cache_mtime: float = 0.0
        self._cache_size: int = 0

    def _maybe_load_cache(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            self._cache_rows = []
            self._cache_mtime = 0.0
            self._cache_size = 0
            return self._cache_rows
        try:
            stat = self.path.stat()
        except OSError:
            return self._cache_rows or []
        mtime = float(stat.st_mtime)
        size = int(stat.st_size)
        if (
            self._cache_rows is not None
            and mtime == self._cache_mtime
            and size == self._cache_size
        ):
            return self._cache_rows
        rows: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = str(line or "").strip()
                if not stripped:
                    continue
                try:
                    rows.append(json.loads(stripped))
                except json.JSONDecodeError:
                    continue
        self._cache_rows = rows
        self._cache_mtime = mtime
        self._cache_size = size
        return rows

    def record(
        self,
        *,
        case_id: str,
        findings: StructuredFindings,
        evidence: EvidenceBundle,
        retrieval_stats: RetrievalRankingStats,
        validated: bool = False,
    ) -> None:
        promoted_items = [
            item
            for item in evidence.items
            if str(item.verification_status or "").strip().lower() == "verified"
            and str(item.relation_type or "").strip().lower() in {"supports", "refutes"}
            and bool(item.linked_hypotheses)
        ]
        if not promoted_items:
            return
        linked_hypotheses = list(
            dict.fromkeys(
                label
                for item in promoted_items
                for label in (item.linked_hypotheses or [])
                if str(label or "").strip()
            )
        )[:8]
        relation_type = str(promoted_items[0].relation_type or "neutral").strip().lower()
        evidence_refs = [
            {
                "title": item.title,
                "citation": item.citation,
                "pmid": item.pmid,
                "doi": item.doi,
                "linked_hypotheses": list(item.linked_hypotheses or []),
                "relation_type": item.relation_type,
            }
            for item in promoted_items[:12]
        ]
        payload: dict[str, Any] = {
            "recorded_at": _utc_now(),
            "case_id": case_id,
            "summary": findings.summary,
            "validated": bool(validated),
            "state_signature": sorted(_state_signature(findings)),
            "queries": [item.query_hint for item in evidence.retrieval_intents[:12]],
            "evidence_titles": [item["title"] for item in evidence_refs[:10]],
            "evidence_refs": evidence_refs,
            "linked_hypotheses": linked_hypotheses,
            "relation_type": relation_type,
            "retrieval_stats": retrieval_stats.model_dump(mode="json"),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        # Keep cache hot — append in-memory and refresh stat fingerprint so
        # the next recall() picks up the new entry without a re-read.
        if self._cache_rows is not None:
            self._cache_rows.append(payload)
            try:
                stat = self.path.stat()
                self._cache_mtime = float(stat.st_mtime)
                self._cache_size = int(stat.st_size)
            except OSError:
                pass

    def recall(
        self,
        *,
        summary: str,
        findings: StructuredFindings | None = None,
        case_id: str = "",
        hypotheses: list[str] | None = None,
        discriminators: list[str] | None = None,
        limit: int = 3,
        validated_only: bool = True,
    ) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        query_tokens = _tokenize(summary)
        query_hypotheses = {_token for item in (hypotheses or []) for _token in _tokenize(item.replace("_", " "))}
        query_discriminators = {_token for item in (discriminators or []) for _token in _tokenize(item)}
        if not query_tokens:
            return []
        query_signature = _state_signature(findings) if findings else set()
        scored: list[tuple[int, dict[str, Any]]] = []
        for payload in self._maybe_load_cache():
            if validated_only and not bool(payload.get("validated", False)):
                continue
            if case_id and str(payload.get("case_id", "")) == case_id:
                continue
            memory_tokens = _tokenize(
                " ".join(
                    [
                        str(payload.get("summary", "")),
                        *[str(item) for item in list(payload.get("evidence_titles", []) or [])[:8]],
                        *[str(item) for item in list(payload.get("linked_hypotheses", []) or [])[:6]],
                    ]
                )
            )
            overlap = len(query_tokens & memory_tokens)
            if overlap <= 1:
                continue
            memory_hypothesis_tokens = {_token for item in list(payload.get("linked_hypotheses", []) or []) for _token in _tokenize(item)}
            hypothesis_overlap = len(query_hypotheses & memory_hypothesis_tokens) if query_hypotheses else 0
            discriminator_overlap = len(query_discriminators & memory_tokens) if query_discriminators else 0
            signature_tokens = set(str(item) for item in list(payload.get("state_signature", []) or []))
            signature_jaccard = 0.0
            if query_signature and signature_tokens:
                union = len(query_signature | signature_tokens)
                signature_jaccard = (len(query_signature & signature_tokens) / union) if union else 0.0
            if query_signature and signature_tokens and signature_jaccard < 0.2:
                continue
            blended = int((overlap * 10) + (signature_jaccard * 10) + (hypothesis_overlap * 8) + (discriminator_overlap * 5))
            scored.append((blended, payload))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [payload for _, payload in scored[:limit]]
