"""Source-built syndrome registry backed by a generated SQLite artifact."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

_log = logging.getLogger("rrrie-cdss")

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr


class SyndromeCombo(BaseModel):
    """Declarative pathognomonic / near-pathognomonic atom conjunction.

    A combo says: when ALL of ``required_atoms`` are present (and any
    ``trigger_context`` constraints hold), the parent syndrome should be
    force-promoted into the differential with the given priority. This is a
    *symbolic* layer that complements the bag-of-tokens prototype matcher.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    required_atoms: list[str] = Field(default_factory=list)
    optional_atoms: list[str] = Field(default_factory=list)
    forbidden_atoms: list[str] = Field(default_factory=list)
    trigger_context: dict = Field(default_factory=dict)
    promotes_to_priority: str = "must_not_miss"
    rationale: str = ""


class ChainedTrigger(BaseModel):
    """Temporal mechanism chain: a trigger event followed by a syndrome.

    Example: ``trigger_atom='dental_procedure'`` within ``window_days=30`` plus
    consequent atoms ``['acral_purple_lesions','new_dyspnea']`` ⇒ promote
    ``infective_endocarditis``. Stored as data; clinical reviewers add new
    chains without code changes.
    """

    model_config = ConfigDict(extra="ignore")

    name: str
    trigger_atom: str
    window_days: int = 30
    consequent_atoms: list[str] = Field(default_factory=list)
    optional_atoms: list[str] = Field(default_factory=list)
    consequent_dx: str = ""
    promotes_to_priority: str = "must_not_miss"
    rationale: str = ""


class SyndromeProfile(BaseModel):
    """Declarative syndrome-family profile used by ranker and planner."""

    model_config = ConfigDict(extra="ignore")

    id: str
    label: str
    summary: str = ""
    cue_lexicon: list[str] = Field(default_factory=list)
    evidence_needs: list[str] = Field(default_factory=list)
    challenge_queries: list[str] = Field(default_factory=list)
    immediate_actions: list[str] = Field(default_factory=list)
    must_not_miss: bool = False
    dangerous_if_missed: bool = False
    dangerous_if_treated_as: str = ""
    unsafe_interventions: list[str] = Field(default_factory=list)
    trigger_requirements: list[str] = Field(default_factory=list)
    trigger_exclusions: list[str] = Field(default_factory=list)
    clinical_role: str = ""
    source_disease_resolved: bool | None = None
    source_manifest: str = ""
    source_ref: str = ""
    artifact_version: str = ""
    # Populated at runtime by demographic_profiler (LLM-synthesized, never hand-written).
    # Stored in separate learned SQLite store; injected here after load.
    demographic_risk_profile: dict = Field(default_factory=dict)
    # Hierarchical placement (W1 Module A): level 0=family, 1=genus, 2=species.
    # Legacy profiles default to level=2 with empty parent_id; overlay loads from
    # data/cdss/knowledge/hierarchy.json (optional asset). Consumers that don't
    # know about hierarchy treat every profile as a flat species.
    level: int = 2
    parent_id: str = ""
    # Symbolic promotion layer. Populated from the optional combo-overlay JSON
    # at ``data/cdss/knowledge/syndrome_combos.json``; consumers tolerate empty
    # lists. See ``SyndromeCombo`` / ``ChainedTrigger`` and
    # ``prototype_memory.detect_combo_promotions``.
    pathognomonic_combos: list[SyndromeCombo] = Field(default_factory=list)
    chained_triggers: list[ChainedTrigger] = Field(default_factory=list)


class SyndromeProfileRegistry(BaseModel):
    """In-memory registry loaded from a generated SQLite artifact."""

    model_config = ConfigDict(extra="ignore")

    profiles: list[SyndromeProfile] = Field(default_factory=list)
    artifact_path: str = ""
    source_sha256: str = ""
    built_at: str = ""

    # 4b: lazy lookup indices. The registry is loaded once and treated as
    # immutable, so a dict-based id index and a level-bucket map cut the
    # per-call O(N) scan down to O(1) / O(level_size). Reset is automatic
    # because new instances ship fresh PrivateAttr defaults.
    _id_index: dict[str, "SyndromeProfile"] | None = PrivateAttr(default=None)
    _level_index: dict[int, list["SyndromeProfile"]] | None = PrivateAttr(default=None)

    def by_id(self, profile_id: str) -> SyndromeProfile | None:
        if self._id_index is None:
            self._id_index = {item.id: item for item in self.profiles}
        return self._id_index.get(profile_id)

    # ------------------------------------------------------------------
    # Hierarchy helpers (W1 Module A). All operate on in-memory profiles.
    # Safe when `level`/`parent_id` are absent: every profile counts as level=2
    # with parent_id="" so `by_level(0)` returns empty until hierarchy.json ships.
    # ------------------------------------------------------------------
    def by_level(self, level: int) -> list[SyndromeProfile]:
        try:
            target = int(level)
        except Exception:
            return []
        if self._level_index is None:
            buckets: dict[int, list[SyndromeProfile]] = {}
            for item in self.profiles:
                buckets.setdefault(int(item.level), []).append(item)
            self._level_index = buckets
        return list(self._level_index.get(target, ()))

    def children(self, parent_id: str) -> list[SyndromeProfile]:
        pid = str(parent_id or "").strip()
        if not pid:
            return []
        return [item for item in self.profiles if str(item.parent_id or "").strip() == pid]

    def ancestors(self, profile_id: str) -> list[SyndromeProfile]:
        chain: list[SyndromeProfile] = []
        seen: set[str] = set()
        current = self.by_id(profile_id)
        while current is not None and current.parent_id:
            pid = str(current.parent_id).strip()
            if not pid or pid in seen:
                break
            seen.add(pid)
            parent = self.by_id(pid)
            if parent is None:
                break
            chain.append(parent)
            current = parent
        return chain

    def descendants(self, profile_id: str, *, level: int | None = None) -> list[SyndromeProfile]:
        """Collect all descendants (optionally filtered to a specific level)."""
        target = str(profile_id or "").strip()
        if not target:
            return []
        # BFS via children() to avoid cycles.
        out: list[SyndromeProfile] = []
        seen: set[str] = set()
        frontier: list[str] = [target]
        while frontier:
            next_frontier: list[str] = []
            for pid in frontier:
                for child in self.children(pid):
                    cid = str(child.id or "").strip()
                    if not cid or cid in seen:
                        continue
                    seen.add(cid)
                    if level is None or int(child.level) == int(level):
                        out.append(child)
                    next_frontier.append(cid)
            frontier = next_frontier
        return out

    def siblings(self, profile_id: str) -> list[SyndromeProfile]:
        me = self.by_id(profile_id)
        if me is None or not me.parent_id:
            return []
        return [
            item
            for item in self.children(me.parent_id)
            if str(item.id or "").strip() != str(me.id or "").strip()
        ]


_REPO_ROOT = Path(__file__).resolve().parents[3]
_SOURCE_PATH = Path(__file__).resolve().parent / "seed" / "syndrome_profiles.json"
_ARTIFACT_PATH = _REPO_ROOT / "data" / "cdss" / "knowledge" / "registry" / "syndrome_profiles.sqlite"
_GENERATED_GRAPH_PATH = _REPO_ROOT / "data" / "cdss" / "knowledge" / "generated" / "clinical_graph.jsonl"
_HIERARCHY_PATH = _REPO_ROOT / "data" / "cdss" / "knowledge" / "hierarchy.json"
_COMBO_OVERLAY_PATH = _REPO_ROOT / "data" / "cdss" / "knowledge" / "syndrome_combos.json"
_ARTIFACT_VERSION = "syndrome-registry-v2"
_LEARNING_DIR = _REPO_ROOT / "data" / "cdss" / "learning"
_EVIDENCE_MEMORY_PATH = _LEARNING_DIR / "evidence_memory.jsonl"
_PROFILE_STOPWORDS = {
    "acute",
    "adult",
    "case",
    "clinical",
    "diagnosis",
    "differential",
    "evaluation",
    "history",
    "patient",
    "presentation",
    "review",
    "rule",
    "state",
    "story",
    "syndrome",
    "testing",
    "woman",
    "man",
}


def _dedupe(values: list[str], *, limit: int | None = None) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for value in values:
        item = " ".join(str(value or "").split()).strip()
        key = item.lower()
        if not item or key in seen:
            continue
        seen.add(key)
        output.append(item)
        if limit and len(output) >= limit:
            break
    return output


def _canonical_label(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value or "").strip().lower()).strip("_")


def _tokenize(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]{4,}", str(value or "").lower())
        if token not in _PROFILE_STOPWORDS
    ]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = str(line or "").strip()
        if not raw:
            continue
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def _validated_learning_rows(path: Path = _EVIDENCE_MEMORY_PATH) -> list[dict]:
    rows: list[dict] = []
    for payload in _read_jsonl(path):
        if not bool(payload.get("validated", False)):
            continue
        review_tier = str(payload.get("review_tier") or payload.get("promotion_status") or "").strip().lower()
        if review_tier and review_tier not in {"validated", "promoted", "gold", "golden"}:
            continue
        linked = [
            _canonical_label(item)
            for item in list(payload.get("linked_hypotheses", []) or [])
            if _canonical_label(item)
        ]
        if not linked:
            continue
        stats = dict(payload.get("retrieval_stats", {}) or {})
        query_encoder = str(stats.get("query_encoder_used", "") or "").strip().lower()
        specificity = float(stats.get("specificity_gain", 0.0) or 0.0)
        citation_alignment = float(stats.get("citation_alignment", 0.0) or 0.0)
        novelty_gain = float(stats.get("novelty_gain", 0.0) or 0.0)
        query_hygiene = float(stats.get("query_hygiene_score", 1.0) or 1.0)
        hazard_leakage_rate = float(stats.get("hazard_leakage_rate", 0.0) or 0.0)
        hazard_leakage_detected = bool(stats.get("hazard_leakage_detected", False))
        retrieved_count = int(stats.get("retrieved_count", 0) or 0)
        quality = round((specificity * 0.45) + (citation_alignment * 0.45) + (novelty_gain * 0.10), 2)
        if hazard_leakage_detected or hazard_leakage_rate > 0.0:
            continue
        if query_hygiene < 0.55:
            continue
        if specificity < 0.52 or citation_alignment < 0.56:
            continue
        if retrieved_count < 2 and citation_alignment < 0.8:
            continue
        if query_encoder == "heuristic_fallback" and citation_alignment < 0.72:
            continue
        payload = dict(payload)
        payload["_quality"] = quality
        payload["_linked_hypotheses"] = linked
        rows.append(payload)
    return rows


def _runtime_substrate_mode() -> str:
    raw = str(os.getenv("CDSS_RUNTIME_SUBSTRATE") or "").strip().lower()
    if raw in {"adaptive_only", "adaptive_first", "compatibility"}:
        return raw
    return "adaptive_first"


def _runtime_substrate_min_profiles() -> int:
    raw = str(os.getenv("CDSS_RUNTIME_SUBSTRATE_MIN_PROFILES") or "").strip()
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 24
    return max(1, value)


def _adaptive_learning_registry() -> SyndromeProfileRegistry:
    grouped: dict[str, list[dict]] = {}
    for row in _validated_learning_rows():
        for label in list(row.get("_linked_hypotheses", []) or []):
            key = _canonical_label(label)
            if not key:
                continue
            grouped.setdefault(key, []).append(row)

    profiles: list[SyndromeProfile] = []
    for label, rows in grouped.items():
        payload = _synthesize_learning_profile(label, rows, existing=None)
        if not payload:
            continue
        payload["source_manifest"] = "adaptive_evidence_memory"
        payload["source_ref"] = str(_EVIDENCE_MEMORY_PATH)
        payload["artifact_version"] = "adaptive-learning-v1"
        profiles.append(SyndromeProfile.model_validate(payload))

    profiles.sort(key=lambda item: (not bool(item.must_not_miss), item.label))
    return SyndromeProfileRegistry(
        profiles=profiles,
        artifact_path="adaptive://learning-memory",
        source_sha256="adaptive-learning-memory",
        built_at=datetime.now(timezone.utc).isoformat(),
    )


def _generated_graph_registry(path: Path = _GENERATED_GRAPH_PATH) -> SyndromeProfileRegistry:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _log.warning("[REGISTRY] Cannot create generated graph directory %s: %s", path.parent, e)
    profiles: list[SyndromeProfile] = []
    for payload in _read_jsonl(path):
        if not isinstance(payload, dict):
            continue
        node_type = str(payload.get("node_type") or payload.get("kind") or "profile").strip().lower()
        if node_type not in {"profile", "disease", "syndrome_profile"}:
            continue
        label = _canonical_label(str(payload.get("id") or payload.get("label") or ""))
        if not label:
            continue
        row = dict(payload)
        row["id"] = label
        row["label"] = label
        row["source_manifest"] = str(row.get("source_manifest") or "generated_clinical_graph")
        row["source_ref"] = str(row.get("source_ref") or path)
        row["artifact_version"] = str(row.get("artifact_version") or "generated-graph-v1")
        try:
            profiles.append(SyndromeProfile.model_validate(row))
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue

    return SyndromeProfileRegistry(
        profiles=profiles,
        artifact_path=str(path),
        source_sha256="generated-clinical-graph",
        built_at=datetime.now(timezone.utc).isoformat(),
    )


def _merge_registry_profiles(primary: SyndromeProfileRegistry, secondary: SyndromeProfileRegistry) -> SyndromeProfileRegistry:
    merged: list[SyndromeProfile] = []
    seen: set[str] = set()
    for item in list(primary.profiles) + list(secondary.profiles):
        key = str(item.id or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return SyndromeProfileRegistry(
        profiles=merged,
        artifact_path=primary.artifact_path or secondary.artifact_path,
        source_sha256=primary.source_sha256 or secondary.source_sha256,
        built_at=primary.built_at or secondary.built_at,
    )


def _summary_fragments(summary: str) -> list[str]:
    fragments = re.split(r"[;,]", str(summary or ""))
    return _dedupe(
        [
            fragment.strip()
            for fragment in fragments
            if 12 <= len(fragment.strip()) <= 120
        ],
        limit=6,
    )


def _usable_query(query: str) -> bool:
    text = " ".join(str(query or "").split()).strip()
    return len(_tokenize(text)) >= 3 and 16 <= len(text) <= 160


def _synthesize_learning_profile(label: str, rows: list[dict], existing: dict | None = None) -> dict | None:
    if not label or label.endswith("_process"):
        return None
    if not rows:
        return None
    max_quality = max(float(item.get("_quality", 0.0) or 0.0) for item in rows)
    if existing is None and len(rows) < 2 and max_quality < 0.58:
        return None

    summary_fragments: list[str] = []
    cue_terms: list[str] = []
    challenge_queries: list[str] = []
    high_acuity = False
    hemodynamic_instability = False
    role_votes: list[str] = []
    source_resolution_votes: list[bool] = []
    for row in rows:
        summary_fragments.extend(_summary_fragments(str(row.get("summary", "") or "")))
        challenge_queries.extend(
            query for query in list(row.get("queries", []) or [])[:8] if _usable_query(str(query or ""))
        )
        role = str(row.get("clinical_role") or "").strip().lower()
        if role:
            role_votes.append(role)
        resolved = row.get("source_disease_resolved", None)
        if isinstance(resolved, bool):
            source_resolution_votes.append(resolved)
        signature = [str(item or "").strip() for item in list(row.get("state_signature", []) or [])[:24]]
        cue_terms.extend(
            item.replace("_", " ")
            for item in signature
            if item and not item.endswith("_profile") and len(_tokenize(item.replace("_", " "))) >= 1
        )
        for item in signature:
            lowered = item.lower()
            if lowered in {"tachycardic_profile", "hypotensive_profile"}:
                high_acuity = True
            if lowered == "hypotensive_profile":
                hemodynamic_instability = True
        for ref in list(row.get("evidence_refs", []) or [])[:8]:
            title = str((ref or {}).get("title", "") or "").strip()
            if 8 <= len(title) <= 120:
                cue_terms.append(title)

    cue_lexicon = _dedupe([*summary_fragments, *cue_terms], limit=12)
    evidence_needs = [f"objective_discriminator_for_{label}", "objective_vitals"]
    if high_acuity:
        evidence_needs.append("high_risk_rule_out_testing")
    if hemodynamic_instability:
        evidence_needs.append("hemodynamic_assessment")
    evidence_needs = _dedupe(evidence_needs, limit=6)
    merged = dict(existing or {})
    clinical_role = max(set(role_votes), key=role_votes.count) if role_votes else ""
    source_disease_resolved: bool | None = None
    if source_resolution_votes:
        positives = sum(1 for item in source_resolution_votes if item)
        negatives = sum(1 for item in source_resolution_votes if not item)
        source_disease_resolved = positives >= negatives
    if source_disease_resolved is False:
        evidence_needs.append("underlying_cause_resolution")
        evidence_needs = _dedupe(evidence_needs, limit=6)
    merged.update(
        {
            "id": label,
            "label": label,
            "summary": str(
                merged.get("summary")
                or f"Adaptive learned profile for {label.replace('_', ' ')} grounded by validated evidence-memory cases."
            ),
            "source_manifest": str(merged.get("source_manifest", "") or _EVIDENCE_MEMORY_PATH.name),
            "source_ref": str(merged.get("source_ref", "") or "validated_evidence_memory"),
            "cue_lexicon": _dedupe([*list(merged.get("cue_lexicon", []) or []), *cue_lexicon], limit=16),
            "evidence_needs": _dedupe([*list(merged.get("evidence_needs", []) or []), *evidence_needs], limit=8),
            "challenge_queries": _dedupe(
                [*list(merged.get("challenge_queries", []) or []), *challenge_queries],
                limit=8,
            ),
            "immediate_actions": _dedupe(list(merged.get("immediate_actions", []) or []), limit=6),
            "must_not_miss": bool(merged.get("must_not_miss", False) or (hemodynamic_instability and max_quality >= 0.55)),
            "dangerous_if_missed": bool(
                merged.get("dangerous_if_missed", False) or (hemodynamic_instability and max_quality >= 0.55)
            ),
            "dangerous_if_treated_as": str(merged.get("dangerous_if_treated_as", "") or ""),
            "unsafe_interventions": _dedupe(list(merged.get("unsafe_interventions", []) or []), limit=6),
            "trigger_requirements": _dedupe(list(merged.get("trigger_requirements", []) or []), limit=6),
            "trigger_exclusions": _dedupe(list(merged.get("trigger_exclusions", []) or []), limit=6),
        }
    )
    if clinical_role and not str(merged.get("clinical_role") or "").strip():
        merged["clinical_role"] = clinical_role
    if merged.get("source_disease_resolved", None) is None and source_disease_resolved is not None:
        merged["source_disease_resolved"] = source_disease_resolved
    return merged


def _merge_learning_profiles(base_profiles: list[dict]) -> list[dict]:
    profiles_by_id: dict[str, dict] = {
        str(item.get("id", "") or ""): dict(item)
        for item in list(base_profiles or [])
        if str(item.get("id", "") or "").strip()
    }
    grouped: dict[str, list[dict]] = {}
    for row in _validated_learning_rows():
        for label in list(row.get("_linked_hypotheses", []) or []):
            grouped.setdefault(label, []).append(row)
    for label, rows in grouped.items():
        learned = _synthesize_learning_profile(label, rows, profiles_by_id.get(label))
        if learned:
            profiles_by_id[label] = learned
    return [profiles_by_id[key] for key in sorted(profiles_by_id)]


def _source_payload(source_path: Path = _SOURCE_PATH) -> dict:
    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("syndrome source manifest must be an object")
    payload.setdefault("profiles", [])
    payload.setdefault(
        "sources",
        [
            {
                "source_id": "seed_curated_manifest",
                "kind": "curated_manifest",
                "citation": f"file://{source_path.name}",
            }
        ],
    )
    if _EVIDENCE_MEMORY_PATH.exists():
        payload["sources"] = [
            *list(payload.get("sources", []) or []),
            {
                "source_id": "validated_evidence_memory",
                "kind": "validated_evidence_memory",
                "citation": f"file://{_EVIDENCE_MEMORY_PATH.relative_to(_REPO_ROOT).as_posix()}",
            },
        ]
    payload["profiles"] = _merge_learning_profiles(list(payload.get("profiles", []) or []))
    return payload


def _source_sha256(source_path: Path = _SOURCE_PATH) -> str:
    return hashlib.sha256(_json_text(_source_payload(source_path)).encode("utf-8")).hexdigest()


def _ensure_artifact_parent(artifact_path: Path) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)


def _json_text(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_syndrome_registry_artifact(
    source_path: Path = _SOURCE_PATH,
    artifact_path: Path = _ARTIFACT_PATH,
) -> Path:
    """Materialize the syndrome registry into a generated SQLite artifact."""

    payload = _source_payload(source_path)
    source_sha = _source_sha256(source_path)
    _ensure_artifact_parent(artifact_path)
    if artifact_path.exists():
        artifact_path.unlink()

    conn = sqlite3.connect(str(artifact_path))
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute(
            """
            CREATE TABLE metadata (
                artifact_version TEXT NOT NULL,
                source_path TEXT NOT NULL,
                source_sha256 TEXT NOT NULL,
                built_at TEXT NOT NULL,
                source_manifest TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE syndrome_profiles (
                id TEXT PRIMARY KEY,
                label TEXT NOT NULL,
                summary TEXT NOT NULL,
                cue_lexicon TEXT NOT NULL,
                evidence_needs TEXT NOT NULL,
                challenge_queries TEXT NOT NULL,
                immediate_actions TEXT NOT NULL,
                must_not_miss INTEGER NOT NULL,
                dangerous_if_missed INTEGER NOT NULL,
                dangerous_if_treated_as TEXT NOT NULL,
                unsafe_interventions TEXT NOT NULL,
                trigger_requirements TEXT NOT NULL,
                trigger_exclusions TEXT NOT NULL,
                source_manifest TEXT NOT NULL,
                source_ref TEXT NOT NULL,
                artifact_version TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            INSERT INTO metadata (
                artifact_version,
                source_path,
                source_sha256,
                built_at,
                source_manifest
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (
                _ARTIFACT_VERSION,
                str(source_path),
                source_sha,
                datetime.now(timezone.utc).isoformat(),
                _json_text(payload.get("sources", [])),
            ),
        )

        for raw_profile in list(payload.get("profiles", []) or []):
            profile = SyndromeProfile.model_validate(
                {
                    **dict(raw_profile or {}),
                    "source_manifest": str(dict(raw_profile or {}).get("source_manifest") or source_path.name),
                    "source_ref": str(
                        dict(raw_profile or {}).get("source_ref")
                        or (payload.get("sources") or [{}])[0].get("source_id")
                        or "seed_curated_manifest"
                    ),
                    "artifact_version": _ARTIFACT_VERSION,
                }
            )
            conn.execute(
                """
                INSERT INTO syndrome_profiles (
                    id,
                    label,
                    summary,
                    cue_lexicon,
                    evidence_needs,
                    challenge_queries,
                    immediate_actions,
                    must_not_miss,
                    dangerous_if_missed,
                    dangerous_if_treated_as,
                    unsafe_interventions,
                    trigger_requirements,
                    trigger_exclusions,
                    source_manifest,
                    source_ref,
                    artifact_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    profile.id,
                    profile.label,
                    profile.summary,
                    _json_text(profile.cue_lexicon),
                    _json_text(profile.evidence_needs),
                    _json_text(profile.challenge_queries),
                    _json_text(profile.immediate_actions),
                    int(profile.must_not_miss),
                    int(profile.dangerous_if_missed),
                    profile.dangerous_if_treated_as,
                    _json_text(profile.unsafe_interventions),
                    _json_text(profile.trigger_requirements),
                    _json_text(profile.trigger_exclusions),
                    profile.source_manifest,
                    profile.source_ref,
                    profile.artifact_version,
                ),
            )
        conn.commit()
    finally:
        conn.close()
    return artifact_path


def _artifact_is_fresh(source_path: Path = _SOURCE_PATH, artifact_path: Path = _ARTIFACT_PATH) -> bool:
    if not artifact_path.exists():
        return False
    try:
        conn = sqlite3.connect(str(artifact_path))
        row = conn.execute(
            "SELECT artifact_version, source_sha256 FROM metadata LIMIT 1"
        ).fetchone()
    except Exception:
        return False
    finally:
        try:
            conn.close()
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass
    if not row:
        return False
    artifact_version, source_sha = row
    return str(artifact_version or "") == _ARTIFACT_VERSION and str(source_sha or "") == _source_sha256(source_path)


def ensure_syndrome_registry_artifact(
    source_path: Path = _SOURCE_PATH,
    artifact_path: Path = _ARTIFACT_PATH,
) -> Path:
    """Ensure the generated SQLite artifact exists and is in sync with its source manifest."""

    if not _artifact_is_fresh(source_path, artifact_path):
        build_syndrome_registry_artifact(source_path, artifact_path)
    return artifact_path


def _load_profiles_from_artifact(artifact_path: Path) -> tuple[list[SyndromeProfile], dict[str, str]]:
    conn = sqlite3.connect(str(artifact_path))
    conn.row_factory = sqlite3.Row
    try:
        metadata_row = conn.execute(
            "SELECT artifact_version, source_sha256, built_at FROM metadata LIMIT 1"
        ).fetchone()
        profile_rows = conn.execute(
            """
            SELECT
                id,
                label,
                summary,
                cue_lexicon,
                evidence_needs,
                challenge_queries,
                immediate_actions,
                must_not_miss,
                dangerous_if_missed,
                dangerous_if_treated_as,
                unsafe_interventions,
                trigger_requirements,
                trigger_exclusions,
                source_manifest,
                source_ref,
                artifact_version
            FROM syndrome_profiles
            ORDER BY id
            """
        ).fetchall()
    finally:
        conn.close()

    profiles = [
        SyndromeProfile.model_validate(
            {
                "id": row["id"],
                "label": row["label"],
                "summary": row["summary"],
                "cue_lexicon": json.loads(row["cue_lexicon"]),
                "evidence_needs": json.loads(row["evidence_needs"]),
                "challenge_queries": json.loads(row["challenge_queries"]),
                "immediate_actions": json.loads(row["immediate_actions"]),
                "must_not_miss": bool(row["must_not_miss"]),
                "dangerous_if_missed": bool(row["dangerous_if_missed"]),
                "dangerous_if_treated_as": row["dangerous_if_treated_as"],
                "unsafe_interventions": json.loads(row["unsafe_interventions"]),
                "trigger_requirements": json.loads(row["trigger_requirements"]),
                "trigger_exclusions": json.loads(row["trigger_exclusions"]),
                "source_manifest": row["source_manifest"],
                "source_ref": row["source_ref"],
                "artifact_version": row["artifact_version"],
            }
        )
        for row in profile_rows
    ]
    metadata = {
        "artifact_version": str(metadata_row["artifact_version"] if metadata_row else ""),
        "source_sha256": str(metadata_row["source_sha256"] if metadata_row else ""),
        "built_at": str(metadata_row["built_at"] if metadata_row else ""),
    }
    return profiles, metadata


def _runtime_substrate_mode() -> str:
    raw = os.getenv("CDSS_RUNTIME_SUBSTRATE", "generated").strip().lower()
    if raw == "compatibility":
        return "compatibility"
    return "generated"

def _load_hierarchy_overlay(path: Path = _HIERARCHY_PATH) -> dict[str, dict]:
    """Read optional hierarchy.json and return {profile_id: {level, parent_id, ...}}.

    Structure:
        {
          "families": [{"id": "...", "label": "...", "summary": "..."}],
          "genera":   [{"id": "...", "parent_id": "<family_id>", "label": "...", ...}],
          "assignments": {"<species_id>": "<genus_id_or_family_id>"}
        }

    Missing file → empty overlay (legacy flat behavior preserved).
    Virtual level-0/1 profiles (families/genera) get synthesized on-the-fly.
    """
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    overlay: dict[str, dict] = {}
    for item in list(data.get("families", []) or []):
        fid = _canonical_label(str((item or {}).get("id", "") or ""))
        if not fid:
            continue
        overlay[fid] = {
            "level": 0,
            "parent_id": "",
            "label": str((item or {}).get("label") or fid),
            "summary": str((item or {}).get("summary") or ""),
            "_virtual": True,
        }
    for item in list(data.get("genera", []) or []):
        gid = _canonical_label(str((item or {}).get("id", "") or ""))
        if not gid:
            continue
        overlay[gid] = {
            "level": 1,
            "parent_id": _canonical_label(str((item or {}).get("parent_id", "") or "")),
            "label": str((item or {}).get("label") or gid),
            "summary": str((item or {}).get("summary") or ""),
            "_virtual": True,
        }
    assignments = data.get("assignments") or {}
    if isinstance(assignments, dict):
        for raw_sid, raw_pid in assignments.items():
            sid = _canonical_label(str(raw_sid or ""))
            pid = _canonical_label(str(raw_pid or ""))
            if not sid:
                continue
            overlay[sid] = {
                "level": 2,
                "parent_id": pid,
                "_virtual": False,
            }
    return overlay


def _apply_hierarchy_overlay(registry: SyndromeProfileRegistry) -> SyndromeProfileRegistry:
    overlay = _load_hierarchy_overlay()
    if not overlay:
        return registry
    updated: list[SyndromeProfile] = []
    existing_ids: set[str] = set()
    for profile in registry.profiles:
        meta = overlay.get(profile.id)
        if meta and not meta.get("_virtual", False):
            updated.append(
                profile.model_copy(update={
                    "level": int(meta.get("level", profile.level)),
                    "parent_id": str(meta.get("parent_id", profile.parent_id) or ""),
                })
            )
        else:
            updated.append(profile)
        existing_ids.add(profile.id)
    # Synthesize virtual family/genus profiles absent from seed registry.
    for pid, meta in overlay.items():
        if not meta.get("_virtual", False) or pid in existing_ids:
            continue
        try:
            updated.append(SyndromeProfile.model_validate({
                "id": pid,
                "label": str(meta.get("label", pid)),
                "summary": str(meta.get("summary", "")),
                "level": int(meta.get("level", 2)),
                "parent_id": str(meta.get("parent_id", "") or ""),
                "source_manifest": "hierarchy_overlay",
                "source_ref": str(_HIERARCHY_PATH),
                "artifact_version": "hierarchy-overlay-v1",
            }))
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue
    return SyndromeProfileRegistry(
        profiles=updated,
        artifact_path=registry.artifact_path,
        source_sha256=registry.source_sha256,
        built_at=registry.built_at,
    )


def _load_combo_overlay(path: Path = _COMBO_OVERLAY_PATH) -> dict[str, dict]:
    """Read optional ``syndrome_combos.json`` overlay.

    Schema (top-level dict, profile_id → entry)::

        {
          "<profile_id>": {
            "label":   "human-readable diagnosis name",   # optional
            "summary": "short rationale",                 # optional
            "must_not_miss":      true,                    # optional, default false
            "dangerous_if_missed": true,                   # optional, default false
            "pathognomonic_combos": [SyndromeCombo, ...],
            "chained_triggers":    [ChainedTrigger, ...]
          },
          ...
        }

    The overlay can both decorate existing profiles (matching id) and add
    *synthetic* profiles for diagnoses absent from the seed registry. This
    matches the pattern used by ``hierarchy.json`` for virtual families.

    Missing file → empty overlay (legacy zero-combo behaviour preserved).
    """

    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    overlay: dict[str, dict] = {}
    for raw_id, raw_payload in data.items():
        pid = _canonical_label(str(raw_id or ""))
        if not pid or not isinstance(raw_payload, dict):
            continue
        combos: list[SyndromeCombo] = []
        for raw in list(raw_payload.get("pathognomonic_combos") or []):
            if not isinstance(raw, dict):
                continue
            try:
                combos.append(SyndromeCombo.model_validate(raw))
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                continue
        chains: list[ChainedTrigger] = []
        for raw in list(raw_payload.get("chained_triggers") or []):
            if not isinstance(raw, dict):
                continue
            try:
                chains.append(ChainedTrigger.model_validate(raw))
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                continue
        if not combos and not chains:
            continue
        overlay[pid] = {
            "label": str(raw_payload.get("label") or pid),
            "summary": str(raw_payload.get("summary") or ""),
            "must_not_miss": bool(raw_payload.get("must_not_miss", False)),
            "dangerous_if_missed": bool(raw_payload.get("dangerous_if_missed", False)),
            "pathognomonic_combos": combos,
            "chained_triggers": chains,
        }
    return overlay


def _apply_combo_overlay(registry: SyndromeProfileRegistry) -> SyndromeProfileRegistry:
    overlay = _load_combo_overlay()
    if not overlay:
        return registry
    updated: list[SyndromeProfile] = []
    existing_ids: set[str] = set()
    for profile in registry.profiles:
        meta = overlay.get(profile.id)
        if not meta:
            updated.append(profile)
        else:
            updated.append(
                profile.model_copy(
                    update={
                        "pathognomonic_combos": list(meta.get("pathognomonic_combos") or []),
                        "chained_triggers": list(meta.get("chained_triggers") or []),
                    }
                )
            )
        existing_ids.add(profile.id)
    # Synthesize standalone profiles for combo entries whose id is not in the
    # seed registry. Lets clinical reviewers add new symbolic-promotion entries
    # (e.g. infective_endocarditis, horner_syndrome) without touching the seed.
    for pid, meta in overlay.items():
        if pid in existing_ids:
            continue
        try:
            updated.append(
                SyndromeProfile.model_validate(
                    {
                        "id": pid,
                        "label": str(meta.get("label") or pid),
                        "summary": str(meta.get("summary") or ""),
                        "must_not_miss": bool(meta.get("must_not_miss", False)),
                        "dangerous_if_missed": bool(meta.get("dangerous_if_missed", False)),
                        "pathognomonic_combos": list(meta.get("pathognomonic_combos") or []),
                        "chained_triggers": list(meta.get("chained_triggers") or []),
                        "level": 2,
                        "parent_id": "",
                        "source_manifest": "combo_overlay",
                        "source_ref": str(_COMBO_OVERLAY_PATH),
                        "artifact_version": "combo-overlay-v1",
                    }
                )
            )
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue
    return SyndromeProfileRegistry(
        profiles=updated,
        artifact_path=registry.artifact_path,
        source_sha256=registry.source_sha256,
        built_at=registry.built_at,
    )


@lru_cache(maxsize=1)
def load_syndrome_registry() -> SyndromeProfileRegistry:
    """Load registry with a graceful fallback chain.

    Priority: generated clinical graph → adaptive learning overlay → SQLite seed artifact.
    Each tier is tried in order; the first non-empty result wins.  Merging stops at the
    first tier that satisfies the minimum-profile threshold so the fastest path is used
    in production (generated graph), while the system degrades gracefully if that file
    has not yet been built.
    """
    generated = _generated_graph_registry()
    adaptive = _adaptive_learning_registry()
    merged = _merge_registry_profiles(generated, adaptive) if generated.profiles else adaptive

    if len(merged.profiles) >= _runtime_substrate_min_profiles():
        return _apply_combo_overlay(_apply_hierarchy_overlay(merged))

    # Generated graph missing or too sparse → fall back to the SQLite seed artifact.
    try:
        artifact_path = ensure_syndrome_registry_artifact()
        seed_profiles, metadata = _load_profiles_from_artifact(artifact_path)
        seed_registry = SyndromeProfileRegistry(
            profiles=seed_profiles,
            artifact_path=str(artifact_path),
            source_sha256=metadata.get("source_sha256", ""),
            built_at=metadata.get("built_at", ""),
        )
        if seed_registry.profiles:
            # Overlay adaptive learning on top of seed so learned entries are included.
            combined = _merge_registry_profiles(seed_registry, adaptive) if adaptive.profiles else seed_registry
            return _apply_combo_overlay(_apply_hierarchy_overlay(combined))
    except Exception as e:
        _log.error(
            "[REGISTRY] Seed artifact load failed — running with partial registry (%d profiles): %s",
            len(merged.profiles), e, exc_info=True,
        )

    # Last resort: return whatever we have (may be empty).
    return _apply_combo_overlay(_apply_hierarchy_overlay(merged))
