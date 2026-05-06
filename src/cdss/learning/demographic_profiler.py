"""Autonomous demographic risk profiler.

Architecture
------------
Rather than hardcoding demographic rules in Python, this module lets the LLM
derive structured demographic_risk_profile entries from the existing syndrome
knowledge (summary, cue_lexicon, challenge_queries).  Results are persisted in
a separate SQLite store — syndrome_profiles.json is never touched.

Two learning phases
-------------------
1. Bootstrap  — On first access (or when a syndrome changes), the LLM synthesises
               a demographic_risk_profile from the syndrome text.  Stored in
               data/cdss/knowledge/learned/demographic_profiles.sqlite.
2. Refinement — Every confirmed/rejected case outcome updates the confidence score
               for the relevant profile via update_from_case_outcome().  Over time
               the profile's weight reflects real clinical experience, not just
               LLM prior knowledge.

Integration points
------------------
- load_and_enrich_registry()  : called by registry consumers to get SyndromeProfile
                                objects with demographic_risk_profile pre-filled.
- get_demographic_profile()   : fast lookup by syndrome_id (uses module-level cache).
- update_from_case_outcome()  : called by promotion.py after each confirmed case.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.cdss.knowledge.registry import SyndromeProfile, SyndromeProfileRegistry

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[4]
_STORE_PATH = _REPO_ROOT / "data" / "cdss" / "knowledge" / "learned" / "demographic_profiles.sqlite"
_STORE_LOCK = threading.Lock()

# Module-level profile cache: {syndrome_id: profile_dict}
_PROFILE_CACHE: dict[str, dict] | None = None
_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Store bootstrap
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS demographic_profiles (
    syndrome_id      TEXT    PRIMARY KEY,
    profile_json     TEXT    NOT NULL,
    confidence       REAL    NOT NULL DEFAULT 0.5,
    synthesized_at   TEXT    NOT NULL,
    source_sha       TEXT    NOT NULL,
    case_count       INTEGER NOT NULL DEFAULT 0,
    confirmed_count  INTEGER NOT NULL DEFAULT 0
);
"""


def _ensure_store() -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _STORE_LOCK:
        conn = sqlite3.connect(str(_STORE_PATH))
        try:
            conn.execute(_DDL)
            conn.commit()
        finally:
            conn.close()


def _syndrome_sha(profile: "SyndromeProfile") -> str:
    """Stable hash of the parts of the profile the LLM reads for synthesis."""
    blob = json.dumps(
        {
            "summary": profile.summary,
            "cue_lexicon": profile.cue_lexicon[:16],
            "challenge_queries": profile.challenge_queries[:8],
            "must_not_miss": profile.must_not_miss,
            "dangerous_if_missed": profile.dangerous_if_missed,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# SQLite read / write helpers
# ---------------------------------------------------------------------------

def _load_all_from_store() -> dict[str, dict]:
    if not _STORE_PATH.exists():
        return {}
    with _STORE_LOCK:
        conn = sqlite3.connect(str(_STORE_PATH))
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                "SELECT syndrome_id, profile_json, confidence, source_sha FROM demographic_profiles"
            ).fetchall()
        finally:
            conn.close()
    out: dict[str, dict] = {}
    for row in rows:
        try:
            profile = json.loads(row["profile_json"])
            profile["_confidence"] = float(row["confidence"])
            profile["_source_sha"] = row["source_sha"]
            out[row["syndrome_id"]] = profile
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue
    return out


def _upsert_profile(syndrome_id: str, profile: dict, sha: str, confidence: float = 0.5) -> None:
    _ensure_store()
    payload = {k: v for k, v in profile.items() if not k.startswith("_")}
    with _STORE_LOCK:
        conn = sqlite3.connect(str(_STORE_PATH))
        try:
            conn.execute(
                """
                INSERT INTO demographic_profiles
                    (syndrome_id, profile_json, confidence, synthesized_at, source_sha)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(syndrome_id) DO UPDATE SET
                    profile_json   = excluded.profile_json,
                    confidence     = excluded.confidence,
                    synthesized_at = excluded.synthesized_at,
                    source_sha     = excluded.source_sha
                """,
                (
                    syndrome_id,
                    json.dumps(payload, ensure_ascii=False),
                    confidence,
                    datetime.now(timezone.utc).isoformat(),
                    sha,
                ),
            )
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# LLM synthesis
# ---------------------------------------------------------------------------

_SYNTHESIS_PROMPT = """\
You are a clinical decision support expert. Analyse the syndrome below and extract its demographic risk profile.

Syndrome ID   : {id}
Summary       : {summary}
Cue lexicon   : {cue_lexicon}
Challenge qs  : {challenge_queries}
Must-not-miss : {must_not_miss}
Dangerous     : {dangerous_if_missed}

Return ONLY a JSON object — no prose, no markdown fences.

Required keys:
{{
  "sex": "female" | "male" | "any",
  "age_min": <integer or null>,
  "age_max": <integer or null>,
  "hemodynamic_flags": [ "shock_index_high" | "shock_index_critical" | "hypotension" | "tachycardia" ],
  "complaint_domains": [ "abdominal" | "pelvic" | "chest" | "neuro" | "musculoskeletal" | "any" ],
  "constitutional_flags": ["<constitutional_marker>" | "any"],
  "priority": "must_not_miss" | "high" | "moderate" | "low",
  "rationale": "<one sentence clinical reasoning>"
}}

Rules:
- Use "any" for sex / complaint_domains if the syndrome affects all demographics.
- hemodynamic_flags and complaint_domains may be empty arrays [].
- age_min / age_max are null if there is no meaningful age restriction.
- priority = "must_not_miss" only for immediately life-threatening presentations.
- constitutional_flags: If this syndrome has a known body habitus or constitutional phenotype risk association (e.g., specific stature, weight, build, genetic phenotype), list the relevant markers as short token strings a clinician would use in free text. Use [] or ["any"] if body habitus is not a discriminating factor.
"""

_JSON_RE = __import__("re").compile(r"\{[\s\S]*\}", __import__("re").DOTALL)


def _synthesize_via_llm(profile: "SyndromeProfile", llm_client) -> dict | None:  # noqa: ANN001
    """Call the LLM to produce a demographic_risk_profile dict for one syndrome."""
    prompt = _SYNTHESIS_PROMPT.format(
        id=profile.id,
        summary=(profile.summary or "")[:300],
        cue_lexicon=json.dumps(profile.cue_lexicon[:12], ensure_ascii=False),
        challenge_queries=json.dumps(profile.challenge_queries[:6], ensure_ascii=False),
        must_not_miss=profile.must_not_miss,
        dangerous_if_missed=profile.dangerous_if_missed,
    )
    try:
        raw = llm_client.chat(
            system="You are a structured data extractor. Output only valid JSON.",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            stage="DEMO_PROFILE",
            json_mode=True,
        )
    except Exception as exc:
        logger.warning("demographic_profiler: LLM call failed for %s — %s", profile.id, exc)
        return None

    text = str(raw or "").strip()
    m = _JSON_RE.search(text)
    if not m:
        logger.warning("demographic_profiler: no JSON in LLM response for %s", profile.id)
        return None
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError as exc:
        logger.warning("demographic_profiler: JSON parse error for %s — %s", profile.id, exc)
        return None

    # Normalise and validate required keys
    validated: dict = {}
    validated["sex"] = str(data.get("sex") or "any").lower()
    if validated["sex"] not in ("female", "male", "any"):
        validated["sex"] = "any"

    for key in ("age_min", "age_max"):
        val = data.get(key)
        try:
            validated[key] = int(val) if val is not None else None
        except (TypeError, ValueError):
            validated[key] = None

    hemo_flags_raw = data.get("hemodynamic_flags") or []
    valid_hemo = {"shock_index_high", "shock_index_critical", "hypotension", "tachycardia"}
    validated["hemodynamic_flags"] = [f for f in hemo_flags_raw if f in valid_hemo]

    domains_raw = data.get("complaint_domains") or []
    valid_domains = {"abdominal", "pelvic", "chest", "neuro", "musculoskeletal", "any"}
    validated["complaint_domains"] = [d for d in domains_raw if d in valid_domains]

    priority = str(data.get("priority") or "moderate").lower()
    if priority not in ("must_not_miss", "high", "moderate", "low"):
        priority = "high" if profile.must_not_miss else "moderate"
    validated["priority"] = priority

    validated["rationale"] = str(data.get("rationale") or "")[:200]
    return validated


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_demographic_profile(syndrome_id: str) -> dict:
    """Return the learned demographic_risk_profile for a syndrome, or {}."""
    global _PROFILE_CACHE
    with _CACHE_LOCK:
        if _PROFILE_CACHE is None:
            _PROFILE_CACHE = _load_all_from_store()
    return _PROFILE_CACHE.get(syndrome_id, {})


def invalidate_cache() -> None:
    """Force reload from store on next access (call after synthesis runs)."""
    global _PROFILE_CACHE
    with _CACHE_LOCK:
        _PROFILE_CACHE = None


def ensure_profiles_built(
    registry: "SyndromeProfileRegistry",
    llm_client,  # noqa: ANN001
    *,
    force_resynthesis: bool = False,
) -> int:
    """Synthesise missing or stale demographic profiles for all registry syndromes.

    Runs once per server startup (or whenever the syndrome source changes).
    Profiles already in the store with a matching source_sha are skipped.

    Returns the number of profiles newly synthesised.
    """
    _ensure_store()
    existing = _load_all_from_store()
    synthesised = 0

    for profile in registry.profiles:
        sha = _syndrome_sha(profile)
        existing_entry = existing.get(profile.id)

        if not force_resynthesis and existing_entry and existing_entry.get("_source_sha") == sha:
            # Already up-to-date — skip
            continue

        result = _synthesize_via_llm(profile, llm_client)
        if result is None:
            # LLM failed: keep old entry if present; insert placeholder otherwise
            if not existing_entry:
                placeholder = {
                    "sex": "any",
                    "age_min": None,
                    "age_max": None,
                    "hemodynamic_flags": [],
                    "complaint_domains": [],
                    "priority": "high" if profile.must_not_miss else "moderate",
                    "rationale": "synthesis_failed",
                }
                _upsert_profile(profile.id, placeholder, sha, confidence=0.2)
                synthesised += 1
            continue

        initial_confidence = 0.65 if profile.must_not_miss else 0.5
        _upsert_profile(profile.id, result, sha, confidence=initial_confidence)
        synthesised += 1
        logger.info("demographic_profiler: synthesised profile for %s (priority=%s)", profile.id, result.get("priority"))

    if synthesised:
        invalidate_cache()
        logger.info("demographic_profiler: synthesised %d profile(s)", synthesised)

    return synthesised


def load_and_enrich_registry(
    registry: "SyndromeProfileRegistry",
) -> "SyndromeProfileRegistry":
    """Return registry with demographic_risk_profile injected into each SyndromeProfile.

    Does NOT call the LLM — only reads what is already in the learned store.
    Call ensure_profiles_built() first (once at startup) to guarantee freshness.
    """
    profiles_store = _load_all_from_store()
    if not profiles_store:
        return registry

    enriched = []
    for p in registry.profiles:
        learned = profiles_store.get(p.id, {})
        clean = {k: v for k, v in learned.items() if not k.startswith("_")}
        if clean:
            enriched.append(p.model_copy(update={"demographic_risk_profile": clean}))
        else:
            enriched.append(p)

    return registry.model_copy(update={"profiles": enriched})


def update_from_case_outcome(
    syndrome_id: str,
    *,
    confirmed: bool,
) -> None:
    """Refine confidence score for a syndrome based on a confirmed/rejected case outcome.

    - confirmed=True  : the demographic profile correctly identified a must-not-miss case
    - confirmed=False : the profile fired but the syndrome was ruled out

    Bayesian-style running average: confidence converges toward 1.0 for true positives
    and decays toward 0.0 for false positives over accumulated case observations.
    """
    if not _STORE_PATH.exists():
        return

    with _STORE_LOCK:
        conn = sqlite3.connect(str(_STORE_PATH))
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT confidence, case_count, confirmed_count FROM demographic_profiles WHERE syndrome_id = ?",
                (syndrome_id,),
            ).fetchone()
            if row is None:
                return
            old_conf = float(row["confidence"])
            n = int(row["case_count"]) + 1
            c = int(row["confirmed_count"]) + (1 if confirmed else 0)
            # Weighted update: new observations carry less weight as history accumulates
            # (Laplace-smoothed running average, cap at 50 observations)
            effective_n = min(n, 50)
            new_conf = round((old_conf * (effective_n - 1) + (1.0 if confirmed else 0.0)) / effective_n, 4)
            conn.execute(
                """
                UPDATE demographic_profiles
                SET confidence = ?, case_count = ?, confirmed_count = ?
                WHERE syndrome_id = ?
                """,
                (new_conf, n, c, syndrome_id),
            )
            conn.commit()
        finally:
            conn.close()

    # Invalidate in-memory cache so next get_demographic_profile() reflects update
    invalidate_cache()
    logger.debug(
        "demographic_profiler: updated %s confidence → %.3f (confirmed=%s, n=%d)",
        syndrome_id,
        new_conf,  # type: ignore[possibly-undefined]
        confirmed,
        n,  # type: ignore[possibly-undefined]
    )
