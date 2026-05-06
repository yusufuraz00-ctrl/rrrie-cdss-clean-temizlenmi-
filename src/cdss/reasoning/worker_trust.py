"""Cross-case EMA worker reliability (W6.2 Module J.3).

Per (profile, family) trust τ ∈ [TRUST_FLOOR, 1.0]. Updated post-verification:

    quality   = 1.0 if profile's top-1 == verified_dx
              = 0.5 if verified_dx ∈ profile's top-3
              = 0.0 otherwise
    τ_new(profile, family) = ETA · quality + (1 − ETA) · τ_prev      # TeamMedAgents EMA

Cold-start at COLD_TRUST. Floor at TRUST_FLOOR so a bad streak never silences a
worker. Read sites: rank-fusion Trust-Weighted Borda (J.1) consumes τ.

Persistence
-----------

Stored as JSON at `data/cdss/learning/worker_reliability.json`:

    {
      "version": "1",
      "trust": {
        "<family>": { "<profile>": <float>, ... },
        ...
      }
    }

`update(profile, family, quality)` reads the JSON, applies one EMA step, writes
back atomically (tempfile + replace) so concurrent workers don't corrupt the
file. `load_worker_trust(family)` returns the per-profile τ map for the active
family — exactly the shape rank_fusion's `trust` kwarg expects.

Pure I/O + arithmetic. No LLM.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Mapping

logger = logging.getLogger(__name__)

_log = logging.getLogger("rrrie-cdss")

ETA = 0.7              # weight on new quality signal (TeamMedAgents)
COLD_TRUST = 0.8       # cold-start before any updates
TRUST_FLOOR = 0.4      # never let a bad streak silence a worker

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3]
    / "data" / "cdss" / "learning" / "worker_reliability.json"
)


def _coerce_float(v, default: float) -> float:
    try:
        f = float(v)
    except Exception:
        return default
    if f != f:  # NaN
        return default
    return f


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"version": "1", "trust": {}}
    except Exception as exc:
        _log.warning("[WORKER_TRUST] read failed (%s) — starting empty.", exc)
        return {"version": "1", "trust": {}}


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".worker_trust_", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            pass
        raise


def load_worker_trust(
    family: str = "",
    *,
    path: str | Path | None = None,
) -> dict[str, float]:
    """Return per-profile trust map for `family`. Empty family → cross-family fallback.

    Output shape matches rank_fusion.trust_weighted_borda's `trust` kwarg.
    Profiles not present default to COLD_TRUST at the call site (rank_fusion
    handles missing keys via `cold_trust=0.8`).
    """
    p = Path(path) if path is not None else _DEFAULT_PATH
    data = _read(p)
    trust_root: dict = data.get("trust") or {}
    fam = str(family or "").strip()
    if fam and fam in trust_root:
        return {k: _coerce_float(v, COLD_TRUST) for k, v in (trust_root[fam] or {}).items()}
    # Cross-family fallback: average per-profile trust across all families seen.
    if not trust_root:
        return {}
    accum: dict[str, list[float]] = {}
    for fam_block in trust_root.values():
        if not isinstance(fam_block, Mapping):
            continue
        for prof, val in fam_block.items():
            accum.setdefault(prof, []).append(_coerce_float(val, COLD_TRUST))
    return {prof: (sum(vals) / len(vals)) for prof, vals in accum.items() if vals}


def quality_score(top_k: list[str], verified_dx: str) -> float:
    """Map (top-3 hypothesis ids, gold dx id) to quality ∈ {0.0, 0.5, 1.0}."""
    if not top_k or not verified_dx:
        return 0.0
    truth = str(verified_dx).strip().lower()
    if not truth:
        return 0.0
    norm = [str(x).strip().lower() for x in top_k[:3]]
    if not norm:
        return 0.0
    if norm[0] == truth:
        return 1.0
    if truth in norm:
        return 0.5
    return 0.0


def update(
    profile: str,
    family: str,
    quality: float,
    *,
    path: str | Path | None = None,
) -> float:
    """One EMA step. Returns the post-update τ. Atomic on disk."""
    prof = str(profile or "").strip()
    fam = str(family or "").strip() or "_global"
    if not prof:
        return COLD_TRUST
    p = Path(path) if path is not None else _DEFAULT_PATH
    data = _read(p)
    trust_root = data.setdefault("trust", {})
    fam_block = trust_root.setdefault(fam, {})
    prev = _coerce_float(fam_block.get(prof, COLD_TRUST), COLD_TRUST)
    q = max(0.0, min(1.0, _coerce_float(quality, 0.0)))
    new = ETA * q + (1.0 - ETA) * prev
    new = max(TRUST_FLOOR, min(1.0, new))
    fam_block[prof] = round(new, 4)
    data["version"] = data.get("version", "1")
    try:
        _atomic_write(p, data)
    except Exception as exc:
        _log.warning("[WORKER_TRUST] write failed: %s", exc)
    return new


def update_panel(
    panel: list[str],
    family: str,
    quality_per_profile: Mapping[str, float],
    *,
    path: str | Path | None = None,
) -> dict[str, float]:
    """Batch-update τ for an entire panel. Returns post-update τ map for the
    family. One read + one write per call (cheaper than N updates)."""
    fam = str(family or "").strip() or "_global"
    p = Path(path) if path is not None else _DEFAULT_PATH
    data = _read(p)
    trust_root = data.setdefault("trust", {})
    fam_block = trust_root.setdefault(fam, {})
    for prof in panel:
        if not prof:
            continue
        q = max(0.0, min(1.0, _coerce_float(quality_per_profile.get(prof, 0.0), 0.0)))
        prev = _coerce_float(fam_block.get(prof, COLD_TRUST), COLD_TRUST)
        new = ETA * q + (1.0 - ETA) * prev
        new = max(TRUST_FLOOR, min(1.0, new))
        fam_block[prof] = round(new, 4)
    data["version"] = data.get("version", "1")
    try:
        _atomic_write(p, data)
    except Exception as exc:
        _log.warning("[WORKER_TRUST] write failed: %s", exc)
    return {k: _coerce_float(v, COLD_TRUST) for k, v in fam_block.items()}


__all__ = [
    "ETA",
    "COLD_TRUST",
    "TRUST_FLOOR",
    "load_worker_trust",
    "quality_score",
    "update",
    "update_panel",
]
