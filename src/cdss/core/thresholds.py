"""Centralized clinical-threshold loader.

Loads `config/clinical_thresholds.json` once and caches the parsed structure.
Each call site reads its threshold via the typed accessor `get(path, default)`.
The accessor *always* returns a numeric value: if the file is missing,
malformed, or the requested path is absent, the supplied `default` is used and
the substitution is logged once per missing key. This guarantees that removing
or breaking the file cannot crash the pipeline — the system silently falls
back to the in-code defaults that were the original implementation.

Threshold values must be numeric (int or float). The JSON file may store each
threshold either as a bare number (legacy form) or as an object with a `value`
field plus optional `source`/`last_reviewed` metadata (preferred form). Both
are accepted; only the numeric `value` is used at runtime.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

_LOG = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: dict[str, Any] | None = None
_MISSING_KEYS_LOGGED: set[str] = set()


def _config_path() -> Path:
    override = os.environ.get("CDSS_CLINICAL_THRESHOLDS_PATH")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "config" / "clinical_thresholds.json"


def _load() -> dict[str, Any]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        path = _config_path()
        try:
            raw = path.read_text(encoding="utf-8")
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise TypeError("clinical_thresholds.json must contain a JSON object at the top level")
            _CACHE = data
        except FileNotFoundError:
            _LOG.info("clinical_thresholds.json not found at %s; using in-code defaults", path)
            _CACHE = {}
        except (json.JSONDecodeError, TypeError, OSError) as exc:
            _LOG.warning("clinical_thresholds.json failed to load (%s); using in-code defaults", exc)
            _CACHE = {}
    return _CACHE


def reload() -> None:
    """Force re-read of the thresholds file. Mainly for tests."""
    global _CACHE, _MISSING_KEYS_LOGGED
    with _LOCK:
        _CACHE = None
        _MISSING_KEYS_LOGGED = set()


def _resolve(path: str) -> Any | None:
    data = _load()
    cursor: Any = data
    for part in path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _coerce_numeric(node: Any) -> float | int | None:
    if isinstance(node, bool):
        return None
    if isinstance(node, (int, float)):
        return node
    if isinstance(node, dict):
        value = node.get("value")
        if isinstance(value, bool):
            return None
        if isinstance(value, (int, float)):
            return value
    return None


def get(path: str, default: float | int) -> float | int:
    """Look up a threshold by dotted path; return `default` if missing."""
    node = _resolve(path)
    coerced = _coerce_numeric(node) if node is not None else None
    if coerced is None:
        if path not in _MISSING_KEYS_LOGGED:
            _MISSING_KEYS_LOGGED.add(path)
            _LOG.debug("clinical threshold %s not in config; falling back to default %s", path, default)
        return default
    return coerced


def get_int(path: str, default: int) -> int:
    return int(get(path, default))


def get_float(path: str, default: float) -> float:
    return float(get(path, default))


def metadata(path: str) -> dict[str, Any]:
    """Return the full threshold node (value + source + last_reviewed) or empty dict."""
    node = _resolve(path)
    if isinstance(node, dict):
        return dict(node)
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        return {"value": node}
    return {}


__all__ = ["get", "get_int", "get_float", "metadata", "reload"]
