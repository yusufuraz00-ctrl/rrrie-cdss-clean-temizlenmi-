"""Content-hash LRU cache for deterministic LLM responses.

Deterministic generations (temperature == 0, same model, same prompt, same
params) produce the same output every time. Across a benchmark of repeated
cases this duplicate work can account for a meaningful share of wall time.
This cache memoizes those outputs so a re-run returns instantly.

Usage:
    from src.cdss.runtime.response_cache import get_response_cache

    cache = get_response_cache()
    key = cache.make_key(model, messages, temperature=..., top_p=..., ...)
    hit = cache.get(key)
    if hit is not None:
        return hit
    result = call_llm(...)
    cache.set(key, result)

The cache is intentionally opt-in via the CDSS_RESPONSE_CACHE env flag so
existing deterministic-but-stateful behaviour is never changed without an
explicit signal from the operator.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Any

_DEFAULT_CAPACITY = 512
_DEFAULT_TTL_S = 60 * 60  # 1h
_ENV_FLAG = "CDSS_RESPONSE_CACHE"


class ResponseCache:
    """Thread-safe LRU cache keyed on sha256(model + prompt + params)."""

    def __init__(self, capacity: int = _DEFAULT_CAPACITY, ttl_s: float = _DEFAULT_TTL_S) -> None:
        self._capacity = int(max(1, capacity))
        self._ttl_s = float(max(1.0, ttl_s))
        self._store: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def enabled() -> bool:
        return str(os.environ.get(_ENV_FLAG, "") or "").strip().lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def make_key(
        model: str,
        messages: list[dict[str, Any]] | str,
        *,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int | None = None,
        max_tokens: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> str:
        try:
            serialized_msgs = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
        except Exception:
            serialized_msgs = str(messages)
        payload = {
            "model": str(model or ""),
            "messages": serialized_msgs,
            "temperature": round(float(temperature or 0.0), 4),
            "top_p": round(float(top_p or 1.0), 4),
            "top_k": int(top_k) if top_k is not None else None,
            "max_tokens": int(max_tokens) if max_tokens is not None else None,
            "extra": extra or {},
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self.misses += 1
                return None
            ts, value = entry
            if now - ts > self._ttl_s:
                self._store.pop(key, None)
                self.misses += 1
                return None
            # LRU touch
            self._store.move_to_end(key)
            self.hits += 1
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self._capacity:
                self._store.popitem(last=False)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "size": len(self._store),
                "capacity": self._capacity,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": (self.hits / total) if total else 0.0,
            }

    def clear(self) -> None:
        with self._lock:
            self._store.clear()


_singleton: ResponseCache | None = None
_singleton_lock = threading.Lock()


def get_response_cache() -> ResponseCache:
    global _singleton
    if _singleton is None:
        with _singleton_lock:
            if _singleton is None:
                _singleton = ResponseCache()
    return _singleton
