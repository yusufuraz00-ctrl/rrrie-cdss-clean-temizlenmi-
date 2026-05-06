"""Centralized async rate limiter + TTL cache for external API tools.

Prevents hitting rate limits on PubMed, OpenFDA, ClinicalTrials.gov etc.
Also provides a simple TTL cache to avoid redundant API calls within a session.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Any


class AsyncRateLimiter:
    """Token-bucket rate limiter for async API calls.

    Usage:
        limiter = AsyncRateLimiter(calls_per_second=3)
        async with limiter:
            resp = await client.get(...)
    """

    def __init__(self, calls_per_second: float = 3.0):
        self._min_interval = 1.0 / calls_per_second
        self._last_call = 0.0
        self._lock = asyncio.Lock()

    async def __aenter__(self):
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class TTLCache:
    """Simple in-memory TTL cache for API responses.

    Usage:
        cache = TTLCache(ttl_seconds=600)
        key = cache.make_key("pubmed", query="pneumonia")
        if cached := cache.get(key):
            return cached
        result = await actual_api_call(...)
        cache.set(key, result)
    """

    def __init__(self, ttl_seconds: int = 600, max_size: int = 200):
        self._ttl = ttl_seconds
        self._max_size = max_size
        self._store: dict[str, tuple[float, Any]] = {}

    @staticmethod
    def make_key(tool_name: str, **kwargs) -> str:
        """Create a deterministic cache key from tool name + parameters."""
        raw = json.dumps({"tool": tool_name, **kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def get(self, key: str) -> Any | None:
        """Get cached value if still valid, else None."""
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.time() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        """Store value with current timestamp. Evicts oldest if full."""
        if len(self._store) >= self._max_size:
            # Evict oldest entry
            oldest_key = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest_key]
        self._store[key] = (time.time(), value)

    def clear(self) -> None:
        """Clear all cached entries."""
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


# ── Global instances (one per tool category) ────────────────────────
# PubMed: 3 req/s without key, 10 req/s with key
pubmed_limiter = AsyncRateLimiter(calls_per_second=3.0)

# OpenFDA: ~40 req/min ≈ 0.67/s
openfda_limiter = AsyncRateLimiter(calls_per_second=0.6)

# ClinicalTrials.gov: conservative
clinicaltrials_limiter = AsyncRateLimiter(calls_per_second=2.0)

# WHO ICD-11: conservative
who_limiter = AsyncRateLimiter(calls_per_second=1.0)

# General web tools (Tavily, Wikipedia, MedlinePlus)
web_limiter = AsyncRateLimiter(calls_per_second=2.0)

# Shared TTL cache for all tools (10 min TTL)
tool_cache = TTLCache(ttl_seconds=600, max_size=300)
