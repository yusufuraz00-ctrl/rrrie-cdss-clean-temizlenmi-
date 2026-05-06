"""In-process structured span emitter for the CDSS pipeline.

Provides a lightweight, dependency-free `Span` context manager that records
stage timing, parent/child nesting, llm-call counts, model wait time, and
errors. Spans are written as JSONL to `output/traces/<run_id>.jsonl` and are
also delivered to any registered subscriber callbacks so the WebSocket layer
can mirror them as `trace` events without owning two parallel sources of
truth.

This module deliberately avoids any external dependency (no OpenTelemetry).
An OTLP exporter can be layered on top later as a subscriber without changing
the call sites.

Threading model: each pipeline run is single-threaded inside the state
machine, but parallel swarm workers can emit child spans concurrently. A
threading-local stack tracks the current parent so concurrent spawns do not
crosslink their parents.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

_LOG = logging.getLogger(__name__)

_TRACE_DIR_ENV = "CDSS_TRACE_DIR"
_TRACE_DISABLE_ENV = "CDSS_TRACE_DISABLE"


@dataclass
class SpanRecord:
    span_id: str
    parent_id: str | None
    run_id: str
    name: str
    start_ts: float
    end_ts: float | None = None
    duration_ms: float | None = None
    status: str = "ok"
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    counters: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "run_id": self.run_id,
            "name": self.name,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error": self.error,
            "attributes": self.attributes,
            "counters": self.counters,
        }


_local = threading.local()
_subscribers_lock = threading.Lock()
_subscribers: list[Callable[[SpanRecord], None]] = []
_writers_lock = threading.Lock()
_writers: dict[str, Any] = {}


def _stack() -> list[SpanRecord]:
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = []
        _local.stack = stack
    return stack


def _run_id() -> str:
    rid = getattr(_local, "run_id", None)
    if rid is None:
        rid = uuid.uuid4().hex[:12]
        _local.run_id = rid
    return rid


def set_run_id(run_id: str | None) -> None:
    """Bind a run_id to the current thread. Pass None to clear."""
    if run_id is None:
        if hasattr(_local, "run_id"):
            del _local.run_id
        return
    _local.run_id = str(run_id)


def current_run_id() -> str:
    return _run_id()


def _trace_dir() -> Path:
    override = os.environ.get(_TRACE_DIR_ENV)
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[3] / "output" / "traces"


def _writer(run_id: str):
    if os.environ.get(_TRACE_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}:
        return None
    with _writers_lock:
        existing = _writers.get(run_id)
        if existing is not None:
            return existing
        try:
            base = _trace_dir()
            base.mkdir(parents=True, exist_ok=True)
            handle = (base / f"{run_id}.jsonl").open("a", encoding="utf-8")
        except OSError as exc:
            _LOG.debug("trace writer disabled for %s: %s", run_id, exc)
            _writers[run_id] = None
            return None
        _writers[run_id] = handle
        return handle


def close_run(run_id: str | None = None) -> None:
    """Close the trace file for the given run (or current run)."""
    rid = run_id or _run_id()
    with _writers_lock:
        handle = _writers.pop(rid, None)
    if handle is not None:
        try:
            handle.close()
        except OSError:
            pass


def subscribe(callback: Callable[[SpanRecord], None]) -> Callable[[], None]:
    """Register a callback that receives every completed span. Returns an unsubscribe handle."""
    with _subscribers_lock:
        _subscribers.append(callback)

    def _unsubscribe() -> None:
        with _subscribers_lock:
            if callback in _subscribers:
                _subscribers.remove(callback)

    return _unsubscribe


def _emit(record: SpanRecord) -> None:
    handle = _writer(record.run_id)
    if handle is not None:
        try:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
        except OSError as exc:
            _LOG.debug("trace write failed for %s: %s", record.span_id, exc)
    with _subscribers_lock:
        listeners = list(_subscribers)
    for cb in listeners:
        try:
            cb(record)
        except Exception:  # noqa: BLE001 — subscribers must never break the pipeline
            _LOG.exception("trace subscriber raised; ignoring")


@contextmanager
def span(name: str, **attributes: Any) -> Iterator[SpanRecord]:
    """Open a structured span. Use as a context manager; sets duration on exit."""
    parent = _stack()[-1] if _stack() else None
    record = SpanRecord(
        span_id=uuid.uuid4().hex[:16],
        parent_id=parent.span_id if parent else None,
        run_id=_run_id(),
        name=str(name or "span"),
        start_ts=time.time(),
        attributes=dict(attributes),
    )
    _stack().append(record)
    try:
        yield record
    except Exception as exc:  # noqa: BLE001
        record.status = "error"
        record.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record.end_ts = time.time()
        record.duration_ms = round((record.end_ts - record.start_ts) * 1000.0, 3)
        if _stack() and _stack()[-1] is record:
            _stack().pop()
        _emit(record)


def add_counter(record: SpanRecord, key: str, delta: float = 1.0) -> None:
    record.counters[key] = float(record.counters.get(key, 0.0)) + float(delta)


def set_attribute(record: SpanRecord, key: str, value: Any) -> None:
    record.attributes[key] = value


__all__ = [
    "SpanRecord",
    "span",
    "subscribe",
    "set_run_id",
    "current_run_id",
    "close_run",
    "add_counter",
    "set_attribute",
]
