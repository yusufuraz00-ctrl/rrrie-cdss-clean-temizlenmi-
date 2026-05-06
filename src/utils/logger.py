"""Structured logging with structlog — JSON-formatted output for RRRIE-CDSS."""

from __future__ import annotations

import logging
import sys

import structlog

_configured = False


def _configure_console_encoding() -> None:
    """Force UTF-8 console streams where the Python runtime supports it."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if not callable(reconfigure):
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


def setup_logging(log_level: str = "INFO") -> None:
    """Configure structlog with JSON output and standard library integration."""
    global _configured
    if _configured:
        return

    level = getattr(logging, log_level.upper(), logging.INFO)
    _configure_console_encoding()

    # Configure standard library root logger
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # structlog configuration
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    if not _configured:
        setup_logging()
    return structlog.get_logger(name)


# ---------------------------------------------------------------------------
# OpenTelemetry tracing helpers (merged from tracing.py)
# ---------------------------------------------------------------------------
import os as _os
import atexit as _atexit

_TRACE_PROVIDER_SETUP = False


def setup_tracing(service_name: str = "rrrie-cdss-agent") -> None:
    global _TRACE_PROVIDER_SETUP
    if _TRACE_PROVIDER_SETUP:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource, SERVICE_NAME
        endpoint = _os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318/v1/traces")
        resource = Resource.create({SERVICE_NAME: service_name})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        trace.set_tracer_provider(provider)
        _atexit.register(provider.shutdown)
        _TRACE_PROVIDER_SETUP = True
    except ImportError:
        pass


def get_tracer():
    try:
        from opentelemetry import trace
        return trace.get_tracer(__name__)
    except ImportError:
        return None
