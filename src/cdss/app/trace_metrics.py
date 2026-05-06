"""Shared helpers for trace timing and token-stage metric aggregation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from src.cdss.contracts.models import DecisionPacket


def compute_stage_metrics_from_trace(
    packet: DecisionPacket,
    stage_map: dict[str, str],
    *,
    stage_bucket_tokens_as_int: bool = False,
) -> tuple[dict[str, dict[str, float]], float, dict[str, Any]]:
    stage_metrics: dict[str, dict[str, float]] = {
        "R1": {"time": 0.0, "tokens": 0.0, "tok_s": 0.0, "model_wait_s": 0.0, "prompt_chars": 0.0, "output_chars": 0.0, "cache_hits": 0.0, "web_hits": 0.0, "llm_calls": 0.0},
        "R2": {"time": 0.0, "tokens": 0.0, "tok_s": 0.0, "model_wait_s": 0.0, "prompt_chars": 0.0, "output_chars": 0.0, "cache_hits": 0.0, "web_hits": 0.0, "llm_calls": 0.0},
        "R3": {"time": 0.0, "tokens": 0.0, "tok_s": 0.0, "model_wait_s": 0.0, "prompt_chars": 0.0, "output_chars": 0.0, "cache_hits": 0.0, "web_hits": 0.0, "llm_calls": 0.0},
        "IE": {"time": 0.0, "tokens": 0.0, "tok_s": 0.0, "model_wait_s": 0.0, "prompt_chars": 0.0, "output_chars": 0.0, "cache_hits": 0.0, "web_hits": 0.0, "llm_calls": 0.0},
    }
    token_budget: dict[str, Any] = {}
    total_time = 0.0

    for item in packet.trace:
        stage_name = stage_map.get(item.stage)
        if not stage_name:
            continue

        metrics = dict((item.payload or {}).get("metrics", {}) or {})
        prompt_tokens = float(metrics.get("prompt_tokens", 0) or 0)
        completion_tokens = float(metrics.get("completion_tokens", 0) or 0)
        total_tokens = float(metrics.get("total_tokens", prompt_tokens + completion_tokens) or 0)
        time_s = float(metrics.get("time_s", 0) or 0)
        model_wait_s = float(metrics.get("model_wait_s", time_s) or 0)
        prompt_chars = float(metrics.get("prompt_chars", 0) or 0)
        output_chars = float(metrics.get("output_chars", 0) or 0)
        cache_hits = float(metrics.get("cache_hits", 0) or 0)
        web_hits = float(metrics.get("web_hits", 0) or 0)
        llm_calls = float(metrics.get("llm_calls", 0) or 0)

        stage_metrics[stage_name]["time"] += time_s
        stage_metrics[stage_name]["tokens"] += total_tokens
        stage_metrics[stage_name]["model_wait_s"] += model_wait_s
        stage_metrics[stage_name]["prompt_chars"] += prompt_chars
        stage_metrics[stage_name]["output_chars"] += output_chars
        stage_metrics[stage_name]["cache_hits"] += cache_hits
        stage_metrics[stage_name]["web_hits"] += web_hits
        stage_metrics[stage_name]["llm_calls"] += llm_calls
        total_time += model_wait_s

        current = token_budget.get(stage_name, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        token_budget[stage_name] = {
            "prompt_tokens": int(current["prompt_tokens"] + prompt_tokens),
            "completion_tokens": int(current["completion_tokens"] + completion_tokens),
            "total_tokens": int(current["total_tokens"] + total_tokens),
        }

    for bucket in stage_metrics.values():
        bucket["tok_s"] = round(bucket["tokens"] / bucket["time"], 1) if bucket["time"] > 0 else 0.0
        bucket["time"] = round(bucket["time"], 2)
        bucket["model_wait_s"] = round(bucket["model_wait_s"], 2)
        bucket["prompt_chars"] = int(bucket["prompt_chars"])
        bucket["output_chars"] = int(bucket["output_chars"])
        bucket["cache_hits"] = int(bucket["cache_hits"])
        bucket["web_hits"] = int(bucket["web_hits"])
        bucket["llm_calls"] = int(bucket["llm_calls"])
        if stage_bucket_tokens_as_int:
            bucket["tokens"] = int(bucket["tokens"])
        else:
            bucket["tokens"] = round(bucket["tokens"], 2)

    return stage_metrics, round(total_time, 2), token_budget


def compute_trace_wall_time(packet: DecisionPacket) -> float:
    if len(packet.trace) < 2:
        return 0.0
    try:
        started = datetime.fromisoformat(str(packet.trace[0].timestamp).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(packet.trace[-1].timestamp).replace("Z", "+00:00"))
    except ValueError:
        return 0.0
    return round(max(0.0, (finished - started).total_seconds()), 2)
