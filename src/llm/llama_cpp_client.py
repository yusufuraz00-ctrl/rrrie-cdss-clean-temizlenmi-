"""llama.cpp GGUF inference client for Qwen3.5-4B -- ADAPTIVE THINKING edition.

Backend: llama-server (b8192+) running as a local HTTP server on port 8080.
Communicates via OpenAI-compatible /v1/chat/completions endpoint.

WHY llama-server INSTEAD OF llama-cpp-python:
  - llama-cpp-python 0.3.16 (latest PyPI) bundles an older llama.cpp backend
    that does NOT support the Qwen3.5 architecture (Gated DeltaNet + MoE).
  - The pre-built llama-server binary (b8192) includes full Qwen3.5 support
    with CUDA acceleration, flash attention, and graph optimisations.
  - No Windows CUDA wheels exist for llama-cpp-python.
  - llama-server provides identical OpenAI-compatible API with better performance.

QWEN3.5 THINKING:
  - Does NOT support /think or /no_think soft switches (unlike Qwen3)
  - Thinking controlled via enable_thinking parameter in chat template
  - Default: thinking ENABLED -> generates <think>...</think> blocks
  - Recommended: temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5

ADAPTIVE THINKING STRATEGY:
  Token budgets (thinking + answer):
    R1 (diagnosis):   answer_budget * 2.5
    R3 (synthesis):   answer_budget * 2.0
    IE (evaluation):  answer_budget * 1.5

Expected RTX 4050 (6 GB VRAM):
  - Qwen3-4B bitsandbytes NF4 (HF): ~10 tok/s
  - Qwen3.5-4B GGUF Q4_K_M (server): ~30-60 tok/s
"""

from __future__ import annotations

import re
import subprocess
import threading
import time
import os
import asyncio
from dataclasses import dataclass, field
from typing import Any

import requests as http_requests

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _coerce_text_block(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            candidate = value.get(key)
            text = _coerce_text_block(candidate)
            if text:
                return text
        return ""
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            text = _coerce_text_block(item)
            if text:
                parts.append(text)
        return "\n".join(parts)
    return ""


def _extract_reasoning_text(message: dict[str, Any] | None) -> str:
    if not isinstance(message, dict):
        return ""
    candidates = [
        _coerce_text_block(message.get("reasoning", "")),
        _coerce_text_block(message.get("reasoning_content", "")),
        _coerce_text_block(message.get("thinking", "")),
    ]
    for item in candidates:
        if str(item or "").strip():
            return str(item)
    return ""


def _extract_message_tool_calls(message: dict[str, Any] | None) -> list[dict[str, Any]] | None:
    """Normalize tool_call payloads from OpenAI-compatible chat responses."""
    if not isinstance(message, dict):
        return None
    raw_tool_calls = message.get("tool_calls")
    if not isinstance(raw_tool_calls, list):
        return None
    parsed: list[dict[str, Any]] = []
    for item in raw_tool_calls:
        if not isinstance(item, dict):
            continue
        function = dict(item.get("function", {}) or {})
        parsed.append(
            {
                "id": str(item.get("id", "") or ""),
                "type": str(item.get("type", "function") or "function"),
                "function": {
                    "name": str(function.get("name", "") or ""),
                    "arguments": str(function.get("arguments", "") or ""),
                },
            }
        )
    return parsed or None

# -- Configuration -----------------------------------------------------------
def _normalize_server_url(url: str, default: str) -> str:
    normalized = (url or default).strip().rstrip("/")
    if normalized.endswith("/v1"):
        normalized = normalized[:-3]
    return normalized or default


def _read_runtime_server_url(env_key: str, settings_value: str, default: str) -> str:
    env_value = str(os.environ.get(env_key, "") or "").strip()
    return _normalize_server_url(env_value or settings_value, default)


def resolve_main_server_url() -> str:
    settings = get_settings()
    return _read_runtime_server_url("LLAMA_SERVER_URL", settings.local_api_url, "http://127.0.0.1:8080")


def resolve_dllm_server_url() -> str:
    settings = get_settings()
    return _read_runtime_server_url("DLLM_SERVER_URL", settings.dllm_api_url, "http://127.0.0.1:8081")


_settings = get_settings()
LLAMA_SERVER_URL = resolve_main_server_url()
DLLM_SERVER_URL = resolve_dllm_server_url()
LLAMA_SERVER_TIMEOUT = 600  # seconds per request (long R1/R3 thinking calls)

# -- Singleton Lock ----------------------------------------------------------
_lock = threading.Lock()
_instance: "LlamaCppClient | None" = None
_server_failed: bool = False


@dataclass
class LlamaCppChatResponse:
    """Mimics OpenAI ChatCompletion structure for drop-in compatibility."""

    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tool_calls: list[dict[str, Any]] | None = None
    raw_content: str = ""
    reasoning_text: str = ""

    class _Message:
        def __init__(self, content: str, tool_calls: list[dict[str, Any]] | None = None):
            self.content = content
            self.tool_calls = tool_calls

    class _Usage:
        def __init__(self, prompt_tokens: int, completion_tokens: int):
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    class _Choice:
        def __init__(self, message: "LlamaCppChatResponse._Message"):
            self.message = message

    @property
    def choices(self) -> list:
        return [self._Choice(self._Message(self.content, self.tool_calls))]

    @property
    def usage(self) -> "_Usage":
        return self._Usage(self.prompt_tokens, self.completion_tokens)


@dataclass
class TokenLogProb:
    """Single token/logprob pair returned by a logprob probe."""

    token: str
    logprob: float


@dataclass
class ProbeToken:
    """One generated token plus its top-k alternatives."""

    token: str
    logprob: float | None = None
    top_logprobs: list[TokenLogProb] = field(default_factory=list)


@dataclass
class LogprobProbeResponse:
    """Structured response for capability-safe logprob probing."""

    content: str = ""
    tokens: list[ProbeToken] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    supported: bool = True
    error: str = ""


def _coerce_float(value: Any) -> float | None:
    """Best-effort float conversion for heterogeneous API payloads."""
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_top_logprobs(raw_top_logprobs: Any) -> list[TokenLogProb]:
    """Parse OpenAI-style or llama.cpp-style top_logprobs payloads."""
    parsed: list[TokenLogProb] = []

    if isinstance(raw_top_logprobs, dict):
        for token, logprob in raw_top_logprobs.items():
            numeric = _coerce_float(logprob)
            if numeric is not None:
                parsed.append(TokenLogProb(token=str(token), logprob=numeric))
        return parsed

    if not isinstance(raw_top_logprobs, list):
        return parsed

    for item in raw_top_logprobs:
        if isinstance(item, dict) and "token" in item:
            numeric = _coerce_float(item.get("logprob"))
            if numeric is not None:
                parsed.append(TokenLogProb(token=str(item.get("token", "")), logprob=numeric))
            continue
        if isinstance(item, dict):
            for token, logprob in item.items():
                numeric = _coerce_float(logprob)
                if numeric is not None:
                    parsed.append(TokenLogProb(token=str(token), logprob=numeric))

    return parsed


def _parse_probe_tokens(logprobs: Any) -> list[ProbeToken]:
    """Parse token-level logprob structures from chat completion responses."""
    if not isinstance(logprobs, dict):
        return []

    content_items = logprobs.get("content")
    if isinstance(content_items, list):
        parsed_tokens: list[ProbeToken] = []
        for item in content_items:
            if not isinstance(item, dict):
                continue
            parsed_tokens.append(
                ProbeToken(
                    token=str(item.get("token", "")),
                    logprob=_coerce_float(item.get("logprob")),
                    top_logprobs=_parse_top_logprobs(item.get("top_logprobs")),
                )
            )
        return parsed_tokens

    raw_tokens = logprobs.get("tokens")
    if not isinstance(raw_tokens, list):
        return []

    raw_token_logprobs = logprobs.get("token_logprobs", [])
    raw_top_logprobs = logprobs.get("top_logprobs", [])
    parsed_tokens = []

    for idx, token in enumerate(raw_tokens):
        token_logprob = raw_token_logprobs[idx] if idx < len(raw_token_logprobs) else None
        top_logprobs = raw_top_logprobs[idx] if idx < len(raw_top_logprobs) else []
        parsed_tokens.append(
            ProbeToken(
                token=str(token),
                logprob=_coerce_float(token_logprob),
                top_logprobs=_parse_top_logprobs(top_logprobs),
            )
        )

    return parsed_tokens


class LlamaCppClient:
    """High-speed local LLM client -- talks to llama-server via HTTP.

    Optimised for Qwen3.5-4B with adaptive thinking.
    Drop-in replacement for HFLocalClient with identical API surface
    (.chat(), .get_instance(), .is_loaded, .get_vram_usage()).
    """

    def __init__(self, model_name: str, max_ctx: int = 4096) -> None:
        self.model_name = model_name
        self.max_ctx = max_ctx
        self._session = http_requests.Session()
        self._server_url = resolve_main_server_url()
        self._verify_server()

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        try:
            self._session.close()
        except Exception:
            pass

    # -- Singleton -----------------------------------------------------------

    @classmethod
    def get_instance(
        cls, model_name: str = "Qwen/Qwen3.5-4B", max_ctx: int = 4096
    ) -> "LlamaCppClient":
        """Get or create the singleton LlamaCppClient instance."""
        global _instance, _server_failed
        if _instance is None:
            with _lock:
                if _instance is None:
                    try:
                        _instance = cls(model_name, max_ctx)
                        _server_failed = False
                    except Exception:
                        _server_failed = True
                        raise
        elif _server_failed:
            # Retry connection instead of permanent death lock
            try:
                _instance._verify_server()
                _server_failed = False
            except Exception:
                raise ConnectionError("llama-server is currently offline and failed health ping. Start server and retry.")
        return _instance

    @classmethod
    def reset_server_state(cls) -> None:
        """Clear cached failure flag to allow re-trying connection."""
        global _server_failed
        _server_failed = False

    # -- Server Health Check -------------------------------------------------

    def _verify_server(self) -> None:
        """Verify llama-server is running and healthy."""
        try:
            resp = self._session.get(
                f"{self._server_url}/health",
                timeout=5,
            )
            if resp.status_code == 200:
                logger.info(
                    "llama_server_connected",
                    model=self.model_name,
                    url=self._server_url,
                    status="healthy",
                )
                return
        except http_requests.ConnectionError:
            pass

        raise ConnectionError(
            f"llama-server not running at {self._server_url}. "
            "Start it with:\n"
            "  llama-server\\llama-server.exe -m <gguf_path> -ngl 99 "
            "--host 127.0.0.1 --port 8080 -c 4096 -fa 1"
        )

    # -- Adaptive Thinking Token Multipliers ---------------------------------
    # Per-stage thinking budget = answer_tokens * multiplier. Profile selectable
    # via CDSS_THINKING_PROFILE env (balanced|fast|deep). `balanced` is the new
    # default — it cuts ~40% of thinking tokens vs the previous `deep` profile
    # without measurable top-3 differential regression on gold10. Operators
    # can still scale uniformly via QWEN_THINKING_POOL_SCALE on top of the
    # profile multipliers.
    THINKING_PROFILES: dict[str, dict[str, float]] = {
        "deep": {  # legacy behaviour (pre-2026-05 optimization)
            "R1": 2.4,
            "R3": 1.8,
            "IE": 1.4,
            "default": 2.0,
        },
        "balanced": {  # new default — quality-preserving cut
            "R1": 1.5,
            "R3": 1.3,
            "IE": 1.2,
            "default": 1.4,
        },
        "fast": {  # aggressive — accept small quality risk
            "R1": 1.2,
            "R3": 1.1,
            "IE": 1.05,
            "default": 1.15,
        },
    }

    @classmethod
    def _active_thinking_multipliers(cls) -> dict[str, float]:
        profile = str(os.environ.get("CDSS_THINKING_PROFILE", "balanced") or "balanced").strip().lower()
        return cls.THINKING_PROFILES.get(profile, cls.THINKING_PROFILES["balanced"])

    # Backward-compat alias — existing call sites read THINKING_MULTIPLIERS.
    # Resolved at class-load via the default profile so static reads still work,
    # but runtime resolution should call _active_thinking_multipliers().
    THINKING_MULTIPLIERS = THINKING_PROFILES["balanced"]
    STAGE_GENERATION_PROFILES = {
        "R1": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.02, "presence_penalty": 0.15},
        "R3": {"temperature": 0.45, "top_p": 0.90, "top_k": 20, "repeat_penalty": 1.03, "presence_penalty": 0.05},
        "IE": {"temperature": 0.25, "top_p": 0.85, "top_k": 12, "repeat_penalty": 1.02, "presence_penalty": 0.0},
        "POST_MORTEM": {"temperature": 0.30, "top_p": 0.90, "top_k": 16, "repeat_penalty": 1.02, "presence_penalty": 0.0},
        "DEFAULT": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "repeat_penalty": 1.02, "presence_penalty": 0.0},
    }

    # Regex to strip <think>...</think> blocks from output
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

    # Reasoning stages: when ATOM mode is active, thinking is ON here even when
    # json_mode=True.  The <think>...</think> block is stripped by _THINK_RE before
    # returning to the caller, so they always receive clean JSON.
    # In normal mode these stages behave identically to all other stages (no thinking).
    _REASONING_STAGES: frozenset[str] = frozenset({"R1", "R3", "IE"})

    # Per-stage minimum token floors (answer budget before multiplier).
    # These override the caller-supplied max_tokens when the caller passes a
    # value below the floor — e.g. the policy default of 256 is too small for IE.
    STAGE_BASE_TOKENS: dict[str, int] = {
        "R1": 512,   # differential hypothesis generation
        "R3": 768,   # swarm panel synthesis
        "IE": 1024,  # devil's advocate challenger
    }

    # -- Chat Completion -----------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 2048,
        json_mode: bool = False,
        structured_output: bool = False,
        stage: str = "default",
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LlamaCppChatResponse:
        """Generate a chat completion via llama-server HTTP API.

        Args:
            messages: List of {"role": ..., "content": ...} dicts.
            temperature: Sampling temperature (0.0 = greedy).
            max_tokens: Maximum new tokens for the ANSWER (thinking extra).
            json_mode: If True, append JSON instruction to system prompt.
            structured_output: If True, disable enlarged thinking pools and use a
                deterministic profile so protocol stages return actual answer text.
            stage: RRRIE stage ("R1", "R3", "IE") for adaptive thinking budget.

        Returns:
            LlamaCppChatResponse with .choices[0].message.content and .usage.
        """
        if json_mode:
            messages = self._inject_json_instruction(messages)

        # ATOM mode: deep reasoning active only when explicitly enabled.
        # Normal mode: all stages run without thinking (fast, old behaviour).
        atom_mode = bool(get_settings().atom_mode)
        is_reasoning_stage = atom_mode and (stage in self._REASONING_STAGES)

        # Apply per-stage token floor only in ATOM mode — in normal mode the caller's
        # max_tokens governs as before.
        stage_floor = self.STAGE_BASE_TOKENS.get(stage, 0) if atom_mode else 0
        effective_max_tokens = max(max_tokens, stage_floor)

        # Token budget & thinking:
        #   ATOM mode OFF (normal):                          → no thinking, caller's tokens
        #   ATOM mode ON + reasoning stage + structured/json → full multiplier + thinking ON
        #   ATOM mode ON + reasoning stage + plain           → full multiplier + thinking ON
        #   ATOM mode ON + non-reasoning stage               → tight, no thinking
        disable_thinking = (json_mode or structured_output) and not is_reasoning_stage

        if structured_output and not is_reasoning_stage:
            multiplier = 1.0
            total_budget = int(effective_max_tokens)
        else:
            active = self._active_thinking_multipliers()
            multiplier = active.get(stage, active["default"])
            try:
                pool_scale = float(os.environ.get("QWEN_THINKING_POOL_SCALE", "1.0") or "1.0")
            except ValueError:
                pool_scale = 1.0
            pool_scale = max(0.5, min(3.0, pool_scale))
            multiplier = round(multiplier * pool_scale, 2)
            total_budget = int(effective_max_tokens * multiplier)
            try:
                max_ctx_fraction = float(os.environ.get("QWEN_THINKING_MAX_CTX_FRACTION", "0.75") or "0.75")
            except ValueError:
                max_ctx_fraction = 0.75
            max_ctx_fraction = max(0.25, min(0.95, max_ctx_fraction))
            total_budget = min(total_budget, int(self.max_ctx * max_ctx_fraction))

        profile_mode = str(get_settings().qwen_stage_profile_mode or "official")
        if disable_thinking:
            generation_profile = {
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
            }
        elif profile_mode == "official":
            generation_profile = dict(
                self.STAGE_GENERATION_PROFILES.get(stage, self.STAGE_GENERATION_PROFILES["DEFAULT"])
            )
        else:
            generation_profile = dict(self.STAGE_GENERATION_PROFILES["DEFAULT"])
            generation_profile["temperature"] = temperature if temperature > 0 else 1.0

        t0 = time.time()

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": total_budget,
            "temperature": generation_profile["temperature"],
            "top_p": generation_profile["top_p"],
            "top_k": generation_profile["top_k"],
            "repeat_penalty": generation_profile["repeat_penalty"],
            "presence_penalty": generation_profile["presence_penalty"],
        }

        # Response cache — opt-in via CDSS_RESPONSE_CACHE=1. Only safe for
        # deterministic generations (temperature == 0) with no tools/response_format,
        # because cached output must be byte-identical to a fresh call.
        _cache_enabled = False
        _cache = None
        _cache_key = ""
        try:
            from src.cdss.runtime.response_cache import ResponseCache, get_response_cache
            _cache_enabled = (
                ResponseCache.enabled()
                and float(generation_profile.get("temperature", 0.0) or 0.0) == 0.0
                and not tools
                and not response_format
            )
            if _cache_enabled:
                _cache = get_response_cache()
                _cache_key = ResponseCache.make_key(
                    self.model_name,
                    messages,
                    temperature=float(generation_profile.get("temperature", 0.0) or 0.0),
                    top_p=float(generation_profile.get("top_p", 1.0) or 1.0),
                    top_k=generation_profile.get("top_k"),
                    max_tokens=total_budget,
                    extra={
                        "stage": stage,
                        "json_mode": bool(json_mode),
                        "structured_output": bool(structured_output),
                        "disable_thinking": bool(disable_thinking),
                    },
                )
                cached = _cache.get(_cache_key)
                if cached is not None:
                    logger.info(
                        "gguf_response_cache_hit",
                        stage=stage,
                        key_prefix=_cache_key[:12],
                        stats=_cache.stats(),
                    )
                    return cached
        except Exception:
            _cache_enabled = False
            _cache = None
        if disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        if response_format:
            payload["response_format"] = response_format

        # Quick health check before committing to a 600s inference timeout.
        # Converts a silent 10-minute hang into an immediate, actionable error.
        try:
            _health = self._session.get(f"{self._server_url}/health", timeout=5)
            _health.raise_for_status()
        except Exception as _he:
            raise RuntimeError(f"llama-server unreachable before inference: {_he}") from _he

        _last_exc: Exception | None = None
        for _attempt in range(3):
            try:
                resp = self._session.post(
                    f"{self._server_url}/v1/chat/completions",
                    json=payload,
                    timeout=LLAMA_SERVER_TIMEOUT,
                )
                resp.raise_for_status()
                break
            except http_requests.exceptions.HTTPError:
                raise  # HTTP errors (4xx/5xx) are not transient — reraise immediately
            except (
                http_requests.exceptions.ConnectionError,
                http_requests.exceptions.Timeout,
                http_requests.exceptions.ChunkedEncodingError,
            ) as exc:
                _last_exc = exc
                if _attempt < 2:
                    logger.warning(
                        "llama_server_transient_error",
                        attempt=_attempt + 1,
                        error=str(exc)[:120],
                    )
                    time.sleep(2 ** _attempt)
                else:
                    raise _last_exc
        data = resp.json()

        # Extract raw content (may include <think>...</think> blocks)
        message = data["choices"][0]["message"]
        raw_content = _coerce_text_block(message.get("content") or "")
        reasoning_text = _extract_reasoning_text(message)
        tool_calls = _extract_message_tool_calls(message)
        usage = data.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Strip thinking blocks, keep only the answer
        think_match = self._THINK_RE.search(raw_content)
        inline_thinking_text = think_match.group(0) if think_match else ""
        inline_thinking_tokens = len(inline_thinking_text.split()) if inline_thinking_text else 0
        reasoning_tokens = len(reasoning_text.split()) if reasoning_text else 0
        thinking_tokens = min(
            int(completion_tokens or 0),
            max(inline_thinking_tokens, reasoning_tokens),
        )

        clean_content = self._THINK_RE.sub("", raw_content).strip()

        elapsed = round(time.time() - t0, 2)
        tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0

        logger.info(
            "gguf_inference_complete",
            stage=stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            thinking_tokens_approx=thinking_tokens,
            answer_tokens_approx=max(0, completion_tokens - thinking_tokens),
            time_s=elapsed,
            tok_per_sec=tok_per_sec,
            thinking_multiplier=multiplier,
            reasoning_chars=len(reasoning_text),
        )

        if not clean_content and completion_tokens > 5:
            if reasoning_text.strip():
                logger.info(
                    "llm_reasoning_only_response",
                    stage=stage,
                    raw_len=len(raw_content),
                    completion_tokens=completion_tokens,
                    thinking_tokens_approx=thinking_tokens,
                    reasoning_chars=len(reasoning_text),
                )
            else:
                logger.warning(
                    "llm_content_empty_after_strip",
                    stage=stage,
                    raw_len=len(raw_content),
                    completion_tokens=completion_tokens,
                    thinking_tokens_approx=thinking_tokens,
                )

        if structured_output and not clean_content and completion_tokens > 5:
            retry_messages = [
                *messages,
                {
                    "role": "user",
                    "content": (
                        "Return ONLY the final structured answer now. "
                        "Do not emit hidden reasoning, commentary, or markdown. "
                        "Output only the protocol rows requested in the prior instruction."
                    ),
                },
            ]
            retry_payload = {
                "model": self.model_name,
                "messages": retry_messages,
                "max_tokens": max(64, min(int(max_tokens), 256)),
                "temperature": 0.0,
                "top_p": 1.0,
                "top_k": 1,
                "repeat_penalty": 1.0,
                "presence_penalty": 0.0,
            }
            retry_payload["chat_template_kwargs"] = {"enable_thinking": False}
            if tools:
                retry_payload["tools"] = tools
                retry_payload["tool_choice"] = tool_choice or "auto"
            if response_format:
                retry_payload["response_format"] = response_format
            retry_started = time.time()
            retry_resp = self._session.post(
                f"{self._server_url}/v1/chat/completions",
                json=retry_payload,
                timeout=LLAMA_SERVER_TIMEOUT,
            )
            retry_resp.raise_for_status()
            retry_data = retry_resp.json()
            retry_message = retry_data["choices"][0]["message"]
            retry_raw_content = _coerce_text_block(retry_message.get("content") or "")
            retry_reasoning_text = _extract_reasoning_text(retry_message)
            retry_clean_content = self._THINK_RE.sub("", retry_raw_content).strip()
            retry_usage = retry_data.get("usage", {})
            retry_prompt_tokens = int(retry_usage.get("prompt_tokens", 0) or 0)
            retry_completion_tokens = int(retry_usage.get("completion_tokens", 0) or 0)
            retry_elapsed = round(time.time() - retry_started, 2)
            if retry_clean_content:
                logger.info(
                    "llm_structured_retry_recovered",
                    stage=stage,
                    prompt_tokens=retry_prompt_tokens,
                    completion_tokens=retry_completion_tokens,
                    time_s=retry_elapsed,
                    reasoning_chars=len(retry_reasoning_text),
                )
                clean_content = retry_clean_content
                prompt_tokens = retry_prompt_tokens
                completion_tokens = retry_completion_tokens
                tool_calls = _extract_message_tool_calls(retry_message)

        _response = LlamaCppChatResponse(
            content=clean_content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            tool_calls=tool_calls,
            raw_content=raw_content,
            reasoning_text=reasoning_text,
        )
        if _cache_enabled and _cache is not None and _cache_key:
            try:
                _cache.set(_cache_key, _response)
            except Exception:
                pass
        return _response

    def probe_logprobs(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 3,
        top_logprobs: int = 8,
        temperature: float = 0.0,
    ) -> LogprobProbeResponse:
        """Request token logprobs without disturbing the main generation flow."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": max(top_logprobs, 1),
            "repeat_penalty": 1.0,
            "presence_penalty": 0.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        try:
            resp = self._session.post(
                f"{self._server_url}/v1/chat/completions",
                json=payload,
                timeout=LLAMA_SERVER_TIMEOUT,
            )
            resp.raise_for_status()
        except http_requests.HTTPError as exc:
            error_text = ""
            if exc.response is not None:
                error_text = exc.response.text[:300]
            return LogprobProbeResponse(
                supported=False,
                error=error_text or str(exc),
            )
        except Exception as exc:
            return LogprobProbeResponse(supported=False, error=str(exc))

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tokens = _parse_probe_tokens(choice.get("logprobs"))
        usage = data.get("usage", {})

        if not tokens:
            return LogprobProbeResponse(
                content=message.get("content", "") or "",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                supported=False,
                error="Server did not return token logprobs",
            )

        return LogprobProbeResponse(
            content=message.get("content", "") or "",
            tokens=tokens,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            supported=True,
        )

    # -- Helpers -------------------------------------------------------------

    @staticmethod
    def _inject_json_instruction(
        messages: list[dict[str, str]],
    ) -> list[dict[str, str]]:
        """Append a JSON-mode instruction to the system message."""
        messages = [m.copy() for m in messages]
        json_hint = (
            "\n\nIMPORTANT: You MUST respond with valid JSON only. "
            "No markdown, no explanation outside JSON."
        )

        for m in messages:
            if m["role"] == "system":
                m["content"] += json_hint
                return messages

        messages.insert(0, {"role": "system", "content": json_hint.strip()})
        return messages

    @property
    def is_loaded(self) -> bool:
        """Check if the server is running and responsive."""
        try:
            resp = self._session.get(
                f"{self._server_url}/health", timeout=2
            )
            return resp.status_code == 200
        except Exception:
            return False

    def get_vram_usage(self) -> float:
        """Return estimated VRAM usage in GB."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            first_line = next(
                (line.strip() for line in result.stdout.splitlines() if line.strip()),
                "",
            )
            if first_line:
                return float(first_line) / 1024.0
        except Exception:
            pass
        return 2.6  # estimated for Qwen3.5-4B Q4_K_M


# ═══════════════════════════════════════════════════════════════════════
# DLLMClient — Lightweight client for DLLM R0
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class DLLMResponse:
    """Response from the DLLM 0.8B model, preserving thinking + output."""
    output: str = ""
    thinking: str = ""
    raw: str = ""
    tokens: int = 0
    elapsed: float = 0.0
    tool_calls: list[dict[str, Any]] | None = None


class DLLMClient:
    """Lightweight HTTP client for DLLM R0 reasoning engine.

    Differences from LlamaCppClient:
      - Targets the resolved DLLM server URL, which may be port 8081 or the
        main 4B server in shared_main_server topology
      - Preserves <think> blocks for logging (doesn't strip them)
      - Lower timeout because R0-style probe calls should stay compact
      - Not a singleton — created per pipeline run
    """

    TIMEOUT = 30  # 0.8B should respond in <10s, 30s is generous

    _THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)

    def __init__(self, base_url: str | None = None) -> None:
        # Strip trailing /v1 or /v1/ to prevent double /v1/v1/ in URL
        url = (base_url or resolve_dllm_server_url()).rstrip("/")
        if url.endswith("/v1"):
            url = url[:-3]
        self._base_url = url
        self._session = http_requests.Session()

    def is_healthy(self) -> bool:
        """Check if the DLLM server is running."""
        try:
            resp = self._session.get(f"{self._base_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 512,
        json_mode: bool = False,
        json_schema: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> DLLMResponse:
        """Single chat completion — returns both thinking and output."""
        if not self.is_healthy():
            raise RuntimeError(f"DLLM server unreachable before inference: {self._base_url}")
        t0 = time.time()
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature if temperature > 0 else 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repeat_penalty": 1.05,
        }
        if json_mode:
            # Keep the transport compatible with the current llama-server build.
            # The DLLM pipeline may pass a stage schema, but this endpoint only
            # reliably supports json_object mode in the current setup.
            payload["response_format"] = {"type": "json_object"}
        elif response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice or "auto"
        resp = self._session.post(
            f"{self._base_url}/v1/chat/completions",
            json=payload,
            timeout=self.TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()

        message = data["choices"][0]["message"]
        raw = message.get("content") or ""
        tool_calls = _extract_message_tool_calls(message)
        tokens = data.get("usage", {}).get("completion_tokens", 0)
        elapsed = round(time.time() - t0, 2)

        # Separate thinking from output (keep both)
        thinking = ""
        output = raw
        think_match = self._THINK_RE.search(raw)
        if think_match:
            thinking = think_match.group(1).strip()
            output = self._THINK_RE.sub("", raw).strip()

        logger.info(
            "dllm_inference",
            tokens=tokens,
            time_s=elapsed,
            thinking_len=len(thinking),
            output_len=len(output),
        )

        return DLLMResponse(
            output=output,
            thinking=thinking,
            raw=raw,
            tokens=tokens,
            elapsed=elapsed,
            tool_calls=tool_calls,
        )

    def probe_logprobs(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 3,
        top_logprobs: int = 8,
        temperature: float = 0.0,
    ) -> LogprobProbeResponse:
        """Request token logprobs from the fast 0.8B DLLM server."""
        payload = {
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 1.0,
            "top_k": max(top_logprobs, 1),
            "repeat_penalty": 1.0,
            "logprobs": True,
            "top_logprobs": top_logprobs,
        }

        try:
            resp = self._session.post(
                f"{self._base_url}/v1/chat/completions",
                json=payload,
                timeout=self.TIMEOUT,
            )
            resp.raise_for_status()
        except http_requests.HTTPError as exc:
            error_text = ""
            if exc.response is not None:
                error_text = exc.response.text[:300]
            return LogprobProbeResponse(
                supported=False,
                error=error_text or str(exc),
            )
        except Exception as exc:
            return LogprobProbeResponse(supported=False, error=str(exc))

        data = resp.json()
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        tokens = _parse_probe_tokens(choice.get("logprobs"))
        usage = data.get("usage", {})

        if not tokens:
            return LogprobProbeResponse(
                content=message.get("content", "") or "",
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                supported=False,
                error="Server did not return token logprobs",
            )

        return LogprobProbeResponse(
            content=message.get("content", "") or "",
            tokens=tokens,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            supported=True,
        )
