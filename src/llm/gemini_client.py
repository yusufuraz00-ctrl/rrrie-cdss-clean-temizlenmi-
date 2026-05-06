"""Google Gemini cloud LLM client with lane-aware model fallback."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from enum import Enum
from typing import Any, AsyncIterator

import google.generativeai as genai
from google.generativeai import types
from pydantic import BaseModel as PydanticBaseModel

logger = logging.getLogger("rrrie-cdss")

GEMINI_TIMEOUT = 180

# Rate-limit / transient-error retry for Gemini.
_RETRY_MAX_ATTEMPTS = 3
_RETRY_BASE_DELAY_S = 2.0
_RETRY_KEYWORDS = (
    "429",
    "resource_exhausted",
    "resource exhausted",
    "rate_limit",
    "rate limit",
    "quota",
    "503",
    "unavailable",
    "overloaded",
    "deadline_exceeded",
)


def _is_retryable_gemini_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(keyword in message for keyword in _RETRY_KEYWORDS)


class GeminiLane(str, Enum):
    STABLE = "stable"
    EXPERIMENTAL = "experimental"


class GeminiClient:
    """Async Gemini client used by cloud board and case generation flows."""

    def __init__(
        self,
        api_key: str,
        flash_model: str = "",
        pro_model: str = "",
        *,
        stable_flash_model: str = "gemini-2.5-flash",
        stable_pro_model: str = "gemini-2.5-pro",
        experimental_flash_model: str = "gemini-3-flash-preview",
        experimental_pro_model: str = "gemini-3.1-pro-preview",
        default_lane: str = "experimental",
    ):
        self.api_key = api_key
        self.stable_flash_model = stable_flash_model
        self.stable_pro_model = stable_pro_model
        self.experimental_flash_model = experimental_flash_model
        self.experimental_pro_model = experimental_pro_model
        self.default_lane = (
            GeminiLane(default_lane)
            if str(default_lane or "").strip() in {lane.value for lane in GeminiLane}
            else GeminiLane.EXPERIMENTAL
        )
        self.flash_model = flash_model or self.resolve_model(use_pro=False, lane=self.default_lane.value)
        self.pro_model = pro_model or self.resolve_model(use_pro=True, lane=self.default_lane.value)
        self._available: bool | None = None
        self._client: genai.Client | None = None

    @staticmethod
    def _fallback_model_candidates(model_name: str) -> list[str]:
        name = str(model_name or "").strip()
        if not name:
            return []
        alias_map = {
            "gemini-3-flash": ["gemini-3-flash-preview", "gemini-2.5-flash"],
            "gemini-3-flash-preview": ["gemini-2.5-flash"],
            "gemini-3-pro-preview": ["gemini-3.1-pro-preview", "gemini-2.5-pro"],
            "gemini-3.1-pro-preview": ["gemini-2.5-pro"],
        }
        candidates = [name]
        for item in alias_map.get(name, []):
            if item not in candidates:
                candidates.append(item)
        return candidates

    def _get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    @property
    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        self._available = bool(self.api_key and len(self.api_key) > 10)
        if self._available:
            logger.info(
                "gemini_client_ready lane=%s flash=%s pro=%s", 
                self.default_lane.value, 
                self.flash_model, 
                self.pro_model
            )
        else:
            logger.info("gemini_client_unavailable - no API key")
        return self._available

    @property
    def model(self) -> str:
        return self.flash_model

    def resolve_model(self, *, use_pro: bool, lane: str | None = None) -> str:
        requested_lane = str(lane or self.default_lane.value).strip().lower()
        lane_enum = (
            GeminiLane(requested_lane)
            if requested_lane in {item.value for item in GeminiLane}
            else self.default_lane
        )
        if lane_enum == GeminiLane.STABLE:
            return self.stable_pro_model if use_pro else self.stable_flash_model
        return self.experimental_pro_model if use_pro else self.experimental_flash_model

    async def _generate_content_with_fallback(
        self,
        *,
        model_name: str,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        stream: bool,
    ) -> Any:
        client = self._get_client()
        last_exc: Exception | None = None
        for candidate_model in self._fallback_model_candidates(model_name):
            for attempt in range(_RETRY_MAX_ATTEMPTS):
                try:
                    if stream:
                        def _stream_all():
                            chunks = []
                            stream_iter = client.models.generate_content_stream(
                                model=candidate_model,
                                contents=contents,
                                config=config,
                            )
                            for chunk in stream_iter:
                                chunks.append(chunk)
                            return chunks

                        result = await asyncio.wait_for(
                            asyncio.to_thread(_stream_all),
                            timeout=GEMINI_TIMEOUT,
                        )
                    else:
                        result = await asyncio.wait_for(
                            asyncio.to_thread(
                                client.models.generate_content,
                                model=candidate_model,
                                contents=contents,
                                config=config,
                            ),
                            timeout=GEMINI_TIMEOUT,
                        )
                    if candidate_model != model_name:
                        logger.warning("[GEMINI] fallback %s -> %s", model_name, candidate_model)
                    return result
                except asyncio.TimeoutError:
                    raise RuntimeError(f"Gemini model {candidate_model} timed out after {GEMINI_TIMEOUT}s")
                except Exception as exc:
                    last_exc = exc
                    message = str(exc).lower()
                    if _is_retryable_gemini_error(exc) and attempt < _RETRY_MAX_ATTEMPTS - 1:
                        delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                        logger.warning(
                            "[GEMINI] transient error on %s (attempt %d/%d): %s — retrying in %.1fs",
                            candidate_model, attempt + 1, _RETRY_MAX_ATTEMPTS, str(exc)[:120], delay,
                        )
                        await asyncio.sleep(delay)
                        continue
                    if "not_found" in message or "not found" in message:
                        break  # fall through to next candidate_model
                    raise
        raise last_exc or RuntimeError(f"Gemini model unavailable: {model_name}")

    async def stream_chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.3,
        use_pro: bool = False,
        json_schema: Any | None = None,
        lane: str | None = None,
    ) -> AsyncIterator[dict]:
        model_name = self.resolve_model(use_pro=use_pro, lane=lane)

        system_instruction = None
        contents: list[types.Content] = []
        for msg in messages:
            role = msg.get("role", "user")
            text = msg.get("content", "")
            if role == "system":
                system_instruction = text
            else:
                contents.append(
                    types.Content(
                        role="user" if role == "user" else "model",
                        parts=[types.Part(text=text)],
                    )
                )

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": max_tokens,
            "temperature": temperature,
            "system_instruction": system_instruction,
            "safety_settings": [
                types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
            ],
        }
        if use_pro:
            config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=8192)
        if json_schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = json_schema
        config = types.GenerateContentConfig(**config_kwargs)

        all_chunks = await self._generate_content_with_fallback(
            model_name=model_name,
            contents=contents,
            config=config,
            stream=True,
        )

        prompt_tokens = 0
        completion_tokens = 0
        thinking_tokens = 0
        for chunk in all_chunks:
            if chunk.candidates:
                for part in chunk.candidates[0].content.parts:
                    if getattr(part, "thought", False):
                        yield {"type": "thinking", "content": part.text or ""}
                    elif part.text:
                        yield {"type": "token", "content": part.text}
            if chunk.usage_metadata:
                prompt_tokens = getattr(chunk.usage_metadata, "prompt_token_count", 0) or 0
                completion_tokens = getattr(chunk.usage_metadata, "candidates_token_count", 0) or 0
                thinking_tokens = getattr(chunk.usage_metadata, "thoughts_token_count", 0) or thinking_tokens

        yield {
            "type": "done",
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "thinking_tokens": thinking_tokens,
            },
        }

    async def chat_complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 8192,
        temperature: float = 0.3,
        use_pro: bool = False,
        json_schema: Any | None = None,
        lane: str | None = None,
    ) -> dict:
        t0 = time.time()
        content = ""
        thinking_text = ""
        prompt_tokens = 0
        completion_tokens = 0

        if json_schema is not None:
            model_name = self.resolve_model(use_pro=use_pro, lane=lane)
            system_instruction = None
            contents: list[types.Content] = []
            for msg in messages:
                role = msg.get("role", "user")
                text = msg.get("content", "")
                if role == "system":
                    system_instruction = text
                else:
                    contents.append(
                        types.Content(
                            role="user" if role == "user" else "model",
                            parts=[types.Part(text=text)],
                        )
                    )

            config = types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
                system_instruction=system_instruction,
                response_mime_type="application/json",
                response_schema=json_schema,
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_NONE"),
                    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
                ],
            )
            result = await self._generate_content_with_fallback(
                model_name=model_name,
                contents=contents,
                config=config,
                stream=False,
            )
            usage = getattr(result, "usage_metadata", None)
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            parsed = getattr(result, "parsed", None)
            if parsed is not None:
                if isinstance(parsed, PydanticBaseModel):
                    content = parsed.model_dump_json()
                elif isinstance(parsed, (dict, list)):
                    content = json.dumps(parsed, ensure_ascii=False)
                else:
                    content = str(parsed)
            if not content:
                content = getattr(result, "text", "") or ""
            elapsed = round(time.time() - t0, 2)
            tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0
            return {
                "content": content,
                "thinking_text": "",
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "elapsed": elapsed,
                "tok_per_sec": tok_per_sec,
            }

        async for chunk in self.stream_chat(
            messages,
            max_tokens,
            temperature,
            use_pro=use_pro,
            json_schema=json_schema,
            lane=lane,
        ):
            if chunk["type"] == "token":
                content += chunk["content"]
            elif chunk["type"] == "thinking":
                thinking_text += chunk["content"]
            elif chunk["type"] == "done":
                usage = chunk.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)

        elapsed = round(time.time() - t0, 2)
        tok_per_sec = round(completion_tokens / elapsed, 1) if elapsed > 0 else 0
        return {
            "content": content,
            "thinking_text": thinking_text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "elapsed": elapsed,
            "tok_per_sec": tok_per_sec,
        }
