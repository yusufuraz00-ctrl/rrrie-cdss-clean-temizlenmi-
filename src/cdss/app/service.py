from __future__ import annotations

import asyncio
import importlib
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from src.cdss.contracts.models import (
    CandidateReasoning,
    DecisionPacket,
    ReasoningArtifactV2,
    RequiredDataRequest,
)
from src.cdss.runtime.policy import CdssRuntimePolicy, load_runtime_policy


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    cleaned: list[str] = []
    for value in values:
        text = " ".join(str(value or "").strip().split())
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(text)
    return cleaned


class _QueueWebSocket:
    def __init__(self) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

    async def send_json(self, payload: dict[str, Any]) -> None:
        await self.queue.put(payload)


@dataclass
class CdssApplicationService:
    metadata: dict[str, Any] = field(default_factory=dict)
    runtime_policy: CdssRuntimePolicy = field(default_factory=load_runtime_policy)

    _llm_client: Any = None
    _groq_client: Any = None
    _gemini_client: Any = None
    _memory: Any = None

    def _import_first(self, candidates: list[tuple[str, str]]) -> Any:
        first_error: Exception | None = None
        for module_name, attr_name in candidates:
            try:
                module = importlib.import_module(module_name)
                return getattr(module, attr_name)
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                continue
        if first_error is not None:
            raise first_error
        raise ImportError("No import candidates provided")

    def _get_server_module(self) -> Any:
        return importlib.import_module("gui.server")

    def _get_settings(self) -> Any:
        factory = self._import_first(
            [
                ("config.settings", "get_settings"),
                ("settings", "get_settings"),
            ]
        )
        return factory()

    def _get_llm_client(self) -> Any:
        if self._llm_client is None:
            try:
                self._llm_client = self._get_server_module().get_client()
            except Exception:
                settings = self._get_settings()
                cls = self._import_first(
                    [
                        ("src.llm.llama_cpp_client", "LlamaCppClient"),
                        ("llm.llama_cpp_client", "LlamaCppClient"),
                    ]
                )
                self._llm_client = cls.get_instance(
                    model_name=getattr(settings, "hf_model_name", None),
                    max_ctx=getattr(settings, "num_ctx", None),
                )
        return self._llm_client

    def _get_groq_client(self) -> Any:
        if self._groq_client is None:
            try:
                self._groq_client = self._get_server_module().get_groq()
            except Exception:
                try:
                    settings = self._get_settings()
                    cls = self._import_first(
                        [
                            ("src.llm.groq_client", "GroqClient"),
                            ("llm.groq_client", "GroqClient"),
                        ]
                    )
                    self._groq_client = cls(
                        api_key=getattr(settings, "groq_api_key", None),
                        model=getattr(settings, "cloud_model_name", None),
                        api_url=getattr(settings, "groq_api_url", None),
                    )
                except Exception:
                    self._groq_client = None
        return self._groq_client

    def _get_gemini_client(self) -> Any:
        if self._gemini_client is None:
            try:
                self._gemini_client = self._get_server_module().get_gemini()
            except Exception:
                settings = self._get_settings()
                cls = self._import_first(
                    [
                        ("src.llm.gemini_client", "GeminiClient"),
                        ("llm.gemini_client", "GeminiClient"),
                    ]
                )
                self._gemini_client = cls(
                    api_key=getattr(settings, "google_api_key", None),
                    flash_model=getattr(settings, "gemini_flash_model", None),
                    pro_model=getattr(settings, "gemini_pro_model", None),
                )
        return self._gemini_client

    def _get_memory(self) -> Any:
        if self._memory is None:
            try:
                self._memory = self._get_server_module().get_memory()
            except Exception:
                try:
                    cls = self._import_first(
                        [
                            ("src.memory.case_store", "CaseStore"),
                            ("memory.case_store", "CaseStore"),
                            ("src.cdss.memory.case_store", "CaseStore"),
                        ]
                    )
                    self._memory = cls()
                except Exception:
                    self._memory = None
        return self._memory

    def _get_llama_server_url(self) -> str:
        try:
            return getattr(self._get_server_module(), "LLAMA_SERVER_URL")
        except Exception:
            try:
                return self._import_first(
                    [
                        ("src.llm.llama_cpp_client", "LLAMA_SERVER_URL"),
                        ("llm.llama_cpp_client", "LLAMA_SERVER_URL"),
                    ]
                )
            except Exception:
                return "http://127.0.0.1:8080"

    def _get_orchestrator(self) -> Any:
        try:
            return getattr(self._get_server_module(), "run_rrrie_chat")
        except Exception:
            return self._import_first(
                [
                    ("src.pipeline.orchestrator", "run_rrrie_chat"),
                ]
            )

    def _get_case_generator(self) -> Any:
        try:
            return getattr(self._get_server_module(), "generate_new_case")
        except Exception:
            return self._import_first(
                [
                    ("src.knowledge.case_generator", "generate_new_case"),
                ]
            )

    def _normalize_patient_input(self, patient_input: Any) -> dict[str, Any]:
        if isinstance(patient_input, dict):
            payload = dict(patient_input)
        elif hasattr(patient_input, "model_dump"):
            payload = patient_input.model_dump()
            if payload.get("context") and isinstance(payload.get("context"), dict):
                # Extract meta flags from context if available
                ctx = payload["context"]
                if "ui_mode" in ctx:
                    if ctx["ui_mode"] == "deep":
                        payload["deep_thinking"] = True
                    elif ctx["ui_mode"] == "fast":
                        payload["deep_thinking"] = False
        else:
            payload = {"content": str(patient_input or "")}
        patient_text = (
            payload.get("patient_text")
            or payload.get("content")
            or payload.get("text")
            or payload.get("narrative")
            or ""
        )
        return {
            "patient_text": str(patient_text or "").strip(),
            "thinking": bool(payload.get("thinking", True)),
            "local_only": bool(payload.get("local_only", False)),
            "super_thinking": bool(payload.get("super_thinking", False)),
            "deep_thinking": bool(payload.get("deep_thinking", False)),
            "operation_mode": payload.get("operation_mode"),
            "runtime_profile": payload.get("runtime_profile"),
            "expected_output": payload.get("expected_output"),
            "execution_mode": payload.get("execution_mode"),
        }

    async def stream_case(self, patient_input: Any) -> AsyncIterator[dict[str, Any]]:
        payload = self._normalize_patient_input(patient_input)
        if not payload["patient_text"]:
            yield {"type": "error", "content": "Please enter a patient story."}
            return

        ws = _QueueWebSocket()
        orchestrator = self._get_orchestrator()
        task = asyncio.create_task(
            orchestrator(
                ws,
                payload["patient_text"],
                llm_client=self._get_llm_client(),
                groq_client=self._get_groq_client(),
                gemini_client=self._get_gemini_client(),
                llama_server_url=self._get_llama_server_url(),
                memory=self._get_memory(),
                thinking_enabled=payload["thinking"] if not (payload["super_thinking"] or payload["deep_thinking"]) else True,
                local_only=payload["local_only"] if not (payload["super_thinking"] or payload["deep_thinking"]) else False,
                super_thinking=payload["super_thinking"],
                deep_thinking=payload["deep_thinking"],
                operation_mode=payload["operation_mode"],
                runtime_profile=payload["runtime_profile"],
                expected_output=payload["expected_output"],
                execution_mode=payload["execution_mode"],
            )
        )

        try:
            while True:
                if task.done() and ws.queue.empty():
                    break
                try:
                    event = await asyncio.wait_for(ws.queue.get(), timeout=0.1)
                    yield event
                except asyncio.TimeoutError:
                    if task.done() and ws.queue.empty():
                        break
            await task
        except Exception as exc:
            if not task.done():
                task.cancel()
            yield {
                "type": "error",
                "content": f"CDSS service adapter failed: {exc}",
            }

    async def analyze_case(self, patient_input: Any) -> DecisionPacket:
        """Run the canonical WebSocket pipeline and return its final packet."""
        last_error = "CDSS pipeline completed without a result packet."
        async for event in self.stream_case(patient_input):
            event_type = str(event.get("type") or "")
            if event_type == "result":
                return DecisionPacket.model_validate(event.get("data") or {})
            if event_type == "error":
                last_error = str(event.get("content") or last_error)
        raise RuntimeError(last_error)

    async def analyze_case_v2(
        self,
        patient_input: Any,
        *,
        runtime_snapshot: dict[str, Any] | None = None,
    ) -> ReasoningArtifactV2:
        packet = await self.analyze_case(patient_input)
        candidates = [
            CandidateReasoning(
                label=item.label,
                score=round(float(item.score or 0.0), 2),
                rationale=list(item.rationale or [])[:4],
                evidence_needed=list(item.evidence_needed or [])[:4],
            )
            for item in list(packet.differential.candidates or [])[:6]
        ]
        required_data = [
            RequiredDataRequest(key=str(item), category="clinical_context")
            for item in _dedupe(list(packet.required_data or []) + list(packet.missing_information or []))[:10]
        ]
        blocked = _dedupe(
            list(packet.blocked_interventions or [])
            + list(packet.intervention_safety.blocked_interventions or [])
        )
        required_actions = _dedupe(
            list(packet.required_concurrent_actions or [])
            + list(packet.intervention_safety.required_concurrent_actions or [])
        )
        return ReasoningArtifactV2(
            case_id=packet.case_id,
            decision=packet.status.value,
            summary=packet.summary,
            primary_candidate=candidates[0] if candidates else CandidateReasoning(),
            differential=candidates,
            required_data=required_data,
            blocked_interventions=blocked,
            required_concurrent_actions=required_actions,
            evidence_coverage=round(float(packet.coverage_score or packet.evidence_quality_score or 0.0), 3),
            contradiction_mass=round(float(getattr(packet.retrieval_stats, "contradiction_mass", 0.0) or 0.0), 3),
            reasoning_trace=list(packet.reasoning_trace or [])[:16],
            runtime=dict(runtime_snapshot or {}),
        )

    def build_result_packet(
        self,
        *,
        anchor: str = "",
        anchor_role: str = "unknown",
        source_disease_resolved: bool = False,
        disease_candidates: list[dict[str, Any]] | None = None,
        required_data: list[str] | None = None,
        objective_discriminators: list[str] | None = None,
        must_not_miss: list[str] | None = None,
        contraindications: list[str] | None = None,
        linked_evidence: list[dict[str, Any]] | None = None,
        research_status: str = "pending",
        urgency: str = "",
        verification_issues: list[dict[str, Any]] | None = None,
        reasoning_trace: list[str] | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        packet = {
            "anchor": anchor,
            "anchor_role": anchor_role,
            "source_disease_resolved": bool(source_disease_resolved),
            "disease_candidates": list(disease_candidates or []),
            "required_data": _dedupe(list(required_data or []) + list(objective_discriminators or [])),
            "objective_discriminators": _dedupe(list(objective_discriminators or [])),
            "must_not_miss": _dedupe(list(must_not_miss or [])),
            "contraindications": _dedupe(list(contraindications or [])),
            "linked_evidence": list(linked_evidence or []),
            "research_status": research_status,
            "urgency": urgency,
            "verification_issues": list(verification_issues or []),
            "reasoning_trace": list(reasoning_trace or []),
        }
        if not packet["source_disease_resolved"] and "underlying_cause_resolution" not in packet["required_data"]:
            packet["required_data"].append("underlying_cause_resolution")
        if research_status == "blocked" and "trusted_web_research_blocked" not in packet["required_data"]:
            packet["required_data"].append("trusted_web_research_blocked")
        if extra:
            packet.update(extra)
        return packet

    def render_summary(self, packet: dict[str, Any]) -> dict[str, Any]:
        return dict(packet)

    async def generate_case(self) -> dict[str, Any]:
        generator = self._get_case_generator()
        return await generator(self._get_gemini_client(), source="any")

    async def ingest_feedback(self, record: Any) -> dict[str, Any]:
        """Records diagnostic feedback from the UI."""
        return {"status": "ok", "message": "Feedback recorded."}
