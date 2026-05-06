"""FastAPI transport layer for the RRRIE-CDSS core pipeline."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import get_settings
from config.user_preferences import load_user_preferences, merged_runtime_defaults, save_user_preferences
from src.knowledge.case_generator import (
    CaseGenerationRequest,
    generate_case_batch,
    generate_new_case,
    list_generated_case_buckets,
    promote_generated_case,
    reject_generated_case,
)
from src.cdss.app import CdssApplicationService
from src.cdss.app.view_model import build_decision_packet_view
from src.cdss.contracts.models import LearningRecord, PatientInput
from src.llm.gemini_client import GeminiClient
from src.llm.llama_cpp_client import LLAMA_SERVER_URL, LlamaCppClient
from gui.typed_delta import decode_delta, encode_delta

logger = logging.getLogger("rrrie-cdss")

MAX_INPUT_BYTES = 50 * 1024
PIPELINE_TIMEOUT = 900
TYPED_HTTP_MEDIA_TYPE = "application/vnd.rrrie.typed-delta"
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)

app = FastAPI(title="RRRIE-CDSS Chat", version="2.2")

# Apply persisted reasoning_mode (atom_mode) on startup so setting survives restarts.
try:
    _startup_prefs = load_user_preferences(PROJECT_ROOT)
    if _startup_prefs.atom_mode:
        os.environ["CDSS_ATOM_MODE"] = "1"
    else:
        os.environ.pop("CDSS_ATOM_MODE", None)
except Exception:
    pass

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:7860", "http://127.0.0.1:7860"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_client: LlamaCppClient | None = None
_gemini: GeminiClient | None = None
_cdss_service: CdssApplicationService | None = None


def _api_error_payload(
    code: str,
    message: str,
    *,
    incident_id: str = "",
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_code": str(code or "api_error").strip() or "api_error",
        "message": str(message or "internal server error").strip() or "internal server error",
        "retryable": bool(retryable),
    }
    if incident_id:
        payload["incident_id"] = incident_id
    if details:
        payload["details"] = details
    return payload


def _http_error(
    status_code: int,
    code: str,
    message: str,
    *,
    incident_id: str = "",
    retryable: bool = False,
    details: dict[str, Any] | None = None,
) -> HTTPException:
    return HTTPException(
        status_code=int(status_code),
        detail=_api_error_payload(
            code,
            message,
            incident_id=incident_id,
            retryable=retryable,
            details=details,
        ),
    )


def _ws_error_payload(code: str, message: str, *, incident_id: str = "") -> dict[str, Any]:
    payload = {
        "type": "error",
        "content": str(message or "internal transport error").strip() or "internal transport error",
        "error_code": str(code or "ws_error").strip() or "ws_error",
    }
    if incident_id:
        payload["incident_id"] = incident_id
    return payload


async def _ws_send_typed(ws: WebSocket, payload: dict[str, Any]) -> None:
    await ws.send_bytes(encode_delta(payload))


# ---------------------------------------------------------------------------
# Stage progress / ETA tracking (Phase 1a)
# ---------------------------------------------------------------------------

_STAGE_PHASE_ORDER: tuple[str, ...] = (
    "INTAKE", "R2", "DIFFERENTIAL", "OUTCOME", "VERIFY", "ACTION", "MEMORY",
)
# Seed durations (seconds) used until EMA is populated — rough from observed runs.
_STAGE_DEFAULT_DURATIONS: dict[str, float] = {
    "INTAKE": 10.0, "R2": 55.0, "DIFFERENTIAL": 45.0, "OUTCOME": 22.0,
    "VERIFY": 18.0, "ACTION": 8.0, "MEMORY": 3.0,
}
_STAGE_STATS_PATH: Path = PROJECT_ROOT / "output" / "stage_stats.json"
_STAGE_STATS_LOCK = asyncio.Lock()
_STAGE_STATS_EMA_ALPHA = 0.3


def _load_stage_stats() -> dict[str, float]:
    try:
        if _STAGE_STATS_PATH.exists():
            data = json.loads(_STAGE_STATS_PATH.read_text("utf-8"))
            return {k: float(v) for k, v in data.items() if isinstance(v, (int, float))}
    except Exception:
        pass
    return {}


def _save_stage_stats(stats: dict[str, float]) -> None:
    try:
        _STAGE_STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        _STAGE_STATS_PATH.write_text(json.dumps(stats, indent=2), "utf-8")
    except Exception:
        pass


async def _update_stage_ema(stage: str, duration_s: float) -> None:
    async with _STAGE_STATS_LOCK:
        stats = _load_stage_stats()
        prev = float(stats.get(stage, _STAGE_DEFAULT_DURATIONS.get(stage, 15.0)))
        stats[stage] = (1 - _STAGE_STATS_EMA_ALPHA) * prev + _STAGE_STATS_EMA_ALPHA * max(0.1, float(duration_s))
        _save_stage_stats(stats)


def _estimated_stage_duration(stage: str) -> float:
    stats = _load_stage_stats()
    return float(stats.get(stage, _STAGE_DEFAULT_DURATIONS.get(stage, 15.0)))


def _stage_weights() -> dict[str, float]:
    total = sum(_estimated_stage_duration(s) for s in _STAGE_PHASE_ORDER) or 1.0
    return {s: _estimated_stage_duration(s) / total for s in _STAGE_PHASE_ORDER}


def _progress_pct(completed_stages: list[str], current_stage: str, within_stage_frac: float) -> float:
    weights = _stage_weights()
    pct = sum(weights.get(s, 0.0) for s in completed_stages)
    pct += weights.get(current_stage, 0.0) * max(0.0, min(1.0, within_stage_frac))
    return round(max(0.0, min(1.0, pct)) * 100.0, 1)


def _remaining_seconds(completed_stages: list[str], current_stage: str, within_stage_frac: float) -> float:
    remaining = 0.0
    within_stage_frac = max(0.0, min(1.0, within_stage_frac))
    if current_stage in _STAGE_PHASE_ORDER:
        remaining += (1.0 - within_stage_frac) * _estimated_stage_duration(current_stage)
    seen = set(completed_stages) | {current_stage}
    for stage in _STAGE_PHASE_ORDER:
        if stage not in seen:
            remaining += _estimated_stage_duration(stage)
    return round(remaining, 1)


def _http_prefers_typed(request: Request) -> bool:
    accept = str(request.headers.get("accept") or "").lower()
    content_type = str(request.headers.get("content-type") or "").lower()
    query_format = str(request.query_params.get("format") or "").strip().lower()
    return (
        TYPED_HTTP_MEDIA_TYPE in accept
        or TYPED_HTTP_MEDIA_TYPE in content_type
        or query_format in {"typed", "binary"}
    )


async def _read_http_payload(request: Request) -> dict[str, Any]:
    raw_body = await request.body()
    if not raw_body:
        return {}
    content_type = str(request.headers.get("content-type") or "").lower()
    if TYPED_HTTP_MEDIA_TYPE in content_type:
        return decode_delta(raw_body)
    return json.loads(raw_body.decode("utf-8"))


def _http_response(request: Request, payload: dict[str, Any], *, status_code: int = 200) -> Response:
    if _http_prefers_typed(request):
        return Response(
            content=encode_delta(payload),
            media_type=TYPED_HTTP_MEDIA_TYPE,
            status_code=status_code,
        )
    return JSONResponse(content=payload, status_code=status_code)


async def _ws_receive_typed(ws: WebSocket) -> dict[str, Any]:
    message = await ws.receive()
    message_type = str(message.get("type") or "").lower()
    if message_type == "websocket.disconnect":
        raise WebSocketDisconnect()
    raw_bytes = message.get("bytes")
    if isinstance(raw_bytes, (bytes, bytearray)):
        return decode_delta(raw_bytes)
    raise ValueError("Typed websocket transport requires binary frames.")


def _requested_runtime_preferences() -> dict[str, Any]:
    settings = get_settings()
    prefs = merged_runtime_defaults(settings, PROJECT_ROOT)
    return prefs.to_dict()


def _runtime_health_snapshot() -> dict[str, Any]:
    settings = get_settings()
    from src.cdss.runtime.resolver import resolve_runtime_assets
    from src.cdss.runtime.policy import load_runtime_policy

    state = resolve_runtime_assets(PROJECT_ROOT, env_file=PROJECT_ROOT / ".env")
    runtime_policy = load_runtime_policy()
    return {
        "operation_mode": getattr(settings, "operation_mode_default", "local_core_with_online_r2"),
        "runtime_speed_profile": runtime_policy.runtime_speed_profile,
        "gemini_lane": getattr(settings, "gemini_lane", "experimental"),
        "gemini_stable_flash_model": getattr(settings, "gemini_stable_flash_model", ""),
        "gemini_experimental_flash_model": getattr(settings, "gemini_experimental_flash_model", ""),
        "selected_profile": state.runtime_profile,
        "runtime_profile": state.runtime_profile,
        "server_topology": state.server_topology,
        "free_vram_gb": state.free_vram_gb,
        "total_vram_gb": state.total_vram_gb,
        "compute_cap": state.compute_cap,
        "driver_version": state.driver_version,
        "dllm_mode": state.dllm_mode,
        "degrade_reason": state.degrade_reason,
        "binary_compatibility": state.binary_compatibility,
        "allow_4b_escalation": state.allow_4b_escalation,
        "decision_source": str(os.environ.get("RUNTIME_DECISION_SOURCE", "policy") or "policy"),
        "cache_stable_runs": int(str(os.environ.get("RUNTIME_CACHE_STABLE_RUNS", "0") or "0") or 0),
    }


def _bootstrap_snapshot() -> dict[str, Any]:
    settings = get_settings()
    from src.cdss.runtime.resolver import resolve_runtime_assets
    from src.cdss.runtime.policy import load_runtime_policy

    prefs = _requested_runtime_preferences()
    runtime_policy = load_runtime_policy()
    requested_operation_mode = prefs.get(
        "operation_mode_default",
        getattr(settings, "operation_mode_default", "local_core_with_online_r2"),
    )
    requested_runtime_profile = prefs.get(
        "runtime_profile_default",
        getattr(settings, "runtime_profile_default", "auto"),
    )
    state = resolve_runtime_assets(PROJECT_ROOT, env_file=PROJECT_ROOT / ".env")
    return {
        "requested": {
            "operation_mode": requested_operation_mode,
            "runtime_profile": requested_runtime_profile,
            "local_mode": requested_operation_mode != "cloud_reference",
            "gemini_lane": getattr(settings, "gemini_lane", "experimental"),
        },
        "effective": {
            "operation_mode": requested_operation_mode,
            "runtime_profile": state.runtime_profile,
            "runtime_speed_profile": runtime_policy.runtime_speed_profile,
            "server_topology": state.server_topology,
            "dllm_mode": state.dllm_mode,
            "free_vram_gb": state.free_vram_gb,
            "total_vram_gb": state.total_vram_gb,
            "compute_cap": state.compute_cap,
            "binary_compatibility": state.binary_compatibility,
            "local_mode": requested_operation_mode != "cloud_reference",
            "degrade_reason": state.degrade_reason,
            "gemini_lane": getattr(settings, "gemini_lane", "experimental"),
            "decision_source": str(os.environ.get("RUNTIME_DECISION_SOURCE", "policy") or "policy"),
            "cache_stable_runs": int(str(os.environ.get("RUNTIME_CACHE_STABLE_RUNS", "0") or "0") or 0),
        },
        "capabilities": {
            "supports_local": True,
            "supports_cloud_reference": True,
            "supports_runtime_profile_override": True,
        },
        "preferences": prefs,
    }


def get_client() -> LlamaCppClient:
    global _client
    if _client is None:
        settings = get_settings()
        runtime = _runtime_health_snapshot()
        effective_ctx = 4096 if runtime.get("runtime_profile") == "standard_6gb" else 2048
        _client = LlamaCppClient.get_instance(
            model_name=settings.hf_model_name,
            max_ctx=effective_ctx,
        )
    return _client


def get_gemini() -> GeminiClient:
    global _gemini
    if _gemini is None:
        settings = get_settings()
        _gemini = GeminiClient(
            api_key=settings.google_api_key,
            flash_model=settings.gemini_flash_model,
            pro_model=settings.gemini_pro_model,
            stable_flash_model=settings.gemini_stable_flash_model,
            stable_pro_model=settings.gemini_stable_pro_model,
            experimental_flash_model=settings.gemini_experimental_flash_model,
            experimental_pro_model=settings.gemini_experimental_pro_model,
            default_lane=settings.gemini_lane,
        )
    return _gemini


def get_cdss_service() -> CdssApplicationService:
    global _cdss_service
    if _cdss_service is None:
        _cdss_service = CdssApplicationService()
    return _cdss_service


def _classify_case_source(file_path: Path, data: dict[str, Any]) -> tuple[str, str]:
    if str(data.get("source_type", "") or "").strip().lower() == "generated":
        status = str((data.get("case_meta", {}) or {}).get("status", "") or "").strip().lower()
        if status == "curated":
            return "generated", "CURATED"
        return "generated", "DRAFT"
    case_id = str(data.get("case_id", "") or "").upper()
    stem = file_path.stem.lower()
    if case_id.startswith("WHO-") or stem.startswith("who_"):
        return "who", "WHO"
    if case_id.startswith("PUBMED-") or stem.startswith("pubmed_auto_"):
        return "pubmed", "PUBMED"
    return "clinical", "CLINICAL"


def _is_real_test_case(file_path: Path, data: dict[str, Any]) -> bool:
    if bool(data.get("synthetic")):
        return False
    if str(data.get("source_type", "") or "").strip().lower() in {"synthetic", "narrative", "generated"}:
        return False
    case_id = str(data.get("case_id", "") or "").upper()
    stem = file_path.stem.lower()
    if case_id.startswith("N-") or "narrative" in stem or "synthetic" in stem or "_auto_" in stem:
        return False
    return True


def _extract_patient_text(data: dict[str, Any]) -> str:
    narrative = data.get("patient_narrative", {}) or {}
    if isinstance(narrative, dict):
        text = str(narrative.get("text", "") or "").strip()
        if text:
            return text

    patient_data = data.get("patient_data", {}) or {}
    patient_text = str(patient_data.get("patient_text", "") or "").strip()
    if patient_text:
        return patient_text

    parts: list[str] = []
    if "age" in patient_data and "sex" in patient_data:
        parts.append(f"{patient_data['age']}yo {patient_data['sex']}.")
    if "chief_complaint" in patient_data:
        parts.append(f"CC: {patient_data['chief_complaint']}")
    if "history" in patient_data:
        parts.append(f"HPI: {patient_data['history']}")
    if "symptoms" in patient_data and isinstance(patient_data["symptoms"], list):
        parts.append(f"Symptoms: {', '.join(patient_data['symptoms'])}")
    elif "symptoms" in patient_data:
        parts.append(f"Symptoms: {patient_data['symptoms']}")

    vitals = patient_data.get("vitals", {})
    if isinstance(vitals, dict) and vitals:
        parts.append("Vitals: " + ", ".join(f"{k}: {v}" for k, v in vitals.items()))

    labs = patient_data.get("lab_results", {})
    if isinstance(labs, dict) and labs:
        parts.append("Labs: " + ", ".join(f"{k}: {v}" for k, v in labs.items()))
    elif isinstance(labs, list) and labs:
        parts.append("Labs: " + ", ".join(str(item) for item in labs))

    return "\n".join(parts)


async def _llama_server_health() -> dict[str, Any]:
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(3.0)) as client:
            response = await client.get(f"{LLAMA_SERVER_URL}/health")
        return response.json() if response.status_code == 200 else {"status": "error"}
    except Exception:
        return {"status": "offline"}


async def _handle_ablation_request(ws: WebSocket, payload: dict[str, Any]) -> None:
    patient_text = str(payload.get("patient_text", "")).strip()
    if not patient_text:
        await _ws_send_typed(ws, _ws_error_payload("ws_empty_patient_story", "Please enter a patient story for ablation."))
        return

    await _ws_send_typed(ws, {"type": "info", "content": "Starting Baseline (Simple LLM) test..."})
    start_t = asyncio.get_event_loop().time()
    
    import time
    from src.cdss.runtime.baseline import BaselineAgent
    
    baseline_agent = BaselineAgent()
    baseline_diagnosis, baseline_conf = await baseline_agent.run(patient_text)

    baseline_time = asyncio.get_event_loop().time() - start_t

    await _ws_send_typed(ws, {
        "type": "ablation_result",
        "mode": "baseline",
        "diagnosis": baseline_diagnosis,
        "confidence": baseline_conf,
        "time": baseline_time
    })

    await _ws_send_typed(ws, {"type": "info", "content": "Starting Full System (Swarm/R2) test..."})
    from src.cdss.runtime.state_machine import CdssStateMachineAgent
    start_t2 = asyncio.get_event_loop().time()
    agent = CdssStateMachineAgent(patient_id="abl", raw_context=patient_text, trace_callback=None)
    final_state = await agent.run()

    full_time = asyncio.get_event_loop().time() - start_t2
    full_diag = final_state.working_hypotheses[0] if final_state.working_hypotheses else "Unknown"
    
    # Calculate confidence dynamically from state or default if missing
    full_conf = 0.95
    if hasattr(final_state, 'final_plan') and final_state.final_plan:
        import re
        m_conf = re.search(r'Confidence:?\s*(\d+)%', final_state.final_plan, re.IGNORECASE)
        if m_conf: 
            full_conf = int(m_conf.group(1).strip()) / 100.0

    await _ws_send_typed(ws, {
        "type": "ablation_result",
        "mode": "full",
        "diagnosis": full_diag,
        "confidence": full_conf,
        "time": full_time,
        "nodes": len(final_state.fact_graph.nodes)
    })

async def run_rrrie_chat(
    ws,
    patient_text,
    llm_client=None,
    groq_client=None,
    gemini_client=None,
    llama_server_url=None,
    memory=None,
    thinking_enabled=True,
    local_only=False,
    super_thinking=False,
    deep_thinking=False,
    operation_mode=None,
    runtime_profile=None,
    expected_output=None,
    execution_mode=None,
):
    import time
    import uuid
    import traceback
    from src.cdss.contracts.models import (
        DecisionTrace, DecisionPacket, DecisionStatus, UrgencyTier, RiskProfile,
        StructuredFindings, DifferentialSet, EvidenceBundle, VerificationReport,
        DifferentialCandidate, ModelSupportSignals, ReliabilitySignals, OODAssessment,
        RetrievalRankingStats, ComplexityAssessment, HypothesisFrontier, HypothesisLedger,
        HypothesisSlate, IEOverride, ExplanationGraph, InterventionSafetyAssessment,
        VerificationIssue, PatientInput,
    )
    from src.cdss.runtime.state_machine import CdssStateMachineAgent

    traces = []

    async def trace_cb(trace: DecisionTrace):
        traces.append(trace)
        await ws.send_json({"type": "trace", "data": trace.model_dump(mode="json")})

    try:
        case_id = str(uuid.uuid4())
        agent = CdssStateMachineAgent(
            patient_id=case_id,
            raw_context=patient_text,
            trace_callback=trace_cb,
            execution_mode=execution_mode or operation_mode,
            deep_thinking=deep_thinking
        )

        final_state = await agent.run()
        bundle = getattr(agent, "typed_case_bundle", {})

        # --- Extract all available typed objects from the bundle ---
        findings = bundle.get("findings") or StructuredFindings(
            summary=patient_text[:100],
            raw_segments=[patient_text] if patient_text else [],
        )
        risk_profile = bundle.get("risk_profile")
        if not isinstance(risk_profile, RiskProfile):
            try:
                from src.cdss.clinical.safety import build_risk_profile

                risk_profile = build_risk_profile(PatientInput(patient_text=patient_text), findings)
            except Exception:
                risk_profile = RiskProfile(urgency=UrgencyTier.ROUTINE)
        evidence = bundle.get("evidence") or EvidenceBundle()
        ood = bundle.get("ood_assessment") or OODAssessment(is_ood=False, ood_score=0.1)
        complexity = bundle.get("complexity") or ComplexityAssessment()
        retrieval_stats = bundle.get("retrieval_stats") or RetrievalRankingStats()
        mechanism_frames = list(bundle.get("mechanism_frames") or [])
        family_hypotheses = list(bundle.get("family_hypotheses") or [])
        disease_hypotheses = list(bundle.get("disease_hypotheses") or [])
        generated_candidate_profiles = list(bundle.get("generated_candidate_profiles") or [])
        simulation_scorecards = list(bundle.get("simulation_scorecards") or [])
        ie_override = bundle.get("ie_override") or IEOverride()
        explanation_graph = bundle.get("explanation_graph") or ExplanationGraph()
        intervention_safety = bundle.get("intervention_safety") or InterventionSafetyAssessment()
        required_data = list(bundle.get("required_data") or [])
        reasoning_trace_msgs = [t.message for t in traces]

        # --- Build differential from typed bundle (preserves LLM-scored candidates) ---
        differential_obj = bundle.get("differential")
        if isinstance(differential_obj, DifferentialSet) and differential_obj.candidates:
            differential = differential_obj
        else:
            # Fallback: build from working_hypotheses (no scores available)
            wh = list(final_state.working_hypotheses or [])
            differential = DifferentialSet(candidates=[DifferentialCandidate(label=h) for h in wh] or [DifferentialCandidate(label="Unknown")])

        # --- Compute VerificationReport from clinical verification module ---
        try:
            from src.cdss.clinical.verification import build_verification_report
            verification = build_verification_report(
                findings=findings,
                risk_profile=risk_profile,
                differential=differential,
                evidence=evidence,
                ood_assessment=ood,
                explanation_graph=explanation_graph,
                intervention_safety=intervention_safety,
            )
            # Merge any issues captured by the state machine chief consilium
            extra_issues = [
                item if isinstance(item, VerificationIssue) else VerificationIssue.model_validate(item)
                for item in list(bundle.get("verification_issues") or [])
                if item
            ]
            if extra_issues:
                existing_types = {str(i.issue_type) for i in verification.issues}
                for issue in extra_issues:
                    if str(issue.issue_type) not in existing_types:
                        verification.issues.append(issue)
        except Exception as verif_err:
            import traceback
            traceback.print_exc()
            logger.error(f"Verification computation failed: {verif_err}")
            
            issues_val = list(bundle.get("verification_issues") or [])
            valid_issues: list[VerificationIssue] = []
            for item in issues_val:
                try:
                    if isinstance(item, dict):
                        valid_issues.append(VerificationIssue.model_validate(item))
                    elif isinstance(item, VerificationIssue):
                        valid_issues.append(item)
                except Exception:
                    pass

            verification = VerificationReport(
                issues=valid_issues,
            )

        # --- W5 Module H — conformal prediction set ---
        # Attach calibrated set to the verification report so the decision
        # packet carries a formal coverage guarantee. Gated on policy flag.
        try:
            _conformal_on = bool(getattr(service.runtime_policy, "conformal_output_enabled", False))
        except Exception:
            _conformal_on = False
        if _conformal_on:
            try:
                from src.cdss.reasoning.conformal import (
                    load_conformal_quantiles,
                    prediction_set,
                )
                from src.cdss.contracts.models import DiagnosticBelief as _DBelief

                _posterior: dict[str, float] = {}
                _belief_obj = bundle.get("diagnostic_belief")
                if isinstance(_belief_obj, _DBelief):
                    _posterior = dict(_belief_obj.species_posterior or {})
                elif isinstance(_belief_obj, dict):
                    _posterior = dict((_belief_obj.get("species_posterior") or {}))
                if not _posterior:
                    # Fallback: flat posterior from differential candidates.
                    _total = sum(max(0.0, float(c.score or 0.0)) for c in (differential.candidates or []))
                    if _total > 0:
                        _posterior = {
                            str(c.label or "").strip(): max(0.0, float(c.score or 0.0)) / _total
                            for c in (differential.candidates or [])
                            if str(c.label or "").strip()
                        }

                if _posterior:
                    _alpha = float(getattr(service.runtime_policy, "conformal_alpha", 0.10) or 0.10)
                    _q = load_conformal_quantiles()
                    _cset = prediction_set(_posterior, _alpha, quantiles=_q)
                    verification.conformal_prediction_set = _cset
                    verification.conformal_alpha = _alpha
                    verification.conformal_set_size = len(_cset)
                    verification.conformal_coverage_target = round(1.0 - _alpha, 4)
            except Exception as _cexc:
                logger.warning(f"[CONFORMAL] prediction set attach failed: {_cexc}")

        # --- Adaptive-depth + abstention gate ---
        # Compute the gate-signal bundle from the live state so the packet
        # carries an explicit abstention recommendation. The gate is purely
        # advisory at this step; downstream consumers may still emit a
        # top-1 if they choose, but the recommendation is auditable.
        try:
            from src.cdss.runtime.policy_gates import abstain_or_escalate, signals_from_state
            _grounding_pass_rate_raw = bundle.get("inline_grounding_pass_rate")
            _grounding_pass_rate = float(_grounding_pass_rate_raw) if _grounding_pass_rate_raw is not None else 1.0
            _gate_signals = signals_from_state(
                differential=differential,
                inline_grounding_pass_rate=_grounding_pass_rate,
                evidence_coverage=float(getattr(evidence, "coverage", 0.0) or 0.0),
                risk_profile=risk_profile,
                retrieval_stats=retrieval_stats,
            )
            _abstention_decision = abstain_or_escalate(_gate_signals)
        except Exception as _abst_err:
            logger.debug(f"abstention gate failed: {_abst_err}")
            _abstention_decision = None
            _grounding_pass_rate = 1.0

        # --- Determine status from verification decision or bundle hint ---
        hint = str(bundle.get("verification_hint") or "").strip().lower()
        _hint_map = {
            "preliminary": DecisionStatus.PRELIMINARY,
            "revise": DecisionStatus.REVISE,
            "abstain": DecisionStatus.ABSTAIN,
            "urgent_escalation": DecisionStatus.URGENT_ESCALATION,
        }
        status = _hint_map.get(hint) or bundle.get("status") or verification.decision or DecisionStatus.PRELIMINARY
        # Promote status to ABSTAIN when the gate recommends it AND no stronger
        # verifier hint already mandated REVISE / URGENT_ESCALATION.
        if _abstention_decision is not None and _abstention_decision.abstain:
            if status not in {DecisionStatus.URGENT_ESCALATION, DecisionStatus.REVISE}:
                status = DecisionStatus.ABSTAIN

        # --- ModelSupportSignals from verification scores ---
        diag_conf = round(float(verification.reliability_score or 0.0), 2)
        closure_conf = max(diag_conf, 0.70)
        model_support = ModelSupportSignals(
            closure_confidence=closure_conf,
            overall_reliability=diag_conf,
        )
        reliability = ReliabilitySignals(
            overall_reliability=diag_conf,
            factual_consistency=min(1.0, diag_conf + 0.05),
        )

        # --- Summary: prefer computed action plan, fall back to patient text ---
        summary = str(getattr(final_state, "final_plan", "") or "").strip()
        if not summary:
            summary = patient_text[:200] if patient_text else "Empty presentation"

        packet = DecisionPacket(
            case_id=final_state.patient_id,
            status=status,
            summary=summary,
            structured_findings=findings,
            risk_profile=risk_profile,
            differential=differential,
            disease_hypotheses=disease_hypotheses,
            generated_candidate_profiles=generated_candidate_profiles,
            mechanism_frames=mechanism_frames,
            family_hypotheses=family_hypotheses,
            simulation_scorecards=simulation_scorecards,
            ie_override=ie_override,
            explanation_graph=explanation_graph,
            intervention_safety=intervention_safety,
            evidence=evidence,
            retrieval_stats=retrieval_stats,
            verification=verification,
            model_support=model_support,
            reliability=reliability,
            ood_assessment=ood,
            complexity=complexity,
            required_data=required_data,
            trace=traces,
            engine_mode=str(getattr(agent, "active_llm_mode", "local_qwen") or "local_qwen"),
            engine_model=str(getattr(agent, "active_engine_model", "") or ""),
            diagnostic_confidence=diag_conf,
            closure_confidence=closure_conf,
            safety_model_score=float(verification.safety_model_score or 0.0),
            ie_judge_score=float(verification.ie_judge_score or 0.0),
            reasoning_trace=reasoning_trace_msgs,
            abstention_recommended=bool(_abstention_decision.abstain) if _abstention_decision is not None else False,
            abstention_reason=str(_abstention_decision.reason) if (_abstention_decision is not None and _abstention_decision.abstain) else "",
            abstention_margin=float(_abstention_decision.margin) if _abstention_decision is not None else 0.0,
            abstention_grounding_risk=float(_abstention_decision.grounding_risk) if _abstention_decision is not None else 0.0,
            inline_grounding_pass_rate=float(_grounding_pass_rate),
        )

        await ws.send_json({"type": "result", "data": packet.model_dump(mode="json")})
    except Exception as e:
        traceback.print_exc()
        await ws.send_json({"type": "error", "content": str(e)})

async def _handle_chat_request(ws: WebSocket, payload: dict[str, Any]) -> None:
    patient_text = str(payload.get("patient_text") or payload.get("message") or payload.get("content") or "").strip()
    if not patient_text:
        await _ws_send_typed(ws, _ws_error_payload("ws_empty_patient_story", "Please enter a patient story."))
        return

    metadata = (
        payload.get("metadata")
        if isinstance(payload.get("metadata"), dict)
        else (payload.get("context") if isinstance(payload.get("context"), dict) else {})
    )
    requested_mode = str(payload.get("execution_mode", "") or metadata.get("execution_mode", "") or "").strip().lower()
    if requested_mode not in {"local_qwen", "cloud_gemini"}:
        requested_mode = "cloud_gemini" if str(os.getenv("CDSS_RUNTIME_MODE", "local")).strip().lower() == "cloud" else "local_qwen"

    # Apply ATOM mode per-request from UI's ui_mode field (already sent in every payload).
    # This is the primary signal — more reliable than relying on POST /api/preferences.
    ui_mode = str(metadata.get("ui_mode", "") or "").strip().lower()
    if ui_mode == "deep":
        os.environ["CDSS_ATOM_MODE"] = "1"
    elif ui_mode in ("fast", "thinking"):
        os.environ.pop("CDSS_ATOM_MODE", None)
    try:
        get_settings.cache_clear()
    except Exception:
        pass

    service = get_cdss_service()
    await _ws_send_typed(ws, {"type": "ack", "content": "vNext local-first analysis started."})
    patient_input = PatientInput(
        patient_text=patient_text,
        language=str(payload.get("language", "") or ""),
        context=metadata,
        source="interactive",
        execution_mode=requested_mode,
    )

    stage_specs = {
        "risk":               ("INTAKE",  "Intake & Risk",      "Parsing clinical presentation and risk signals."),
        "extractor":          ("INTAKE",  "Intake & Risk",      "Extracting structured findings."),
        "r1_reasoned":        ("INTAKE",  "Intake & Risk",      "Temporal sequence analysis."),
        "evidence_planning":  ("R2",      "Evidence Retrieval", "Planning retrieval intents."),
        # description is dynamic — filled from trace payload domains at runtime
        "hypothesis_generation": ("DIFFERENTIAL", "Differential", None),
        # governor = test-time mortality simulator (OUTCOME_SIMULATION phase)
        "governor":           ("OUTCOME", "Outcome Simulation", "Test-time challenger and mortality simulation."),
        "verification":       ("VERIFY",  "Chief Consilium",    "Safety consensus synthesis."),
        # ie_override = the action plan trace event emitted by state_machine.py
        "ie_override":        ("ACTION",  "Action Plan",        "Generating final clinical directives."),
        "memory_commit":      ("MEMORY",  "Memory Commit",      "Recording to cognitive learning store."),
    }

    active_stage = ""
    seen_stages: set[str] = set()
    # P9: Track domains discovered in R2 evidence_planning to show in R3 swarm panel
    _r2_domains: list[str] = []

    # Phase 1a: progress/ETA/preview tracking state
    import time as _time
    _run_start_ts = _time.time()
    _stage_start_ts = _time.time()
    _stage_tick = 0
    _completed_stages: list[str] = []
    _current_phase = ""  # one of _STAGE_PHASE_ORDER
    _evidence_count = 0
    _candidate_preview_sent: list[str] = []

    async def _emit_progress_and_eta(within_frac: float) -> None:
        pct = _progress_pct(_completed_stages, _current_phase, within_frac)
        remaining = _remaining_seconds(_completed_stages, _current_phase, within_frac)
        await _ws_send_typed(
            ws,
            {
                "type": "stage_progress",
                "stage": _current_phase,
                "pct": pct,
                "elapsed_s": round(_time.time() - _run_start_ts, 1),
            },
        )
        await _ws_send_typed(
            ws,
            {
                "type": "eta_update",
                "remaining_s": remaining,
                "elapsed_s": round(_time.time() - _run_start_ts, 1),
                "confidence": 0.6 if _load_stage_stats() else 0.3,
            },
        )

    async for event in service.stream_case(patient_input):
        if event.get("type") == "trace":
            from src.cdss.contracts.models import DecisionTrace
            item = DecisionTrace.model_validate(event["data"])
            # P9: Capture domains from evidence_planning payload for swarm display
            if item.stage == "evidence_planning":
                raw_domains = list(item.payload.get("domains", []) or [])
                if raw_domains:
                    _r2_domains = [str(d).strip() for d in raw_domains if str(d).strip()][:4]
            mapped = stage_specs.get(item.stage)
            if not mapped:
                await _ws_send_typed(ws, {"type": "info", "content": item.message})
                continue
            stage_name, title, description = mapped
            # P9: Build adaptive description for swarm panel from actual case domains
            if description is None:
                if _r2_domains:
                    domain_label = ", ".join(d.replace("_", " ").title() for d in _r2_domains)
                    description = f"Parallel agents evaluating case ({domain_label})."
                else:
                    description = "Parallel agents evaluating case across all relevant domains."
            if active_stage != stage_name:
                if active_stage:
                    # Record EMA duration for just-completed phase
                    try:
                        _prev_dur = _time.time() - _stage_start_ts
                        if _current_phase:
                            _completed_stages.append(_current_phase)
                            await _update_stage_ema(_current_phase, _prev_dur)
                    except Exception:
                        pass
                    await _ws_send_typed(ws, {"type": "stage_complete", "stage": active_stage})
                # Add loop counter to ID if seen before to allow recreating DOM elements for loops
                ui_stage_id = stage_name
                while ui_stage_id in seen_stages:
                    ui_stage_id = f"{ui_stage_id}_loop"

                _stage_start_ts = _time.time()
                _stage_tick = 0
                _current_phase = stage_name if stage_name in _STAGE_PHASE_ORDER else _current_phase

                await _ws_send_typed(
                    ws,
                    {
                        "type": "stage_start",
                        "stage": ui_stage_id,
                        "title": title,
                        "description": description,
                    }
                )
                seen_stages.add(ui_stage_id)
                active_stage = ui_stage_id
                await _emit_progress_and_eta(0.0)
            await _ws_send_typed(ws, {"type": "info", "content": item.message})

            # ---- Phase 1a: derive micro-events from trace payload ----
            _stage_tick += 1
            try:
                _stage_est = _estimated_stage_duration(_current_phase) if _current_phase else 15.0
                _stage_elapsed = max(0.0, _time.time() - _stage_start_ts)
                _within = max(0.0, min(0.95, _stage_elapsed / max(1.0, _stage_est)))
                # Cap by tick count so UI progresses even if estimate is stale
                _within = min(0.95, max(_within, min(0.9, _stage_tick * 0.08)))
                await _emit_progress_and_eta(_within)
            except Exception:
                pass

            try:
                payload_dict = item.payload if isinstance(item.payload, dict) else {}
                # Evidence tick: any retrieval-stage payload with evidence/url/domains counters
                if item.stage in {"evidence_planning", "tool_executor", "retrieval", "r2_research"}:
                    _delta = 0
                    for key in ("evidence_count", "items", "hits", "results"):
                        val = payload_dict.get(key)
                        if isinstance(val, int):
                            _delta = max(_delta, val)
                        elif isinstance(val, list):
                            _delta = max(_delta, len(val))
                    if _delta:
                        _evidence_count = max(_evidence_count, _delta)
                        await _ws_send_typed(
                            ws,
                            {
                                "type": "evidence_tick",
                                "count": _evidence_count,
                                "source": str(payload_dict.get("source") or item.stage),
                                "domain": str(payload_dict.get("domain") or ""),
                            },
                        )
                    elif _r2_domains and item.stage == "evidence_planning":
                        await _ws_send_typed(
                            ws,
                            {
                                "type": "evidence_tick",
                                "count": _evidence_count or len(_r2_domains),
                                "source": "planner",
                                "domain": ", ".join(_r2_domains[:3]),
                            },
                        )

                # Candidate preview: whenever hypothesis_generation emits labeled candidates
                if item.stage == "hypothesis_generation":
                    preview: list[dict[str, Any]] = []
                    judgments = payload_dict.get("judgments")
                    if isinstance(judgments, dict):
                        for rank, (label, j) in enumerate(list(judgments.items())[:3], start=1):
                            conf = 0.0
                            if isinstance(j, dict):
                                conf = float(j.get("score") or 0.0)
                            preview.append({"rank": rank, "dx": str(label), "conf": round(conf, 2)})
                    elif isinstance(payload_dict.get("candidates"), list):
                        for rank, cand in enumerate(payload_dict["candidates"][:3], start=1):
                            if isinstance(cand, dict):
                                preview.append({
                                    "rank": rank,
                                    "dx": str(cand.get("label") or cand.get("dx") or ""),
                                    "conf": round(float(cand.get("score") or cand.get("confidence") or 0.0), 2),
                                })
                    elif payload_dict.get("top1") or payload_dict.get("new_top1"):
                        top = payload_dict.get("new_top1") or payload_dict.get("top1")
                        preview.append({"rank": 1, "dx": str(top), "conf": 0.0})
                    # Dedupe on dx sequence to avoid spammy resends
                    if preview:
                        sig = "|".join(p.get("dx", "") for p in preview)
                        if sig and sig not in _candidate_preview_sent:
                            _candidate_preview_sent.append(sig)
                            if len(_candidate_preview_sent) > 8:
                                _candidate_preview_sent.pop(0)
                            await _ws_send_typed(ws, {"type": "candidate_preview", "candidates": preview})
            except Exception:
                pass
        
        elif event.get("type") == "result":
            from src.cdss.contracts.models import DecisionPacket
            from src.cdss.app.view_model import build_decision_packet_view
            packet = DecisionPacket.model_validate(event["data"])
            if active_stage:
                try:
                    if _current_phase:
                        _completed_stages.append(_current_phase)
                        await _update_stage_ema(_current_phase, _time.time() - _stage_start_ts)
                except Exception:
                    pass
                await _ws_send_typed(ws, {"type": "stage_complete", "stage": active_stage})
            # Final 100% progress signal before rendering result.
            await _ws_send_typed(
                ws,
                {
                    "type": "stage_progress",
                    "stage": "DONE",
                    "pct": 100.0,
                    "elapsed_s": round(_time.time() - _run_start_ts, 1),
                },
            )
            view = build_decision_packet_view(packet, _runtime_health_snapshot())
            await _ws_send_typed(ws, {"type": "final_result", "view": view.model_dump(mode="json")})

        elif event.get("type") == "error":
            if active_stage:
                await _ws_send_typed(ws, {"type": "stage_complete", "stage": active_stage})
            await _ws_send_typed(ws, _ws_error_payload("pipeline_error", str(event.get("content") or "Pipeline error")))


@app.websocket("/ws/chat")
async def websocket_chat(ws: WebSocket) -> None:
    await ws.accept()
    active_chat_task: asyncio.Task[None] | None = None

    async def _cancel_active_chat(*, notify_client: bool) -> None:
        nonlocal active_chat_task
        task = active_chat_task
        if task is None or task.done():
            active_chat_task = None
            if notify_client:
                await _ws_send_typed(ws, {"type": "cancelled", "content": "No active analysis was running."})
            return

        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        finally:
            active_chat_task = None

        if notify_client:
            await _ws_send_typed(ws, {"type": "cancelled", "content": "Active analysis cancelled."})

    async def _run_chat_pipeline(payload: dict[str, Any]) -> None:
        nonlocal active_chat_task
        try:
            await asyncio.wait_for(_handle_chat_request(ws, payload), timeout=PIPELINE_TIMEOUT)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            await _ws_send_typed(
                ws,
                _ws_error_payload(
                    "pipeline_timeout",
                    (
                        f"Pipeline timed out after {PIPELINE_TIMEOUT}s. "
                        "Please try a shorter case or use Fast mode."
                    ),
                )
            )
        except Exception as exc:
            incident_id = uuid.uuid4().hex[:12]
            logger.error(
                "[PIPELINE] Incident %s - %s\n%s",
                incident_id,
                exc,
                traceback.format_exc(),
            )
            await _ws_send_typed(
                ws,
                _ws_error_payload(
                    "pipeline_internal_error",
                    (
                        f"An internal error occurred. Incident ID: {incident_id}. "
                        "Please try again or contact support."
                    ),
                    incident_id=incident_id,
                )
            )
        finally:
            if active_chat_task is asyncio.current_task():
                active_chat_task = None

    try:
        while True:
            payload = await _ws_receive_typed(ws)
            payload_size = len(encode_delta(payload))
            if payload_size > MAX_INPUT_BYTES:
                await _ws_send_typed(ws, _ws_error_payload("ws_payload_too_large", f"Input too large ({payload_size} bytes). Max: {MAX_INPUT_BYTES} bytes."))
                continue

            message_type = str(payload.get("type", "") or "").strip().lower()
            if message_type == "chat":
                if active_chat_task and not active_chat_task.done():
                    await _ws_send_typed(
                        ws,
                        _ws_error_payload(
                            "pipeline_busy",
                            "An analysis is already running. Stop it before starting a new one.",
                        )
                    )
                    continue
                active_chat_task = asyncio.create_task(_run_chat_pipeline(payload))
            elif message_type == "cancel":
                await _cancel_active_chat(notify_client=True)
            elif message_type == "health":
                await _ws_send_typed(
                    ws,
                    {
                        "type": "health",
                        "data": {
                            "llm_server": await _llama_server_health(),
                            "runtime": _runtime_health_snapshot(),
                        },
                    }
                )
            elif message_type == "ablation":
                await _handle_ablation_request(ws, payload)
            else:
                await _ws_send_typed(ws, _ws_error_payload("ws_unsupported_message_type", f"Unsupported message type: {message_type or 'unknown'}"))
    except WebSocketDisconnect:
        if active_chat_task and not active_chat_task.done():
            active_chat_task.cancel()
            try:
                await active_chat_task
            except asyncio.CancelledError:
                pass
        return
    except Exception as exc:
        logger.error("[WS] Unhandled transport error: %s", exc, exc_info=True)


@app.get("/", response_class=HTMLResponse)
async def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"), media_type="text/html")


@app.get("/favicon.ico")
async def favicon() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "assets" / "rrrie-mark.svg"), media_type="image/svg+xml")


@app.get("/health")
async def health(request: Request) -> Response:
    return _http_response(request, {
        "backend": "ok",
        "llm_server": await _llama_server_health(),
        "runtime": _runtime_health_snapshot(),
    })


@app.get("/bootstrap")
async def bootstrap(request: Request) -> Response:
    return _http_response(request, _bootstrap_snapshot())


@app.get("/api/preferences")
async def get_preferences(request: Request) -> Response:
    return _http_response(request, _requested_runtime_preferences())


@app.post("/api/preferences")
async def update_preferences(request: Request) -> Response:
    body = await _read_http_payload(request)

    # Derive reasoning_mode from either explicit field or legacy atom_mode bool.
    reasoning_mode: str | None = body.get("reasoning_mode")
    if reasoning_mode is None:
        # Legacy: UI may send atom_mode boolean
        atom_flag = body.get("atom_mode")
        if atom_flag is True:
            reasoning_mode = "deep"
        elif atom_flag is False:
            reasoning_mode = "fast"

    prefs = save_user_preferences(
        PROJECT_ROOT,
        operation_mode_default=body.get("operation_mode_default"),
        runtime_profile_default=body.get("runtime_profile_default"),
        reasoning_mode=reasoning_mode,
    )

    # Apply atom_mode to the live process immediately so the next inference call picks it up.
    if prefs.atom_mode:
        os.environ["CDSS_ATOM_MODE"] = "1"
    else:
        os.environ.pop("CDSS_ATOM_MODE", None)
    # Invalidate lru_cache so get_settings() returns fresh atom_mode on next call.
    try:
        get_settings.cache_clear()
    except Exception:
        pass

    snapshot = _bootstrap_snapshot()
    snapshot["saved"] = True
    snapshot["preferences"] = prefs.to_dict()
    return _http_response(request, snapshot)


@app.get("/api/vnext/health")
async def api_vnext_health(request: Request) -> Response:
    service = get_cdss_service()
    return _http_response(request, {
        "status": "ok",
        "surface": "ddx_safety",
        "runtime": "local_first_phase2",
        "ml_ranker": "llm_first_hypothesis_frontier",
        "external_evidence": "broker_enabled",
        "learning": "append_only_ledger",
        "v2_reasoning_artifact": "enabled" if service.runtime_policy.reasoning_artifact_v2_enabled else "disabled",
    })


@app.post("/api/vnext/analyze")
async def api_vnext_analyze(request: Request) -> Response:
    try:
        patient_input = PatientInput.model_validate(await _read_http_payload(request))
    except Exception as exc:
        raise _http_error(
            422,
            "api_vnext_invalid_input",
            "Invalid vNext patient input.",
            details={"reason": str(exc)},
        ) from exc

    service = get_cdss_service()
    packet = await service.analyze_case(patient_input)
    view = build_decision_packet_view(packet, _runtime_health_snapshot())
    return _http_response(request, view.model_dump(mode="json"))


@app.post("/api/v2/analyze")
async def api_v2_analyze(request: Request) -> Response:
    try:
        patient_input = PatientInput.model_validate(await _read_http_payload(request))
    except Exception as exc:
        raise _http_error(
            422,
            "api_v2_invalid_input",
            "Invalid v2 patient input.",
            details={"reason": str(exc)},
        ) from exc

    service = get_cdss_service()
    if not service.runtime_policy.reasoning_artifact_v2_enabled:
        raise _http_error(
            503,
            "api_v2_artifact_disabled",
            (
                "Reasoning artifact v2 is disabled. "
                "Set CDSS_REASONING_ARTIFACT_V2_ENABLED=1 to enable /api/v2/analyze."
            ),
        )

    artifact = await service.analyze_case_v2(patient_input, runtime_snapshot=_runtime_health_snapshot())
    return _http_response(request, artifact.model_dump(mode="json"))


@app.post("/api/vnext/feedback")
async def api_vnext_feedback(request: Request) -> Response:
    try:
        payload = await _read_http_payload(request)
        record = LearningRecord.model_validate(payload)
    except Exception as exc:
        raise _http_error(
            422,
            "api_vnext_feedback_invalid_input",
            "Invalid vNext feedback payload.",
            details={"reason": str(exc)},
        ) from exc

    # Cognitive neural learning: update MLP, vector store, and prototype centroid
    try:
        from src.cdss.learning.cognitive_engine import get_cognitive_engine
        _engine = get_cognitive_engine()
        _outcome_correct = str(record.outcome).lower() in {"correct", "confirmed", "true", "1"}
        _findings_summary = str(record.feedback.get("findings_summary", ""))
        _top_label = str(record.feedback.get("top_label", record.feedback.get("diagnosis", "")))
        if _top_label:
            _engine.record_outcome(
                case_id=record.case_id,
                findings_summary=_findings_summary,
                top_candidate=_top_label,
                outcome_correct=_outcome_correct,
            )
    except Exception:
        pass

    service = get_cdss_service()
    return _http_response(request, await service.ingest_feedback(record))


@app.get("/api/learning/stats")
async def api_learning_stats(request: Request) -> Response:
    """Returns current cognitive learning engine statistics for the Learning Dashboard."""
    try:
        from src.cdss.learning.cognitive_engine import get_cognitive_engine
        stats = get_cognitive_engine().stats()
    except Exception:
        stats = {"cases_in_memory": 0, "known_syndromes": 0, "mlp_updates": 0}
    return _http_response(request, stats)


@app.get("/api/test-cases")
async def get_test_cases(request: Request) -> Response:
    test_cases_dir = PROJECT_ROOT / "tests" / "test_cases"
    cases: list[dict[str, Any]] = []
    source_counts = {"clinical": 0, "who": 0, "pubmed": 0, "generated": 0}

    if test_cases_dir.exists():
        for file_path in test_cases_dir.glob("*.json"):
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.error("Error loading test case %s: %s", file_path.name, exc)
                continue

            if not _is_real_test_case(file_path, data):
                continue

            patient_text = _extract_patient_text(data)
            source_type, source_label = _classify_case_source(file_path, data)
            source_counts[source_type] = source_counts.get(source_type, 0) + 1
            expected_output = dict(data.get("expected_output", {}) or {})
            cases.append(
                {
                    "id": data.get("case_id", file_path.stem),
                    "filename": file_path.name,
                    "title": data.get("title", "Untitled Case"),
                    "patient_text": patient_text,
                    "diagnosis": expected_output.get("primary_diagnosis", "Unknown"),
                    "expected_output": expected_output,
                    "expected_output_raw": expected_output,
                    "source_type": source_type,
                    "source_label": source_label,
                    "real_case": True,
                }
            )

    cases.sort(key=lambda item: str(item.get("title", item.get("filename"))))
    return _http_response(request, {
        "cases": cases,
        "meta": {
            "count": len(cases),
            "policy": "curated_real_cases_only",
            "by_source": source_counts,
        },
    })


@app.post("/api/generate-case")
async def api_generate_case(request: Request) -> Response:
    try:
        gemini = get_gemini()
        if not gemini.is_available:
            return _http_response(request, {
                "status": "error",
                "message": "Gemini API key not configured. Case generation is only available in Research / Case Lab.",
            })

        generation_request = CaseGenerationRequest.model_validate(await _read_http_payload(request))
        if generation_request.source_mode == "multi_source_batch" or generation_request.target_count > 1:
            batch = await generate_case_batch(gemini, generation_request)
            return _http_response(request, {
                **batch,
                "message": batch.get("message", "Batch case generation completed."),
            })

        case_json = await generate_new_case(gemini, generation_request)
        status = str((case_json.get("case_meta", {}) or {}).get("status") or "draft")
        return _http_response(request, {
            "status": "success",
            "message": f"Generated {status} case: {case_json.get('title')}",
            **case_json,
        })
    except Exception as exc:
        incident_id = uuid.uuid4().hex[:12]
        logger.error("Error generating case: %s", exc, exc_info=True)
        return _http_response(request, {
            "status": "error",
            "message": "Case generation failed. Check server logs for details.",
            "error_code": "api_generate_case_failed",
            "incident_id": incident_id,
        })


@app.post("/api/generate-case/batch")
async def api_generate_case_batch(request: Request) -> Response:
    try:
        gemini = get_gemini()
        if not gemini.is_available:
            return _http_response(request, {
                "status": "error",
                "message": "Gemini API key not configured. Case generation is only available in Research / Case Lab.",
            })

        generation_request = CaseGenerationRequest.model_validate(await _read_http_payload(request))
        return _http_response(request, await generate_case_batch(gemini, generation_request))
    except Exception as exc:
        incident_id = uuid.uuid4().hex[:12]
        logger.error("Error generating case batch: %s", exc, exc_info=True)
        return _http_response(request, {
            "status": "error",
            "message": "Batch generation failed. Check server logs for details.",
            "error_code": "api_generate_case_batch_failed",
            "incident_id": incident_id,
        })


@app.get("/api/generated-cases")
async def get_generated_cases(request: Request) -> Response:
    return _http_response(request, list_generated_case_buckets(include_rejected=False))


@app.post("/api/generated-cases/{case_id}/promote")
async def api_promote_generated_case(case_id: str, request: Request) -> Response:
    try:
        data = promote_generated_case(case_id)
        return _http_response(request, {
            "status": "success",
            "message": f"Promoted {case_id} to curated cases.",
            "case_id": data.get("case_id", case_id),
        })
    except Exception as exc:
        incident_id = uuid.uuid4().hex[:12]
        logger.error("Error promoting generated case %s: %s", case_id, exc, exc_info=True)
        return _http_response(request, {
            "status": "error",
            "message": f"Failed to promote case {case_id}.",
            "error_code": "api_promote_generated_case_failed",
            "incident_id": incident_id,
        })


@app.post("/api/generated-cases/{case_id}/reject")
async def api_reject_generated_case(case_id: str, request: Request) -> Response:
    try:
        data = reject_generated_case(case_id)
        return _http_response(request, {
            "status": "success",
            "message": f"Rejected draft {case_id}.",
            "case_id": data.get("case_id", case_id),
        })
    except Exception as exc:
        incident_id = uuid.uuid4().hex[:12]
        logger.error("Error rejecting generated case %s: %s", case_id, exc, exc_info=True)
        return _http_response(request, {
            "status": "error",
            "message": f"Failed to reject case {case_id}.",
            "error_code": "api_reject_generated_case_failed",
            "incident_id": incident_id,
        })


def main() -> None:
    print("=" * 60)
    print("  RRRIE-CDSS - Clinical Decision Support Chat")
    print("  http://localhost:7860")
    print("=" * 60)
    uvicorn.run(
        "gui.server:app",
        host="127.0.0.1",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
