"""Application settings — loaded from environment variables via .env file."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration for RRRIE-CDSS."""

    # ── Local LLM (HuggingFace Transformers) ───────────────────────
    hf_model_name: str = Field(
        default="Qwen/Qwen3.5-4B",
        description="HuggingFace model name for local inference",
    )
    local_api_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        validation_alias=AliasChoices("LOCAL_API_URL", "LLAMA_SERVER_URL"),
        description="Local OpenAI-compatible API URL",
    )

    # ── DLLM R0 (Neuron-Logic Preprocessing) ───────────────────────
    dllm_api_url: str = Field(
        default="http://127.0.0.1:8081/v1",
        validation_alias=AliasChoices("DLLM_API_URL", "DLLM_SERVER_URL"),
        description="Local DLLM (Qwen3.5-0.8B) API URL",
    )
    dllm_model_name: str = Field(
        default="Qwen3.5-0.8B",
    )
    dllm_temperature: float = Field(
        default=0.2, 
        description="Low but non-zero temp for neuron-logic diversity",
    )
    dllm_max_tokens: int = Field(
        default=768, 
        description="Budget for thinking + JSON extraction",
    )
    entropy_router_mode: str = Field(
        default="disabled",
        pattern="^(disabled|shadow|enforce)$",
        description="Feature-flagged entropy routing mode",
    )
    entropy_router_threshold_simple: float = Field(
        default=0.45,
        description="Average entropy ceiling for simple-track routing",
    )
    entropy_router_threshold_moderate: float = Field(
        default=0.90,
        description="Average entropy ceiling for moderate-track routing",
    )
    entropy_router_threshold_complex: float = Field(
        default=1.35,
        description="Average entropy ceiling for complex-track routing",
    )
    entropy_router_margin: float = Field(
        default=0.12,
        description="Borderline band that triggers 4B verification",
    )
    entropy_router_top_logprobs: int = Field(
        default=8,
        ge=2,
        le=20,
        description="How many top logprobs to request for entropy probing",
    )
    entropy_router_probe_tokens: int = Field(
        default=3,
        ge=1,
        le=8,
        description="How many output tokens to inspect during entropy probing",
    )
    entropy_router_use_4b_fallback: bool = Field(
        default=True,
        description="Verify borderline entropy probes with the 4B model",
    )
    math_ie_mode: str = Field(
        default="disabled",
        pattern="^(disabled|shadow|enforce)$",
        description="Feature-flagged mathematical IE overlay mode",
    )
    math_ie_top_logprobs: int = Field(
        default=6,
        ge=2,
        le=20,
        description="How many top logprobs to request for math IE probes",
    )
    math_ie_probe_tokens: int = Field(
        default=1,
        ge=1,
        le=4,
        description="How many generated tokens to inspect during math IE probing",
    )
    math_ie_entropy_threshold: float = Field(
        default=0.52,
        ge=0.0,
        le=1.0,
        description="Binary entropy threshold that triggers conservative iterate escalation",
    )
    math_ie_jsd_threshold: float = Field(
        default=0.08,
        ge=0.0,
        le=1.0,
        description="Jensen-Shannon divergence threshold for skeptical-vs-supportive probe disagreement",
    )
    dllm_4b_escalation_mode: str = Field(
        default="critical_only",
        pattern="^(disabled|critical_only|complex_and_critical)$",
        description="Controls whether DLLM R0 may escalate L3/L5 to the 4B model",
    )

    # ── Cloud LLM (Groq — R2 stage) ────────────────────────────────
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_api_url: str = Field(
        default="https://api.groq.com/openai/v1",
        description="Groq OpenAI-compatible API URL",
    )
    cloud_model_name: str = Field(
        default="llama-3.3-70b-versatile",
        description="Cloud model name for R2 function-calling",
    )

    # ── Fallback Cloud (Together AI) ────────────────────────────────
    together_api_key: str = Field(default="", description="Together AI API key")
    together_model_name: str = Field(default="Qwen/Qwen3-32B")

    # ── Google Gemini (Super + Deep Thinking modes) ─────────────────
    google_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("GOOGLE_API_KEY", "GEMINI_API_KEY"),
        description="Google AI API key for Gemini",
    )
    gemini_lane: str = Field(
        default="experimental",
        pattern="^(stable|experimental)$",
        description="Default Gemini model lane for cloud board routing",
    )
    gemini_stable_flash_model: str = Field(
        default="gemini-2.5-flash",
        description="Stable Gemini Flash model for worker/coordinator duties",
    )
    gemini_stable_pro_model: str = Field(
        default="gemini-2.5-pro",
        description="Stable Gemini Pro model for second-opinion and safe fallback work",
    )
    gemini_experimental_flash_model: str = Field(
        default="gemini-3-flash-preview",
        description="Experimental Gemini Flash model for cloud board intake and orchestration",
    )
    gemini_experimental_pro_model: str = Field(
        default="gemini-3.1-pro-preview",
        description="Experimental Gemini Pro model for chief-board reasoning",
    )
    gemini_flash_model: str = Field(
        default="gemini-3-flash-preview",
        description="Compatibility alias for the primary Gemini Flash model",
    )
    gemini_pro_model: str = Field(
        default="gemini-3.1-pro-preview",
        description="Compatibility alias for the primary Gemini Pro model",
    )

    # ── Medical APIs ────────────────────────────────────────────────
    ncbi_email: str = Field(default="", description="NCBI email for E-Utilities (required by NCBI policy)")
    ncbi_api_key: str = Field(default="", description="NCBI API key for higher PubMed rate limits")
    tavily_api_key: str = Field(default="", description="Tavily Search API key")

    # ── WHO ICD-11 API (OAuth2 Client Credentials) ──────────────────
    icd11_client_id: str = Field(default="", description="WHO ICD-11 API client ID")
    icd11_client_secret: str = Field(default="", description="WHO ICD-11 API client secret")

    # ── Cache ───────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0")
    cache_ttl: int = Field(default=3600, description="Cache TTL in seconds")

    # ── RRRIE Protocol ──────────────────────────────────────────────
    operation_mode_default: str = Field(
        default="local_core_with_online_r2",
        pattern="^(strict_offline|local_core_with_online_r2|cloud_reference)$",
        description="Default execution mode when the GUI/request does not override it",
    )
    runtime_profile_default: str = Field(
        default="auto",
        pattern="^(auto|standard_6gb|compact_4gb|legacy_4gb|cpu_only)$",
        description="Default runtime profile for auto/manual resource-aware execution",
    )
    runtime_profile_vram_gb_override: float = Field(
        default=0.0,
        ge=0.0,
        le=48.0,
        description="Optional manual VRAM override for testing profile selection",
    )
    runtime_profile_min_free_vram_gb: float = Field(
        default=0.9,
        ge=0.0,
        le=16.0,
        description="Minimum remaining free VRAM required to start the DLLM server",
    )
    runtime_profile_dllm_mode: str = Field(
        default="auto",
        pattern="^(auto|full|lite|deterministic|off)$",
        description="Optional manual override for DLLM runtime mode selection",
    )
    runtime_profile_compact_ctx: int = Field(
        default=2048,
        ge=1024,
        le=8192,
        description="Effective context window used by the compact 4 GB runtime profile",
    )
    runtime_profile_compact_max_iterations: int = Field(default=2, ge=1, le=5)
    runtime_profile_compact_min_iterations: int = Field(default=1, ge=1, le=3)
    runtime_profile_compact_r2_max_queries: int = Field(default=4, ge=1, le=8)
    swarm_concurrency_limit: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Max concurrent specialist workers in expert swarm (default 4; increase for >6GB VRAM)",
    )
    llama_server_exe_legacy: str = Field(
        default="",
        description="Optional path to a legacy-compatible llama-server.exe for older NVIDIA GPUs",
    )
    max_rrrie_iterations: int = Field(default=3, ge=1, le=10)
    min_rrrie_iterations: int = Field(default=2, ge=1, le=5)
    confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    improvement_delta: float = Field(default=0.05, ge=0.0, le=1.0)
    min_evidence_sources: int = Field(default=2, ge=1)
    stagnation_threshold: int = Field(default=2, ge=1, le=5,
        description="If same primary dx repeats this many times with no improvement, force perspective shift")

    # ── LLM Parameters ─────────────────────────────────────────────
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=256, le=16384)
    num_ctx: int = Field(default=8192, description="Model context window size (8K = optimal for 4B)")
    # ATOM mode: deep reasoning with <think>...</think> enabled on R1/R3/IE stages.
    # Off by default — normal mode is fast, no thinking overhead.
    # Enable via: --atom CLI flag, CDSS_ATOM_MODE=1 env var, or ATOM_MODE=1, or .env file.
    atom_mode: bool = Field(
        default=False,
        validation_alias=AliasChoices("atom_mode", "ATOM_MODE", "CDSS_ATOM_MODE"),
        description="Enable deep reasoning (ATOM mode): thinking ON for R1/R3/IE stages. Slower but more thorough.",
    )

    qwen_stage_profile_mode: str = Field(
        default="official",
        pattern="^(official|legacy)$",
        description="Use official Qwen best-practice stage presets or the legacy generic sampling policy",
    )
    qwen_context_strategy: str = Field(
        default="working",
        pattern="^(working|legacy)$",
        description="Prefer a compact working-context strategy over naive full-prompt growth",
    )
    qwen_target_ctx: int = Field(
        default=32768,
        ge=4096,
        le=262144,
        description="Preferred effective context target for Qwen-compatible runtimes when resources allow",
    )
    retrieval_backend: str = Field(
        default="qwen_hybrid",
        pattern="^(local_hybrid|qwen_hybrid)$",
        description="Local retrieval backend contract used by evidence patch collection",
    )
    retrieval_patch_limit: int = Field(
        default=4,
        ge=1,
        le=12,
        description="Maximum number of local evidence patches appended during R2",
    )
    retrieval_embedding_api_url: str = Field(
        default="http://127.0.0.1:8080/v1",
        description="OpenAI-compatible embeddings endpoint used by the qwen_hybrid retrieval backend",
    )
    retrieval_embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Embedding model name requested by the qwen_hybrid retrieval backend",
    )
    retrieval_embedding_timeout_s: float = Field(
        default=12.0,
        ge=1.0,
        le=120.0,
        description="Timeout for retrieval embedding requests",
    )
    retrieval_qwen_candidate_pool: int = Field(
        default=12,
        ge=4,
        le=40,
        description="Candidate pool size before Qwen embedding rerank is applied",
    )
    retrieval_qwen_similarity_weight: float = Field(
        default=0.35,
        ge=0.0,
        le=1.0,
        description="Similarity weight blended into rerank_score by the qwen_hybrid backend",
    )

    # ── Application ─────────────────────────────────────────────────
    log_level: str = Field(default="INFO")
    debug: bool = Field(default=False)

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings singleton."""
    return Settings()
