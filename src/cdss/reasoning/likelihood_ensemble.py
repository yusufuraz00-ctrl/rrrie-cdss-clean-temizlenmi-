"""LLM-as-likelihood temperature ensemble (W1 Module B.1).

Purpose: estimate P(evidence | hypothesis) = p with variance, by firing the
same ranking prompt at multiple temperatures and aggregating the responses
into a Beta(α, β) posterior over p.

Math:
  Given n independent samples x_1..x_n ∈ [0,1]:
    μ = mean(x), σ² = var(x)
    method-of-moments Beta:
      κ = μ(1 - μ) / σ² - 1    (concentration parameter, κ > 0)
      α = μ · κ,  β = (1 - μ) · κ
  Fallback (σ² ≈ 0, all samples identical):
      α = μ · n_effective + 1,  β = (1 - μ) · n_effective + 1
  where n_effective = 3 (pseudo-count for low-disagreement prior).

We return (mean, variance, alpha, beta) so downstream Bayes update picks the
posterior mean; variance feeds MCTS/EVI and Dempster-Shafer (higher variance
= more epistemic uncertainty = more reason to ask for info).

Response format (prompt-enforced):
  `LIK|<prob_float>`  — one line, prob ∈ [0, 1].

No LLM in this file → caller injects `llm_client`. Parallel fan-out uses
asyncio.gather under whatever semaphore the caller holds.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any, Iterable

import logging
logger = logging.getLogger(__name__)

_LIK_LINE_RE = re.compile(r"LIK\|([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Default temperature ensemble — small, covers mode + near-mode samples.
_DEFAULT_TEMPS: tuple[float, ...] = (0.0, 0.2, 0.4)

# Floor/ceiling so probabilities never collapse to 0/1 (kills log-likelihood).
_P_MIN = 1e-4
_P_MAX = 1.0 - 1e-4


@dataclass
class LikelihoodEstimate:
    hypothesis: str
    finding: str
    mean: float
    variance: float
    alpha: float  # Beta α
    beta: float   # Beta β
    samples: list[float]
    cache_key: str = ""
    grounded: bool = True  # False when fallback to uniform 0.5

    @property
    def log_mean(self) -> float:
        import math
        return math.log(max(_P_MIN, min(_P_MAX, float(self.mean))))


def _clamp_prob(v: float) -> float:
    return max(_P_MIN, min(_P_MAX, float(v)))


def _build_prompt(finding: str, hypothesis: str, *, context: str = "") -> str:
    ctx = f"\nCONTEXT\n{context}\n" if context else ""
    return (
        "You are estimating a clinical likelihood P(finding | hypothesis).\n"
        "Return ONE line only, no extra text, in the format:\n"
        "  LIK|<probability between 0 and 1>\n\n"
        f"FINDING: {finding}\n"
        f"HYPOTHESIS (diagnosis): {hypothesis}\n"
        f"{ctx}"
        "Probability scale:\n"
        "  0.95+ : finding is almost always present when this hypothesis is true (pathognomonic)\n"
        "  0.70  : finding is common with this hypothesis\n"
        "  0.40  : finding is occasional / nonspecific\n"
        "  0.10  : finding rarely occurs with this hypothesis\n"
        "  0.02  : finding is essentially incompatible with this hypothesis\n\n"
        "Give only the LIK line:\n"
    )


def _parse_prob(raw: str) -> float | None:
    if not raw or not isinstance(raw, str):
        return None
    m = _LIK_LINE_RE.search(raw)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


async def _one_call(
    llm_client: Any,
    prompt: str,
    *,
    temperature: float,
    max_tokens: int = 24,
) -> float | None:
    try:
        if hasattr(llm_client, "complete"):
            raw = await llm_client.complete(prompt, max_tokens=max_tokens, temperature=temperature)
        elif hasattr(llm_client, "complete_sync"):
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(
                None, lambda: llm_client.complete_sync(prompt, max_tokens=max_tokens, temperature=temperature)
            )
        elif hasattr(llm_client, "call_sync"):
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, lambda: llm_client.call_sync(prompt, max_tokens=max_tokens))
        elif callable(llm_client):
            loop = asyncio.get_event_loop()
            raw = await loop.run_in_executor(None, lambda: llm_client(prompt))
        else:
            return None
    except Exception:
        return None
    return _parse_prob(str(raw or ""))


def _method_of_moments_beta(mean: float, var: float) -> tuple[float, float]:
    """Return (α, β) from (mean, var) via method of moments.

    κ = μ(1-μ)/σ² - 1. Valid when var < μ(1-μ); clamp otherwise with pseudo-count prior.
    """
    mu = _clamp_prob(mean)
    max_var = mu * (1.0 - mu)
    v = max(1e-6, min(max_var - 1e-6, float(var)))
    if v <= 1e-6 or max_var <= 1e-6:
        # Low-variance prior: pseudo-count of 3 samples.
        return mu * 3.0 + 1.0, (1.0 - mu) * 3.0 + 1.0
    kappa = max(1e-3, max_var / v - 1.0)
    return max(1e-3, mu * kappa), max(1e-3, (1.0 - mu) * kappa)


def _cache_key(finding: str, hypothesis: str, temperature: float) -> str:
    h = hashlib.sha256()
    h.update(str(finding or "").strip().lower().encode("utf-8"))
    h.update(b"|")
    h.update(str(hypothesis or "").strip().lower().encode("utf-8"))
    h.update(b"|")
    h.update(f"{float(temperature):.3f}".encode("utf-8"))
    return h.hexdigest()[:16]


async def estimate_likelihood(
    finding: str,
    hypothesis: str,
    *,
    llm_client: Any,
    context: str = "",
    temperatures: Iterable[float] | None = None,
    response_cache: Any | None = None,
    max_tokens: int = 24,
) -> LikelihoodEstimate:
    """Estimate P(finding | hypothesis) as a Beta posterior via temperature ensemble.

    On total LLM failure → returns uniform prior (mean=0.5, wide variance).
    """
    temps = tuple(temperatures) if temperatures else _DEFAULT_TEMPS
    prompt = _build_prompt(finding, hypothesis, context=context)
    # Cache lookup (optional; cache must expose .get_sync / .put_sync-ish API).
    cache_prefix = _cache_key(finding, hypothesis, -1.0)

    async def sampled(t: float) -> float | None:
        if response_cache is not None:
            key = _cache_key(finding, hypothesis, t)
            try:
                hit = response_cache.get(key) if hasattr(response_cache, "get") else None
            except Exception:
                hit = None
            if hit is not None:
                return _clamp_prob(float(hit))
        val = await _one_call(llm_client, prompt, temperature=t, max_tokens=max_tokens)
        if val is None:
            return None
        val = _clamp_prob(val)
        if response_cache is not None and hasattr(response_cache, "put"):
            try:
                response_cache.put(_cache_key(finding, hypothesis, t), float(val))
            except Exception:  # noqa: BLE001
                logger.debug("swallowed exception", exc_info=True)
                pass
        return val

    samples_raw = await asyncio.gather(*(sampled(t) for t in temps), return_exceptions=False)
    samples = [s for s in samples_raw if isinstance(s, float)]

    if not samples:
        # Uniform fallback: no information.
        return LikelihoodEstimate(
            hypothesis=hypothesis,
            finding=finding,
            mean=0.5,
            variance=0.25,  # maximum variance for [0,1] support
            alpha=1.0,
            beta=1.0,
            samples=[],
            cache_key=cache_prefix,
            grounded=False,
        )

    n = len(samples)
    mu = sum(samples) / n
    var = sum((x - mu) ** 2 for x in samples) / max(1, n - 1) if n >= 2 else 0.0
    alpha, beta = _method_of_moments_beta(mu, var)
    return LikelihoodEstimate(
        hypothesis=hypothesis,
        finding=finding,
        mean=_clamp_prob(mu),
        variance=max(0.0, var),
        alpha=alpha,
        beta=beta,
        samples=list(samples),
        cache_key=cache_prefix,
        grounded=True,
    )


async def estimate_likelihood_matrix(
    findings: list[str],
    hypotheses: list[str],
    *,
    llm_client: Any,
    context: str = "",
    temperatures: Iterable[float] | None = None,
    response_cache: Any | None = None,
    semaphore: asyncio.Semaphore | None = None,
) -> dict[tuple[str, str], LikelihoodEstimate]:
    """Fan out estimate_likelihood across (finding, hypothesis) grid.

    Rate-limited by `semaphore` if provided (reuse `_swarm_limit` from the
    state machine to respect VRAM budget).
    """
    async def one(f: str, h: str) -> tuple[tuple[str, str], LikelihoodEstimate]:
        if semaphore is not None:
            async with semaphore:
                est = await estimate_likelihood(
                    f, h,
                    llm_client=llm_client,
                    context=context,
                    temperatures=temperatures,
                    response_cache=response_cache,
                )
        else:
            est = await estimate_likelihood(
                f, h,
                llm_client=llm_client,
                context=context,
                temperatures=temperatures,
                response_cache=response_cache,
            )
        return (f, h), est

    tasks = [one(f, h) for f in findings for h in hypotheses]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return {key: est for key, est in results}


__all__ = [
    "LikelihoodEstimate",
    "estimate_likelihood",
    "estimate_likelihood_matrix",
]
