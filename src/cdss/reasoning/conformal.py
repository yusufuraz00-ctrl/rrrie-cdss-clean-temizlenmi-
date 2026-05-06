"""Inductive conformal prediction wrapper (W5 Module H).

Replaces the naked top-1 disposition with a calibrated *prediction set* over
species hypotheses with a formal coverage guarantee: under exchangeability,
`P(true_dx ∈ C_α(x)) ≥ 1 − α`.

Nonconformity score:

    s(x, h) = -log P(h | x)

Prediction set at level α:

    C_α(x) = { h ∈ S : s(x, h) ≤ q̂_{1-α} }

`q̂_{1-α}` is the `(1-α)·(n+1)/n`-quantile of nonconformity scores on a held-
out calibration split, loaded from `data/cdss/learning/conformal_quantiles.json`.
If the asset is missing or does not cover the requested α, we fall back to
*score-mass* prediction sets: add hypotheses in descending posterior until
cumulative mass ≥ 1-α (standard top-p coverage). This keeps the runtime
correct and non-empty even before calibration has been run.

Set size serves as honest uncertainty:
  |C_{0.1}| = 1  → confident
  |C_{0.1}| ≥ 5 → "differential" view
  |C_{0.1}| ≥ 10 → hand to clinician (matches `_ESCALATE_CONFORMAL_SIZE`)

Pure math + one asset load. No LLM, no state mutation.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

logger = logging.getLogger(__name__)

try:
    from src.cdss.knowledge.ontology import normalize_candidate_label as _norm_label
except Exception:  # pragma: no cover
    import re as _re

    def _norm_label(value: str) -> str:
        s = str(value or "").lower()
        return _re.sub(r"[^a-z0-9]+", "_", s).strip("_")

_log = logging.getLogger("rrrie-cdss")


def _normalize_posterior(posterior: Mapping[str, float]) -> dict[str, float]:
    """Return posterior with canonicalized keys; merges duplicates by sum."""
    out: dict[str, float] = {}
    for k, v in (posterior or {}).items():
        key = _norm_label(str(k))
        if not key:
            continue
        try:
            fv = float(v or 0.0)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue
        out[key] = out.get(key, 0.0) + fv
    return out


_DEFAULT_ASSET = (
    Path(__file__).resolve().parents[3] / "data" / "cdss" / "learning" / "conformal_quantiles.json"
)
_SUPPORTED_ALPHAS = (0.05, 0.10, 0.20)


@dataclass
class ConformalQuantiles:
    """Calibrated per-α nonconformity quantiles."""

    quantiles: dict[float, float] = field(default_factory=dict)   # α → q̂_{1-α}
    n_calibration: int = 0
    split_fraction: float = 0.0
    version: str = ""

    def quantile_for(self, alpha: float) -> float | None:
        a = float(alpha)
        if a in self.quantiles:
            return float(self.quantiles[a])
        # Nearest-α fallback.
        if not self.quantiles:
            return None
        closest = min(self.quantiles.keys(), key=lambda k: abs(k - a))
        return float(self.quantiles[closest])


# -----------------------------------------------------------------
# Asset I/O
# -----------------------------------------------------------------

def load_conformal_quantiles(path: str | Path | None = None) -> ConformalQuantiles:
    """Load per-α quantile asset. Returns empty struct if missing."""
    p = Path(path) if path is not None else _DEFAULT_ASSET
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        _log.info("[CONFORMAL] quantile asset unavailable (%s) — falling back to score-mass sets.", exc)
        return ConformalQuantiles()
    qs_raw = data.get("quantiles") or {}
    quantiles: dict[float, float] = {}
    for k, v in qs_raw.items():
        try:
            quantiles[float(k)] = float(v)
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue
    return ConformalQuantiles(
        quantiles=quantiles,
        n_calibration=int(data.get("n_calibration", 0) or 0),
        split_fraction=float(data.get("split_fraction", 0.0) or 0.0),
        version=str(data.get("version", "")),
    )


def save_conformal_quantiles(
    quantiles: ConformalQuantiles, path: str | Path | None = None
) -> None:
    p = Path(path) if path is not None else _DEFAULT_ASSET
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": quantiles.version or "unversioned",
        "n_calibration": int(quantiles.n_calibration),
        "split_fraction": float(quantiles.split_fraction),
        "quantiles": {f"{k:.4f}": float(v) for k, v in sorted(quantiles.quantiles.items())},
    }
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# -----------------------------------------------------------------
# Calibration math
# -----------------------------------------------------------------

def nonconformity_score(posterior: Mapping[str, float], hypothesis: str) -> float:
    """`s(x,h) = -log P(h|x)`. Floored at 1e-12 posterior.

    Both posterior keys and the hypothesis label are canonicalized via
    `normalize_candidate_label` so that space-separated labels like
    ``"peptic ulcer disease"`` match underscored keys ``"peptic_ulcer_disease"``.
    """
    norm_post = _normalize_posterior(posterior)
    key = _norm_label(str(hypothesis))
    p = float(norm_post.get(key, 0.0) or 0.0)
    p = max(1e-12, min(1.0, p))
    return -math.log(p)


def quantile_from_scores(scores: Sequence[float], alpha: float) -> float:
    """Finite-sample conformal quantile: q̂ = ceil((n+1)(1-α))-th order statistic.

    Returns +inf when `scores` is empty so the fallback set covers everything.
    """
    if not scores:
        return float("inf")
    n = len(scores)
    k = max(1, min(n, int(math.ceil((n + 1) * (1.0 - float(alpha))))))
    sorted_s = sorted(scores)
    return float(sorted_s[k - 1])


def calibrate(
    calibration_records: Sequence[tuple[Mapping[str, float], str]],
    alphas: Sequence[float] = _SUPPORTED_ALPHAS,
) -> ConformalQuantiles:
    """Compute quantiles from held-out (posterior, true_dx) pairs.

    Records where the true dx is missing from the posterior get posterior=1e-12
    (max nonconformity) so the system is penalized honestly for hypotheses it
    never enumerated.
    """
    scores: list[float] = []
    for post, true_dx in calibration_records:
        s = nonconformity_score(post or {}, str(true_dx))
        scores.append(s)
    qs: dict[float, float] = {}
    for a in alphas:
        qs[float(a)] = quantile_from_scores(scores, float(a))
    return ConformalQuantiles(
        quantiles=qs,
        n_calibration=len(scores),
        split_fraction=1.0,
        version="calibrated",
    )


# -----------------------------------------------------------------
# Runtime prediction set
# -----------------------------------------------------------------

def _score_mass_set(posterior: Mapping[str, float], alpha: float) -> list[str]:
    """Fallback when no calibration: top-p set with cumulative mass ≥ 1-α."""
    items = sorted((posterior or {}).items(), key=lambda kv: kv[1], reverse=True)
    target = 1.0 - float(alpha)
    out: list[str] = []
    acc = 0.0
    for h, p in items:
        if not h:
            continue
        out.append(str(h))
        acc += float(p)
        if acc >= target:
            break
    return out


def prediction_set(
    posterior: Mapping[str, float],
    alpha: float = 0.1,
    *,
    quantiles: ConformalQuantiles | None = None,
    max_size: int = 20,
) -> list[str]:
    """Return the calibrated prediction set at level α.

    If quantiles are available and finite, use the conformal rule; otherwise
    fall back to a score-mass top-p set. Always returns a non-empty list when
    the posterior is non-empty.
    """
    if not posterior:
        return []
    q = None
    if quantiles is not None:
        q = quantiles.quantile_for(float(alpha))
    if q is None or not math.isfinite(q):
        return _score_mass_set(posterior, alpha)[: max(1, int(max_size))]
    out: list[str] = []
    for h, p in posterior.items():
        if not h:
            continue
        s = nonconformity_score(posterior, h)
        if s <= q:
            out.append(str(h))
    # Never return empty — conformal guarantee allows it only on degenerate
    # posteriors. Fall back to argmax so callers don't see an empty list.
    if not out:
        top = max(posterior.items(), key=lambda kv: kv[1])
        out = [str(top[0])]
    # Rank inside the set by posterior, descending.
    out = sorted(out, key=lambda h: float(posterior.get(h, 0.0)), reverse=True)
    return out[: max(1, int(max_size))]


def set_size(posterior: Mapping[str, float], alpha: float = 0.1,
             *, quantiles: ConformalQuantiles | None = None) -> int:
    return len(prediction_set(posterior, alpha, quantiles=quantiles))


def prediction_report(
    posterior: Mapping[str, float],
    *,
    alphas: Sequence[float] = _SUPPORTED_ALPHAS,
    quantiles: ConformalQuantiles | None = None,
) -> dict[str, Any]:
    """Return set + metadata across multiple α levels for UI/audit."""
    q = quantiles if quantiles is not None else load_conformal_quantiles()
    out: dict[str, Any] = {"n_calibration": q.n_calibration, "version": q.version, "by_alpha": {}}
    for a in alphas:
        s = prediction_set(posterior, a, quantiles=q)
        out["by_alpha"][f"{float(a):.2f}"] = {
            "set": s,
            "size": len(s),
            "coverage_target": round(1.0 - float(a), 4),
            "quantile": (q.quantile_for(float(a)) if q.quantiles else None),
        }
    return out


__all__ = [
    "ConformalQuantiles",
    "load_conformal_quantiles",
    "save_conformal_quantiles",
    "nonconformity_score",
    "quantile_from_scores",
    "calibrate",
    "prediction_set",
    "set_size",
    "prediction_report",
]
