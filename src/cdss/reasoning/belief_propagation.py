"""Belief-propagation primitives for the 3-level diagnostic hierarchy.

Pure functions (no LLM, no I/O). All operate on `DiagnosticBelief` instances
and a registry-provided `parent_of: dict[child_id, parent_id]` map.

Math (W1 Module A.3):
- Dirichlet posterior mean: P(h) = α_h / Σ α
- Normalization: ensures Σ posteriors = 1 per level.
- Shannon entropy (log base 2): H(p) = -Σ p log₂ p, with 0 log 0 := 0.
- Down-propagation: P(genus g) = Σ_{f parent of g} P(f) · P(g | f)
  (requires conditional P(g|f) which is derived from current genus posterior
  restricted to g's family siblings, then renormalized within family).
- Up-propagation: aggregates species α into genus α, genus α into family α.
  Used when species-level evidence arrives and we want the belief at higher
  levels to reflect it.

All operations clone the input belief (immutable pattern) and return a new
instance; entropy_history is appended to on each call that mutates species.
"""

from __future__ import annotations

import math
from typing import Iterable

from ..contracts.models import DiagnosticBelief

import logging
logger = logging.getLogger(__name__)

_LEVELS = ("family", "genus", "species")


def _alpha_field(level: int | str) -> str:
    if level in (0, "family"):
        return "family_alpha"
    if level in (1, "genus"):
        return "genus_alpha"
    return "species_alpha"


def _posterior_field(level: int | str) -> str:
    if level in (0, "family"):
        return "family_posterior"
    if level in (1, "genus"):
        return "genus_posterior"
    return "species_posterior"


def _get_dict(belief: DiagnosticBelief, field: str) -> dict[str, float]:
    return dict(getattr(belief, field) or {})


# ---------------------------------------------------------------------------
# Core numeric helpers
# ---------------------------------------------------------------------------

def normalize(counts: dict[str, float]) -> dict[str, float]:
    """Renormalize a non-negative count dict into a probability distribution."""
    if not counts:
        return {}
    total = sum(max(0.0, float(v)) for v in counts.values())
    if total <= 0.0:
        # Uniform fallback.
        n = len(counts)
        return {k: 1.0 / n for k in counts}
    return {k: max(0.0, float(v)) / total for k, v in counts.items()}


def dirichlet_mean(alpha: dict[str, float]) -> dict[str, float]:
    """Dirichlet posterior mean = α_h / Σ α. Equivalent to normalize(α)."""
    return normalize(alpha)


def entropy(distribution: dict[str, float], base: float = 2.0) -> float:
    """Shannon entropy of a probability distribution. 0 log 0 := 0."""
    if not distribution:
        return 0.0
    total = sum(max(0.0, float(v)) for v in distribution.values())
    if total <= 0.0:
        return 0.0
    log_base = math.log(base)
    h = 0.0
    for v in distribution.values():
        p = max(0.0, float(v)) / total
        if p > 0.0:
            h -= p * math.log(p) / log_base
    return h


def kl_divergence(p: dict[str, float], q: dict[str, float], base: float = 2.0) -> float:
    """KL(p || q) over shared support. Missing-in-q keys contribute ∞ → clamped."""
    if not p or not q:
        return 0.0
    log_base = math.log(base)
    d = 0.0
    for k, pv in p.items():
        pv = max(0.0, float(pv))
        if pv <= 0.0:
            continue
        qv = max(1e-12, float(q.get(k, 0.0)))
        d += pv * (math.log(pv / qv) / log_base)
    return max(0.0, d)


# ---------------------------------------------------------------------------
# Belief manipulation
# ---------------------------------------------------------------------------

def normalize_level(belief: DiagnosticBelief, level: int | str) -> DiagnosticBelief:
    """Return a new belief with the target level's posterior re-normalized from α."""
    alpha_field = _alpha_field(level)
    post_field = _posterior_field(level)
    alpha = _get_dict(belief, alpha_field)
    posterior = dirichlet_mean(alpha)
    return belief.model_copy(update={post_field: posterior})


def entropy_of_level(belief: DiagnosticBelief, level: int | str = 2) -> float:
    return entropy(_get_dict(belief, _posterior_field(level)))


def propagate_down(
    belief: DiagnosticBelief,
    *,
    parent_of_genus: dict[str, str],
    parent_of_species: dict[str, str],
) -> DiagnosticBelief:
    """Enforce parent-consistency: child posteriors scale by parent posterior.

    For each genus g with family parent f:
        P(g) ← P(g | f) · P(f)
    where P(g | f) is derived by renormalizing current genus posteriors within
    each family cluster. Same logic applied species → genus.

    If a child has no known parent in the registry map, its mass is preserved
    at 1.0 (treated as its own family — degrades to flat when hierarchy.json
    isn't shipped).
    """
    updates: dict[str, dict[str, float]] = {}

    # Families → genera
    family_posterior = _get_dict(belief, "family_posterior")
    genus_posterior = _get_dict(belief, "genus_posterior")
    if family_posterior and genus_posterior:
        # Group genus posterior by family.
        by_family: dict[str, dict[str, float]] = {}
        for gid, gp in genus_posterior.items():
            fid = str(parent_of_genus.get(gid, "") or "")
            by_family.setdefault(fid, {})[gid] = max(0.0, float(gp))
        new_genus: dict[str, float] = {}
        for fid, group in by_family.items():
            conditional = normalize(group)
            fp = float(family_posterior.get(fid, 1.0)) if fid else 1.0
            for gid, cp in conditional.items():
                new_genus[gid] = cp * fp
        updates["genus_posterior"] = normalize(new_genus)
    elif genus_posterior:
        updates["genus_posterior"] = normalize(genus_posterior)

    # Genera → species
    # Re-read genus_posterior from updates if we just recomputed it.
    effective_genus = updates.get("genus_posterior", genus_posterior)
    species_posterior = _get_dict(belief, "species_posterior")
    if effective_genus and species_posterior:
        by_genus: dict[str, dict[str, float]] = {}
        for sid, sp in species_posterior.items():
            gid = str(parent_of_species.get(sid, "") or "")
            by_genus.setdefault(gid, {})[sid] = max(0.0, float(sp))
        new_species: dict[str, float] = {}
        for gid, group in by_genus.items():
            conditional = normalize(group)
            gp = float(effective_genus.get(gid, 1.0)) if gid else 1.0
            for sid, cp in conditional.items():
                new_species[sid] = cp * gp
        updates["species_posterior"] = normalize(new_species)
    elif species_posterior:
        updates["species_posterior"] = normalize(species_posterior)

    return belief.model_copy(update=updates)


def propagate_up(
    belief: DiagnosticBelief,
    *,
    parent_of_genus: dict[str, str],
    parent_of_species: dict[str, str],
) -> DiagnosticBelief:
    """Marginalize species α up to genus α, genus α up to family α.

    Unlike propagate_down (which rescales posteriors), propagate_up aggregates
    Dirichlet counts so parent-level α reflects evidence gathered at child level.
    """
    species_alpha = _get_dict(belief, "species_alpha")
    genus_alpha = _get_dict(belief, "genus_alpha")
    family_alpha = _get_dict(belief, "family_alpha")

    # Species → genus
    if species_alpha:
        agg_genus: dict[str, float] = dict(genus_alpha)
        for sid, av in species_alpha.items():
            gid = str(parent_of_species.get(sid, "") or "")
            if not gid:
                continue
            agg_genus[gid] = agg_genus.get(gid, 0.0) + max(0.0, float(av))
        genus_alpha = agg_genus

    # Genus → family
    if genus_alpha:
        agg_family: dict[str, float] = dict(family_alpha)
        for gid, av in genus_alpha.items():
            fid = str(parent_of_genus.get(gid, "") or "")
            if not fid:
                continue
            agg_family[fid] = agg_family.get(fid, 0.0) + max(0.0, float(av))
        family_alpha = agg_family

    updated = belief.model_copy(update={
        "genus_alpha": genus_alpha,
        "family_alpha": family_alpha,
        "genus_posterior": dirichlet_mean(genus_alpha),
        "family_posterior": dirichlet_mean(family_alpha),
    })
    return updated


def record_entropy(belief: DiagnosticBelief) -> DiagnosticBelief:
    """Append current H(b_S) to entropy_history."""
    h = entropy_of_level(belief, 2)
    history = list(belief.entropy_history or []) + [round(h, 6)]
    return belief.model_copy(update={"entropy_history": history})


# ---------------------------------------------------------------------------
# Registry → parent-map helpers
# ---------------------------------------------------------------------------

def build_parent_maps(profiles: Iterable) -> tuple[dict[str, str], dict[str, str]]:
    """Turn a registry profile list into (parent_of_genus, parent_of_species) dicts."""
    parent_of_genus: dict[str, str] = {}
    parent_of_species: dict[str, str] = {}
    for p in profiles:
        try:
            level = int(getattr(p, "level", 2))
            pid = str(getattr(p, "id", "") or "")
            par = str(getattr(p, "parent_id", "") or "")
        except Exception:  # noqa: BLE001
            logger.debug("swallowed exception", exc_info=True)
            continue
        if not pid:
            continue
        if level == 1 and par:
            parent_of_genus[pid] = par
        elif level == 2 and par:
            parent_of_species[pid] = par
    return parent_of_genus, parent_of_species


__all__ = [
    "normalize",
    "dirichlet_mean",
    "entropy",
    "kl_divergence",
    "normalize_level",
    "entropy_of_level",
    "propagate_down",
    "propagate_up",
    "record_entropy",
    "build_parent_maps",
]
