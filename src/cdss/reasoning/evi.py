"""Expected Value of Information for candidate actions (W3 Module D.3).

    EVI(a, b) = E_{o ∼ P(o|a, b)} [ U(b' | do(a), o) − U(b) ] − cost_scalar(a)

MCTS calls this at each rollout to score a proposed action. Expectation is
estimated via `n_samples` observation draws. Each draw simulates a one-step
belief update via a *lightweight* rollout: apply the observation as a soft
likelihood boost on the targeted hypothesis (or a uniform ablation) and score
ΔU.

Rollouts are deliberately cheap — we do NOT fire a real LLM here. The LLM is
reserved for the real action execution; EVI is a model-based estimator over
the already-known posterior shape. This keeps the controller loop within
latency budget (plan target: +15 s P95).

Pure math. No state-machine mutation.
"""

from __future__ import annotations

from typing import Mapping

from ..contracts.models import DiagnosticBelief
from ..runtime.actions import Action, ActionKind
from .utility import utility


def _apply_soft_update(
    belief: DiagnosticBelief,
    target: str,
    *,
    boost: float = 1.5,
) -> DiagnosticBelief:
    """Cheap one-step posterior update — boost `target`, renormalize."""
    new_post: dict[str, float] = {}
    species = dict(belief.species_posterior or {})
    if not species or target not in species:
        return belief
    for h, p in species.items():
        new_post[h] = float(p) * (float(boost) if h == target else 1.0)
    z = sum(new_post.values())
    if z <= 0.0:
        return belief
    new_post = {h: v / z for h, v in new_post.items()}
    return belief.model_copy(update={"species_posterior": new_post})


def _apply_soft_downweight(
    belief: DiagnosticBelief,
    target: str,
    *,
    decay: float = 0.5,
) -> DiagnosticBelief:
    """Cheap disconfirming update — shrink `target`, renormalize."""
    new_post: dict[str, float] = {}
    species = dict(belief.species_posterior or {})
    if not species or target not in species:
        return belief
    for h, p in species.items():
        new_post[h] = float(p) * (float(decay) if h == target else 1.0)
    z = sum(new_post.values())
    if z <= 0.0:
        return belief
    new_post = {h: v / z for h, v in new_post.items()}
    return belief.model_copy(update={"species_posterior": new_post})


def simulate_rollout(
    belief: DiagnosticBelief,
    action: Action,
    *,
    severity: Mapping[str, float] | None = None,
    treatment_correct: Mapping[str, float] | None = None,
    utility_kwargs: dict | None = None,
) -> float:
    """One lookahead rollout. Returns ΔU estimate.

    Strategy per action:
      - STOP          : 0.0 (no change, cost baseline)
      - ESCALATE      : 0.0 (delegates; no immediate posterior change)
      - CONSULT_SPECIALIST / QUERY_ONTOLOGY / SEARCH_LITERATURE / RUN_SWARM_LEVEL
                      : expected boost/decay weighted by current top-k.
      - ABLATE_FINDING
                      : disconfirming downweight on current top-1.
      - REQUEST_TEST / REQUEST_HISTORY
                      : expected sharpening proportional to top-1 mass.
      - EXPAND_HIERARCHY
                      : approximate variance injection (zero ΔU mean).
    """
    u_kwargs = dict(utility_kwargs or {})
    u0 = utility(belief, severity=severity, treatment_correct=treatment_correct, **u_kwargs)
    species = dict(belief.species_posterior or {})

    if action.kind in (ActionKind.STOP, ActionKind.ESCALATE):
        return 0.0
    if not species:
        return 0.0

    top_items = sorted(species.items(), key=lambda kv: kv[1], reverse=True)
    top_h, top_p = top_items[0]

    # Target resolution.
    target = action.target or top_h
    if target not in species:
        target = top_h

    if action.kind == ActionKind.ABLATE_FINDING:
        # Half downweight on top-1 (we are asking "if this finding vanished").
        after = _apply_soft_downweight(belief, top_h, decay=0.5)
    elif action.kind in (
        ActionKind.CONSULT_SPECIALIST,
        ActionKind.QUERY_ONTOLOGY,
        ActionKind.SEARCH_LITERATURE,
        ActionKind.RUN_SWARM_LEVEL,
        ActionKind.REQUEST_TEST,
        ActionKind.REQUEST_HISTORY,
    ):
        # Expected: confirming evidence sharpens target.
        boost = 1.6 if action.kind == ActionKind.REQUEST_TEST else 1.4
        after = _apply_soft_update(belief, target, boost=boost)
    elif action.kind == ActionKind.EXPAND_HIERARCHY:
        # Adds candidates → flatter posterior by a small mix toward uniform.
        n = max(1, len(species))
        uniform = 1.0 / n
        alpha = 0.15
        new_post = {h: (1.0 - alpha) * float(p) + alpha * uniform for h, p in species.items()}
        z = sum(new_post.values()) or 1.0
        new_post = {h: v / z for h, v in new_post.items()}
        after = belief.model_copy(update={"species_posterior": new_post})
    else:
        after = belief

    u1 = utility(after, severity=severity, treatment_correct=treatment_correct, **u_kwargs)
    return float(u1) - float(u0)


def estimate_evi(
    belief: DiagnosticBelief,
    action: Action,
    *,
    n_samples: int = 3,
    severity: Mapping[str, float] | None = None,
    treatment_correct: Mapping[str, float] | None = None,
    time_weight: float = 0.02,
    token_weight: float = 1e-4,
    utility_kwargs: dict | None = None,
) -> tuple[float, float]:
    """Monte-Carlo estimate of EVI with cost subtraction.

    Returns `(evi, cost_scalar)`. Positive EVI → action worth taking.
    `n_samples` drawn from `action.sample_observations()`; each drives one
    `simulate_rollout` with soft updates. Mean ΔU minus cost.
    """
    observations = action.sample_observations(belief, n=max(1, int(n_samples)))
    if not observations:
        mean_du = 0.0
    else:
        total_p = sum(max(1e-6, p) for _o, p in observations) or 1.0
        weighted = 0.0
        for _o, p in observations:
            du = simulate_rollout(
                belief,
                action,
                severity=severity,
                treatment_correct=treatment_correct,
                utility_kwargs=utility_kwargs,
            )
            weighted += (max(1e-6, p) / total_p) * du
        mean_du = weighted

    cost = action.estimated_cost().scalarize(time_weight=time_weight, token_weight=token_weight)
    return round(float(mean_du) - float(cost), 6), round(float(cost), 6)


__all__ = [
    "estimate_evi",
    "simulate_rollout",
]
