"""Action registry for the cognitive controller (W3 Module D.1).

Typed action payloads consumed by MCTS. Each action exposes three methods:

  - `estimated_cost()`       — wall-time / token / network cost vector
  - `simulate(belief, ...)`  — LLM-driven observation sampler for lookahead
  - `execute(belief, ...)`   — real call, returns (observation, updated_belief)

Controller chooses the `a* = argmax_a EVI(a)` at every tick. Actions have
deliberate coarse granularity — one action = one "clinical move".

10 action kinds per Plan §D (architecture section):
  1. QUERY_ONTOLOGY
  2. SEARCH_LITERATURE
  3. RUN_SWARM_LEVEL
  4. CONSULT_SPECIALIST
  5. ABLATE_FINDING
  6. REQUEST_TEST
  7. REQUEST_HISTORY
  8. EXPAND_HIERARCHY
  9. STOP
 10. ESCALATE

`execute` bodies are thin — they dispatch to existing state-machine helpers
(`_run_r2_research`, `_run_hierarchical_swarm`, etc.) where available, and
otherwise return a no-op updated belief so the controller remains well-formed
even when bridging work for later waves isn't done yet.

No state_machine mutation here — controller owns its own belief snapshot and
re-applies it into the bundle only at loop termination.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from ..contracts.models import DiagnosticBelief

_log = logging.getLogger("rrrie-cdss")


class ActionKind(str, Enum):
    QUERY_ONTOLOGY = "query_ontology"
    SEARCH_LITERATURE = "search_literature"
    RUN_SWARM_LEVEL = "run_swarm_level"
    CONSULT_SPECIALIST = "consult_specialist"
    ABLATE_FINDING = "ablate_finding"
    REQUEST_TEST = "request_test"
    REQUEST_HISTORY = "request_history"
    EXPAND_HIERARCHY = "expand_hierarchy"
    STOP = "stop"
    ESCALATE = "escalate"


@dataclass(frozen=True)
class Cost:
    """Resource cost of invoking an action."""

    wall_s: float = 0.0
    tokens: int = 0
    network_ms: float = 0.0

    def scalarize(self, *, time_weight: float = 1.0, token_weight: float = 1e-4) -> float:
        """Collapse into a single scalar so EVI can subtract it from ΔU."""
        return float(time_weight) * float(self.wall_s) + float(token_weight) * float(self.tokens)


@dataclass
class Action:
    """Controller-executable action payload."""

    kind: ActionKind
    target: str = ""                         # hypothesis id, finding key, test name, etc.
    level: int | None = None                 # for RUN_SWARM_LEVEL / EXPAND_HIERARCHY
    payload: dict[str, Any] = field(default_factory=dict)  # free-form kwargs

    def estimated_cost(self) -> Cost:
        """Cost estimate by kind. Heuristics tuned per plan latency accounting."""
        k = self.kind
        if k == ActionKind.STOP or k == ActionKind.ESCALATE:
            return Cost(wall_s=0.0, tokens=0)
        if k == ActionKind.QUERY_ONTOLOGY:
            return Cost(wall_s=0.2, tokens=0, network_ms=0.0)
        if k == ActionKind.SEARCH_LITERATURE:
            return Cost(wall_s=4.0, tokens=800, network_ms=800.0)
        if k == ActionKind.RUN_SWARM_LEVEL:
            # Fan-out 3-5 workers, each a short LLM call.
            return Cost(wall_s=8.0, tokens=2500)
        if k == ActionKind.CONSULT_SPECIALIST:
            return Cost(wall_s=3.5, tokens=700)
        if k == ActionKind.ABLATE_FINDING:
            return Cost(wall_s=1.5, tokens=300)
        if k == ActionKind.REQUEST_TEST:
            # User-facing — wall_s = 0 (async with user), token=0 until answer
            return Cost(wall_s=0.0, tokens=0)
        if k == ActionKind.REQUEST_HISTORY:
            return Cost(wall_s=0.0, tokens=0)
        if k == ActionKind.EXPAND_HIERARCHY:
            return Cost(wall_s=0.1, tokens=0)
        return Cost()

    # -----------------------------------------------------------------
    # Observation sampler — used by MCTS for lookahead (not ground truth).
    # -----------------------------------------------------------------

    def sample_observations(
        self,
        belief: DiagnosticBelief,
        *,
        n: int = 3,
        llm_client: Any | None = None,
    ) -> list[tuple[str, float]]:
        """Return list of (observation_token, prob) used to simulate outcomes.

        Default heuristic samples: each of the top-k hypotheses in the current
        species posterior may show a confirming or disconfirming signal. Concrete
        action kinds override this in their LLM-augmented variants (which the
        EVI module may inject).
        """
        species = dict(belief.species_posterior or {})
        if not species:
            return [("no_change", 1.0)]
        # Take top-k and treat each as a plausible "what we might learn" bucket.
        items = sorted(species.items(), key=lambda kv: kv[1], reverse=True)[: max(1, int(n))]
        total = sum(max(1e-6, p) for _h, p in items)
        return [(f"{self.kind.value}:{h}", max(1e-6, p) / total) for h, p in items]

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "target": self.target,
            "level": self.level,
            "payload": dict(self.payload),
            "cost": {
                "wall_s": self.estimated_cost().wall_s,
                "tokens": self.estimated_cost().tokens,
            },
        }


# -----------------------------------------------------------------
# Candidate action generator
# -----------------------------------------------------------------

def enumerate_actions(
    belief: DiagnosticBelief,
    *,
    budget_spent_frac: float = 0.0,
    findings_keys: list[str] | None = None,
    top_k: int = 3,
    allow_user_asks: bool = True,
) -> list[Action]:
    """Heuristic action expander.

    Produces a small, MCTS-tractable action set from the current belief.
    Rules:
      - Always offer STOP and ESCALATE (terminal choices).
      - For each top-k species, offer ABLATE_FINDING of its strongest evidence
        and CONSULT_SPECIALIST targeted at its family.
      - If species entropy high → RUN_SWARM_LEVEL at level 2.
      - If family entropy high → RUN_SWARM_LEVEL at level 0.
      - If uncertain critical hypothesis → REQUEST_TEST.
      - Budget near exhaustion → inject STOP with higher priority (handled by
        EVI weighting, not here).
    """
    actions: list[Action] = []
    species = dict(belief.species_posterior or {})
    family = dict(belief.family_posterior or {})

    # Terminal options always available.
    actions.append(Action(kind=ActionKind.STOP))
    actions.append(Action(kind=ActionKind.ESCALATE))

    top_species = sorted(species.items(), key=lambda kv: kv[1], reverse=True)[: max(1, top_k)]

    for h, _p in top_species:
        actions.append(Action(kind=ActionKind.CONSULT_SPECIALIST, target=h))
        actions.append(Action(kind=ActionKind.QUERY_ONTOLOGY, target=h))
        actions.append(Action(kind=ActionKind.SEARCH_LITERATURE, target=h))

    # Ablation probes against the strongest-signal findings.
    for fk in (findings_keys or [])[:3]:
        actions.append(Action(kind=ActionKind.ABLATE_FINDING, target=str(fk)))

    # Coarse swarm re-runs when any level is noisy.
    if species and _shannon_high(species):
        actions.append(Action(kind=ActionKind.RUN_SWARM_LEVEL, level=2))
    if family and _shannon_high(family):
        actions.append(Action(kind=ActionKind.RUN_SWARM_LEVEL, level=0))

    # User-facing asks (unless suppressed).
    if allow_user_asks:
        if top_species:
            actions.append(Action(kind=ActionKind.REQUEST_TEST, target=top_species[0][0]))
            actions.append(Action(kind=ActionKind.REQUEST_HISTORY, target=top_species[0][0]))

    # De-dup — same kind+target pair only once.
    seen: set[tuple[str, str, int | None]] = set()
    dedup: list[Action] = []
    for a in actions:
        key = (a.kind.value, a.target, a.level)
        if key in seen:
            continue
        seen.add(key)
        dedup.append(a)
    return dedup


def _shannon_high(post: Mapping[str, float], *, threshold: float = 1.0) -> bool:
    """Cheap check: is entropy above `threshold` bits?"""
    import math
    total = sum(max(0.0, float(v)) for v in post.values())
    if total <= 0.0:
        return False
    h = 0.0
    for v in post.values():
        p = max(1e-12, float(v) / total)
        h -= p * math.log2(p)
    return h > float(threshold)


__all__ = [
    "ActionKind",
    "Action",
    "Cost",
    "enumerate_actions",
]
