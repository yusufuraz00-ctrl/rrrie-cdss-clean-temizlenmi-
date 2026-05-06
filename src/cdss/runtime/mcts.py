"""MCTS / UCT cognitive controller (W3 Module D.4 + Module I termination).

Short-horizon lookahead over the ten-action space defined in `actions.py`.
Budget-capped per plan: depth 2, rollout budget 20. Parallel rollouts via
`asyncio.gather` (respect `_swarm_limit` passed in by caller).

UCB1 / UCT:
    a* = argmax_a  Q̂(a) + c · √( ln(N) / n(a) )
with c = √2. Q̂(a) = mean ΔU over completed rollouts at that node. Early-cut
when one action dominates (Q gap > 0.3 after 5 rollouts) — plan §D.4.

Termination (Module I.1 / I.2):
  - STOP chosen when entropy < ε₁ AND miss_risk < ε₂.
  - ESCALATE chosen when conformal-set-size ≥ 10 at α=0.1 OR
    Dempster-Shafer conflict K > 0.7 OR must-not-miss Pl > 0.3 ∧ Bel < 0.1.

Self-refine (Module I.3) is a separate module (`self_refine.py`) invoked after
the loop terminates.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
from dataclasses import dataclass, field
from typing import Mapping

from ..contracts.models import DiagnosticBelief
from ..reasoning.bayes_update import miss_risk as _miss_risk
from ..reasoning.belief_propagation import entropy as _entropy
from ..reasoning.evi import simulate_rollout
from .actions import Action, ActionKind, enumerate_actions

_log = logging.getLogger("rrrie-cdss")
_UCT_C = math.sqrt(2.0)
_EARLY_CUT_GAP = 0.3
_EARLY_CUT_MIN_N = 5

# Termination thresholds (plan §I.1).
_EPS_ENTROPY = 0.5
_EPS_MISS_RISK = 0.05
# Escalation thresholds (plan §I.2).
_ESCALATE_CONFORMAL_SIZE = 10
_ESCALATE_DS_CONFLICT_K = 0.70
_ESCALATE_MNM_PL = 0.30
_ESCALATE_MNM_BEL = 0.10


@dataclass
class _Node:
    action: Action
    n: int = 0
    q_sum: float = 0.0
    children_du: list[float] = field(default_factory=list)

    @property
    def q_hat(self) -> float:
        return self.q_sum / self.n if self.n > 0 else 0.0


@dataclass
class MctsTickResult:
    chosen: Action
    q_hat: float
    visit_count: int
    explored: list[tuple[Action, float, int]]  # (action, Q̂, n)
    total_rollouts: int
    terminated: bool = False
    termination_reason: str = ""
    escalation_reason: str = ""


def _uct(parent_n: int, child: _Node) -> float:
    if child.n == 0:
        return math.inf  # unvisited nodes always take priority
    return child.q_hat + _UCT_C * math.sqrt(math.log(max(1, parent_n)) / child.n)


def should_stop(
    belief: DiagnosticBelief,
    *,
    severity: Mapping[str, float] | None = None,
    treatment_correct: Mapping[str, float] | None = None,
    eps_entropy: float = _EPS_ENTROPY,
    eps_miss_risk: float = _EPS_MISS_RISK,
) -> tuple[bool, str]:
    """Module I.1 termination check."""
    species = dict(belief.species_posterior or {})
    if not species:
        return False, ""
    h = float(_entropy(species))
    mr = float(_miss_risk(belief, severity=severity or {}, treatment_correct=treatment_correct or {}))
    if h < float(eps_entropy) and mr < float(eps_miss_risk):
        return True, f"confident: H={h:.2f}<{eps_entropy:.2f} and miss_risk={mr:.3f}<{eps_miss_risk:.3f}"
    return False, ""


def should_escalate(
    belief: DiagnosticBelief,
    *,
    conformal_set_size: int | None = None,
    ds_conflict_K: float = 0.0,
    mnm_brackets: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[bool, str]:
    """Module I.2 escalation check.

    `conformal_set_size` from Module H when available; `ds_conflict_K` from
    Module E fusion step; `mnm_brackets` is {hypothesis: (Bel, Pl)} on the
    must-not-miss subset.
    """
    if conformal_set_size is not None and int(conformal_set_size) >= _ESCALATE_CONFORMAL_SIZE:
        return True, f"conformal_set_size={int(conformal_set_size)}≥{_ESCALATE_CONFORMAL_SIZE}"
    if float(ds_conflict_K) > _ESCALATE_DS_CONFLICT_K:
        return True, f"ds_conflict_K={float(ds_conflict_K):.2f}>{_ESCALATE_DS_CONFLICT_K}"
    if mnm_brackets:
        for h, (bel, pl) in mnm_brackets.items():
            if float(pl) > _ESCALATE_MNM_PL and float(bel) < _ESCALATE_MNM_BEL:
                return True, f"must_not_miss:{h} Pl={pl:.2f}>{_ESCALATE_MNM_PL} Bel={bel:.2f}<{_ESCALATE_MNM_BEL}"
    return False, ""


async def _score_one(
    belief: DiagnosticBelief,
    action: Action,
    *,
    severity: Mapping[str, float] | None,
    treatment_correct: Mapping[str, float] | None,
    utility_kwargs: dict,
) -> float:
    """Single rollout ΔU — cheap model-based estimate."""
    # `estimate_evi` already subtracts cost; we only need its raw ΔU here so
    # MCTS can assemble Q̂ from ΔU across rollouts before charging cost once.
    return simulate_rollout(
        belief,
        action,
        severity=severity,
        treatment_correct=treatment_correct,
        utility_kwargs=utility_kwargs,
    )


async def tick(
    belief: DiagnosticBelief,
    *,
    severity: Mapping[str, float] | None = None,
    treatment_correct: Mapping[str, float] | None = None,
    budget_spent_frac: float = 0.0,
    rollout_budget: int = 20,
    parallelism: int = 4,
    findings_keys: list[str] | None = None,
    conformal_set_size: int | None = None,
    ds_conflict_K: float = 0.0,
    mnm_brackets: Mapping[str, tuple[float, float]] | None = None,
    allow_user_asks: bool = True,
    utility_kwargs: dict | None = None,
    rng: random.Random | None = None,
) -> MctsTickResult:
    """Run one MCTS tick and pick an action.

    Termination / escalation checked *first* — if either fires, we short-circuit
    and return the terminal action without burning rollouts.
    """
    u_kwargs = dict(utility_kwargs or {})
    rng = rng or random.Random(0xC0FFEE)

    # Terminal checks (Module I.1/I.2).
    stop, stop_reason = should_stop(
        belief,
        severity=severity,
        treatment_correct=treatment_correct,
        eps_entropy=float(u_kwargs.get("eps_entropy", _EPS_ENTROPY)),
        eps_miss_risk=float(u_kwargs.get("eps_miss_risk", _EPS_MISS_RISK)),
    )
    if stop:
        return MctsTickResult(
            chosen=Action(kind=ActionKind.STOP),
            q_hat=0.0,
            visit_count=0,
            explored=[],
            total_rollouts=0,
            terminated=True,
            termination_reason=stop_reason,
        )
    escalate, escalate_reason = should_escalate(
        belief,
        conformal_set_size=conformal_set_size,
        ds_conflict_K=ds_conflict_K,
        mnm_brackets=mnm_brackets,
    )
    if escalate:
        return MctsTickResult(
            chosen=Action(kind=ActionKind.ESCALATE),
            q_hat=0.0,
            visit_count=0,
            explored=[],
            total_rollouts=0,
            terminated=True,
            escalation_reason=escalate_reason,
        )

    # Candidate expansion.
    candidates = enumerate_actions(
        belief,
        budget_spent_frac=budget_spent_frac,
        findings_keys=findings_keys or [],
        allow_user_asks=allow_user_asks,
    )
    # Drop STOP/ESCALATE from MCTS scoring — terminal checks above already
    # settle those; keeping them in rollouts only inflates the action set.
    candidates = [a for a in candidates if a.kind not in (ActionKind.STOP, ActionKind.ESCALATE)]
    if not candidates:
        return MctsTickResult(
            chosen=Action(kind=ActionKind.STOP),
            q_hat=0.0,
            visit_count=0,
            explored=[],
            total_rollouts=0,
            terminated=True,
            termination_reason="no_candidate_actions",
        )

    nodes: list[_Node] = [_Node(action=a) for a in candidates]
    total_rollouts = 0
    budget = max(1, int(rollout_budget))
    sem = asyncio.Semaphore(max(1, int(parallelism)))

    async def _run_one(node: _Node) -> None:
        async with sem:
            du = await _score_one(
                belief,
                node.action,
                severity=severity,
                treatment_correct=treatment_correct,
                utility_kwargs=u_kwargs,
            )
        node.q_sum += float(du)
        node.n += 1
        node.children_du.append(float(du))

    # Progressive widening / UCT rollouts until budget or early-cut fires.
    early_cut_triggered = False
    while total_rollouts < budget and not early_cut_triggered:
        # Pick node with max UCT; break ties randomly.
        parent_n = sum(n.n for n in nodes) + 1
        scored = [(_uct(parent_n, n), n) for n in nodes]
        max_score = max(s for s, _ in scored)
        pool = [n for s, n in scored if s >= max_score - 1e-9]
        node = rng.choice(pool)
        await _run_one(node)
        total_rollouts += 1

        # Early-cut: one action's Q̂ dominates.
        visited = [n for n in nodes if n.n >= _EARLY_CUT_MIN_N]
        if len(visited) >= 2:
            sorted_q = sorted(visited, key=lambda n: n.q_hat, reverse=True)
            if (sorted_q[0].q_hat - sorted_q[1].q_hat) > _EARLY_CUT_GAP:
                early_cut_triggered = True

    # Decide.
    chosen = max(nodes, key=lambda n: (n.q_hat, n.n))
    # If best Q̂ is ≤ 0 (no action helps) → stop per plan §D.5.
    if chosen.q_hat <= 0.0:
        return MctsTickResult(
            chosen=Action(kind=ActionKind.STOP),
            q_hat=chosen.q_hat,
            visit_count=chosen.n,
            explored=[(n.action, n.q_hat, n.n) for n in nodes],
            total_rollouts=total_rollouts,
            terminated=True,
            termination_reason=f"no_positive_EVI: best_Qhat={chosen.q_hat:.3f}",
        )

    return MctsTickResult(
        chosen=chosen.action,
        q_hat=chosen.q_hat,
        visit_count=chosen.n,
        explored=[(n.action, n.q_hat, n.n) for n in nodes],
        total_rollouts=total_rollouts,
    )


__all__ = [
    "tick",
    "should_stop",
    "should_escalate",
    "MctsTickResult",
]
