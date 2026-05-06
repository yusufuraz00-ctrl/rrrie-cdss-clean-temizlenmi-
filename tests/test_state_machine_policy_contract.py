import asyncio
from types import SimpleNamespace

from src.cdss.contracts.models import CdssPhase, DifferentialSet, LoopDirective
from src.cdss.runtime.state_machine import CdssStateMachineAgent


class _State:
    def __init__(self):
        self.working_hypotheses = []
        self.transitions = []

    def transition(self, phase):
        self.transitions.append(phase)


class _Gateway:
    def __init__(self):
        self.bridge = SimpleNamespace()

    async def challenge(self, *args, **kwargs):
        return SimpleNamespace(
            metrics={},
            issues=[],
            alt_hypotheses=[],
            loop_directive=LoopDirective(),
            error=None,
        )


def test_outcome_simulation_uses_runtime_policy_contract():
    agent = CdssStateMachineAgent.__new__(CdssStateMachineAgent)
    agent.runtime_policy = SimpleNamespace(
        always_on_challenger_enabled=False,
        steelman_enabled=False,
        counterfactual_ablation_enabled=False,
        premise_conflict_resolver_enabled=False,
        bradley_terry_tournament_enabled=False,
        bma_outcome_enabled=False,
    )
    assert not hasattr(agent, "policy")

    agent.deep_thinking = False
    agent._case_complexity = "routine"
    agent.typed_case_bundle = {"differential": DifferentialSet()}
    agent.state = _State()
    agent._gateway = _Gateway()
    agent._set_profile_stage = lambda stage: None
    agent._ensure_typed_findings = lambda: SimpleNamespace()
    agent._ensure_typed_fact_graph = lambda: SimpleNamespace(nodes=[])
    agent._provisional_frontier = lambda: SimpleNamespace()
    agent._typed_evidence = lambda: SimpleNamespace(atoms=[], contradiction_mass=0.0)
    agent._typed_interventions = lambda: SimpleNamespace()
    agent._typed_risk_profile = lambda: SimpleNamespace()
    agent._record_gateway_metrics = lambda *args, **kwargs: None
    agent._record_stage_profile = lambda *args, **kwargs: None
    agent._stage_metrics = lambda *args, **kwargs: {}
    agent._policy_snapshot = lambda: agent.runtime_policy

    async def emit_trace(*args, **kwargs):
        return None

    agent._emit_trace = emit_trace

    asyncio.run(agent._run_test_time_simulation())

    assert agent.state.transitions == [CdssPhase.VERIFICATION]
