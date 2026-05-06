from src.cdss.app.trace_metrics import compute_stage_metrics_from_trace, compute_trace_wall_time
from src.cdss.contracts.models import (
    DecisionPacket,
    DecisionStatus,
    DecisionTrace,
    DifferentialCandidate,
    DifferentialSet,
    EvidenceBundle,
    RiskProfile,
    StructuredFindings,
    VerificationReport,
)


def _packet_with_trace() -> DecisionPacket:
    return DecisionPacket(
        case_id="case-trace-shared",
        status=DecisionStatus.PRELIMINARY,
        summary="trace metrics",
        structured_findings=StructuredFindings(summary="sample"),
        risk_profile=RiskProfile(),
        differential=DifferentialSet(candidates=[DifferentialCandidate(label="sample", score=0.6)]),
        evidence=EvidenceBundle(coverage=0.5, contradiction_mass=0.1),
        verification=VerificationReport(decision=DecisionStatus.PRELIMINARY, reliability_score=0.6),
        trace=[
            DecisionTrace(
                timestamp="2026-04-03T18:00:00Z",
                stage="extractor",
                message="extractor metrics",
                payload={"metrics": {"prompt_tokens": 10, "completion_tokens": 20, "time_s": 2.0}},
            ),
            DecisionTrace(
                timestamp="2026-04-03T18:00:02Z",
                stage="verification",
                message="verification metrics",
                payload={"metrics": {"prompt_tokens": 5, "completion_tokens": 15, "total_tokens": 22, "time_s": 1.0}},
            ),
            DecisionTrace(
                timestamp="2026-04-03T18:00:04Z",
                stage="ignored_stage",
                message="ignored",
                payload={"metrics": {"prompt_tokens": 100, "completion_tokens": 100, "time_s": 10.0}},
            ),
        ],
    )


def test_compute_stage_metrics_from_trace_handles_rounding_modes():
    packet = _packet_with_trace()
    stage_map = {"extractor": "R1", "verification": "IE"}

    stages_float, total_time, token_budget = compute_stage_metrics_from_trace(packet, stage_map)
    stages_int, _, _ = compute_stage_metrics_from_trace(packet, stage_map, stage_bucket_tokens_as_int=True)

    assert total_time == 3.0
    assert stages_float["R1"]["tokens"] == 30.0
    assert stages_float["IE"]["tokens"] == 22.0
    assert stages_float["R1"]["tok_s"] == 15.0
    assert stages_int["R1"]["tokens"] == 30
    assert token_budget["R1"]["total_tokens"] == 30
    assert token_budget["IE"]["total_tokens"] == 22


def test_compute_trace_wall_time_uses_first_and_last_timestamps():
    packet = _packet_with_trace()

    assert compute_trace_wall_time(packet) == 4.0
