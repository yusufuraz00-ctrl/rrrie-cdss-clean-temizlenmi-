"""Pure, side-effect-free metric functions for the CDSS eval harness.

Each function takes structured inputs (decision packets and expected
labels/families) and returns a typed metric record. The orchestration
script `scripts/run_eval.py` is responsible for running the pipeline,
loading suites, and writing reports — this module never performs I/O.

Metrics
-------

- ``family_hit_at_k``  : 1.0 if the expected disease family appears in the
  top-k predicted candidate families, else 0.0. Family is taken from the
  candidate's ``parent_category``; if that is empty, the candidate label
  itself is used (after lowercasing).

- ``calibration_ece`` : Expected Calibration Error across ``n_bins`` equal-width
  confidence bins. Inputs are aligned arrays of confidences and binary
  correctness flags. ECE is the weighted average of |bin_acc − bin_conf|.

- ``hallucination_rate`` : Average grounding-risk score across packets where
  ``inline_grounding_pass_rate`` is recorded. Higher score means more
  unsupported claims in the rationale of swarm output.

- ``safety_correctness`` : For cases tagged with a must-not-miss expectation,
  did the system flag urgency ≥ URGENT *and* surface that family in the top-3?

- ``abstention_precision`` : When the system abstained, what fraction of
  those cases would have been wrong if the top-1 had been used? Higher is
  better — abstention should fire when the system is genuinely uncertain.

- ``margin_histogram`` : Bucketed counts of the ``top1 − top2`` fused
  score margin across the cohort, for quick distribution sanity checks.

All functions return plain dicts (or float scalars) so they serialize
cleanly to JSON for the eval report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence


def _candidate_family(candidate: Mapping[str, Any] | Any) -> str:
    """Return the family token to compare against the expected family."""
    if isinstance(candidate, Mapping):
        family = str(candidate.get("parent_category") or candidate.get("family_label") or "").strip().lower()
        if family:
            return family
        label = str(candidate.get("label") or "").strip().lower()
        return label
    family = str(getattr(candidate, "parent_category", "") or getattr(candidate, "family_label", "") or "").strip().lower()
    if family:
        return family
    return str(getattr(candidate, "label", "") or "").strip().lower()


def _packet_candidates(packet: Mapping[str, Any] | Any) -> list[Any]:
    if isinstance(packet, Mapping):
        diff = packet.get("differential") or {}
        if isinstance(diff, Mapping):
            return list(diff.get("candidates") or [])
        return list(getattr(diff, "candidates", []) or [])
    diff = getattr(packet, "differential", None)
    if diff is None:
        return []
    return list(getattr(diff, "candidates", []) or [])


def _packet_top_score(packet: Mapping[str, Any] | Any) -> float:
    cands = _packet_candidates(packet)
    if not cands:
        return 0.0
    head = cands[0]
    score = head.get("score") if isinstance(head, Mapping) else getattr(head, "score", 0.0)
    try:
        return float(score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _packet_runner_up_score(packet: Mapping[str, Any] | Any) -> float:
    cands = _packet_candidates(packet)
    if len(cands) < 2:
        return 0.0
    second = cands[1]
    score = second.get("score") if isinstance(second, Mapping) else getattr(second, "score", 0.0)
    try:
        return float(score or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _packet_urgency(packet: Mapping[str, Any] | Any) -> str:
    if isinstance(packet, Mapping):
        risk = packet.get("risk_profile") or {}
        if isinstance(risk, Mapping):
            return str(risk.get("urgency", "") or "").upper()
        return str(getattr(risk, "urgency", "") or "").upper()
    risk = getattr(packet, "risk_profile", None)
    return str(getattr(risk, "urgency", "") or "").upper()


def _packet_decision(packet: Mapping[str, Any] | Any) -> str:
    if isinstance(packet, Mapping):
        return str(packet.get("decision") or packet.get("status") or "").lower()
    return str(getattr(packet, "decision", "") or getattr(packet, "status", "") or "").lower()


def _packet_grounding_pass_rate(packet: Mapping[str, Any] | Any) -> float | None:
    """Return inline grounding pass rate if recorded on the packet, else None."""
    if isinstance(packet, Mapping):
        bundle = packet.get("typed_case_bundle") or packet.get("trace") or {}
        if isinstance(bundle, Mapping):
            value = bundle.get("inline_grounding_pass_rate")
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return None
        value = packet.get("inline_grounding_pass_rate")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    bundle = getattr(packet, "typed_case_bundle", None)
    if isinstance(bundle, Mapping):
        value = bundle.get("inline_grounding_pass_rate")
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


# ---------------------------------------------------------------------------
# Family hit @ k
# ---------------------------------------------------------------------------


def family_hit_at_k(packet: Mapping[str, Any] | Any, expected_family: str, k: int) -> int:
    """Return 1 if expected_family is in top-k families, else 0."""
    expected = (expected_family or "").strip().lower()
    if not expected:
        return 0
    cands = _packet_candidates(packet)[: max(1, int(k))]
    families = [_candidate_family(c) for c in cands]
    return 1 if any(expected in fam or fam in expected for fam in families if fam) else 0


def family_hit_at_k_cohort(
    pairs: Sequence[tuple[Mapping[str, Any] | Any, str]],
    k: int,
) -> float:
    """Mean family-hit@k over a cohort. Returns 0.0 when empty."""
    if not pairs:
        return 0.0
    hits = sum(family_hit_at_k(p, fam, k) for p, fam in pairs)
    return round(hits / len(pairs), 4)


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------


def calibration_ece(
    confidences: Sequence[float],
    correctness: Sequence[int],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error.

    confidences ∈ [0, 1] aligned with correctness ∈ {0, 1}. Returns ECE
    (lower is better, 0 = perfect calibration). Empty inputs return 0.0.
    """
    if not confidences or len(confidences) != len(correctness):
        return 0.0
    n = len(confidences)
    bins = max(1, int(n_bins))
    edges = [i / bins for i in range(bins + 1)]
    ece = 0.0
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [
            (c, k) for c, k in zip(confidences, correctness)
            if (c >= lo and (c < hi or (b == bins - 1 and c <= hi)))
        ]
        if not in_bin:
            continue
        bin_conf = sum(c for c, _ in in_bin) / len(in_bin)
        bin_acc = sum(k for _, k in in_bin) / len(in_bin)
        ece += (len(in_bin) / n) * abs(bin_conf - bin_acc)
    return round(ece, 4)


def reliability_diagram(
    confidences: Sequence[float],
    correctness: Sequence[int],
    n_bins: int = 10,
) -> list[dict[str, float]]:
    """Bin-level reliability data: {bin_lo, bin_hi, count, mean_conf, mean_acc}."""
    if not confidences or len(confidences) != len(correctness):
        return []
    n_bins = max(1, int(n_bins))
    edges = [i / n_bins for i in range(n_bins + 1)]
    rows: list[dict[str, float]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = [
            (c, k) for c, k in zip(confidences, correctness)
            if (c >= lo and (c < hi or (b == n_bins - 1 and c <= hi)))
        ]
        if not in_bin:
            rows.append({"bin_lo": lo, "bin_hi": hi, "count": 0, "mean_conf": 0.0, "mean_acc": 0.0})
            continue
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "count": len(in_bin),
            "mean_conf": round(sum(c for c, _ in in_bin) / len(in_bin), 4),
            "mean_acc": round(sum(k for _, k in in_bin) / len(in_bin), 4),
        })
    return rows


# ---------------------------------------------------------------------------
# Hallucination rate
# ---------------------------------------------------------------------------


def hallucination_rate(packets: Sequence[Mapping[str, Any] | Any]) -> dict[str, float]:
    """Aggregate hallucination metrics across a cohort.

    Uses the inline grounding pass rate recorded on each packet. The
    hallucination rate is 1 − mean(pass_rate). Packets without a recorded
    pass rate are excluded from the average.
    """
    pass_rates = [r for r in (_packet_grounding_pass_rate(p) for p in packets) if r is not None]
    if not pass_rates:
        return {"hallucination_rate": 0.0, "covered_packets": 0, "total_packets": len(packets)}
    mean_pass = sum(pass_rates) / len(pass_rates)
    return {
        "hallucination_rate": round(1.0 - mean_pass, 4),
        "covered_packets": len(pass_rates),
        "total_packets": len(packets),
    }


# ---------------------------------------------------------------------------
# Safety correctness
# ---------------------------------------------------------------------------


_URGENCY_RANK = {"ROUTINE": 0, "URGENT": 1, "EMERGENCY": 2}


def safety_correctness(
    packet: Mapping[str, Any] | Any,
    expected_family: str,
    *,
    must_not_miss: bool,
    require_urgent: bool = True,
) -> dict[str, Any]:
    """Did the system both flag urgency ≥ URGENT and surface the must-not-miss family?"""
    if not must_not_miss:
        return {"applicable": False, "passed": True}
    urgency = _packet_urgency(packet)
    urgent_ok = (not require_urgent) or _URGENCY_RANK.get(urgency, 0) >= 1
    family_in_top3 = bool(family_hit_at_k(packet, expected_family, 3))
    return {
        "applicable": True,
        "passed": bool(urgent_ok and family_in_top3),
        "urgency_observed": urgency,
        "family_in_top3": family_in_top3,
    }


def safety_correctness_cohort(
    cases: Sequence[tuple[Mapping[str, Any] | Any, str, bool]],
) -> dict[str, float]:
    """Cohort version. cases is a sequence of (packet, expected_family, must_not_miss)."""
    applicable = [c for c in cases if c[2]]
    if not applicable:
        return {"safety_pass_rate": 1.0, "applicable_cases": 0}
    passes = sum(1 for p, fam, mn in applicable if safety_correctness(p, fam, must_not_miss=mn).get("passed"))
    return {
        "safety_pass_rate": round(passes / len(applicable), 4),
        "applicable_cases": len(applicable),
    }


# ---------------------------------------------------------------------------
# Abstention precision
# ---------------------------------------------------------------------------


@dataclass
class AbstentionRecord:
    abstained: bool
    top1_family: str
    expected_family: str

    @property
    def top1_correct(self) -> bool:
        a = (self.top1_family or "").strip().lower()
        b = (self.expected_family or "").strip().lower()
        return bool(a and b and (a in b or b in a))


def abstention_precision(records: Sequence[AbstentionRecord]) -> dict[str, float]:
    """Fraction of abstentions where top-1 *would* have been wrong (the good case)."""
    abstained = [r for r in records if r.abstained]
    if not abstained:
        return {"abstention_precision": 0.0, "abstentions": 0, "total": len(records)}
    correct_to_abstain = sum(1 for r in abstained if not r.top1_correct)
    return {
        "abstention_precision": round(correct_to_abstain / len(abstained), 4),
        "abstentions": len(abstained),
        "total": len(records),
    }


# ---------------------------------------------------------------------------
# Margin histogram
# ---------------------------------------------------------------------------


def margin_histogram(
    packets: Sequence[Mapping[str, Any] | Any],
    edges: Sequence[float] = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0),
) -> dict[str, int]:
    """Bucketed counts of (top1 - top2) fused margin."""
    counts: dict[str, int] = {}
    edge_list = list(edges)
    for i in range(len(edge_list) - 1):
        lo, hi = edge_list[i], edge_list[i + 1]
        counts[f"{lo:.2f}-{hi:.2f}"] = 0
    for packet in packets:
        margin = max(0.0, _packet_top_score(packet) - _packet_runner_up_score(packet))
        for i in range(len(edge_list) - 1):
            lo, hi = edge_list[i], edge_list[i + 1]
            if margin >= lo and (margin < hi or (i == len(edge_list) - 2 and margin <= hi)):
                counts[f"{lo:.2f}-{hi:.2f}"] += 1
                break
    return counts


# ---------------------------------------------------------------------------
# Top-level cohort report
# ---------------------------------------------------------------------------


@dataclass
class EvalRecord:
    case_id: str
    expected_family: str
    must_not_miss: bool
    packet: Mapping[str, Any] | Any
    abstained: bool = False
    top1_family: str = ""
    confidence: float = 0.0


def cohort_report(records: Sequence[EvalRecord]) -> dict[str, Any]:
    """Produce a single JSON-serializable eval report from a cohort of records."""
    if not records:
        return {"n": 0}

    pairs = [(r.packet, r.expected_family) for r in records]
    family_at_1 = family_hit_at_k_cohort(pairs, 1)
    family_at_3 = family_hit_at_k_cohort(pairs, 3)
    family_at_5 = family_hit_at_k_cohort(pairs, 5)

    confidences = [float(r.confidence or 0.0) for r in records]
    correctness = [int(family_hit_at_k(r.packet, r.expected_family, 1)) for r in records]
    ece = calibration_ece(confidences, correctness)
    diagram = reliability_diagram(confidences, correctness)

    halluc = hallucination_rate([r.packet for r in records])
    safety = safety_correctness_cohort(
        [(r.packet, r.expected_family, r.must_not_miss) for r in records]
    )
    abst = abstention_precision(
        [AbstentionRecord(r.abstained, r.top1_family, r.expected_family) for r in records]
    )
    margins = margin_histogram([r.packet for r in records])

    return {
        "n": len(records),
        "family_hit_at_1": family_at_1,
        "family_hit_at_3": family_at_3,
        "family_hit_at_5": family_at_5,
        "calibration_ece": ece,
        "reliability_diagram": diagram,
        "hallucination": halluc,
        "safety": safety,
        "abstention": abst,
        "margin_histogram": margins,
    }


__all__ = [
    "AbstentionRecord",
    "EvalRecord",
    "abstention_precision",
    "calibration_ece",
    "cohort_report",
    "family_hit_at_k",
    "family_hit_at_k_cohort",
    "hallucination_rate",
    "margin_histogram",
    "reliability_diagram",
    "safety_correctness",
    "safety_correctness_cohort",
]
