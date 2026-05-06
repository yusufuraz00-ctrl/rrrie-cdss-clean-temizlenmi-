"""Pearl-style `do`-calculus mechanism verifier (W4 Module F.3).

Pure-math utilities over the `MechanismGraph` typed DAG. No LLM, no state
mutation. Callers (`BACKWARD_SIMULATION` in `state_machine.py`) build the
graph from `MechanismFrame.causal_edges` + the pathway-prior asset
(`data/cdss/knowledge/pathway_edges.json`) and then run the probes here.

Three public primitives:

  - `do(graph, intervention)`    — sever incoming edges to the intervened
                                    node (Pearl truncation). Returns a fresh
                                    `MechanismGraph`.
  - `explain_coverage(graph, h, findings)`
                                  — fraction of `findings` reachable from
                                    hypothesis `h` along directed edges in
                                    `graph`.
  - `robustness_score(graph, h, findings)`
                                  — mean `explain_coverage` across single-
                                    finding ablations (observational vs.
                                    counterfactual). High = stable mechanism;
                                    low = brittle, likely overfit story.

Math reference: Pearl 2000, §1.3 (truncated factorization for `do`). Ablation
flavor is a poor-man's ACE (average causal effect) under the assumption that
edge weights are ~1 and paths are boolean-reachable.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable, Mapping

from ..contracts.models import CausalEdge, MechanismGraph


# -----------------------------------------------------------------
# Graph utilities
# -----------------------------------------------------------------

def _adj(graph: MechanismGraph) -> dict[str, list[tuple[str, float]]]:
    """Build forward adjacency list from a `MechanismGraph`."""
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in graph.edges or []:
        adj[str(e.from_node)].append((str(e.to_node), float(e.weight)))
    return adj


def _all_nodes(graph: MechanismGraph) -> set[str]:
    nodes = set(str(n) for n in (graph.nodes or []))
    for e in graph.edges or []:
        nodes.add(str(e.from_node))
        nodes.add(str(e.to_node))
    return nodes


def _reachable(graph: MechanismGraph, source: str) -> set[str]:
    """BFS over directed edges — returns all nodes reachable from `source`."""
    adj = _adj(graph)
    seen: set[str] = set()
    if source not in _all_nodes(graph):
        return seen
    q: deque[str] = deque([source])
    seen.add(source)
    while q:
        u = q.popleft()
        for v, _w in adj.get(u, []):
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


# -----------------------------------------------------------------
# Core primitives
# -----------------------------------------------------------------

def do(graph: MechanismGraph, intervention: Iterable[str]) -> MechanismGraph:
    """Pearl `do(X=x)` — sever all *incoming* edges to intervened nodes.

    Returned graph has the same node set and every edge whose head is NOT in
    `intervention`. Weights and pathway refs preserved on surviving edges.
    """
    intv: set[str] = {str(x) for x in intervention or []}
    if not intv:
        return MechanismGraph(
            nodes=list(graph.nodes or []),
            edges=[CausalEdge.model_validate(e.model_dump()) for e in (graph.edges or [])],
            hypotheses=list(graph.hypotheses or []),
            findings=list(graph.findings or []),
        )
    surviving: list[CausalEdge] = []
    for e in graph.edges or []:
        if str(e.to_node) in intv:
            continue  # severed
        surviving.append(CausalEdge.model_validate(e.model_dump()))
    return MechanismGraph(
        nodes=list(graph.nodes or []),
        edges=surviving,
        hypotheses=list(graph.hypotheses or []),
        findings=list(graph.findings or []),
    )


def explain_coverage(
    graph: MechanismGraph,
    hypothesis: str,
    findings: Iterable[str],
) -> float:
    """Fraction of `findings` reachable from `hypothesis` along directed edges.

    Returns 0.0 when `findings` is empty OR when `hypothesis` not in the graph.
    Reachability is pure connectivity — edge weights not thresholded here
    (they're Pearl-structural, not probabilistic). Use `weighted_coverage` for
    a weight-aware variant.
    """
    f_list = [str(f) for f in (findings or []) if str(f)]
    if not f_list:
        return 0.0
    reach = _reachable(graph, str(hypothesis))
    if not reach:
        return 0.0
    hit = sum(1 for f in f_list if f in reach)
    return float(hit) / float(len(f_list))


def weighted_coverage(
    graph: MechanismGraph,
    hypothesis: str,
    findings: Iterable[str],
) -> float:
    """Weight-aware coverage: each finding's contribution = max path-weight.

    Path weight = product of edge weights along the heaviest reachable path.
    Computed via relaxation (Dijkstra-like on -log weights) capped at graph
    size to keep the probe O(|V|·|E|). Returns mean over findings.
    """
    f_list = [str(f) for f in (findings or []) if str(f)]
    if not f_list:
        return 0.0
    adj = _adj(graph)
    # Best-weight-to-target via BFS with max-weight relaxation (graph is small).
    best: dict[str, float] = {str(hypothesis): 1.0}
    changed = True
    guard = 0
    max_iter = max(4, len(_all_nodes(graph)))
    while changed and guard < max_iter:
        changed = False
        guard += 1
        for u, w_u in list(best.items()):
            for v, w_e in adj.get(u, []):
                cand = float(w_u) * max(0.0, min(1.0, float(w_e)))
                if cand > best.get(v, 0.0):
                    best[v] = cand
                    changed = True
    total = 0.0
    for f in f_list:
        total += float(best.get(f, 0.0))
    return total / float(len(f_list))


def robustness_score(
    graph: MechanismGraph,
    hypothesis: str,
    findings: Iterable[str],
) -> float:
    """Mean explain_coverage across single-finding ablations.

    For each finding `f_i`: drop it from the target list, re-score. Mechanism
    is robust if removing any one piece of evidence doesn't collapse coverage.

    Returns value in [0, 1]. 1.0 = ablating any single finding leaves the
    rest still fully explained by the mechanism. 0.0 = mechanism only
    explained one finding (brittle).
    """
    f_list = [str(f) for f in (findings or []) if str(f)]
    if len(f_list) < 2:
        # Not enough evidence to ablate — fall back to raw coverage.
        return explain_coverage(graph, hypothesis, f_list)
    reach = _reachable(graph, str(hypothesis))
    if not reach:
        return 0.0
    n = len(f_list)
    scores: list[float] = []
    for i in range(n):
        held_out = f_list[:i] + f_list[i + 1 :]
        hit = sum(1 for f in held_out if f in reach)
        scores.append(float(hit) / float(len(held_out) or 1))
    return sum(scores) / float(len(scores)) if scores else 0.0


def counterfactual_drop(
    graph: MechanismGraph,
    hypothesis: str,
    findings: Iterable[str],
    dropped_finding: str,
) -> float:
    """Coverage when `dropped_finding` is severed by `do({dropped_finding})`.

    Useful for "would this mechanism still hold if this finding were NOT
    caused by the hypothesis?" Severs incoming edges to `dropped_finding`,
    then measures coverage over the remaining findings.
    """
    f_list = [str(f) for f in (findings or []) if str(f) and str(f) != str(dropped_finding)]
    if not f_list:
        return 0.0
    perturbed = do(graph, [str(dropped_finding)])
    return explain_coverage(perturbed, str(hypothesis), f_list)


# -----------------------------------------------------------------
# Builder helpers
# -----------------------------------------------------------------

def build_graph_from_edges(
    *,
    hypotheses: Iterable[str],
    findings: Iterable[str],
    edges: Iterable[Mapping | CausalEdge],
) -> MechanismGraph:
    """Assemble a `MechanismGraph` from loose inputs (pathway asset, frame).

    Accepts edges either as `CausalEdge` instances OR as plain dicts with
    `from`/`from_node`, `to`/`to_node`, `weight`, `pathway_ref` keys — the
    pathway-asset JSON uses the short spelling.
    """
    hyps = [str(h) for h in (hypotheses or []) if str(h)]
    fnds = [str(f) for f in (findings or []) if str(f)]
    raw_edges: list[CausalEdge] = []
    for e in edges or []:
        if isinstance(e, CausalEdge):
            raw_edges.append(e)
            continue
        if not isinstance(e, Mapping):
            continue
        frm = str(e.get("from_node") or e.get("from") or "").strip()
        to = str(e.get("to_node") or e.get("to") or "").strip()
        if not frm or not to:
            continue
        w = float(e.get("weight", 1.0) or 1.0)
        ref = str(e.get("pathway_ref") or e.get("pathway") or "")
        raw_edges.append(CausalEdge(from_node=frm, to_node=to, weight=w, pathway_ref=ref))
    nodes: set[str] = set(hyps) | set(fnds)
    for e in raw_edges:
        nodes.add(e.from_node)
        nodes.add(e.to_node)
    return MechanismGraph(
        nodes=sorted(nodes),
        edges=raw_edges,
        hypotheses=hyps,
        findings=fnds,
    )


__all__ = [
    "do",
    "explain_coverage",
    "weighted_coverage",
    "robustness_score",
    "counterfactual_drop",
    "build_graph_from_edges",
]
