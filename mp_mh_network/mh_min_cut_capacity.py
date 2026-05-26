"""
Min-cut / max-flow reference capacity for layered MP-MH BEC (independent of simulator).

Graph: P parallel capacitated links per hop, full mixing at each relay.
Max s–t flow = min over hops h of Σ_p (1 − ε_{p,h}) for this topology.
"""

from __future__ import annotations

import networkx as nx

from mh_epsilon_matrix import article_matrix_for_hops


def _closed_form_min_cut(article_epsilon: list[list[float]]) -> float:
    """article_epsilon[path][hop] = ε; rate per link = 1 - ε."""
    num_paths = len(article_epsilon)
    num_hops = len(article_epsilon[0])
    best = float("inf")
    for h in range(num_hops):
        s = 0.0
        for p in range(num_paths):
            eps = article_epsilon[p][h]
            s += 1.0 - eps
        best = min(best, s)
    return best


def max_flow_capacity_layered_bec(article_epsilon: list[list[float]]) -> float:
    """
    Max flow from source to sink using NetworkX on a directed layered graph.

    article_epsilon[p][h] is the erasure probability on path p+1, hop h+1 (BEC).
    Capacities on forward links are r_{p,h} = 1 - ε_{p,h}.
    Relays are modeled as full mixing (high-capacity edges into/out of each relay).
    """
    num_paths = len(article_epsilon)
    num_hops = len(article_epsilon[0])
    if num_paths == 0 or num_hops == 0:
        return 0.0

    total_rate = sum(
        1.0 - article_epsilon[p][h]
        for p in range(num_paths)
        for h in range(num_hops)
    )
    inf = total_rate + 1.0

    G = nx.DiGraph()
    for h in range(num_hops + 1):
        G.add_node(f"R{h}")

    s, t = "R0", f"R{num_hops}"
    for h in range(num_hops):
        for p in range(num_paths):
            mid = f"M_{h}_{p}"
            cap = 1.0 - article_epsilon[p][h]
            G.add_edge(f"R{h}", mid, capacity=cap)
            G.add_edge(mid, f"R{h + 1}", capacity=inf)

    flow_value, _ = nx.maximum_flow(G, s, t, capacity="capacity")
    expected = _closed_form_min_cut(article_epsilon)
    if abs(flow_value - expected) > 1e-6 * max(1.0, expected):
        raise RuntimeError(
            f"Max-flow {flow_value} != closed-form min-cut {expected}; check graph model."
        )
    return float(flow_value)


def min_cut_capacity_for_epsilons(
    e1: float, e2: float, num_hops_eff: int = 3
) -> float:
    """Min-cut / max-flow for paper-derived article matrix (ε₁, ε₂) and hop count."""
    return max_flow_capacity_layered_bec(
        article_matrix_for_hops(e1, e2, num_hops_eff)
    )
