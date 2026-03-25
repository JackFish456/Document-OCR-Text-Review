"""Bipartite one-to-one assignment via min-cost max-flow (pure Python, no SciPy/NetworkX).

Maximum-weight matching on a padded cost matrix is reduced to a minimum-cost perfect
matching on a square network: ``minimize sum((1 - w) * scale)`` for real edges and
large dummy costs for skipping. A successive shortest augmenting path algorithm with
Dijkstra and potentials (Johnson reweighting) runs in ``O(n^3 log n)`` for dense
``n × n`` matrices (typical ``n`` in the hundreds).

The padded cost matrix is built so a minimum-cost perfect matching corresponds to a
maximum-weight matching on the original (possibly rectangular) weight matrix.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Final

_INF: Final[int] = 10**18


@dataclass(slots=True)
class _MFEdge:
    """Directed edge in a min-cost flow adjacency list."""

    to: int
    rev: int
    cap: int
    cost: int


def _add_mf_edge(graph: list[list[_MFEdge]], fr: int, to: int, cap: int, cost: int) -> None:
    fwd = _MFEdge(to=to, rev=len(graph[to]), cap=cap, cost=cost)
    rev = _MFEdge(to=fr, rev=len(graph[fr]), cap=0, cost=-cost)
    graph[fr].append(fwd)
    graph[to].append(rev)


def _min_cost_max_flow(graph: list[list[_MFEdge]], s: int, t: int, maxf: int) -> int:
    """Minimum total cost to send ``maxf`` units from ``s`` to ``t`` (``maxf`` small)."""
    res = 0
    n = len(graph)
    h = [0] * n
    prevv = [0] * n
    preve = [0] * n

    while maxf > 0:
        dist = [_INF] * n
        dist[s] = 0
        pq: list[tuple[int, int]] = [(0, s)]
        while pq:
            d, v = heapq.heappop(pq)
            if dist[v] < d:
                continue
            for i, e in enumerate(graph[v]):
                if e.cap <= 0:
                    continue
                nd = d + e.cost + h[v] - h[e.to]
                if dist[e.to] > nd:
                    dist[e.to] = nd
                    prevv[e.to] = v
                    preve[e.to] = i
                    heapq.heappush(pq, (nd, e.to))
        if dist[t] == _INF:
            break
        for v in range(n):
            if dist[v] < _INF:
                h[v] += dist[v]
        d = maxf
        v = t
        while v != s:
            pv = prevv[v]
            pe = graph[pv][preve[v]]
            d = min(d, pe.cap)
            v = pv
        maxf -= d
        res += d * h[t]
        v = t
        while v != s:
            pv = prevv[v]
            pe = graph[pv][preve[v]]
            rev = graph[pe.to][pe.rev]
            pe.cap -= d
            rev.cap += d
            v = pv
    return res


def _min_cost_perfect_matching_square(cost: list[list[int]]) -> list[int]:
    """Minimum-cost permutation: ``assignment[i] = j`` for an ``n × n`` matrix."""
    n = len(cost)
    if n == 0:
        return []
    s = 0
    left = 1
    right = 1 + n
    t = 1 + 2 * n
    graph: list[list[_MFEdge]] = [[] for _ in range(t + 1)]

    for i in range(n):
        _add_mf_edge(graph, s, left + i, 1, 0)
    for j in range(n):
        _add_mf_edge(graph, right + j, t, 1, 0)
    for i in range(n):
        for j in range(n):
            _add_mf_edge(graph, left + i, right + j, 1, cost[i][j])

    _min_cost_max_flow(graph, s, t, n)

    assign = [-1] * n
    for i in range(n):
        for e in graph[left + i]:
            if right <= e.to < right + n and e.cap == 0:
                assign[i] = e.to - right
                break
    return assign


def optimal_assignment_max_weight(
    n_rows: int,
    n_cols: int,
    weight: list[list[float | None]],
) -> list[tuple[int, int]]:
    """Maximum-weight matching with at most one pair per row and column.

    ``weight[i][j]`` is ``None`` if the edge is forbidden. Otherwise it is a score in
    ``[0, 1]`` to **maximize**.

    Returns ``(row_index, col_index)`` pairs sorted by ``(row, col)``.
    """
    if n_rows == 0 or n_cols == 0:
        return []
    if len(weight) != n_rows or any(len(row) != n_cols for row in weight):
        raise ValueError("weight shape must match n_rows x n_cols")

    dim = max(n_rows, n_cols) + 1
    scale = 1_000_000
    forbidden = 10**12
    dummy = scale

    cost: list[list[int]] = [[dummy] * dim for _ in range(dim)]

    for i in range(n_rows):
        for j in range(n_cols):
            w = weight[i][j]
            if w is None:
                cost[i][j] = forbidden
            else:
                wc = max(0.0, min(1.0, float(w)))
                cost[i][j] = int(round((1.0 - wc) * scale))

    assign = _min_cost_perfect_matching_square(cost)
    pairs: list[tuple[int, int]] = []
    for i in range(n_rows):
        j = assign[i]
        if j < 0 or j >= n_cols:
            continue
        w = weight[i][j]
        if w is None:
            continue
        pairs.append((i, j))

    pairs.sort()
    return pairs
