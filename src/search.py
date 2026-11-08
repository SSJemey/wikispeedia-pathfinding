"""Search algorithms. Every link costs 1, so path cost equals the number of clicks."""
import heapq
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class SearchResult:
    path: Optional[list]   # article ids from source to target; None if no path exists
    expanded: int          # nodes whose links were explored
    edges_scanned: int     # links examined
    max_stored: int        # most nodes held in memory at once (frontier + explored)

    @property
    def cost(self):
        return None if self.path is None else len(self.path) - 1


def _build_path(parents, node):
    path = [node]
    while parents[node] is not None:
        node = parents[node]
        path.append(node)
    path.reverse()
    return path


def bfs(graph, source, target):
    """Plain BFS: the shortest-path reference answer for this unweighted graph."""
    if source == target:
        return SearchResult([source], 0, 0, 1)
    parents = {source: None}
    frontier = deque([source])
    expanded = edges = 0
    while frontier:
        node = frontier.popleft()
        expanded += 1
        for neighbour in graph.out_edges[node]:
            edges += 1
            if neighbour not in parents:
                parents[neighbour] = node
                if neighbour == target:
                    return SearchResult(_build_path(parents, target), expanded, edges, len(parents))
                frontier.append(neighbour)
    return SearchResult(None, expanded, edges, len(parents))


def bidirectional_bfs(graph, source, target):
    """Search forward from the source and backward from the target until they meet.

    Each turn expands one complete level of the smaller frontier. Because whole
    levels are expanded, the first meeting always gives a shortest path: if the
    searches have reached depths a and b without touching, the shortest path is
    longer than a + b, and any new meeting has a length of exactly a + b + 1.
    The backward search follows links in reverse, since links are one-way.
    """
    if source == target:
        return SearchResult([source], 0, 0, 1)
    parents_f, parents_b = {source: None}, {target: None}
    frontier_f, frontier_b = [source], [target]
    expanded = edges = 0

    while frontier_f and frontier_b:
        forward = len(frontier_f) <= len(frontier_b)
        frontier = frontier_f if forward else frontier_b
        adjacency = graph.out_edges if forward else graph.in_edges
        visited, other = (parents_f, parents_b) if forward else (parents_b, parents_f)

        next_frontier = []
        for node in frontier:
            expanded += 1
            for neighbour in adjacency[node]:
                edges += 1
                if neighbour in visited:
                    continue
                visited[neighbour] = node
                if neighbour in other:
                    head = _build_path(parents_f, neighbour)
                    tail = _build_path(parents_b, neighbour)
                    tail.reverse()
                    stored = len(parents_f) + len(parents_b)
                    return SearchResult(head + tail[1:], expanded, edges, stored)
                next_frontier.append(neighbour)

        if forward:
            frontier_f = next_frontier
        else:
            frontier_b = next_frontier

    return SearchResult(None, expanded, edges, len(parents_f) + len(parents_b))


def astar(graph, source, target, heuristic):
    """A* with f(n) = g(n) + h(n). `heuristic` maps a node id to its estimate.

    Ties on f go to the deepest node. A node is reopened if a cheaper route to it
    appears, which an inconsistent heuristic can cause.
    """
    best_g = {source: 0}
    parents = {source: None}
    frontier = [(heuristic(source), 0, source)]    # (f, -g, node)
    expanded = edges = 0
    max_stored = 1

    while frontier:
        _, negative_g, node = heapq.heappop(frontier)
        g = -negative_g
        if g > best_g[node]:
            continue                               # outdated queue entry
        if node == target:
            return SearchResult(_build_path(parents, target), expanded, edges, max_stored)
        expanded += 1
        for neighbour in graph.out_edges[node]:
            edges += 1
            new_g = g + 1
            if new_g < best_g.get(neighbour, float("inf")):
                best_g[neighbour] = new_g
                parents[neighbour] = node
                heapq.heappush(frontier, (new_g + heuristic(neighbour), -new_g, neighbour))
        max_stored = max(max_stored, len(best_g) + len(frontier))
    return SearchResult(None, expanded, edges, max_stored)
