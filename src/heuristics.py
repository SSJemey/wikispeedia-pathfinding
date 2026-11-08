"""Heuristics for A*. Each factory takes a target and returns h(node)."""
from collections import deque


def zero_heuristic(target):
    """h = 0: A* with no guidance, used as the baseline."""
    return lambda node: 0


def bfs_distances(adjacency, source):
    """Distance from `source` to every node; None marks unreachable nodes."""
    distances = [None] * len(adjacency)
    distances[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbour in adjacency[node]:
            if distances[neighbour] is None:
                distances[neighbour] = distances[node] + 1
                queue.append(neighbour)
    return distances


def largest_scc(graph):
    """Largest strongly connected component (every article reaches every other)."""
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(range(graph.n_nodes))
    g.add_edges_from((u, v) for u, targets in enumerate(graph.out_edges) for v in targets)
    return max(nx.strongly_connected_components(g), key=len)


def select_landmarks(graph, k, strategy):
    """Choose landmarks inside the largest SCC, so distances exist in both directions.

    'hub' takes the most-linked articles. 'farthest' starts at the top hub, then
    repeatedly adds the article farthest from the landmarks chosen so far.
    """
    scc = largest_scc(graph)
    degree = lambda n: (len(graph.out_edges[n]) + len(graph.in_edges[n]), -n)
    by_degree = sorted(scc, key=degree, reverse=True)
    if strategy == "hub":
        return by_degree[:k]
    if strategy != "farthest":
        raise ValueError(f"unknown strategy: {strategy}")
    landmarks = [by_degree[0]]
    nearest = {n: float("inf") for n in scc}
    while len(landmarks) < k:
        forward = bfs_distances(graph.out_edges, landmarks[-1])
        backward = bfs_distances(graph.in_edges, landmarks[-1])
        for n in scc:
            nearest[n] = min(nearest[n], forward[n] + backward[n])
        landmarks.append(max(scc, key=lambda n: (nearest[n], degree(n))))
    return landmarks


class LandmarkHeuristic:
    """Admissible lower bound from the triangle inequality (the ALT technique).

    For a landmark L, the distance left from n to target t is at least
    d(L, t) - d(L, n) and at least d(n, L) - d(t, L). h is the largest such bound,
    so it never overestimates and A* still finds a shortest path.
    """

    def __init__(self, graph, landmarks):
        self.landmarks = list(landmarks)
        self.from_landmark = [bfs_distances(graph.out_edges, L) for L in self.landmarks]
        self.to_landmark = [bfs_distances(graph.in_edges, L) for L in self.landmarks]

    def __call__(self, target):
        tables = []
        for from_l, to_l in zip(self.from_landmark, self.to_landmark):
            if from_l[target] is not None:
                tables.append((from_l, from_l[target], 1))
            if to_l[target] is not None:
                tables.append((to_l, to_l[target], -1))

        def h(node):
            best = 0
            for table, target_value, sign in tables:
                value = table[node]
                if value is not None:
                    bound = sign * (target_value - value)
                    if bound > best:
                        best = bound
            return best
        return h


class CategoryHeuristic:
    """Estimate distance from how far apart two articles sit in the category tree.

    h = weight * (1 - similarity), where similarity is the best shared-prefix score
    between the two articles' categories. It can overestimate, so A* may return a
    path longer than the shortest. That is the price of searching fewer nodes.
    """

    def __init__(self, graph, weight):
        self.graph = graph
        self.weight = weight

    @staticmethod
    def _similarity(cats_a, cats_b):
        best = 0.0
        for a in cats_a:
            for b in cats_b:
                shared = 0
                for level_a, level_b in zip(a, b):
                    if level_a != level_b:
                        break
                    shared += 1
                best = max(best, 2 * shared / (len(a) + len(b)))
        return best

    def __call__(self, target):
        graph, weight = self.graph, self.weight
        target_cats = graph.categories[target]
        cache = {}

        def h(node):
            if node == target:
                return 0.0
            key = tuple(graph.categories[node])
            if key not in cache:
                cache[key] = weight * (1 - self._similarity(key, target_cats))
            return cache[key]
        return h
