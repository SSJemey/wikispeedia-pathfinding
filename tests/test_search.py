"""Check every algorithm against networkx on random graphs, and on the real data if present."""
import random
from pathlib import Path

import networkx as nx
import pytest

from src.data import Graph, load_distance_matrix, load_graph, resolve_back_clicks
from src.evaluate import is_valid_path
from src.heuristics import (CategoryHeuristic, LandmarkHeuristic, bfs_distances,
                            select_landmarks, zero_heuristic)
from src.search import astar, bfs, bidirectional_bfs

DATA_DIR = Path(__file__).parent.parent / "data" / "wikispeedia_paths-and-graph"


def random_graph(n, p, seed):
    rng = random.Random(seed)
    out_edges = [[] for _ in range(n)]
    in_edges = [[] for _ in range(n)]
    for u in range(n):
        for v in range(n):
            if u != v and rng.random() < p:
                out_edges[u].append(v)
                in_edges[v].append(u)
    names = [str(i) for i in range(n)]
    categories = [[("a", str(i % 3))] for i in range(n)]
    return Graph(names, {x: i for i, x in enumerate(names)}, out_edges, in_edges, categories)


@pytest.mark.parametrize("seed", range(30))
@pytest.mark.parametrize("p", [0.02, 0.05, 0.15])
def test_optimal_algorithms_match_networkx(seed, p):
    graph = random_graph(40, p, seed)
    reference = nx.DiGraph((u, v) for u, ts in enumerate(graph.out_edges) for v in ts)
    reference.add_nodes_from(range(40))
    truth = dict(nx.all_pairs_shortest_path_length(reference))
    landmark_h = LandmarkHeuristic(graph, [0, 1, 2])
    category_h = CategoryHeuristic(graph, weight=1)
    for source in range(0, 40, 3):
        for target in range(40):
            expected = truth[source].get(target)
            results = [bfs(graph, source, target),
                       bidirectional_bfs(graph, source, target),
                       astar(graph, source, target, zero_heuristic(target)),
                       astar(graph, source, target, landmark_h(target)),
                       astar(graph, source, target, category_h(target))]
            for result in results:
                assert result.cost == expected
                if expected is not None:
                    assert is_valid_path(graph, result.path, source, target)


def test_one_way_links_are_respected():
    graph = random_graph(3, 0, 0)
    graph.out_edges[0].append(1)
    graph.in_edges[1].append(0)
    assert bidirectional_bfs(graph, 0, 1).cost == 1
    assert bidirectional_bfs(graph, 1, 0).path is None


def test_resolve_back_clicks():
    assert resolve_back_clicks(["A", "B", "<", "C", "D", "<", "<", "E"]) == ["A", "E"]


@pytest.mark.skipif(not DATA_DIR.exists(), reason="dataset not downloaded")
def test_real_data_bfs_matches_dataset_matrix_and_landmarks_never_overestimate():
    graph = load_graph(DATA_DIR)
    matrix = load_distance_matrix(DATA_DIR)
    heuristic = LandmarkHeuristic(graph, select_landmarks(graph, 8, "farthest"))
    rng = random.Random(0)
    for source in rng.sample(range(graph.n_nodes), 50):
        distances = bfs_distances(graph.out_edges, source)
        assert [(-1 if d is None else d) for d in distances] == matrix[source].tolist()
    for target in rng.sample(range(graph.n_nodes), 50):
        h = heuristic(target)
        for node in range(graph.n_nodes):
            if matrix[node, target] >= 0:
                assert h(node) <= matrix[node, target]
