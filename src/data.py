"""Load and clean the Wikispeedia dataset."""
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

import numpy as np
import pandas as pd


@dataclass
class Graph:
    """Directed, unweighted graph stored as adjacency lists over integer ids."""

    names: list            # id -> decoded article name
    index: dict            # decoded article name -> id
    out_edges: list        # id -> sorted list of linked article ids
    in_edges: list         # id -> sorted list of articles linking here
    categories: list       # id -> list of category paths, each a tuple of levels

    @property
    def n_nodes(self):
        return len(self.names)

    @property
    def n_edges(self):
        return sum(len(targets) for targets in self.out_edges)


def _data_lines(path):
    """Yield non-empty lines, skipping the dataset's '#' comment header."""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if line and not line.startswith("#"):
                yield line


def load_graph(data_dir):
    """Build the graph, decoding URL-encoded names and removing duplicate links."""
    data_dir = Path(data_dir)
    names = [unquote(line) for line in _data_lines(data_dir / "articles.tsv")]
    index = {name: i for i, name in enumerate(names)}

    out_sets = [set() for _ in names]
    for line in _data_lines(data_dir / "links.tsv"):
        source, target = (unquote(part) for part in line.split("\t"))
        if source != target:                      # a self-link can never shorten a path
            out_sets[index[source]].add(index[target])

    in_sets = [set() for _ in names]
    for source, targets in enumerate(out_sets):
        for target in targets:
            in_sets[target].add(source)

    categories = [[] for _ in names]
    for line in _data_lines(data_dir / "categories.tsv"):
        article, category = line.split("\t")
        levels = tuple(category.split(".")[1:])   # drop the shared root, 'subject'
        categories[index[unquote(article)]].append(levels)

    return Graph(names, index,
                 [sorted(s) for s in out_sets],
                 [sorted(s) for s in in_sets],
                 categories)


def load_distance_matrix(data_dir):
    """Return the dataset's own all-pairs distances; -1 means unreachable."""
    rows = list(_data_lines(Path(data_dir) / "shortest-path-distance-matrix.txt"))
    raw = np.array([np.frombuffer(row.encode("ascii"), dtype=np.uint8) for row in rows])
    matrix = raw.astype(np.int16) - ord("0")
    matrix[raw == ord("_")] = -1
    return matrix


def resolve_back_clicks(tokens):
    """Turn a click trail containing '<' (back) into the effective forward path."""
    stack = []
    for token in tokens:
        if token == "<":
            if len(stack) > 1:
                stack.pop()
        else:
            stack.append(token)
    return stack


def load_finished_paths(data_dir, graph):
    """Return one row per finished human game, with back-clicks resolved."""
    rows = []
    skipped = 0
    for line in _data_lines(Path(data_dir) / "paths_finished.tsv"):
        _, _, seconds, path, rating = line.split("\t")
        tokens = [unquote(token) for token in path.split(";")]
        if any(t != "<" and t not in graph.index for t in tokens):
            skipped += 1                          # article missing from the graph
            continue
        effective = resolve_back_clicks(tokens)
        rows.append({
            "source": graph.index[effective[0]],
            "target": graph.index[effective[-1]],
            "clicks": len(tokens) - 1,            # includes back-clicks
            "back_clicks": tokens.count("<"),
            "effective_length": len(effective) - 1,
            "seconds": int(seconds),
            "rating": None if rating == "NULL" else int(rating),
        })
    return pd.DataFrame(rows), skipped


def sample_pairs(paths, matrix, n_test, n_validation, n_unreachable, seed):
    """Sample separate validation, test and unreachable source-target pairs."""
    rng = np.random.default_rng(seed)
    pairs = paths[["source", "target"]].drop_duplicates()
    pairs = pairs[pairs["source"] != pairs["target"]]
    pairs = pairs[matrix[pairs["source"].to_numpy(), pairs["target"].to_numpy()] > 0]
    order = rng.permutation(len(pairs))
    shuffled = pairs.iloc[order][["source", "target"]].to_numpy().tolist()
    validation = [tuple(p) for p in shuffled[:n_validation]]
    test = [tuple(p) for p in shuffled[n_validation:n_validation + n_test]]

    sources, targets = np.where(matrix == -1)
    picks = rng.choice(len(sources), size=n_unreachable, replace=False)
    unreachable = [(int(sources[i]), int(targets[i])) for i in picks]
    return validation, test, unreachable
