"""Run algorithms over source-target pairs and collect their measurements."""
import gc
import statistics
import time
import tracemalloc

import pandas as pd


def is_valid_path(graph, path, source, target):
    """Check that a returned path starts, ends and steps along real links."""
    if not path or path[0] != source or path[-1] != target:
        return False
    return all(b in graph.out_edges[a] for a, b in zip(path, path[1:]))


def evaluate(graph, name, search, pairs, distances, repeats=3):
    """Run `search(source, target)` on every pair; return one row per pair.

    Runtime is the median of `repeats` runs, so one slow run doesn't skew a pair.
    `distances` is the matrix of true shortest distances (-1 for unreachable).
    """
    rows = []
    gc.disable()
    try:
        for source, target in pairs:
            times = []
            for _ in range(repeats):
                start = time.perf_counter()
                result = search(source, target)
                times.append(time.perf_counter() - start)
            true_distance = int(distances[source, target])
            found = result.path is not None
            rows.append({
                "algorithm": name,
                "source": source,
                "target": target,
                "true_distance": true_distance,
                "found": found,
                "cost": result.cost,
                "optimal": found and result.cost == true_distance,
                "valid": found and is_valid_path(graph, result.path, source, target),
                "expanded": result.expanded,
                "edges_scanned": result.edges_scanned,
                "max_stored": result.max_stored,
                "time_ms": statistics.median(times) * 1000,
            })
    finally:
        gc.enable()
    return pd.DataFrame(rows)


def peak_memory_kib(search, pairs):
    """Mean peak Python memory per query. Runs separately because tracing slows code."""
    peaks = []
    tracemalloc.start()
    try:
        for source, target in pairs:
            tracemalloc.reset_peak()
            before = tracemalloc.get_traced_memory()[0]
            result = search(source, target)
            peaks.append(tracemalloc.get_traced_memory()[1] - before)
            del result
    finally:
        tracemalloc.stop()
    return sum(peaks) / len(peaks) / 1024


def summarise(results):
    """One row per algorithm, in the order the algorithms first appear."""
    results = results.assign(extra_clicks=results["cost"] - results["true_distance"])
    summary = results.groupby("algorithm", sort=False).agg(
        accuracy_pct=("optimal", lambda s: 100 * s.mean()),
        found_pct=("found", lambda s: 100 * s.mean()),
        valid_pct=("valid", lambda s: 100 * s.mean()),
        mean_path_length=("cost", "mean"),
        mean_extra_clicks=("extra_clicks", "mean"),
        max_extra_clicks=("extra_clicks", "max"),
        mean_expanded=("expanded", "mean"),
        mean_edges_scanned=("edges_scanned", "mean"),
        mean_max_stored=("max_stored", "mean"),
        mean_time_ms=("time_ms", "mean"),
        median_time_ms=("time_ms", "median"),
        p95_time_ms=("time_ms", lambda s: s.quantile(0.95)),
        total_time_s=("time_ms", lambda s: s.sum() / 1000),
    )
    return summary
