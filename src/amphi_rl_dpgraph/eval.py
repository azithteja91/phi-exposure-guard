# Evaluation utilities for AMPHI: latency percentiles, per-policy latency
# aggregation across multiple runs with between-run variance and 95% CI, and
# a thin avg_leaks wrapper over phi_detector. aggregate_policy_latency is the
# intended source for the latency_by_policy block in statistical_robustness.json
# so that file becomes the single source of truth for all latency metrics.

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from .phi_detector import avg_leaks_per_note


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])
    pos  = (n - 1) * float(q)
    lo   = int(pos)
    hi   = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize_latency(times_ms: List[float]) -> Dict[str, float]:
    if not times_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}

    s    = sorted(float(x) for x in times_ms)
    n    = len(s)
    mean = sum(s) / n
    p50  = _percentile(s, 0.50)
    p90  = _percentile(s, 0.90)

    return {"mean_ms": float(mean), "p50_ms": float(p50), "p90_ms": float(p90)}


def aggregate_policy_latency(
    per_run_latencies: Dict[str, List[List[float]]],
) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for policy, runs in per_run_latencies.items():
        all_samples: List[float] = [ms for run in runs for ms in run]
        n = len(all_samples)
        if n == 0:
            result[policy] = {"mean_ms": 0.0, "std_ms": 0.0, "ci95_ms": 0.0,
                               "p50_ms": 0.0, "p90_ms": 0.0, "n": 0}
            continue

        run_means  = [sum(r) / len(r) for r in runs if r]
        grand_mean = sum(all_samples) / n
        n_runs     = len(run_means)

        if n_runs > 1:
            variance = sum((m - grand_mean) ** 2 for m in run_means) / (n_runs - 1)
            std_ms   = math.sqrt(variance)
            ci95_ms  = 1.96 * std_ms / math.sqrt(n_runs)
        else:
            std_ms  = 0.0
            ci95_ms = 0.0

        lat = summarize_latency(all_samples)
        result[policy] = {
            "mean_ms": round(grand_mean, 3),
            "std_ms":  round(std_ms, 3),
            "ci95_ms": round(ci95_ms, 3),
            "p50_ms":  round(lat["p50_ms"], 3),
            "p90_ms":  round(lat["p90_ms"], 3),
            "n":       n,
        }
    return result


def policy_table_simple(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return rows


def avg_leaks(texts) -> float:
    return float(avg_leaks_per_note(texts))
