from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .phi_detector import avg_leaks_per_note


def _percentile(sorted_vals: Sequence[float], q: float) -> float:
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return float(sorted_vals[0])

    pos = (n - 1) * float(q)
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def summarize_latency(times_ms: List[float]) -> Dict[str, float]:
    if not times_ms:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p90_ms": 0.0}

    s = sorted(float(x) for x in times_ms)
    n = len(s)
    mean = sum(s) / n
    p50 = _percentile(s, 0.50)
    p90 = _percentile(s, 0.90)

    return {"mean_ms": float(mean), "p50_ms": float(p50), "p90_ms": float(p90)}


def policy_table_simple(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return rows


def avg_leaks(texts) -> float:
    return float(avg_leaks_per_note(texts))
