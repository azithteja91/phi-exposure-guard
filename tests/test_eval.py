import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_summarize_latency_basic():
    from amphi_rl_dpgraph.eval import summarize_latency
    result = summarize_latency([10.0, 20.0, 30.0, 40.0, 50.0])
    assert result["mean_ms"] == 30.0
    assert result["p90_ms"] >= 40.0

def test_summarize_latency_empty():
    from amphi_rl_dpgraph.eval import summarize_latency
    assert summarize_latency([])["mean_ms"] == 0.0

def test_summarize_latency_single():
    from amphi_rl_dpgraph.eval import summarize_latency
    result = summarize_latency([15.0])
    assert result["mean_ms"] == 15.0
    assert result["p50_ms"] == 15.0

def test_policy_table_simple_returns_list():
    from amphi_rl_dpgraph.eval import policy_table_simple
    records = [
        {"chosen_policy": "redact", "leaks_after": 0, "latency_ms": 10.0},
        {"chosen_policy": "pseudo", "leaks_after": 1, "latency_ms": 5.0},
    ]
    result = policy_table_simple(records)
    assert isinstance(result, list)
    assert len(result) == 2
