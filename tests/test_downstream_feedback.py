import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_rolling_mean_score():
    from amphi_rl_dpgraph.downstream_feedback import RollingUtilityMonitor
    m = RollingUtilityMonitor(window=10, baseline_events=1)
    for v in [0.8, 0.9, 0.7, 0.85, 0.75]:
        m.update(v)
    assert m.mean_score() > 0.0

def test_rolling_window_evicts_old():
    from amphi_rl_dpgraph.downstream_feedback import RollingUtilityMonitor
    m = RollingUtilityMonitor(window=3, baseline_events=1)
    for v in [1.0, 1.0, 1.0, 0.0, 0.0, 0.0]:
        m.update(v)
    assert m.mean_score() == 0.0

def test_rolling_empty():
    from amphi_rl_dpgraph.downstream_feedback import RollingUtilityMonitor
    assert RollingUtilityMonitor(window=5).mean_score() == 0.0

def test_utility_delta_returns_float():
    from amphi_rl_dpgraph.downstream_feedback import RollingUtilityMonitor
    m = RollingUtilityMonitor(window=5, baseline_events=1)
    for v in [0.8, 0.9, 0.7]:
        m.update(v)
    assert isinstance(m.utility_delta(), float)
