import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_adaptive_policy_thresholds():
    from amphi_rl_dpgraph.baseline_experiment import adaptive_policy
    assert adaptive_policy(0.10) == "weak"
    assert adaptive_policy(0.50) == "synthetic"
    assert adaptive_policy(0.70) == "pseudo"
    assert adaptive_policy(0.90) == "redact"

def test_adaptive_policy_boundaries():
    from amphi_rl_dpgraph.baseline_experiment import adaptive_policy
    assert adaptive_policy(0.39) == "weak"
    assert adaptive_policy(0.40) == "synthetic"
    assert adaptive_policy(0.79) == "pseudo"
    assert adaptive_policy(0.80) == "redact"

def test_score_event_returns_tuple():
    from amphi_rl_dpgraph.baseline_experiment import score_event
    privacy, utility = score_event("redact", 0.8)
    assert 0.0 <= privacy <= 1.0 and 0.0 <= utility <= 1.0

def test_score_event_redact_high_privacy():
    from amphi_rl_dpgraph.baseline_experiment import score_event
    assert score_event("redact", 0.8)[0] >= 0.9

def test_score_event_raw_high_utility():
    from amphi_rl_dpgraph.baseline_experiment import score_event
    assert score_event("raw", 0.1)[1] >= 0.9

def test_compare_policies_has_adaptive():
    from amphi_rl_dpgraph.baseline_experiment import compare_policies
    result = compare_policies([0.1, 0.3, 0.5, 0.7, 0.9])
    assert "Adaptive" in result

def test_compare_policies_has_static_baselines():
    from amphi_rl_dpgraph.baseline_experiment import compare_policies
    result = compare_policies([0.1, 0.3, 0.5, 0.7, 0.9])
    assert any("Always" in k for k in result)

def test_monotonic_risks_nondecreasing():
    from amphi_rl_dpgraph.baseline_experiment import _monotonic_risks
    risks = _monotonic_risks([0.1] * 10)
    assert risks[-1] >= risks[0]

def test_bursty_risks_length():
    from amphi_rl_dpgraph.baseline_experiment import _bursty_risks
    base = [0.3] * 20
    assert len(_bursty_risks(base, cycle=6)) == len(base)

def test_mixed_risks_length():
    from amphi_rl_dpgraph.baseline_experiment import _mixed_risks
    base = [0.4] * 15
    assert len(_mixed_risks(base)) == len(base)

def test_compare_policies_score_ranges():
    from amphi_rl_dpgraph.baseline_experiment import compare_policies
    for policy, stats in compare_policies([0.5] * 10).items():
        assert 0.0 <= stats["privacy_mean"] <= 1.0
        assert 0.0 <= stats["utility_mean"] <= 1.0
