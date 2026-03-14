import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ppo_agent_init():
    from amphi_rl_dpgraph.rl_agent import PPOAgent
    assert PPOAgent() is not None

def test_correct_policy_returns_valid():
    from amphi_rl_dpgraph.rl_agent import _correct_policy
    valid = {"raw", "weak", "synthetic", "pseudo", "redact"}
    assert _correct_policy(0.10) in valid
    assert _correct_policy(0.95) in valid

def test_correct_policy_high_risk():
    from amphi_rl_dpgraph.rl_agent import _correct_policy
    assert _correct_policy(0.95) in {"pseudo", "redact"}

def test_compute_reward_range():
    from amphi_rl_dpgraph.rl_agent import compute_reward
    r = compute_reward(r_risk=0.7, delta_auroc=-0.3, latency_ms=15.0, energy_proxy=0.1)
    assert isinstance(r, float)

def test_compute_reward_lower_risk_higher():
    from amphi_rl_dpgraph.rl_agent import compute_reward
    r_low  = compute_reward(r_risk=0.2, delta_auroc=0.0, latency_ms=5.0, energy_proxy=0.0)
    r_high = compute_reward(r_risk=0.9, delta_auroc=0.0, latency_ms=5.0, energy_proxy=0.0)
    assert r_low > r_high

def test_mddmc_state_fields():
    from amphi_rl_dpgraph.rl_agent import MDDMCState
    s = MDDMCState(risk=0.5, units_factor=0.22, recency_factor=0.9, link_bonus=0.0)
    assert s.risk == 0.5

def test_mddmc_state_to_vector():
    from amphi_rl_dpgraph.rl_agent import MDDMCState
    s = MDDMCState(risk=0.5, units_factor=0.22, recency_factor=0.9, link_bonus=0.0)
    v = s.to_vector()
    assert isinstance(v, list) and len(v) > 0
