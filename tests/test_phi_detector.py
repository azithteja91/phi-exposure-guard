from amphi_rl_dpgraph.phi_detector import count_phi


def test_count_phi_positive_case():
    s = "patient alex mrn 1234567 visited on 01/01/1970"
    assert count_phi(s) >= 1


def test_count_phi_none_and_safe_text():
    assert count_phi(None) == 0
    assert count_phi("routine follow up note with no identifiers") == 0
