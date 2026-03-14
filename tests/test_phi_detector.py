import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_finds_name():
    from amphi_rl_dpgraph.phi_detector import find_phi_spans
    spans = find_phi_spans("Patient John Smith presented today.")
    assert len(spans) > 0

def test_finds_dob():
    from amphi_rl_dpgraph.phi_detector import find_phi_spans
    spans = find_phi_spans("DOB 03/22/1955 admitted.")
    assert len(spans) > 0

def test_spans_are_tuples():
    from amphi_rl_dpgraph.phi_detector import find_phi_spans
    spans = find_phi_spans("Patient John Smith, DOB 03/22/1955.")
    assert all(isinstance(s, tuple) and len(s) == 2 for s in spans)

def test_count_phi_empty():
    from amphi_rl_dpgraph.phi_detector import count_phi
    assert count_phi("") == 0

def test_count_phi_with_phi():
    from amphi_rl_dpgraph.phi_detector import count_phi
    assert count_phi("Patient John Smith, DOB 03/22/1955, MRN 4829104.") > 0

def test_count_phi_higher_with_more_phi():
    from amphi_rl_dpgraph.phi_detector import count_phi
    few = count_phi("John Smith, DOB 03/22/1955.")
    many = count_phi("John Smith, DOB 03/22/1955, MRN 4829104, 555-0142, 123 Elm St.")
    assert many >= few

def test_synthetic_name_not_flagged():
    from amphi_rl_dpgraph.phi_detector import count_phi
    assert count_phi("Alex Brooks was seen today.") == 0

def test_leakage_zero_on_clean():
    from amphi_rl_dpgraph.phi_detector import leakage
    assert leakage("No PHI here at all.") == 0
