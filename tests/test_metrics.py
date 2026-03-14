import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_leakage_score_no_phi():
    from amphi_rl_dpgraph.metrics import leakage_score
    assert leakage_score(["No PHI here.", "Clean text."]) == 0.0

def test_leakage_score_with_phi():
    from amphi_rl_dpgraph.metrics import leakage_score
    assert leakage_score(["Patient John Smith, DOB 03/22/1955."]) > 0.0

def test_leakage_score_empty():
    from amphi_rl_dpgraph.metrics import leakage_score
    assert leakage_score([]) == 0.0

def test_utility_proxy_clean_higher_than_redacted():
    from amphi_rl_dpgraph.metrics import utility_proxy_redaction_inverse
    assert (utility_proxy_redaction_inverse(["No redactions."]) >
            utility_proxy_redaction_inverse(["[REDACTED] [REDACTED]"]))

def test_utility_proxy_empty():
    from amphi_rl_dpgraph.metrics import utility_proxy_redaction_inverse
    assert utility_proxy_redaction_inverse([]) == 1.0

def test_compute_delta_auroc_returns_tuple():
    from amphi_rl_dpgraph.metrics import compute_delta_auroc
    texts_orig   = ["Patient John Smith DOB 1955"] * 10 + ["Maria Garcia MRN 111"] * 10
    texts_masked = ["[REDACTED] [REDACTED]"] * 10 + ["[REDACTED] [REDACTED]"] * 10
    labels = [0] * 10 + [1] * 10
    result = compute_delta_auroc(texts_orig, texts_masked, labels)
    assert isinstance(result, tuple)
    assert len(result) >= 1

def test_compute_delta_auroc_fallback_on_empty():
    from amphi_rl_dpgraph.metrics import compute_delta_auroc
    result = compute_delta_auroc([], [], [])
    assert result[0] == 0.0
