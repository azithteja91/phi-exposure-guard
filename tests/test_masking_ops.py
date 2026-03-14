import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_apply_masking_redact():
    from amphi_rl_dpgraph.masking_ops import apply_masking
    result = apply_masking(modality="text", policy="redact",
                           payload="Patient John Smith, DOB 03/22/1955.")
    assert "[REDACTED]" in str(result) or "REDACTED" in str(result)

def test_apply_masking_raw_unchanged():
    from amphi_rl_dpgraph.masking_ops import apply_masking
    text = "Patient John Smith."
    assert apply_masking(modality="text", policy="raw", payload=text) == text

def test_apply_masking_unknown_modality_raises():
    from amphi_rl_dpgraph.masking_ops import apply_masking
    with pytest.raises(ValueError):
        apply_masking(modality="unknown_mod", policy="redact", payload="Some text.")
