import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_redact_removes_phi():
    from amphi_rl_dpgraph.masking import mask_text_redact
    result = mask_text_redact("Patient John Smith, DOB 03/22/1955.")
    assert "John Smith" not in result
    assert "[REDACTED]" in result

def test_weak_generalizes_dob():
    from amphi_rl_dpgraph.masking import mask_text_weak
    result = mask_text_weak("DOB 03/22/1955 presented today.")
    assert "03/22/1955" not in result

def test_pseudo_replaces_phi():
    from amphi_rl_dpgraph.masking import mask_text_pseudo
    result = mask_text_pseudo("Patient John Smith, MRN 4829104.", "PATIENT_001_V0")
    assert "John Smith" not in result

def test_raw_returns_unchanged():
    from amphi_rl_dpgraph.masking import PolicyOutputs
    text = "Patient John Smith."
    po = PolicyOutputs(raw=text, weak="", pseudo="", redact="", synthetic="", adaptive="")
    assert po.raw == text

def test_asr_redact_returns_string():
    from amphi_rl_dpgraph.masking import mask_asr_redact
    result = mask_asr_redact("john smith date of birth march twenty two nineteen fifty five")
    assert isinstance(result, str) and len(result) > 0

def test_image_leak_raw_with_phi():
    from amphi_rl_dpgraph.masking import image_leak_flag
    assert image_leak_flag(1, "raw") == 1

def test_image_leak_raw_no_phi():
    from amphi_rl_dpgraph.masking import image_leak_flag
    assert image_leak_flag(0, "raw") == 0

def test_image_leak_redact():
    from amphi_rl_dpgraph.masking import image_leak_flag
    assert image_leak_flag(1, "redact") == 0

def test_image_leak_pseudo():
    from amphi_rl_dpgraph.masking import image_leak_flag
    assert image_leak_flag(1, "pseudo") == 0

def test_image_leak_weak_passes_through():
    from amphi_rl_dpgraph.masking import image_leak_flag
    assert image_leak_flag(1, "weak") == 1

def test_waveform_leak_raw():
    from amphi_rl_dpgraph.masking import waveform_leak_flag
    assert waveform_leak_flag(1, "raw") == 1

def test_waveform_leak_weak_zero():
    from amphi_rl_dpgraph.masking import waveform_leak_flag
    assert waveform_leak_flag(1, "weak") == 0

def test_audio_leak_raw():
    from amphi_rl_dpgraph.masking import audio_leak_flag
    assert audio_leak_flag(1, "raw") == 1

def test_audio_leak_synthetic():
    from amphi_rl_dpgraph.masking import audio_leak_flag
    assert audio_leak_flag(1, "synthetic") == 0

def test_image_leak_unknown_policy_raises():
    from amphi_rl_dpgraph.masking import image_leak_flag
    with pytest.raises(ValueError):
        image_leak_flag(1, "unknown_policy")
