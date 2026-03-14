import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_synthetic_name_deterministic():
    from amphi_rl_dpgraph.cmo_media import synthetic_name
    assert synthetic_name("John Smith") == synthetic_name("John Smith")

def test_synthetic_name_different_inputs():
    from amphi_rl_dpgraph.cmo_media import synthetic_name
    assert synthetic_name("John Smith") != synthetic_name("Maria Garcia")

def test_synthetic_date_deterministic():
    from amphi_rl_dpgraph.cmo_media import synthetic_date
    assert synthetic_date("03/22/1955") == synthetic_date("03/22/1955")

def test_synthetic_date_replaces():
    from amphi_rl_dpgraph.cmo_media import synthetic_date
    result = synthetic_date("03/22/1955")
    assert result != "03/22/1955"

def test_synthetic_mrn_returns_string():
    from amphi_rl_dpgraph.cmo_media import synthetic_mrn
    result = synthetic_mrn("MRN482910")
    assert isinstance(result, str) and len(result) > 0

def test_replace_names_synthetic():
    from amphi_rl_dpgraph.cmo_media import replace_names_synthetic
    assert isinstance(replace_names_synthetic("Patient John Smith presented today."), str)

def test_replace_dates_synthetic():
    from amphi_rl_dpgraph.cmo_media import replace_dates_synthetic
    result = replace_dates_synthetic("DOB 03/22/1955 on record.")
    assert "03/22/1955" not in result

def test_apply_synthetic_replacement():
    from amphi_rl_dpgraph.cmo_media import apply_synthetic_replacement
    result = apply_synthetic_replacement("Patient John Smith, DOB 03/22/1955, MRN482910.")
    assert isinstance(result, str) and len(result) > 0

def test_image_phi_flag_raw():
    from amphi_rl_dpgraph.cmo_media import image_phi_flag
    assert image_phi_flag({"has_phi": True}, "raw") == 1

def test_image_phi_flag_redact():
    from amphi_rl_dpgraph.cmo_media import image_phi_flag
    assert image_phi_flag({"has_phi": True}, "redact") == 0

def test_waveform_phi_flag_weak():
    from amphi_rl_dpgraph.cmo_media import waveform_phi_flag
    assert waveform_phi_flag({"has_phi": True}, "weak") == 0

def test_audio_phi_flag_pseudo():
    from amphi_rl_dpgraph.cmo_media import audio_phi_flag
    assert audio_phi_flag({"has_phi": True}, "pseudo") == 0

def test_mask_waveform_header():
    from amphi_rl_dpgraph.cmo_media import mask_waveform_header
    result = mask_waveform_header({"patient_name": "John Smith", "dob": "1955-03-22"})
    assert isinstance(result, dict)
