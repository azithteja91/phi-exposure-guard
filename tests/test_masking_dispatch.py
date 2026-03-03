import pytest

from amphi_rl_dpgraph.masking_ops import apply_masking


def test_text_and_asr_raw_passthrough():
    text = "Patient A visited."
    asr = "patient a visited"

    assert apply_masking(modality="text", policy="raw", payload=text) == text
    assert apply_masking(modality="asr", policy="raw", payload=asr) == asr


def test_text_pseudo_uses_token():
    out = apply_masking(
        modality="text",
        policy="pseudo",
        payload="Patient: Alex MRN 1234567",
        patient_token="PATIENT_42_V1",
    )
    assert isinstance(out, str)
    assert "PATIENT_42_V1" in out


def test_unknown_modality_or_policy_raises():
    with pytest.raises(ValueError):
        apply_masking(modality="text", policy="not_a_policy", payload="x")

    with pytest.raises(ValueError):
        apply_masking(modality="not_a_modality", policy="raw", payload="x")
