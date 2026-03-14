import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_registry_has_operators():
    from amphi_rl_dpgraph.cmo_registry import CMORegistry
    assert len(CMORegistry.list_operators()) > 0

def test_apply_via_cmo_redact():
    from amphi_rl_dpgraph.cmo_registry import apply_via_cmo
    result, log = apply_via_cmo(
        modality="text", policy="redact",
        payload="Patient John Smith, DOB 03/22/1955."
    )
    assert result is not None

def test_apply_via_cmo_pseudo():
    from amphi_rl_dpgraph.cmo_registry import apply_via_cmo
    result, log = apply_via_cmo(
        modality="text", policy="pseudo",
        payload="Patient John Smith."
    )
    assert result is not None

def test_apply_via_cmo_returns_log():
    from amphi_rl_dpgraph.cmo_registry import apply_via_cmo, MaskingActionLog
    _, log = apply_via_cmo(modality="text", policy="redact", payload="John Smith.")
    assert isinstance(log, MaskingActionLog)
