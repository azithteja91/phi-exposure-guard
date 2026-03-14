import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_phi_span_fields():
    from amphi_rl_dpgraph.schemas import PHISpan
    s = PHISpan(start=0, end=10, phi_type="NAME", confidence=0.9)
    assert s.start == 0 and s.phi_type == "NAME" and s.confidence == 0.9

def test_phi_span_default_confidence():
    from amphi_rl_dpgraph.schemas import PHISpan
    assert PHISpan(start=0, end=5, phi_type="MRN").confidence == 1.0

def test_data_event_defaults():
    from amphi_rl_dpgraph.schemas import DataEvent
    e = DataEvent(event_id="e1", patient_key="p1", timestamp=1000.0,
                  modality="text", payload="hello")
    assert e.phi_spans == [] and e.phi_units == 0

def test_decision_record_required_fields():
    from amphi_rl_dpgraph.schemas import DecisionRecord
    r = DecisionRecord(event_id="e1", patient_key="p1", policy_run="adaptive",
                       chosen_policy="redact", reason="high risk",
                       risk_pre=0.8, risk_post=0.08, risk_source="dcpg")
    assert r.event_id == "e1"
    assert r.chosen_policy == "redact"
