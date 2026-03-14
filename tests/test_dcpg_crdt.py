import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_node_phi_units():
    from amphi_rl_dpgraph.dcpg_crdt import CRDTNodeState
    n = CRDTNodeState(patient_key="p1", modality="text")
    n.increment_phi("dev_A", 3)
    assert n.total_phi_units == 3

def test_node_link_counts():
    from amphi_rl_dpgraph.dcpg_crdt import CRDTNodeState
    n = CRDTNodeState(patient_key="p1", modality="text")
    n.increment_link("dev_A", 2)
    assert n.total_link_signals == 2

def test_merge_node_gcounter():
    from amphi_rl_dpgraph.dcpg_crdt import CRDTNodeState, merge_node
    a = CRDTNodeState("p1", "text"); b = CRDTNodeState("p1", "text")
    a.phi_unit_counts["dev_A"] = 3
    b.phi_unit_counts["dev_B"] = 2
    assert merge_node(a, b).total_phi_units == 5

def test_merge_node_lww_pseudonym():
    from amphi_rl_dpgraph.dcpg_crdt import CRDTNodeState, merge_node
    a = CRDTNodeState("p1", "text"); b = CRDTNodeState("p1", "text")
    a.set_pseudonym_version(1, ts=100.0)
    b.set_pseudonym_version(2, ts=200.0)
    assert merge_node(a, b).pseudonym_version == 2

def test_crdt_graph_risk():
    from amphi_rl_dpgraph.dcpg_crdt import CRDTGraph
    g = CRDTGraph(device_id="dev_A")
    g.record_exposure("p1", "text", phi_units=5)
    assert 0.0 < g.risk_for("p1") < 1.0

def test_crdt_convergence():
    from amphi_rl_dpgraph.dcpg_crdt import CRDTGraph
    a = CRDTGraph("dev_A"); b = CRDTGraph("dev_B")
    a.record_exposure("p1", "text", phi_units=3)
    b.record_exposure("p1", "asr", phi_units=2)
    a.merge_from(b); b.merge_from(a)
    assert a.risk_for("p1") == b.risk_for("p1")

def test_crdt_risk_empty_patient():
    from amphi_rl_dpgraph.dcpg_crdt import CRDTGraph
    assert CRDTGraph("dev_A").risk_for("nobody") == 0.0

def test_demo_federated_merge():
    from amphi_rl_dpgraph.dcpg_crdt import demo_federated_merge
    result = demo_federated_merge()
    assert result["convergence_guaranteed"] is True
    assert 0.0 < result["merged_risk_patient_1"] < 1.0
