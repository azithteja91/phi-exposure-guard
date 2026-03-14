import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_node_delta_to_dict():
    from amphi_rl_dpgraph.dcpg_federation import NodeDelta
    d = NodeDelta(device_id="dev_A", seq_id=1, patient_key="p1", modality="text",
                  phi_units_added=3, link_signal_added=0,
                  pseudonym_version=0, pseudonym_version_ts=100.0)
    dd = d.to_dict()
    assert dd["device_id"] == "dev_A"
    assert dd["phi_units_added"] == 3

def test_node_delta_roundtrip():
    from amphi_rl_dpgraph.dcpg_federation import NodeDelta
    d = NodeDelta(device_id="dev_B", seq_id=2, patient_key="p1", modality="asr",
                  phi_units_added=2, link_signal_added=1,
                  pseudonym_version=1, pseudonym_version_ts=200.0)
    d2 = NodeDelta.from_dict(d.to_dict())
    assert d2.device_id == d.device_id
    assert d2.phi_units_added == d.phi_units_added

def test_deterministic_pseudonym_stable():
    from amphi_rl_dpgraph.dcpg_federation import deterministic_pseudonym
    key = b"secret"
    assert deterministic_pseudonym("p1", key, version=1) == deterministic_pseudonym("p1", key, version=1)

def test_deterministic_pseudonym_version_differs():
    from amphi_rl_dpgraph.dcpg_federation import deterministic_pseudonym
    key = b"secret"
    assert deterministic_pseudonym("p1", key, version=1) != deterministic_pseudonym("p1", key, version=2)

def test_gossip_bus_publish_and_drain():
    from amphi_rl_dpgraph.dcpg_federation import GossipBus, NodeDelta
    bus = GossipBus()
    bus.register("dev_A"); bus.register("dev_B")
    delta = NodeDelta(device_id="dev_A", seq_id=1, patient_key="p1", modality="text",
                      phi_units_added=3, link_signal_added=0,
                      pseudonym_version=0, pseudonym_version_ts=time.time())
    bus.publish("dev_A", [delta])
    received = bus.drain("dev_B")
    assert len(received) == 1 and received[0].patient_key == "p1"

def test_gossip_bus_no_self_receive():
    from amphi_rl_dpgraph.dcpg_federation import GossipBus, NodeDelta
    bus = GossipBus()
    bus.register("dev_A")
    delta = NodeDelta(device_id="dev_A", seq_id=1, patient_key="p1", modality="text",
                      phi_units_added=1, link_signal_added=0,
                      pseudonym_version=0, pseudonym_version_ts=time.time())
    bus.publish("dev_A", [delta])
    assert len(bus.drain("dev_A")) == 0

def test_demo_live_federation_runs():
    from amphi_rl_dpgraph.dcpg_federation import demo_live_federation
    result = demo_live_federation()
    assert isinstance(result, dict) and len(result) > 0
