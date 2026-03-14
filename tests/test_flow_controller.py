import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_contract(**kwargs):
    from amphi_rl_dpgraph.flow_controller import PolicyContract
    defaults = dict(modality="text", chosen_policy="redact",
                    patient_token="PATIENT_001_V0", risk_score=0.8)
    defaults.update(kwargs)
    return PolicyContract(**defaults)


def test_build_dag_returns_dag():
    from amphi_rl_dpgraph.flow_controller import build_dag, MaskingDAG
    assert isinstance(build_dag(_make_contract()), MaskingDAG)

def test_dag_has_nodes():
    from amphi_rl_dpgraph.flow_controller import build_dag
    dag = build_dag(_make_contract(chosen_policy="pseudo", modality="asr"))
    assert len(dag.nodes) > 0

def test_export_dag_returns_string():
    from amphi_rl_dpgraph.flow_controller import export_dag
    pc = _make_contract()
    result = export_dag(pc)
    assert isinstance(result, str) and len(result) > 0

def test_contract_hash_stable():
    from amphi_rl_dpgraph.flow_controller import _contract_hash
    pc = _make_contract()
    h1 = _contract_hash(pc)
    h2 = _contract_hash(pc)
    assert h1 == h2
    assert len(h1) > 0
