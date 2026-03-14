import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


def test_sha256_consistent():
    from amphi_rl_dpgraph.audit_signing import _sha256
    assert _sha256("hello") == _sha256("hello")
    assert _sha256("hello") != _sha256("world")
    assert len(_sha256("hello")) == 64

def test_merkle_root_single():
    from amphi_rl_dpgraph.audit_signing import build_merkle_root, _sha256
    h = _sha256("leaf")
    assert build_merkle_root([h]) == h

def test_merkle_root_deterministic():
    from amphi_rl_dpgraph.audit_signing import build_merkle_root, _sha256
    leaves = [_sha256(str(i)) for i in range(8)]
    assert build_merkle_root(leaves) == build_merkle_root(leaves)

def test_merkle_root_empty():
    from amphi_rl_dpgraph.audit_signing import build_merkle_root
    assert len(build_merkle_root([])) == 64

def test_audit_chain_append():
    from amphi_rl_dpgraph.audit_signing import AuditChain
    chain = AuditChain()
    entry = chain.append({"event_id": "e1", "policy": "redact"})
    assert entry.record["event_id"] == "e1"
    assert chain.entry_count == 1

def test_audit_chain_checkpoint():
    from amphi_rl_dpgraph.audit_signing import AuditChain
    chain = AuditChain(checkpoint_interval=3)
    for i in range(3):
        chain.append({"event_id": f"e{i}"})
    assert chain.checkpoint_count == 1

def test_sign_and_verify():
    from amphi_rl_dpgraph.audit_signing import generate_signing_key, sign_record, verify_record
    private_key, _ = generate_signing_key()
    if private_key is None:
        pytest.skip("cryptography library not available")
    record = {"event_id": "e1", "policy": "redact", "risk": 0.72}
    sig = sign_record(record, private_key)
    assert verify_record(record, sig, private_key.public_key())

def test_audit_chain_export():
    from amphi_rl_dpgraph.audit_signing import AuditChain
    chain = AuditChain()
    chain.append({"event_id": "e1"}); chain.append({"event_id": "e2"})
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
        path = f.name
    try:
        chain.export_jsonl(path)
        assert len(open(path).readlines()) == 2
    finally:
        os.unlink(path)
