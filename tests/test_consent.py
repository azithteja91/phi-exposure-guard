import pytest
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_token_not_expired_by_default():
    from amphi_rl_dpgraph.consent import ConsentToken, is_expired
    assert not is_expired(ConsentToken(patient_key="p1", max_policy="pseudo"))

def test_token_expired():
    from amphi_rl_dpgraph.consent import ConsentToken, is_expired
    token = ConsentToken(patient_key="p1", max_policy="pseudo", expires_at=1.0)
    assert is_expired(token, now=time.time())

def test_resolve_policy_ok():
    from amphi_rl_dpgraph.consent import ConsentToken, resolve_policy
    token = ConsentToken(patient_key="p1", max_policy="redact")
    policy, status, reason = resolve_policy("pseudo", token, "text")
    assert status == "ok"
    assert policy == "pseudo"
    assert reason is None

def test_resolve_policy_capped():
    from amphi_rl_dpgraph.consent import ConsentToken, resolve_policy
    token = ConsentToken(patient_key="p1", max_policy="weak")
    policy, status, _ = resolve_policy("redact", token, "text")
    assert policy == "weak"
    assert status != "ok"

def test_resolve_policy_modality_denied():
    from amphi_rl_dpgraph.consent import ConsentToken, resolve_policy
    token = ConsentToken(patient_key="p1", max_policy="redact",
                         disallowed_modalities=frozenset(["asr"]))
    _, status, _ = resolve_policy("pseudo", token, "asr")
    assert status == "modality_denied"

def test_resolve_policy_expired():
    from amphi_rl_dpgraph.consent import ConsentToken, resolve_policy
    token = ConsentToken(patient_key="p1", max_policy="redact", expires_at=1.0)
    _, status, _ = resolve_policy("pseudo", token, "text")
    assert status == "expired"
