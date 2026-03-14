import os
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_controller(**kwargs):
    from amphi_rl_dpgraph.context_state import ContextState
    from amphi_rl_dpgraph.controller import ExposurePolicyController
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = f.name
    f.close()
    os.unlink(db_path)
    ctx = ContextState(db_path)
    ctrl = ExposurePolicyController(context=ctx, **kwargs)
    return ctrl, ctx, db_path


def _cleanup(ctx, db_path):
    ctx.close()
    if Path(db_path).exists():
        os.unlink(db_path)


def test_decide_from_risk_low():
    ctrl, ctx, db_path = _make_controller()
    try:
        d = ctrl.decide_from_risk(risk_pre=0.10)
        assert d.policy_name == "weak"
    finally:
        _cleanup(ctx, db_path)


def test_decide_from_risk_medium_low():
    ctrl, ctx, db_path = _make_controller()
    try:
        d = ctrl.decide_from_risk(risk_pre=0.50)
        assert d.policy_name == "synthetic"
    finally:
        _cleanup(ctx, db_path)


def test_decide_from_risk_medium_high():
    ctrl, ctx, db_path = _make_controller()
    try:
        d = ctrl.decide_from_risk(risk_pre=0.70)
        assert d.policy_name == "pseudo"
    finally:
        _cleanup(ctx, db_path)


def test_decide_from_risk_high():
    ctrl, ctx, db_path = _make_controller()
    try:
        d = ctrl.decide_from_risk(risk_pre=0.90)
        assert d.policy_name == "redact"
    finally:
        _cleanup(ctx, db_path)


def test_decide_returns_decision():
    from amphi_rl_dpgraph.controller import Decision
    ctrl, ctx, db_path = _make_controller()
    try:
        d = ctrl.decide_from_risk(risk_pre=0.50)
        assert isinstance(d, Decision)
        assert d.risk_pre == 0.50
    finally:
        _cleanup(ctx, db_path)


def test_custom_thresholds():
    ctrl, ctx, db_path = _make_controller(risk_1=0.30, risk_2=0.70)
    try:
        assert ctrl.decide_from_risk(risk_pre=0.25).policy_name == "weak"
        assert ctrl.decide_from_risk(risk_pre=0.85).policy_name == "redact"
    finally:
        _cleanup(ctx, db_path)


def test_decision_has_reason():
    ctrl, ctx, db_path = _make_controller()
    try:
        d = ctrl.decide_from_risk(risk_pre=0.50)
        assert isinstance(d.reason, str) and len(d.reason) > 0
    finally:
        _cleanup(ctx, db_path)
