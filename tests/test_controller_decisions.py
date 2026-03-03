from amphi_rl_dpgraph.context_state import ContextState
from amphi_rl_dpgraph.controller import ExposurePolicyController


def _controller(tmp_path):
    ctx = ContextState(db_path=str(tmp_path / "ctx.sqlite"))
    return ctx, ExposurePolicyController(context=ctx, risk_1=0.40, risk_2=0.80)


def test_decide_from_risk_low_medium_high(tmp_path):
    ctx, ctl = _controller(tmp_path)
    try:
        low = ctl.decide_from_risk(0.10)
        mid = ctl.decide_from_risk(0.50)
        high = ctl.decide_from_risk(0.90)

        assert low.policy_name == "weak"
        assert mid.policy_name == "pseudo"
        assert high.policy_name == "redact"
    finally:
        ctx.close()


def test_decide_from_risk_boundaries(tmp_path):
    ctx, ctl = _controller(tmp_path)
    try:
        at_r1 = ctl.decide_from_risk(0.40)
        at_r2 = ctl.decide_from_risk(0.80)

        assert at_r1.policy_name == "pseudo"
        assert at_r2.policy_name == "redact"
    finally:
        ctx.close()
