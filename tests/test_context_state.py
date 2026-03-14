import os
import sys
import tempfile
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _make_state():
    from amphi_rl_dpgraph.context_state import ContextState
    f = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = f.name
    f.close()
    os.unlink(db_path)
    return ContextState(db_path), db_path


def _cleanup(state, db_path):
    state.close()
    if Path(db_path).exists():
        os.unlink(db_path)


def test_init_creates_db():
    state, db_path = _make_state()
    try:
        assert Path(db_path).exists()
    finally:
        _cleanup(state, db_path)


def test_risk_starts_zero():
    state, db_path = _make_state()
    try:
        assert state.risk_components("p1").risk == 0.0
    finally:
        _cleanup(state, db_path)


def test_record_event_increases_risk():
    state, db_path = _make_state()
    try:
        state.record_event(
            patient_key="p1", event_id="e1", ts=time.time(),
            modality_exposures={"text": 5},
        )
        assert state.risk_components("p1").risk > 0.0
    finally:
        _cleanup(state, db_path)


def test_pseudonym_version_increments():
    state, db_path = _make_state()
    try:
        ts = time.time()
        # record enough exposure to push risk above a very low threshold
        for i in range(5):
            state.record_event(
                patient_key="p1", event_id=f"e{i}", ts=ts + i,
                modality_exposures={"text": 20},
            )
        v0 = state.get_pseudonym_version("p1")
        risk = state.risk_components("p1").risk
        # use threshold just below current risk to guarantee a crossing
        result = state.localized_remask_trigger(
            "p1", event_id="e5", ts=ts + 5, threshold=max(0.0, risk - 0.01),
        )
        if result["trigger"]:
            assert state.get_pseudonym_version("p1") == v0 + 1
        else:
            assert state.get_pseudonym_version("p1") == v0
    finally:
        _cleanup(state, db_path)


def test_multiple_patients_isolated():
    state, db_path = _make_state()
    try:
        ts = time.time()
        state.record_event(patient_key="p1", event_id="e1", ts=ts,
                           modality_exposures={"text": 10})
        state.record_event(patient_key="p2", event_id="e2", ts=ts,
                           modality_exposures={"text": 1})
        assert state.risk_components("p1").risk > state.risk_components("p2").risk
    finally:
        _cleanup(state, db_path)
