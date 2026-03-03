from amphi_rl_dpgraph.context_state import ContextState


def test_record_event_and_masking_credit(tmp_path):
    ctx = ContextState(db_path=str(tmp_path / "ctx.sqlite"))
    try:
        pk = "patient_1"
        ts = 1_700_000_000.0

        ctx.record_event(
            patient_key=pk,
            event_id="evt_1",
            ts=ts,
            modality_exposures={"text": 2, "asr": 1},
            link_signals={"image_link": 1, "audio_link": 0},
        )

        comps = ctx.risk_components(pk, now_ts=ts)
        assert comps.effective_units == 3
        assert "image_link" in ctx.link_modalities_recent(pk)

        credit = ctx.record_masking_credit(patient_key=pk, masked_units=2)
        assert credit["effective_units"] == 1

        credit2 = ctx.record_masking_credit(patient_key=pk, masked_units=99)
        assert credit2["effective_units"] == 0
    finally:
        ctx.close()
