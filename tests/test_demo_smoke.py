import importlib
from pathlib import Path


def test_run_demo_writes_core_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("AMPHI_RESULTS_DIR", str(tmp_path / "results"))

    import amphi_rl_dpgraph.run_demo as run_demo

    importlib.reload(run_demo)
    run_demo.main()

    out = Path(run_demo.RESULTS_DIR)
    assert (out / "policy_metrics.csv").exists()
    assert (out / "latency_summary.csv").exists()
    assert (out / "audit_log.jsonl").exists()
