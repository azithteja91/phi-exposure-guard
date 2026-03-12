from __future__ import annotations

import json
import csv
import time
from pathlib import Path
from datetime import datetime
from statistics import mean
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from .context_state import ContextState
from .controller import ExposurePolicyController
from .dcpg import DCPGAdapter
from .dcpg_crdt import demo_federated_merge, CRDTGraph
from .audit_signing import AuditChain, generate_signing_key, take_dcpg_snapshot
from .flow_controller import PolicyContract, export_dag, _apply_consent_cap
from .metrics import compute_delta_auroc
from .phi_detector import count_phi
from .schemas import AuditRecord
from .rl_agent import PPOAgent, MDDMCState, compute_reward
from .masking_ops import apply_masking


RESULTS_ROOT = Path("results")


def make_results_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = RESULTS_ROOT / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def synthetic_stream():
    patients = ["A", "B"]
    names = {"A": "John Smith",  "B": "Jane Doe"}
    mrns  = {"A": "MRN-1234567", "B": "MRN-7654321"}
    dobs  = {"A": "01/15/1970",  "B": "03/22/1982"}
    t = time.time()
    for i in range(34):
        p = patients[i % 2]
        yield {
            "event_id":        f"evt_{i}",
            "patient_key":     p,
            "timestamp":       t + i,
            "text":            f"Patient {names[p]} {mrns[p]} DOB {dobs[p]} admitted to Example Hospital ward 4B.",
            "asr":             f"patient {names[p]} date of birth {dobs[p]} {mrns[p]}",
            "image_has_phi":    1 if i % 3 == 0 else 0,
            "waveform_has_phi": 1 if i % 4 == 0 else 0,
            "audio_has_phi":    1 if i % 5 == 0 else 0,
        }


def ppo_pretrain(agent: PPOAgent, episodes: int):
    print(f"\n[PPO] Pre-training for {episodes} episodes\n")
    rewards = []
    for _ in tqdm(range(episodes), desc="PPO Training", unit="episode"):
        state = MDDMCState(
            risk=np.random.rand(),
            units_factor=np.random.rand(),
            recency_factor=np.random.rand(),
            link_bonus=np.random.rand(),
        )
        action = agent.predict(state)
        reward = compute_reward(
            r_risk=state.risk, delta_auroc=0.0,
            latency_ms=1.0, energy_proxy=0.0,
            chosen_policy=str(action.policy),
        )
        agent.update(state, action, reward)
        rewards.append(reward)
    return rewards


def latency_summary(lat: List[float]) -> Dict[str, float]:
    s = sorted(lat)
    return {
        "mean_ms": float(mean(s)),
        "p50_ms":  float(s[int(len(s) * 0.5)]),
        "p90_ms":  float(s[int(len(s) * 0.9)]),
    }


def save_report(outdir):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(outdir / "AMPHI_experiment_report.pdf"), pagesize=letter)
    doc.build([
        Paragraph("AMPHI Multimodal Privacy Experiment", styles["Title"]),
        Spacer(1, 20),
        Paragraph("Artifacts generated automatically.", styles["BodyText"]),
    ])


def main():
    print("\n=== AMPHI Multimodal Privacy Demo ===\n")
    outdir = make_results_dir()
    print("Results directory:", outdir, "\n")

    print("[ST]  Pre-warming SentenceTransformer...")
    try:
        from sentence_transformers import SentenceTransformer as _ST
        from .dcpg import _text_embedding
        _warm_model = _ST("all-MiniLM-L6-v2")
        _warm_model.encode("warmup", show_progress_bar=False)
        _text_embedding._cache["st"] = _warm_model   # share with DCPGAdapter — no second load
        print("[ST]  Ready.\n")
    except Exception as _e:
        print(f"[ST]  Not available ({_e}) — n-gram fallback active.\n")

    try:
        import torch as _tc
        print(f"[RL]  PyTorch {_tc.__version__} — PPO active.\n")
        del _tc
    except ImportError:
        print("[RL]  WARNING: PyTorch not installed. Heuristic mode only.\n")

    rl_agent = PPOAgent(model_path=str(outdir / "ppo_model.pt"), risk_1=0.40, risk_2=0.80)
    rewards  = ppo_pretrain(rl_agent, 200)

    ctx = ContextState(db_path=str(outdir / "dcpg_adaptive.sqlite"), k_units=0.03)
    controller = ExposurePolicyController(
        context=ctx, risk_1=0.40, risk_2=0.80, remask_thresh=0.68,
    )
    graph_adapter = DCPGAdapter(ctx)
    crdt_live     = CRDTGraph(device_id="edge_device_main")

    _CONSENT      = {"A": "research", "B": "standard"}
    _PATIENT_INT  = {"A": 0, "B": 1}
    _POLICY_ORDER = ["raw", "weak", "synthetic", "pseudo", "redact"]

    private_key, _ = generate_signing_key()
    audit_chain = AuditChain(private_key=private_key, checkpoint_interval=20)

    latencies, risks, decisions, audit_rows, rl_event_rewards = [], [], [], [], []
    _buf_orig, _buf_masked, _buf_labels = [], [], []
    _last_delta_auroc = 0.0
    modal_counts = [[], [], [], []]

    for ev in tqdm(list(synthetic_stream()), desc="Event Stream", unit="event"):
        t0 = time.perf_counter()

        text_phi  = count_phi(ev["text"])
        asr_phi   = count_phi(ev["asr"])
        link_sigs = {"asr": 1} if (text_phi > 0 and asr_phi > 0) else {}

        decision = controller.record_and_decide(
            patient_key=ev["patient_key"],
            event_id=ev["event_id"],
            timestamp=ev["timestamp"],
            modality_exposures={
                "text":           text_phi,
                "asr":            asr_phi,
                "image_proxy":    ev["image_has_phi"],
                "waveform_proxy": ev["waveform_has_phi"],
                "audio_proxy":    ev["audio_has_phi"],
            },
            link_signals=link_sigs,
            event_payloads={"text": ev["text"], "asr": ev["asr"]},
        )
        risk = float(decision.risk_pre)

        crdt_live.record_exposure(ev["patient_key"], "text",
            phi_units=text_phi, link_signal=bool(link_sigs))
        crdt_live.record_exposure(ev["patient_key"], "asr",            phi_units=asr_phi)
        crdt_live.record_exposure(ev["patient_key"], "image_proxy",    phi_units=ev["image_has_phi"])
        crdt_live.record_exposure(ev["patient_key"], "waveform_proxy", phi_units=ev["waveform_has_phi"])
        crdt_live.record_exposure(ev["patient_key"], "audio_proxy",    phi_units=ev["audio_has_phi"])
        crdt_risk = round(crdt_live.risk_for(ev["patient_key"]), 4)

        consent_level    = _CONSENT[ev["patient_key"]]
        effective_policy = _apply_consent_cap(decision.policy_name, consent_level)

        patient_token = controller.current_token(
            ev["patient_key"], _PATIENT_INT[ev["patient_key"]]
        )
        masked_text = apply_masking(
            modality="text",
            policy=effective_policy,
            payload=ev["text"],
            patient_token=patient_token,
        )

        _buf_orig.append(ev["text"])
        _buf_masked.append(masked_text)
        _buf_labels.append(_PATIENT_INT[ev["patient_key"]])
        if len(_buf_orig) > 32:
            _buf_orig, _buf_masked, _buf_labels = (
                _buf_orig[-32:], _buf_masked[-32:], _buf_labels[-32:]
            )
        if len(_buf_orig) >= 8 and len(set(_buf_labels)) == 2:
            try:
                delta, _, _ = compute_delta_auroc(_buf_orig, _buf_masked, _buf_labels)
                _last_delta_auroc = float(delta)
            except Exception:
                pass

        comps = decision.risk_components
        rl_state = MDDMCState(
            risk=risk,
            units_factor=float(comps.get("units_factor", 0.0)),
            recency_factor=float(comps.get("recency_factor", 0.0)),
            link_bonus=float(comps.get("link_bonus", 0.0)),
            delta_auroc=_last_delta_auroc,
            latency_ms=float((time.perf_counter() - t0) * 1000),
            energy_proxy=0.0,
            phi_text=text_phi,
            phi_asr=asr_phi,
            phi_image=int(ev["image_has_phi"]),
            phi_waveform=int(ev["waveform_has_phi"]),
            phi_audio=int(ev["audio_has_phi"]),
        )

        rl_action = rl_agent.predict(rl_state, patient_key=ev["patient_key"])
        rl_reward = compute_reward(
            r_risk=risk, delta_auroc=_last_delta_auroc,
            latency_ms=rl_state.latency_ms, energy_proxy=0.0,
            chosen_policy=effective_policy,
        )
        rl_agent.update(rl_state, rl_action, rl_reward)
        rl_event_rewards.append(rl_reward)

        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)
        risks.append(risk)
        try:
            decisions.append(_POLICY_ORDER.index(effective_policy))
        except ValueError:
            decisions.append(2)

        modal_counts[0].append(text_phi)
        modal_counts[1].append(asr_phi)
        modal_counts[2].append(ev["image_has_phi"])
        modal_counts[3].append(ev["audio_has_phi"])

        rec = AuditRecord(
            event_id=ev["event_id"],
            patient_key=ev["patient_key"],
            modality="text",
            policy_run="adaptive",
            chosen_policy=effective_policy,
            reason=decision.reason,
            risk=risk,
            localized_remask_trigger=bool(decision.localized_remask.get("trigger", False)),
            latency_ms=round(lat, 3),
            leaks_after=int(count_phi(masked_text)),
            policy_version="v1",
            extra={
                "delta_auroc":      round(_last_delta_auroc, 5),
                "crdt_risk":        crdt_risk,
                "consent_level":    consent_level,
                "decided_policy":   decision.policy_name,
                "effective_policy": effective_policy,
                "consent_capped":   effective_policy != decision.policy_name,
            },
            decision_blob={
                "rl_policy": rl_action.policy,
                "rl_source": rl_action.source,
                "rl_reward": round(rl_reward, 5),
            },
        )
        audit_rows.append(asdict(rec))
        audit_chain.append(asdict(rec))

    audit_chain.checkpoint()
    audit_chain.export_jsonl(str(outdir / "audit_log_signed_adaptive.jsonl"))
    audit_chain.export_checkpoints_jsonl(str(outdir / "audit_checkpoints_adaptive.jsonl"))
    audit_chain.export_fhir_jsonl(str(outdir / "audit_fhir_adaptive.jsonl"))

    with open(outdir / "audit_log.jsonl", "w") as f:
        for r in audit_rows:
            f.write(json.dumps(r) + "\n")

    snap = take_dcpg_snapshot(ctx, ["A", "B"], "v1", "end_of_run")
    (outdir / "dcpg_snapshot.json").write_text(json.dumps({
        "snapshot_id": snap.snapshot_id, "node_summaries": snap.node_summaries
    }, indent=2))
    (outdir / "dcpg_crdt_demo.json").write_text(json.dumps(demo_federated_merge(), indent=2))

    contract = PolicyContract(modality="text", chosen_policy="pseudo", risk_score=0.8, policy_version="v1")
    (outdir / "sample_dag.dot").write_text(export_dag(contract, fmt="dot"))
    (outdir / "sample_dag.json").write_text(export_dag(contract, fmt="json"))

    rl_agent.save(str(outdir / "ppo_model.pt"))
    (outdir / "rl_reward_stats.json").write_text(json.dumps(rl_agent.reward_stats(), indent=2))

    auroc_vals  = [r["extra"].get("delta_auroc", 0.0) for r in audit_rows]
    crdt_vals   = [r["extra"].get("crdt_risk",   0.0) for r in audit_rows]

    for fname, ys, title, ylabel, kwargs in [
        ("ppo_training_curve.png",    rewards,          "PPO Pretrain Reward",             "Reward",   {}),
        ("ppo_live_reward_curve.png", rl_event_rewards, "Live-Loop RL Reward per Event",   "Reward",   {"marker": "o", "markersize": 3}),
        ("delta_auroc_curve.png",     auroc_vals,       "Delta AUROC per Event",           "Δ AUROC",  {"color": "purple"}),
        ("adaptive_risk_timeline.png",risks,            "Adaptive Risk Timeline",          "Risk",     {}),
        ("policy_switch_timeline.png",decisions,        "Policy Switch Timeline",          "Policy",   {}),
        ("latency_histogram.png",     None,             "Latency Distribution",            "Frequency",{}),
    ]:
        fig, ax = plt.subplots(figsize=(8, 3))
        if fname == "latency_histogram.png":
            ax.hist(latencies, bins=20)
            ax.set_xlabel("Latency (ms)")
        else:
            ax.plot(ys, linewidth=0.8, **kwargs)
            ax.set_xlabel("Event" if fname != "ppo_training_curve.png" else "Episode")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(risks,     label="SQLite risk", linewidth=0.8)
    ax.plot(crdt_vals, label="CRDT risk",   linewidth=0.8, linestyle="--")
    ax.set_title("Risk: SQLite vs CRDT"); ax.set_xlabel("Event"); ax.set_ylabel("Risk"); ax.legend()
    fig.tight_layout(); fig.savefig(outdir / "crdt_vs_sqlite_risk.png", dpi=150); plt.close(fig)


    modal_labels = ["text", "asr", "image", "audio"]
    variable_idx = [i for i, col in enumerate(modal_counts) if len(set(col)) > 1]
    if len(variable_idx) >= 2:
        variable_counts = [modal_counts[i] for i in variable_idx]
        variable_labels = [modal_labels[i] for i in variable_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.nan_to_num(np.corrcoef(variable_counts))
        fig, ax = plt.subplots()
        im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(variable_labels))); ax.set_xticklabels(variable_labels)
        ax.set_yticks(range(len(variable_labels))); ax.set_yticklabels(variable_labels)
        ax.set_title("Multimodal PHI Correlation (variable modalities only)")
        fig.tight_layout()
        fig.savefig(outdir / "multimodal_phi_correlation.png", dpi=150); plt.close(fig)

    lat_sum = latency_summary(latencies)

    with open(outdir / "latency_summary.csv", "w") as f:
        w = csv.DictWriter(f, fieldnames=lat_sum.keys())
        w.writeheader(); w.writerow(lat_sum)

    with open(outdir / "controller_decisions.jsonl", "w") as f:
        for r, d in zip(risks, decisions):
            f.write(json.dumps({"risk": r, "decision": d}) + "\n")

    (outdir / "consent_cap_log.jsonl").write_text("\n".join(
        json.dumps({
            "event_id": r["event_id"], "patient": r["patient_key"],
            "consent":   r["extra"].get("consent_level"),
            "decided":   r["extra"].get("decided_policy"),
            "effective": r["extra"].get("effective_policy"),
            "capped":    r["extra"].get("consent_capped"),
        }) for r in audit_rows
    ))

    (outdir / "delta_auroc_log.jsonl").write_text("\n".join(
        json.dumps({
            "event_idx": i, "event_id": r["event_id"],
            "patient_key": r["patient_key"], "policy": r["chosen_policy"],
            "risk": r["risk"], "delta_auroc": r["extra"].get("delta_auroc", 0.0),
            "crdt_risk": r["extra"].get("crdt_risk", 0.0),
        }) for i, r in enumerate(audit_rows)
    ))

    (outdir / "run_metadata.json").write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "latency_summary": lat_sum,
    }, indent=2))
    (outdir / "experiment_config.json").write_text(json.dumps({"episodes": 200}, indent=2))
    (outdir / "controller_config.json").write_text(json.dumps(
        {"risk1": 0.4, "risk2": 0.8, "remask_thresh": 0.68}, indent=2
    ))

    save_report(outdir)
    print("\nArtifacts written to:", outdir, "\n")


if __name__ == "__main__":
    main()
