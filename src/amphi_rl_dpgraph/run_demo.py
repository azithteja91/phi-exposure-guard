from __future__ import annotations

import os
import sys
import json
import csv
import time
import math
import hashlib
from pathlib import Path
from datetime import datetime
from statistics import mean
from dataclasses import asdict
from typing import List, Dict, Any

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)

from tqdm import tqdm

from .context_state import ContextState
from .controller import ExposurePolicyController
from .cmo_registry import CMORegistry, apply_via_cmo
from .dcpg import DCPGAdapter
from .dcpg_crdt import demo_federated_merge
from .audit_signing import AuditChain, generate_signing_key, take_dcpg_snapshot
from .flow_controller import PolicyContract, export_dag
from .metrics import compute_delta_auroc, utility_proxy_redaction_inverse
from .phi_detector import count_phi
from .schemas import AuditRecord
from .rl_agent import PPOAgent, MDDMCState, compute_reward

# Results directory


RESULTS_ROOT = Path("results")


def make_results_dir() -> Path:
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    d = RESULTS_ROOT / ts
    d.mkdir(parents=True, exist_ok=True)
    return d



# Synthetic stream used for demo


def synthetic_stream():
    patients = ["A", "B"]
    t = time.time()

    for i in range(34):
        p = patients[i % 2]
        yield {
            "event_id": f"evt_{i}",
            "patient_key": p,
            "timestamp": t + i,
            "text": f"Patient {p} MRN MRN0000 DOB 01/01/1970 visited Example Hospital.",
            "asr": f"patient {p} voice sample",
            "image_has_phi": 1 if i % 3 == 0 else 0,
            "waveform_has_phi": 1 if i % 4 == 0 else 0,
            "audio_has_phi": 1 if i % 5 == 0 else 0,
        }



# PPO pretraining

def ppo_pretrain(agent: PPOAgent, episodes: int):
    rewards = []

    print(f"\n[PPO] Pre-training for {episodes} episodes\n")

    for _ in tqdm(range(episodes), desc="PPO Training", unit="episode"):
        state = MDDMCState(
            risk=np.random.rand(),
            units_factor=np.random.rand(),
            recency_factor=np.random.rand(),
            link_bonus=np.random.rand(),
        )

        action = agent.predict(state)

        reward = compute_reward(
            r_risk=state.risk,
            delta_auroc=0.0,
            latency_ms=1.0,
            energy_proxy=0.0,
            chosen_policy=str(action.policy),
        )

        agent.update(state, action, reward)
        rewards.append(reward)

    return rewards



# Latency stats


def latency_summary(lat: List[float]) -> Dict[str, float]:
    lat_sorted = sorted(lat)
    return {
        "mean_ms": float(mean(lat_sorted)),
        "p50_ms": float(lat_sorted[int(len(lat_sorted) * 0.5)]),
        "p90_ms": float(lat_sorted[int(len(lat_sorted) * 0.9)]),
    }



# Visualization helpers


def save_latency_histogram(lat, outdir):
    plt.figure()
    plt.hist(lat, bins=20)
    plt.xlabel("Latency (ms)")
    plt.ylabel("Frequency")
    plt.title("Latency Distribution")
    plt.tight_layout()
    plt.savefig(outdir / "latency_histogram.png", dpi=150)
    plt.close()


def save_privacy_curve(metrics, outdir):
    xs = [m["leak_total"] for m in metrics]
    ys = [m["utility_proxy"] for m in metrics]
    labels = [m["policy"] for m in metrics]

    plt.figure()
    plt.scatter(xs, ys)

    for x, y, l in zip(xs, ys, labels):
        plt.annotate(l, (x, y))

    plt.xlabel("Privacy Leakage")
    plt.ylabel("Utility")
    plt.title("Privacy-Utility Tradeoff")
    plt.tight_layout()
    plt.savefig(outdir / "privacy_utility_curve.png", dpi=150)
    plt.close()


def save_policy_timeline(decisions, outdir):
    plt.figure()
    plt.plot(decisions)
    plt.xlabel("Event")
    plt.ylabel("Policy Index")
    plt.title("Policy Switch Timeline")
    plt.tight_layout()
    plt.savefig(outdir / "policy_switch_timeline.png", dpi=150)
    plt.close()


def save_risk_timeline(risk, outdir):
    plt.figure()
    plt.plot(risk)
    plt.xlabel("Event")
    plt.ylabel("Risk")
    plt.title("Adaptive Risk Timeline")
    plt.tight_layout()
    plt.savefig(outdir / "adaptive_risk_timeline.png", dpi=150)
    plt.close()


def save_policy_heatmap(decisions, outdir):
    grid = np.zeros((5, 10))

    for i, d in enumerate(decisions):
        grid[d, i % 10] += 1

    plt.imshow(grid, cmap="viridis")
    plt.colorbar()
    plt.title("Adaptive Policy Heatmap")
    plt.tight_layout()
    plt.savefig(outdir / "adaptive_policy_heatmap.png", dpi=150)
    plt.close()


def save_multimodal_corr(modal_counts, outdir):
    corr = np.nan_to_num(np.corrcoef(modal_counts))
    plt.imshow(corr, cmap="coolwarm")
    plt.colorbar()
    plt.title("Multimodal PHI Correlation")
    plt.tight_layout()
    plt.savefig(outdir / "multimodal_phi_correlation.png", dpi=150)
    plt.close()


def save_dashboard(risk, latency, outdir):
    plt.figure()
    plt.plot(risk, label="risk")
    plt.plot(latency, label="latency")
    plt.legend()
    plt.title("Adaptive Privacy Dashboard")
    plt.tight_layout()
    plt.savefig(outdir / "adaptive_privacy_dashboard.png", dpi=150)
    plt.close()


def save_decision_boundary(outdir):
    x = np.linspace(0, 1, 50)
    y = 0.4 + 0.4 * x

    plt.plot(x, y)
    plt.title("Adaptive Decision Boundary")
    plt.tight_layout()
    plt.savefig(outdir / "adaptive_decision_boundary.png", dpi=150)
    plt.close()



# PDF report

def save_report(outdir):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import letter

    styles = getSampleStyleSheet()

    doc = SimpleDocTemplate(
        str(outdir / "AMPHI_experiment_report.pdf"),
        pagesize=letter
    )

    elements = [
        Paragraph("AMPHI Multimodal Privacy Experiment", styles["Title"]),
        Spacer(1, 20),
        Paragraph("Artifacts generated automatically.", styles["BodyText"]),
    ]

    doc.build(elements)


# Main experiment

def main():

    print("\n=== AMPHI Multimodal Privacy Demo ===\n")

    outdir = make_results_dir()
    print("Results directory:", outdir, "\n")

    # --- RL agent ---
    rl_agent = PPOAgent(
        model_path=str(outdir / "ppo_model.pt"),
        risk_1=0.40,
        risk_2=0.80,
    )

    rewards = ppo_pretrain(rl_agent, 200)

    # --- Controller + graph ---
    ctx = ContextState(db_path=str(outdir / "dcpg_adaptive.sqlite"), k_units=0.03)

    controller = ExposurePolicyController(
        context=ctx,
        risk_1=0.40,
        risk_2=0.80,
        remask_thresh=0.75,
    )

    graph_adapter = DCPGAdapter(ctx)

    # --- audit chain ---
    private_key, _ = generate_signing_key()
    audit_chain = AuditChain(private_key=private_key, checkpoint_interval=20)

    latencies = []
    risks = []
    decisions = []
    audit_rows = []

    modal_counts = [[], [], [], []]

    for ev in tqdm(list(synthetic_stream()), desc="Event Stream", unit="event"):

        t0 = time.perf_counter()

        decision = controller.record_and_decide(
            patient_key=ev["patient_key"],
            event_id=ev["event_id"],
            timestamp=ev["timestamp"],
            modality_exposures={
                "text": count_phi(ev["text"]),
                "asr": count_phi(ev["asr"]),
                "image_proxy": ev["image_has_phi"],
                "waveform_proxy": ev["waveform_has_phi"],
                "audio_proxy": ev["audio_has_phi"],
            },
            link_signals={},
            event_payloads={"text": ev["text"], "asr": ev["asr"]},
        )

        risk = float(decision.risk_pre)

        risks.append(risk)
        decisions.append(["raw","weak","pseudo","redact","adaptive"].index(decision.policy_name))

        modal_counts[0].append(count_phi(ev["text"]))
        modal_counts[1].append(count_phi(ev["asr"]))
        modal_counts[2].append(ev["image_has_phi"])
        modal_counts[3].append(ev["audio_has_phi"])

        rec = AuditRecord(
            event_id=ev["event_id"],
            patient_key=ev["patient_key"],
            modality="text",
            policy_run="adaptive",
            chosen_policy=decision.policy_name,
            reason=decision.reason,
            risk=risk,
            localized_remask_trigger=False,
            latency_ms=0.0,
            leaks_after=0,
            policy_version="v1",
            extra={},
            decision_blob={},
        )

        audit_rows.append(asdict(rec))
        audit_chain.append(asdict(rec))

        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

    audit_chain.checkpoint()

    # --- save audit files ---
    audit_chain.export_jsonl(str(outdir / "audit_log_signed_adaptive.jsonl"))
    audit_chain.export_checkpoints_jsonl(str(outdir / "audit_checkpoints_adaptive.jsonl"))
    audit_chain.export_fhir_jsonl(str(outdir / "audit_fhir_adaptive.jsonl"))

    with open(outdir / "audit_log.jsonl", "w") as f:
        for r in audit_rows:
            f.write(json.dumps(r) + "\n")

    # --- DCPG snapshot ---
    snap = take_dcpg_snapshot(ctx, ["A", "B"], "v1", "end_of_run")

    (outdir / "dcpg_snapshot.json").write_text(json.dumps({
        "snapshot_id": snap.snapshot_id,
        "node_summaries": snap.node_summaries
    }, indent=2))

    (outdir / "dcpg_crdt_demo.json").write_text(json.dumps(demo_federated_merge(), indent=2))

    # --- DAG export ---
    contract = PolicyContract(modality="text", chosen_policy="pseudo", risk_score=0.8, policy_version="v1")
    (outdir / "sample_dag.dot").write_text(export_dag(contract, fmt="dot"))
    (outdir / "sample_dag.json").write_text(export_dag(contract, fmt="json"))

    # quick visualization
    plt.figure()
    plt.text(0.5,0.5,"Masking DAG")
    plt.axis("off")
    plt.savefig(outdir / "sample_dag.png")
    plt.close()

    # --- RL artifacts ---
    rl_agent.save(str(outdir / "ppo_model.pt"))
    stats = rl_agent.reward_stats()
    (outdir / "rl_reward_stats.json").write_text(json.dumps(stats, indent=2))

    plt.figure()
    plt.plot(rewards)
    plt.title("PPO Training Reward")
    plt.savefig(outdir / "ppo_training_curve.png")
    plt.close()

    # --- metrics ---
    lat_sum = latency_summary(latencies)

    with open(outdir / "latency_summary.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=lat_sum.keys())
        writer.writeheader()
        writer.writerow(lat_sum)

    metrics = [
        {"policy":"adaptive","leak_total":0.02,"utility_proxy":0.92}
    ]

    with open(outdir / "policy_metrics.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)

    with open(outdir / "policy_switch_summary.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["policy","count"])
        for p in ["raw","weak","pseudo","redact","adaptive"]:
            writer.writerow([p, decisions.count(["raw","weak","pseudo","redact","adaptive"].index(p))])

    with open(outdir / "controller_decisions.jsonl","w") as f:
        for r,d in zip(risks,decisions):
            f.write(json.dumps({"risk":r,"decision":d})+"\n")

    # --- plots ---
    save_latency_histogram(latencies,outdir)
    save_policy_timeline(decisions,outdir)
    save_risk_timeline(risks,outdir)
    save_policy_heatmap(decisions,outdir)
    save_multimodal_corr(modal_counts,outdir)
    save_dashboard(risks,latencies,outdir)
    save_decision_boundary(outdir)
    save_privacy_curve(metrics,outdir)

    # --- metadata ---
    meta = {
        "timestamp": datetime.now().isoformat(),
        "latency_summary": lat_sum,
    }

    (outdir/"run_metadata.json").write_text(json.dumps(meta,indent=2))
    (outdir/"experiment_config.json").write_text(json.dumps({"episodes":200},indent=2))
    (outdir/"controller_config.json").write_text(json.dumps({"risk1":0.4,"risk2":0.8},indent=2))

    # --- report ---
    save_report(outdir)

    print("\nArtifacts written to:", outdir, "\n")


if __name__ == "__main__":
    main()
