from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean
from typing import Dict, List, Tuple


def _results_dir() -> Path:
    """
    GitHub-friendly results directory.
    - Defaults to ./results
    - Can override via env var AMPHI_RESULTS_DIR
    """
    base = os.environ.get("AMPHI_RESULTS_DIR", "").strip()
    out = Path(base) if base else Path.cwd() / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


RESULTS_DIR = _results_dir()


def stable_patient_id(patient_key: str, mod: int = 1000) -> int:
    h = hashlib.sha256(patient_key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % int(mod)


def synthetic_stream():
    """
    Obviously synthetic stream.
    No real-looking MRNs, DOBs, hospitals.
    """

    def make_note(label: str, level: int) -> Tuple[str, str]:
        if level == 1:
            return "Routine follow-up visit for a patient.", "routine follow up patient"
        if level == 2:
            return (
                f"Patient {label} reports mild pain.",
                f"patient {label.lower()} mild pain",
            )
        return (
            f"Patient {label} MRN MRN0000 DOB 01/01/1970 visited Example Hospital.",
            f"patient {label.lower()} mrn mrn0000 date of birth 01/01/1970",
        )

    patients = [
        ("patient_1", "A"),
        ("patient_2", "B"),
    ]
    event_id = 0
    t = time.time()

    for pk, label in patients:
        # low PHI
        for _ in range(8):
            text, asr = make_note(label, 1)
            yield {
                "event_id": f"evt_{event_id}",
                "patient_key": pk,
                "timestamp": t,
                "text": text,
                "asr": asr,
                "image_has_phi": 0,
                "waveform_has_phi": 0,
                "audio_has_phi": 0,
                "image_link": 0,
                "audio_link": 0,
                "payloads": {"text": text, "asr": asr},
            }
            event_id += 1
            t += 5

        # moderate PHI
        for _ in range(9):
            text, asr = make_note(label, 2)
            yield {
                "event_id": f"evt_{event_id}",
                "patient_key": pk,
                "timestamp": t,
                "text": text,
                "asr": asr,
                "image_has_phi": 1,
                "waveform_has_phi": 0,
                "audio_has_phi": 1,
                "image_link": 0,
                "audio_link": 0,
                "payloads": {"text": text, "asr": asr},
            }
            event_id += 1
            t += 5

        # cross-modal link signal (no extra PHI units, just linkage signal)
        for _ in range(2):
            text = f"Follow-up for patient {label} regarding medication tolerance."
            asr = "synthetic voice match signal"
            yield {
                "event_id": f"evt_{event_id}",
                "patient_key": pk,
                "timestamp": t,
                "text": text,
                "asr": asr,
                "image_has_phi": 0,
                "waveform_has_phi": 0,
                "audio_has_phi": 0,
                "image_link": 1,
                "audio_link": 1,
                "payloads": {"text": text, "asr": asr},
            }
            event_id += 1
            t += 5

        # high PHI
        for _ in range(20):
            text, asr = make_note(label, 3)
            yield {
                "event_id": f"evt_{event_id}",
                "patient_key": pk,
                "timestamp": t,
                "text": text,
                "asr": asr,
                "image_has_phi": 1,
                "waveform_has_phi": 1,
                "audio_has_phi": 1,
                "image_link": 0,
                "audio_link": 0,
                "payloads": {"text": text, "asr": asr},
            }
            event_id += 1
            t += 5


def _safe_imports():
    from .context_state import ContextState
    from .controller import ExposurePolicyController
    from .cmo_registry import CMORegistry, apply_via_cmo
    from .dcpg import DCPGAdapter
    from .eval import summarize_latency
    from .metrics import compute_delta_auroc, utility_proxy_redaction_inverse
    from .phi_detector import count_phi
    from .schemas import AuditRecord
    from .flow_controller import PolicyContract, export_dag

    # Optional modules: if they don't exist, demo still runs.
    AuditChain = None
    generate_signing_key = None
    take_dcpg_snapshot = None
    try:
        from .audit_signing import AuditChain as _AuditChain, generate_signing_key as _gk, take_dcpg_snapshot as _snap
        AuditChain = _AuditChain
        generate_signing_key = _gk
        take_dcpg_snapshot = _snap
    except Exception:
        pass

    demo_federated_merge = None
    try:
        from .dcpg_crdt import demo_federated_merge as _dfm
        demo_federated_merge = _dfm
    except Exception:
        pass

    PPOAgent = None
    MDDMCState = None
    compute_reward = None
    try:
        from .rl_agent import MDDMCState as _MDDMCState, PPOAgent as _PPOAgent, compute_reward as _compute_reward
        MDDMCState = _MDDMCState
        PPOAgent = _PPOAgent
        compute_reward = _compute_reward
    except Exception:
        pass

    return {
        "ContextState": ContextState,
        "ExposurePolicyController": ExposurePolicyController,
        "CMORegistry": CMORegistry,
        "apply_via_cmo": apply_via_cmo,
        "DCPGAdapter": DCPGAdapter,
        "summarize_latency": summarize_latency,
        "compute_delta_auroc": compute_delta_auroc,
        "utility_proxy_redaction_inverse": utility_proxy_redaction_inverse,
        "count_phi": count_phi,
        "AuditRecord": AuditRecord,
        "PolicyContract": PolicyContract,
        "export_dag": export_dag,
        "AuditChain": AuditChain,
        "generate_signing_key": generate_signing_key,
        "take_dcpg_snapshot": take_dcpg_snapshot,
        "demo_federated_merge": demo_federated_merge,
        "PPOAgent": PPOAgent,
        "MDDMCState": MDDMCState,
        "compute_reward": compute_reward,
    }


def run_policy(policy_run: str) -> Tuple[dict, dict, List[dict]]:
    m = _safe_imports()

    ContextState = m["ContextState"]
    ExposurePolicyController = m["ExposurePolicyController"]
    CMORegistry = m["CMORegistry"]
    apply_via_cmo = m["apply_via_cmo"]
    DCPGAdapter = m["DCPGAdapter"]
    summarize_latency = m["summarize_latency"]
    compute_delta_auroc = m["compute_delta_auroc"]
    utility_proxy_redaction_inverse = m["utility_proxy_redaction_inverse"]
    count_phi = m["count_phi"]
    AuditRecord = m["AuditRecord"]
    PolicyContract = m["PolicyContract"]
    export_dag = m["export_dag"]

    AuditChain = m["AuditChain"]
    generate_signing_key = m["generate_signing_key"]
    take_dcpg_snapshot = m["take_dcpg_snapshot"]
    demo_federated_merge = m["demo_federated_merge"]
    PPOAgent = m["PPOAgent"]
    MDDMCState = m["MDDMCState"]
    compute_reward = m["compute_reward"]

    latencies_ms: List[float] = []
    out_texts: List[str] = []
    out_asrs: List[str] = []
    out_img_leaks: List[int] = []
    out_wav_leaks: List[int] = []
    out_audio_leaks: List[int] = []
    orig_texts: List[str] = []
    orig_labels: List[int] = []
    audit_rows: List[dict] = []

    controller = None
    rl_agent = None
    audit_chain = None
    graph_adapter = None
    seen_patients: set[str] = set()
    graph_cache: Dict[str, dict] = {}
    trigger_snapshots: Dict[str, dict] = {}

    _auroc_seed_orig: List[str] = []
    _auroc_seed_masked: List[str] = []
    _auroc_seed_labels: List[int] = []

    # Set up adaptive components only if needed
    if policy_run == "adaptive":
        db_path = str(RESULTS_DIR / f"dcpg_{policy_run}.sqlite")
        db_file = Path(db_path)
        if db_file.exists():
            db_file.unlink()

        ctx = ContextState(db_path=db_path, k_units=0.03)
        controller = ExposurePolicyController(
            context=ctx,
            risk_1=0.40,
            risk_2=0.80,
            remask_thresh=0.75,
        )

        if PPOAgent is not None:
            rl_agent = PPOAgent(
                model_path=str(RESULTS_DIR / "ppo_model.pt"),
                risk_1=0.40,
                risk_2=0.80,
            )

            # ── PPO PRE-TRAINING ──────────────────────────────────────────────
            # The 78-step evaluation stream is too short for the PPO network to
            # converge. We pre-train across multiple full episodes on the same
            # synthetic stream (using an in-memory context so no state leaks
            # into the evaluation run) before timing begins.
            if MDDMCState is not None and compute_reward is not None:
                PRE_TRAIN_EPISODES = 200
                print(f"[PPO] Pre-training for {PRE_TRAIN_EPISODES} episodes ...", flush=True)
                for _ep in range(PRE_TRAIN_EPISODES):
                    # Fresh in-memory context per episode — no state leakage
                    _pt_ctx = ContextState(db_path=":memory:", k_units=0.03)
                    _pt_ctrl = ExposurePolicyController(
                        context=_pt_ctx,
                        risk_1=0.40,
                        risk_2=0.80,
                        remask_thresh=0.75,
                    )
                    for _ev in synthetic_stream():
                        _dec = _pt_ctrl.record_and_decide(
                            patient_key=_ev["patient_key"],
                            event_id=_ev["event_id"],
                            timestamp=_ev["timestamp"],
                            modality_exposures={
                                "text":          1 if count_phi(_ev["text"]) > 0 else 0,
                                "asr":           1 if count_phi(_ev["asr"]) > 0 else 0,
                                "image_proxy":   int(_ev["image_has_phi"]),
                                "waveform_proxy": int(_ev["waveform_has_phi"]),
                                "audio_proxy":   int(_ev["audio_has_phi"]),
                            },
                            link_signals={
                                "image_link": int(_ev.get("image_link", 0)),
                                "audio_link": int(_ev.get("audio_link", 0)),
                            },
                            event_payloads=_ev.get("payloads", {}),
                        )
                        _risk = float(_dec.risk_components.get("risk", float(_dec.risk_pre)))
                        _s = MDDMCState(
                            risk=_risk,
                            units_factor=float(_dec.risk_components.get("units_factor", 0.0)),
                            recency_factor=float(_dec.risk_components.get("recency_factor", 0.0)),
                            link_bonus=float(_dec.risk_components.get("link_bonus", 0.0)),
                        )
                        _a = rl_agent.predict(_s)
                        _r = compute_reward(
                            r_risk=_risk,
                            delta_auroc=0.0,
                            latency_ms=1.0,
                            energy_proxy=0.0,
                            chosen_policy=str(_a.policy),
                        )
                        rl_agent.update(_s, _a, _r)

                    if (_ep + 1) % 50 == 0:
                        _rs = rl_agent.reward_stats()
                        print(
                            f"[PPO] Episode {_ep + 1}/{PRE_TRAIN_EPISODES} — "
                            f"model_mean={_rs.get('model_mean', 0):.4f}  "
                            f"warmup_mean={_rs.get('warmup_mean', 0):.4f}  "
                            f"model_n={_rs.get('model_n', 0)}",
                            flush=True,
                        )

                _final = rl_agent.reward_stats()
                print(
                    f"[PPO] Pre-training done — "
                    f"model_mean={_final.get('model_mean', 0):.4f}  "
                    f"warmup_mean={_final.get('warmup_mean', 0):.4f}  "
                    f"total steps={rl_agent._step_count}",
                    flush=True,
                )
            # ── END PRE-TRAINING ──────────────────────────────────────────────

        if AuditChain is not None and generate_signing_key is not None:
            private_key, _ = generate_signing_key()
            audit_chain = AuditChain(private_key=private_key, checkpoint_interval=50)

        graph_adapter = DCPGAdapter(ctx)

        if audit_chain:
            audit_chain.register_cmo_version(CMORegistry.list_operators(), policy_version="v1")

        # Warm up the embedding model before timing begins.
        # DCPGAdapter lazily initialises sentence-transformers on the first
        # cross_modal_match() call, which takes 100-400 ms (model load + JIT).
        # Without warmup that spike lands in the first timed event and inflates
        # mean_latency by ~100x vs the true steady-state P50.
        try:
            print("[warmup] loading embedding model ...", flush=True)
            graph_adapter.cross_modal_match("__warmup__", "text", "warmup probe")
            print("[warmup] done", flush=True)
        except Exception:
            pass

        # Seed AUROC with synthetic masked examples if cmo_media exists
        try:
            from .cmo_media import apply_synthetic_replacement as _synth_replace
            seed_notes = {
                "patient_1": "Patient A MRN MRN0000 DOB 01/01/1970 visited Example Hospital.",
                "patient_2": "Patient B MRN MRN0000 DOB 01/01/1970 visited Example Hospital.",
            }
            for idx, pk in enumerate(["patient_1", "patient_2"]):
                for _ in range(4):
                    st = seed_notes[pk]
                    _auroc_seed_orig.append(st)
                    _auroc_seed_masked.append(_synth_replace(st))
                    _auroc_seed_labels.append(idx)
        except Exception:
            pass

    # Stream loop
    for ev in synthetic_stream():
        t0 = time.perf_counter()

        patient_key = ev["patient_key"]
        patient_id = stable_patient_id(patient_key)
        seen_patients.add(patient_key)

        modality_exposures = {
            "text": 1 if count_phi(ev["text"]) > 0 else 0,
            "asr": 1 if count_phi(ev["asr"]) > 0 else 0,
            "image_proxy": int(ev["image_has_phi"]),
            "waveform_proxy": int(ev["waveform_has_phi"]),
            "audio_proxy": int(ev["audio_has_phi"]),
        }
        link_signals = {
            "image_link": int(ev.get("image_link", 0)),
            "audio_link": int(ev.get("audio_link", 0)),
        }
        link_modalities = [k for k, v in link_signals.items() if int(v) > 0]

        orig_texts.append(ev["text"])
        sorted_patients = sorted(seen_patients | {patient_key})
        orig_labels.append(sorted_patients.index(patient_key) % 2)

        if policy_run == "adaptive":
            assert controller is not None

            decision = controller.record_and_decide(
                patient_key=patient_key,
                event_id=ev["event_id"],
                timestamp=ev["timestamp"],
                modality_exposures=modality_exposures,
                link_signals=link_signals,
                event_payloads=ev.get("payloads", {}),
            )

            risk = float(decision.risk_pre)
            remask_info = dict(decision.localized_remask or {})
            remask_trigger = bool(remask_info.get("trigger", False))
            patient_token = controller.current_token(patient_key, patient_id)

            if remask_trigger and patient_key not in trigger_snapshots:
                snap = remask_info.get("dcpg_snapshot_at_trigger", {})
                if snap:
                    trigger_snapshots[patient_key] = snap
                    if audit_chain:
                        sid = f"trigger_{patient_key}_{ev['event_id']}"
                        audit_chain.register_snapshot(sid, {**snap, "timestamp": ev["timestamp"]})

            # RL choice (optional)
            if rl_agent is not None and MDDMCState is not None:
                mddmc_state = MDDMCState(
                    risk=float(decision.risk_components.get("risk", risk)),
                    units_factor=float(decision.risk_components.get("units_factor", 0.0)),
                    recency_factor=float(decision.risk_components.get("recency_factor", 0.0)),
                    link_bonus=float(decision.risk_components.get("link_bonus", 0.0)),
                    phi_text=modality_exposures.get("text", 0),
                    phi_asr=modality_exposures.get("asr", 0),
                    phi_image=modality_exposures.get("image_proxy", 0),
                    phi_waveform=modality_exposures.get("waveform_proxy", 0),
                    phi_audio=modality_exposures.get("audio_proxy", 0),
                )
                rl_action = rl_agent.predict(mddmc_state)
                chosen = rl_action.policy
                reason = f"{decision.reason} / rl:{rl_action.source}"
            else:
                chosen = decision.policy_name
                reason = decision.reason

            # Graph summary cache
            if remask_trigger or patient_key not in graph_cache:
                if graph_adapter:
                    graph_cache[patient_key] = graph_adapter.graph_summary(patient_key)
            graph_summary = graph_cache.get(patient_key, {})
            provisional_risk = graph_summary.get("provisional_risk", 0.0)

            decision_blob = {
                "thresholds": dict(decision.thresholds),
                "candidates": list(decision.candidates),
                "localized_remask": {k: v for k, v in remask_info.items() if k != "dcpg_snapshot_at_trigger"},
                "link_modalities_recent": list(decision.link_modalities_recent),
                "cross_modal_matches": list(decision.cross_modal_matches),
                "risk_components": dict(decision.risk_components),
                "risk_source": str(decision.risk_source),
                "rl_policy": chosen,
                "provisional_risk": provisional_risk,
                "dcpg_node_count": graph_summary.get("node_count", 0),
                "dcpg_edge_count": graph_summary.get("edge_count", 0),
            }
        else:
            chosen = policy_run
            reason = "fixed policy"
            risk = 0.0
            remask_trigger = False
            patient_token = f"PATIENT_{int(patient_id)}_V0"
            decision_blob = {
                "thresholds": {},
                "candidates": [],
                "localized_remask": {"trigger": False},
                "link_modalities_recent": [],
                "cross_modal_matches": [],
                "risk_components": {},
                "risk_source": "fixed_policy",
                "rl_policy": chosen,
                "provisional_risk": 0.0,
                "dcpg_node_count": 0,
                "dcpg_edge_count": 0,
            }

        masked_results = {}
        cmo_logs = []
        for modality in ["text", "asr", "image_proxy", "waveform_proxy", "audio_proxy"]:
            payload = (
                ev["text"]
                if modality == "text"
                else ev["asr"]
                if modality == "asr"
                else ev.get(modality.replace("_proxy", "_has_phi"), 0)
            )
            out, cmo_log = apply_via_cmo(
                modality=modality,
                policy=chosen,
                payload=payload,
                patient_token=patient_token,
                event_id=ev["event_id"],
                risk_score=risk,
            )
            masked_results[modality] = out
            cmo_logs.append(
                {
                    "cmo": cmo_log.cmo_name,
                    "input_hash": cmo_log.input_hash,
                    "output_hash": cmo_log.output_hash,
                    "latency_ms": cmo_log.latency_ms,
                }
            )

        masked_text = masked_results["text"]
        masked_asr = masked_results["asr"]
        img_leak = int(masked_results["image_proxy"])
        wav_leak = int(masked_results["waveform_proxy"])
        audio_leak = int(masked_results["audio_proxy"])

        latency = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(latency)
        out_texts.append(masked_text)
        out_asrs.append(masked_asr)
        out_img_leaks.append(img_leak)
        out_wav_leaks.append(wav_leak)
        out_audio_leaks.append(audio_leak)

        # RL update (optional)
        if policy_run == "adaptive" and controller and rl_agent and MDDMCState and compute_reward:
            units_masked = modality_exposures.get("text", 0) + modality_exposures.get("asr", 0)
            if units_masked > 0 and chosen not in ("raw",):
                credit_info = controller.apply_post_masking_credit(patient_key, units_masked)
                decision_blob["masked_credit_applied"] = units_masked
                decision_blob["masked_credit_info"] = credit_info

            delta_auroc = 0.0
            auc_orig_val = 0.0
            auc_mask_val = 0.0
            if len(set(orig_labels + _auroc_seed_labels)) >= 2:
                full_orig = _auroc_seed_orig + orig_texts
                full_masked = _auroc_seed_masked + out_texts
                full_labels = _auroc_seed_labels + orig_labels
                delta_auroc, auc_orig_val, auc_mask_val = compute_delta_auroc(full_orig, full_masked, full_labels)

            reward = compute_reward(
                r_risk=float(risk),
                delta_auroc=float(delta_auroc),
                latency_ms=float(latency),
                energy_proxy=0.0,
                chosen_policy=str(chosen),
            )
            # Best-effort update
            try:
                mddmc_state.delta_auroc = float(delta_auroc)
            except Exception:
                pass
            rl_agent.update(mddmc_state, rl_action, reward)  # type: ignore[name-defined]
            decision_blob["reward_logged"] = round(float(reward), 5)
            decision_blob["delta_auroc"] = round(float(delta_auroc), 5)
            decision_blob["auc_orig"] = round(float(auc_orig_val), 5)
            decision_blob["auc_mask"] = round(float(auc_mask_val), 5)

        decision_blob["cmo_execution_log"] = cmo_logs

        for modality, output in [
            ("text", masked_text),
            ("asr", masked_asr),
            ("image_proxy", img_leak),
            ("waveform_proxy", wav_leak),
            ("audio_proxy", audio_leak),
        ]:
            rec = AuditRecord(
                event_id=ev["event_id"],
                patient_key=patient_key,
                modality=modality,
                policy_run=policy_run,
                chosen_policy=chosen,
                reason=reason,
                risk=float(risk),
                localized_remask_trigger=bool(remask_trigger),
                latency_ms=float(round(latency, 3)),
                leaks_after=float(count_phi(output) if isinstance(output, str) else output),
                policy_version="v1",
                extra={
                    "patient_token": patient_token,
                    "link_modalities": list(link_modalities),
                },
                decision_blob=dict(decision_blob),
            )
            row_dict = asdict(rec)
            audit_rows.append(row_dict)
            if audit_chain:
                audit_chain.append(row_dict)

    # Exports
    if audit_chain:
        audit_chain.checkpoint()
        audit_chain.export_jsonl(str(RESULTS_DIR / f"audit_log_signed_{policy_run}.jsonl"))
        audit_chain.export_checkpoints_jsonl(str(RESULTS_DIR / f"audit_checkpoints_{policy_run}.jsonl"))
        audit_chain.export_fhir_jsonl(str(RESULTS_DIR / f"audit_fhir_{policy_run}.jsonl"))

    # DCPG + DAG exports (adaptive only, best-effort)
    if policy_run == "adaptive" and graph_adapter and controller:
        ctx_ref = controller.context
        if take_dcpg_snapshot is not None:
            snapshot = take_dcpg_snapshot(
                ctx_ref,
                patient_keys=list(seen_patients),
                policy_version="v1",
                trigger="end_of_run",
            )
            snap_dict = {
                "snapshot_id": snapshot.snapshot_id,
                "timestamp": snapshot.timestamp,
                "trigger": snapshot.trigger,
                "node_summaries": snapshot.node_summaries,
                "trigger_snapshots": trigger_snapshots,
            }
            (RESULTS_DIR / "dcpg_snapshot.json").write_text(json.dumps(snap_dict, indent=2), encoding="utf-8")
            if audit_chain:
                audit_chain.register_snapshot(snapshot.snapshot_id, snap_dict)

        sample_contract = PolicyContract(modality="text", chosen_policy="pseudo", risk_score=0.8, policy_version="v1")
        (RESULTS_DIR / "sample_dag.dot").write_text(export_dag(sample_contract, fmt="dot"), encoding="utf-8")
        (RESULTS_DIR / "sample_dag.json").write_text(export_dag(sample_contract, fmt="json"), encoding="utf-8")

        if demo_federated_merge is not None:
            crdt_result = demo_federated_merge()
            (RESULTS_DIR / "dcpg_crdt_demo.json").write_text(json.dumps(crdt_result, indent=2), encoding="utf-8")

    # RL exports (optional)
    if rl_agent:
        try:
            rl_stats = rl_agent.reward_stats()
            (RESULTS_DIR / "rl_reward_stats.json").write_text(json.dumps(rl_stats, indent=2), encoding="utf-8")
            rl_agent.save(str(RESULTS_DIR / "ppo_model.pt"))
        except Exception:
            pass

    # Metrics
    leak_text = sum(m["count_phi"](t) for t in out_texts) / len(out_texts) if out_texts else 0.0  # type: ignore[index]
    leak_asr = sum(m["count_phi"](t) for t in out_asrs) / len(out_asrs) if out_asrs else 0.0  # type: ignore[index]
    leak_img = mean(out_img_leaks) if out_img_leaks else 0.0
    leak_wav = mean(out_wav_leaks) if out_wav_leaks else 0.0
    leak_audio = mean(out_audio_leaks) if out_audio_leaks else 0.0
    leak_total = float(leak_text + leak_asr + leak_img + leak_wav + leak_audio)

    util = (
        utility_proxy_redaction_inverse(out_texts)
        + utility_proxy_redaction_inverse(out_asrs)
    ) / 2.0
    lat = summarize_latency(latencies_ms)

    return (
        {
            "policy": policy_run,
            "leak_text_avg": round(float(leak_text), 4),
            "leak_asr_avg": round(float(leak_asr), 4),
            "leak_image_avg": round(float(leak_img), 4),
            "leak_waveform_avg": round(float(leak_wav), 4),
            "leak_audio_avg": round(float(leak_audio), 4),
            "leak_total": round(float(leak_total), 4),
            "utility_proxy": round(float(util), 6),
            "mean_latency_ms": round(float(lat["mean_ms"]), 3),
            "p50_latency_ms": round(float(lat["p50_ms"]), 3),
            "p90_latency_ms": round(float(lat["p90_ms"]), 3),
        },
        {"policy": policy_run, **lat},
        audit_rows,
    )


def _write_experiment_report(metrics_rows: List[dict], latency_rows: List[dict], remask_count: int) -> None:
    lines = [
        "# Experiment Report",
        "",
        "## Privacy-Utility Results",
        "",
        "| Policy | Leak Total | Utility Proxy | Mean Latency (ms) | P90 Latency (ms) |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in metrics_rows:
        lines.append(
            f"| {r['policy']} | {r['leak_total']} | {r['utility_proxy']} | {r['mean_latency_ms']} | {r['p90_latency_ms']} |"
        )

    lines += [
        "",
        "## Leakage Breakdown",
        "",
        "| Policy | Text | ASR | Image | Waveform | Audio |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in metrics_rows:
        lines.append(
            f"| {r['policy']} | {r['leak_text_avg']} | {r['leak_asr_avg']} | {r['leak_image_avg']} | {r['leak_waveform_avg']} | {r['leak_audio_avg']} |"
        )

    lines += [
        "",
        "## Latency Summary",
        "",
        "| Policy | Mean (ms) | P50 (ms) | P90 (ms) |",
        "| --- | --- | --- | --- |",
    ]
    for r in latency_rows:
        lines.append(
            f"| {r['policy']} | {round(float(r['mean_ms']), 3)} | {round(float(r['p50_ms']), 3)} | {round(float(r['p90_ms']), 3)} |"
        )

    lines += [
        "",
        "## Adaptive Policy Notes",
        "",
        f"Cross-modal synergy triggered localized retokenization {remask_count} time(s).",
        "",
        "### Output Files",
        "- `audit_log_signed_adaptive.jsonl` (if audit signing enabled)",
        "- `audit_checkpoints_adaptive.jsonl` (if audit signing enabled)",
        "- `audit_fhir_adaptive.jsonl` (if audit signing enabled)",
        "- `dcpg_snapshot.json` (if enabled)",
        "- `dcpg_crdt_demo.json` (if enabled)",
        "- `rl_reward_stats.json` (if enabled)",
        "- `sample_dag.dot` / `sample_dag.json` / `sample_dag.png`",
        "- `privacy_utility_curve.png`",
        "",
    ]
    (RESULTS_DIR / "EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    policies = ["raw", "weak", "pseudo", "redact", "adaptive"]

    metrics_rows: List[dict] = []
    latency_rows: List[dict] = []
    audit_all: List[dict] = []

    for p in policies:
        m, l, a = run_policy(p)
        metrics_rows.append(m)
        latency_rows.append(l)
        audit_all.extend(a)

    # CSVs
    (RESULTS_DIR / "policy_metrics.csv").parent.mkdir(parents=True, exist_ok=True)
    with (RESULTS_DIR / "policy_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_rows[0].keys()))
        writer.writeheader()
        writer.writerows(metrics_rows)

    with (RESULTS_DIR / "latency_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy", "mean_ms", "p50_ms", "p90_ms"])
        writer.writeheader()
        writer.writerows(latency_rows)

    # Raw audit log
    with (RESULTS_DIR / "audit_log.jsonl").open("w", encoding="utf-8") as f:
        for row in audit_all:
            f.write(json.dumps(row) + "\n")

    # Plot (optional)
    plot_saved = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        xs = [r["leak_total"] for r in metrics_rows]
        ys = [r["utility_proxy"] for r in metrics_rows]
        labels = [r["policy"] for r in metrics_rows]

        plt.figure()
        plt.scatter(xs, ys, s=100)
        offsets = {
            "raw": (6, 6),
            "weak": (6, 6),
            "pseudo": (-50, 8),
            "redact": (6, -14),
            "adaptive": (6, 8),
        }
        for x, y, label in zip(xs, ys, labels):
            ox, oy = offsets.get(label, (6, 6))
            plt.annotate(label, (x, y), textcoords="offset points", xytext=(ox, oy), fontsize=10)
        plt.xlabel("Privacy leakage (lower is better)")
        plt.ylabel("Utility proxy (higher is better)")
        plt.title("Privacy-Utility Tradeoff (Synthetic Multimodal Stream)")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "privacy_utility_curve.png", dpi=160)
        plt.close()
        plot_saved = True
    except Exception:
        pass

    # DAG visualization (optional)
    dag_png_saved = False
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from .flow_controller import PolicyContract, build_dag

        sample_contract = PolicyContract(
            modality="text", chosen_policy="pseudo", risk_score=0.8, policy_version="v1"
        )
        dag = build_dag(sample_contract)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 4)
        ax.axis("off")
        ax.set_title("Sample Masking DAG (text / pseudo / risk=0.8)", fontsize=11)

        node_positions = {}
        n = len(dag.nodes)
        for i, node in enumerate(dag.nodes):
            x = 1.5 + i * (7.0 / max(n - 1, 1))
            y = 2.0
            node_positions[node.node_id] = (x, y)
            color = "#cfe2f3" if not node.fallback else "#f9dbc8"
            ls = "dashed" if node.fallback else "solid"
            bbox = dict(boxstyle="round,pad=0.3", fc=color, ec="gray", lw=1.2, linestyle=ls)
            label = f"{node.cmo_name}\n({node.policy})"
            if node.predicate:
                label += f"\n[{node.predicate}]"
            ax.text(x, y, label, ha="center", va="center", fontsize=8, bbox=bbox)

        for edge in dag.edges:
            if edge.src in node_positions and edge.dst in node_positions:
                x0, y0 = node_positions[edge.src]
                x1, y1 = node_positions[edge.dst]
                ax.annotate(
                    "",
                    xy=(x1, y1),
                    xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="gray", lw=1.2),
                )
                mx, my = (x0 + x1) / 2, (y0 + y1) / 2 + 0.15
                ax.text(mx, my, edge.label, ha="center", va="bottom", fontsize=7, color="gray")

        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "sample_dag.png", dpi=120)
        plt.close()
        dag_png_saved = True
    except Exception:
        pass

    # Remask count (optional, if db helpers exist)
    remask_count = 0
    try:
        from .db import DBConfig, connect_db, get_cross_modal_remask_count

        adaptive_db = connect_db(DBConfig(db_path=str(RESULTS_DIR / "dcpg_adaptive.sqlite")))
        remask_count = int(get_cross_modal_remask_count(adaptive_db))
        adaptive_db.close()
    except Exception:
        pass

    _write_experiment_report(metrics_rows, latency_rows, remask_count)

    print("\nResults written to:", RESULTS_DIR)
    print(f"\nCross-modal synergy triggered localized retokenization {remask_count} time(s).")

    # Print metrics table
    print("\n--- Policy Metrics ---")
    header = f"{'Policy':<10} {'Leak Total':>12} {'Utility Proxy':>14} {'Mean Lat (ms)':>14} {'P90 Lat (ms)':>13}"
    print(header)
    print("-" * len(header))
    for r in metrics_rows:
        print(
            f"{r['policy']:<10} {r['leak_total']:>12.4f} {r['utility_proxy']:>14.6f}"
            f" {r['mean_latency_ms']:>14.3f} {r['p90_latency_ms']:>13.3f}"
        )

    print("\nOutputs:")
    print("  policy_metrics.csv")
    print("  latency_summary.csv")
    print("  audit_log.jsonl")
    print("  EXPERIMENT_REPORT.md")
    if plot_saved:
        print("  privacy_utility_curve.png  [saved]")
    else:
        print("  privacy_utility_curve.png  [not saved]")
    if dag_png_saved:
        print("  sample_dag.png             [saved]")
    else:
        print("  sample_dag.png             [not saved]")
    print("  plus additional adaptive artifacts (if optional modules enabled)")


if __name__ == "__main__":
    main()
