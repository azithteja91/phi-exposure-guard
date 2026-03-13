# phi-exposure-guard

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/azithteja91/phi-exposure-guard/blob/main/notebooks/demo_colab.ipynb)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18865882.svg)](https://doi.org/10.5281/zenodo.18865882)

---

Standard PHI de-identification treats each record independently: find the identifiers, remove them, move on. That works fine if you only ever see one document per patient. Clinical data streams don't look like that.

A patient's name shows up in a text note. Their voice is in an ASR transcript two events later. A face region appears in an image proxy after that. None of those alone gets you very far if you're trying to re-identify someone. But together, across a session, they do. The risk isn't in any single event, it's in the accumulation.

This repo builds a controller that tracks that accumulation and adjusts masking strength as it grows.

---

## The core idea

Instead of one masking policy applied to everything, the system picks from five tiers based on a continuously updated risk score:

```
raw -> weak -> synthetic -> pseudo -> redact
```

As a patient's cumulative exposure increases, masking escalates. If cross-modal linkage is detected (same patient appearing in both text and audio in the same event window, for example), the system applies an upward nudge to the risk score before the formal threshold is reached.

The policy decisions come from a PPO agent with an LSTM backbone, pre-trained over 200 episodes and then run live. The exposure state lives in a graph structure called the **Dynamic Contextual Privacy Graph (DCPG)**, which tracks PHI accumulation per modality, cross-modal semantic links, and recency-weighted entropy per patient.

### How the DCPG works

Each patient is a subgraph. Nodes are PHI observations per modality. Edges connect co-occurring observations within the same event, and also flag cross-modal semantic overlap above a cosine similarity of 0.30.

![DCPG graph structure](results/phi_graph_structure.png)

The risk score formula:

```
R = 0.8 * (1 - exp(-k * units)) + 0.2 * recency + link_bonus
link_bonus: +0.20 (>=2 modal links), +0.30 (>=3)
Retokenization at R >= 0.68 -> pseudonym version incremented
```

---

## Results

*Run date: March 13, 2026. 34 live events, 10 jittered replications.*

### Risk trajectory

Risk starts around 0.5 and climbs toward 0.97 by event 33 as PHI accumulates across both patients and modalities. The controller tracks this continuously rather than reacting to individual threshold crossings.

![Adaptive risk timeline](results/adaptive_risk_timeline.png)

The policy switch timeline shows exactly which tier is active at each event, with threshold crossings and consent-cap events marked:

![Policy switch timeline](results/policy_switch_annotated.png)

---

### Re-identification risk reduction

The evaluation metric is delta-AUROC: how much does masking degrade a logistic regression classifier trained to distinguish patients from masked vs. original text? Negative delta means the classifier is losing signal.

**Multi-run mean: -0.9167 +/- 0.0000 (95% CI, n=10).** Same result across all 10 runs. By the end of the stream, the masked output carries almost no re-identification signal.

![Delta AUROC annotated](results/delta_auroc_annotated.png)

![Statistical robustness](results/statistical_robustness.png)

---

### Privacy vs. utility

This is the hard part. Redact everything and clinical utility drops to zero. Apply weak masking and you get great utility but no real privacy protection at high risk. Static policies are locked into one tradeoff.

On the bursty workload (the primary evaluation scenario, where new low-risk patients enter every six events), adaptive is the only policy that hits both the 0.85 privacy floor and keeps utility above 0.50.

![Pareto frontier annotated](results/pareto_frontier_annotated.png)

| Policy | Privacy @ High Risk | Utility @ Low Risk |
|---|---|---|
| Always-Raw | 0.000 | 1.000 |
| Always-Weak | 0.004 | 0.847 |
| Always-Synthetic | 0.564 | 0.676 |
| Always-Pseudo | 0.855 | 0.440 |
| Always-Redact | 1.000 | 0.000 |
| **Adaptive** | **0.991** | **0.847** |

Green bars show where adaptive wins on privacy, blue where it wins on utility, across all three workloads:

![Adaptive vs static delta](results/adaptive_vs_static_delta.png)

A fixed policy applies the same masking strength at risk 0.2 and risk 0.9. That's the root of the problem:

![Risk conditional scores](results/risk_conditional_scores.png)

---

### What happens when an attacker stays sub-threshold

The obvious evasion strategy: keep each individual event just below the lowest risk threshold so the system never escalates. We modeled this explicitly.

The attacker spaces PHI at risk 0.34-0.39, staying below `risk_1 = 0.40`, and sends a cross-modal probe every 5th event to build linkage signal before the DCPG link bonus fires.

![Adversarial algorithm](results/adversarial_algorithm.png)

The system catches it. Cross-modal probes are detected via cosine similarity, a +0.15 risk nudge is applied, and the policy escalates to pseudo on every probe event. Always-Weak never moves.

![Adversarial workload detail](results/adversarial_workload_detail.png)

We also ran modality-imbalanced streams (85% text-only, 15% multimodal bursts) and alternating-burst cycles (4 low-risk events followed by 4 high-risk events, repeating):

![Messy workload analysis](results/messy_workload_analysis.png)

---

### Does the risk score actually track re-identification threat?

The DCPG risk score is based on exposure entropy, not directly on reconstruction probability. To check they track together, we correlated the score against a closed-form combinatorial reconstruction probability computed independently. Pearson r = 0.881 across all 34 events.

![Risk model validation](results/risk_model_validation.png)

---

### Cross-modal PHI correlation

The cross-modal link threshold sits at cosine similarity 0.30. To confirm that threshold won't produce false positives under normal conditions, we measured off-diagonal correlation between image and audio PHI signals across the run. It came out at r = 0.081, which means the two modalities are carrying largely independent information and routine events won't trip the link detector.

![Multimodal PHI correlation](results/multimodal_phi_correlation.png)

---

### Latency

Canonical multi-run latency is **17.2-17.9 ms** across pseudo, redact, and synthetic, well inside the 50 ms real-time threshold. Latency stays flat with respect to risk score, so the controller isn't getting slower as exposure accumulates.

![Latency by policy](results/latency_by_policy.png)

---

### PPO training

200 stratified pre-training episodes before live deployment. By the final epoch, pseudo and redact make up a larger share of decisions than they did early in training, which is what you'd expect from a policy converging toward risk-appropriate choices.

![PPO training stability](results/rl_training_stability.png)

Live-loop rewards settle into the 0.62-0.67 range after the first five events:

![PPO live reward](results/ppo_live_reward_curve.png)

One thing we had to fix: the original reward function used `alpha*(1-risk)` as the privacy credit term. As cumulative risk grows over a session, that term shrinks toward zero, which means the agent gets penalized just for being in a high-risk environment, regardless of whether its decisions are good. We replaced it with a risk-normalized term that stays stable as risk accumulates:

![PPO reward fix](results/ppo_reward_fix.png)

---

### Federated deployment via CRDT

The DCPG graph state can be merged across edge devices. Two devices with overlapping patient observations can merge their graphs and arrive at the same result regardless of update ordering. The plot below compares CRDT-backed risk against the SQLite-backed centralized version over the live run (k=0.012 offset keeps them visually distinct):

![CRDT vs SQLite risk](results/crdt_vs_sqlite_risk.png)

---

## Architecture

```
Event stream
    |
    v
PHI Detector  -->  DCPG (per-patient, per-modality graph)
                        |
                    Risk score
                    (entropy + recency + cross-modal bonus)
                        |
                    PPO Agent  -->  Policy decision
                        |
                    Consent layer  -->  Cap enforcement
                        |
                    Masking CMO
                        |
                    Audit log (signed)
```

Default thresholds (in `controller_config.json`):

```
risk_1   = 0.40   weak -> synthetic
risk_mid = 0.60   synthetic -> pseudo
risk_2   = 0.80   pseudo -> redact
remask   = 0.68   retokenization trigger
```

---

## Quickstart

```bash
git clone https://github.com/azithteja91/phi-exposure-guard
cd phi-exposure-guard
pip install -e .
python -m amphi_rl_dpgraph.run_demo
```

Results go to `results/`. Tests:

```bash
pytest -vv
```

Or run directly in Colab via the badge at the top.

---

## Files

| File | What it does |
|---|---|
| `dcpg.py` | Dynamic Contextual Privacy Graph: PHI accumulation, risk entropy, cross-modal links |
| `controller.py` | Risk-threshold controller, consent cap enforcement |
| `rl_agent.py` | PPO agent with LSTM policy network (128-dim hidden, 2 layers, 14-dim state) |
| `masking.py` / `masking_ops.py` | Five-tier masking CMOs |
| `dcpg_crdt.py` | CRDT merge for federated edge deployment |
| `consent.py` | Per-patient consent tier enforcement |
| `audit_signing.py` | Structured, signed audit log |
| `eval.py` | AUROC-based re-identification evaluation |
| `flow_controller.py` | Streaming event loop |

---

## Scope

This is a research implementation on synthetic data. No real patient records anywhere in the repo. It's not a drop-in compliance tool. Using it in a regulated clinical environment would require independent security review, data governance sign-off, and validation under whatever legal framework applies.

---

## Citation

```bibtex
@software{phi_exposure_guard,
  title  = {Stateful Exposure-Aware De-Identification for Multimodal Streaming Data},
  doi    = {10.5281/zenodo.18865882},
  url    = {https://doi.org/10.5281/zenodo.18865882}
}
```

See `CITATION.cff` for full details.

Patent notice: U.S. provisional patent application filed 2025-07-05. Public release: 2026-03-02.

MIT License.
