# Stateful Exposure-Aware De-Identification for Multimodal Streaming Data

Research implementation of a stateful, risk-aware de-identification architecture for streaming multimodal systems.

This project demonstrates an alternative to static, document-level anonymization. Instead of treating privacy protection as a one-time preprocessing step, the system models cumulative identity exposure over time and dynamically adjusts masking strength in response to quantified re-identification risk.

## Overview

Most de-identification pipelines operate per document:

detect PHI -> remove PHI -> store result

This approach assumes that risk is isolated within individual records. In practice, re-identification risk accumulates across events, modalities, and time.

A name fragment, identifier token, or cross-modal linkage that appears harmless in isolation may become identifying when repeated or combined with other signals.

This repository implements a stateful exposure-aware controller that:

-  Maintains subject-level exposure state
-  Computes rolling re-identification risk
-  Incorporates recency and cross-modal linkage signals
-  Dynamically selects masking strength
-  Supports pseudonym versioning upon risk escalation
-  Produces structured, reproducible audit logs

De-identification becomes a longitudinal control problem rather than a static transformation.

## Architectural Characteristics

The system differs from conventional masking pipelines in several concrete ways:

**Longitudinal Exposure Tracking:**
Identity exposure is accumulated and tracked over time at the subject level.

**Risk-Governed Policy Selection:**
Masking strength is selected dynamically based on quantified risk thresholds.

**Cross-Modal Linkage Modeling:**
Signals from text, ASR transcripts, image proxies, waveform headers, and audio metadata are aggregated to evaluate identity-level exposure.

**Localized Retokenization**
When risk increases, pseudonym tokens can be versioned forward, containing linkage continuity without global reprocessing.

**Auditability:**
All masking decisions are logged with structured metadata and can be reproduced deterministically from exposure state.

## Demonstration

The repository includes a fully synthetic streaming simulation.

Five policies are evaluated:

- raw
- weak
- pseudo
- redact
- adaptive

The adaptive controller escalates masking strength only when cumulative exposure justifies it.

Outputs include:

- `policy_metrics.csv`
- `latency_summary.csv`
- `audit_log.jsonl`
- `EXPERIMENT_REPORT.md`
- `privacy_utility_curve.png`
- sample_dag.png

All experiments are reproducible from source using synthetic data generated within the repository.

Run:

```
python src/amphi_rl_dpgraph/run_demo.py
```

Results are written to the `results/` directory.

## Data Description

This repository does not contain real clinical data, personal information, or protected health information.

All experiments operate on synthetically generated streams designed to simulate longitudinal healthcare data workflows. The synthetic data includes structured representations of:

- Clinical note text
- Speech transcription output
- Image proxy signals
- Waveform and monitoring features

The streams are constructed to model realistic structural properties relevant to privacy evaluation, including:

- Repeated subject mentions over time
- Identifier recurrence
- Variable disclosure frequency
- Cross-modal co-occurrence patterns

These properties allow controlled evaluation of cumulative identity exposure and adaptive masking behavior without exposing real individuals.

Synthetic data is used to ensure reproducibility, transparency, and safe public distribution of the research implementation.

## Privacy–Utility Evaluation

The demo evaluates:

- Residual PHI leakage
- Utility proxy metrics
- Latency distribution
- Adaptive escalation behavior

The objective is not to eliminate utility through maximal redaction, but to demonstrate controlled escalation based on exposure accumulation.

## Intended Use

This repository is intended for:

- Research in privacy-preserving machine learning
- Streaming system design
- Exposure-aware masking strategies
- Longitudinal risk modeling
- Reproducible evaluation of privacy–utility tradeoffs

It is not a production-ready compliance system.

# Security and Data Safety Policy

## Data Restrictions

This repository must not contain real patient data, protected health information (PHI), or identifiable personal data.

All demonstrations and experiments run exclusively on synthetic data generated within the repository or on publicly permitted datasets.

The following must never be uploaded:

- Clinical notes derived from real individuals
- Hospital records or EHR exports
- Medical images associated with identifiable persons
- Audio recordings of patients
- Any dataset containing direct or indirect identifiers

If sensitive data is discovered, do not open a public issue. Contact the maintainer directly for immediate removal.

# Ethical Scope and Research Boundaries

This project studies adaptive privacy control mechanisms for streaming and multimodal systems.

It does not collect, process, or distribute real clinical data.

The methods demonstrated here are intended to strengthen privacy protection. They are not designed to weaken safeguards or enable re-identification.

When adapting this code to real-world systems, implementers must ensure:

- Institutional and regulatory compliance
- Independent security controls
- Data governance review
- Validation under applicable legal frameworks

Privacy protection in regulated domains requires layered safeguards. This repository addresses one technical layer: exposure-aware masking.

It should not be treated as a substitute for comprehensive compliance infrastructure.

## Citation

If you use this software in academic or technical work, please cite it via the included `CITATION.cff` file.

Title:

Stateful Exposure-Aware De-Identification for Multimodal Streaming Data

## License

MIT License. See `LICENSE`.
