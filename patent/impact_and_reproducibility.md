# Impact and Reproducibility

## Impact

The architectural shift introduced by this system has implications for how privacy is managed in longitudinal data environments.

Modern healthcare and biomedical systems increasingly operate on streaming, multi-visit, and multimodal data. In such settings, re-identification risk does not arise from isolated records but from cumulative disclosure across time. Traditional de-identification pipelines are structurally misaligned with this reality because they operate without exposure memory.

By introducing persistent entity-level exposure modeling and risk-conditioned masking escalation, this method enables:

- controlled preservation of longitudinal analytical signal
- prevention of cumulative re-identification risk
- cross-modal exposure aggregation
- adaptive escalation without uniform data destruction

This allows privacy protection to scale with exposure rather than being fixed in advance.

In streaming systems, this distinction is material. Early low-risk observations remain analytically useful. As exposure accumulates, masking strength increases automatically. This reduces unnecessary information loss while preventing reconstruction risk in later stages.

The architecture is directly applicable to:

- longitudinal clinical record systems
- multimodal health monitoring pipelines
- streaming model training environments
- federated or distributed data processing systems

The contribution lies in re-framing de-identification as a sequential control mechanism operating over entities rather than as static record filtering.

## Reproducibility

This repository provides a fully executable reference implementation.

All experiments are performed using synthetic streaming data generated within the repository. The synthetic streams are designed to emulate:

- repeated entity mentions across time
- progressive identifier disclosure
- multimodal exposure signals
- cross-modal linkage indicators

Because no real clinical data is used, independent researchers can replicate all experiments without regulatory constraints.

The demonstration includes:

- adaptive policy control
- cumulative exposure tracking
- cross-modal aggregation
- dynamic masking escalation
- leakage measurement
- utility proxy evaluation
- latency reporting
- audit logging of all decisions

Running the demo produces reproducible artifacts including:

- policy comparison metrics
- privacy–utility tradeoff visualization
- full audit logs
- exposure graph snapshots

The implementation is intentionally transparent. Masking policies, exposure accumulation, and decision logic are observable and inspectable. No external datasets or proprietary infrastructure are required.

This enables independent verification of:

- cumulative exposure behavior
- adaptive policy transitions
- privacy–utility tradeoff dynamics
- sequential decision effects

The repository is structured so that the exposure controller can be embedded into other streaming systems with minimal dependencies.

## Independent Evaluation Potential

Because the system operates on synthetic but structurally realistic streams, external researchers can:

- modify exposure thresholds
- adjust modality contributions
- plug in alternative masking operators
- evaluate downstream learning performance
- test different escalation strategies

The architecture is modular by design, enabling independent experimentation and extension without access to sensitive data.

## Transparency

All masking decisions are logged and reproducible.

Each transformation records:

- selected policy
- exposure state
- escalation triggers
- post-masking leakage
- latency metrics

This ensures that privacy decisions are inspectable rather than opaque.
