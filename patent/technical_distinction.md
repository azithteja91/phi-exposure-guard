# Technical Distinction

This architecture introduces a fundamentally different model for de-identification in longitudinal and multimodal systems.

Conventional systems treat privacy as a static filtering task applied to isolated records. The present design models privacy as a dynamic, cumulative property of entities across time.

This distinction changes the operational category of de-identification from document preprocessing to stateful risk control.

## 1. Limitation of Record-Level Masking Pipelines

Most deployed de-identification systems operate as stateless processors:

* detect identifiers within a single record
* apply predefined masking rules
* emit a transformed record

The decision boundary is confined to the current input. Once a record is processed, its contribution to future disclosure risk is not retained.

### Structural Failure

In longitudinal datasets, re-identification risk emerges through accumulation. Individually benign disclosures combine across time to reconstruct identity. Stateless pipelines are structurally incapable of preventing this because they do not model cumulative exposure.

This is not a tuning issue. It is a memory limitation inherent to the architecture.

## 2. Limitation of Stable Pseudonymization

Stable pseudonymization preserves entity continuity by replacing identifiers with consistent tokens.

While this supports longitudinal modeling, it introduces a persistent linkage handle. As additional attributes accumulate, the pseudonym becomes a surrogate identity.

### Structural Failure

Because exposure accumulation is not quantified, pseudonym continuity is maintained indefinitely, even after risk surpasses acceptable thresholds. The system cannot determine when continuity itself becomes unsafe.

This creates delayed re-identification risk that is invisible to the masking mechanism.

## 3. Limitation of Uniform Redaction

Full redaction enforces maximal masking across all records regardless of context.

### Structural Failure

Uniform redaction treats all records as high risk, eliminating longitudinal learning signal even when exposure is minimal. This approach avoids cumulative modeling by discarding continuity entirely.

The result is privacy preservation through utility collapse.

## 4. Entity-Level Longitudinal Exposure Modeling

The architecture implemented here introduces persistent exposure state at the entity level.

Each identity-related signal contributes to a cumulative exposure representation. This representation evolves across time and modalities and directly governs masking strength.

### Structural Advancement

Privacy is no longer defined as a property of a document. It is defined as a function of historical disclosure.

This enables the system to reason about how much identifying evidence has already been revealed, not merely what appears in the current record.

## 5. Risk-Conditioned Adaptive Escalation

Masking policy is selected based on exposure state.

The same record may receive:

- light transformation when exposure is low
- pseudonymization when exposure grows
- full redaction when exposure exceeds threshold

### Structural Advancement

Masking strength becomes state-dependent and sequential.

This transforms de-identification into a controlled escalation process rather than a fixed transformation rule.

## 6. Cross-Modal Aggregation of Identity Signals

Identity risk does not arise from text alone. It emerges from combined signals across notes, transcripts, images, and structured observations.

The architecture aggregates exposure contributions across modalities into a unified entity state.

### Structural Advancement

Re-identification risk is modeled holistically rather than in isolated modality silos.

This prevents cross-modal reconstruction that would evade single-channel systems.

## 7. Closed-Loop Sequential Control

Each masking action affects future exposure trajectories.

Post-masking state updates alter subsequent policy decisions, forming a closed feedback loop.

### Structural Advancement

De-identification becomes a sequential decision process operating over streams, not a stateless preprocessing stage.

This reclassifies anonymization as a dynamic control problem.

## Summary of Significance

The advancement is architectural, not incremental.

Conventional systems:

- operate per record
- apply static rules
- lack exposure memory
- cannot regulate cumulative disclosure

This architecture:

- maintains persistent entity-level exposure state
- aggregates disclosure across time and modalities
- escalates masking strength based on measured risk
- enforces privacy without abandoning longitudinal continuity

By shifting from document-level filtering to entity-level exposure control, the system addresses a structural failure mode in existing de-identification approaches.

The result is a mechanism capable of preserving analytical continuity when safe and enforcing strong privacy when necessary, within the same continuous stream.
