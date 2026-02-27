# Experiment Report

## Privacy-Utility Results

| Policy | Leak Total | Utility Proxy | Mean Latency (ms) | P90 Latency (ms) |
| --- | --- | --- | --- | --- |
| raw | 3.0256 | 1.0 | 0.123 | 0.154 |
| weak | 2.0 | 0.512195 | 0.141 | 0.18 |
| pseudo | 0.5128 | 1.0 | 0.159 | 0.188 |
| redact | 0.5128 | 0.505051 | 0.157 | 0.192 |
| adaptive | 0.5641 | 1.0 | 1.16 | 1.237 |

## Leakage Breakdown

| Policy | Text | ASR | Image | Waveform | Audio |
| --- | --- | --- | --- | --- | --- |
| raw | 0.5128 | 0.5128 | 0.7436 | 0.5128 | 0.7436 |
| weak | 0.5128 | 0.0 | 0.7436 | 0.0 | 0.7436 |
| pseudo | 0.5128 | 0.0 | 0.0 | 0.0 | 0.0 |
| redact | 0.5128 | 0.0 | 0.0 | 0.0 | 0.0 |
| adaptive | 0.359 | 0.0 | 0.1026 | 0.0 | 0.1026 |

## Latency Summary

| Policy | Mean (ms) | P50 (ms) | P90 (ms) |
| --- | --- | --- | --- |
| raw | 0.123 | 0.12 | 0.154 |
| weak | 0.141 | 0.142 | 0.18 |
| pseudo | 0.159 | 0.14 | 0.188 |
| redact | 0.157 | 0.156 | 0.192 |
| adaptive | 1.16 | 1.027 | 1.237 |

## Adaptive Policy Notes

Cross-modal synergy triggered localized retokenization 2 time(s).

### Output Files
- `audit_log_signed_adaptive.jsonl` (if audit signing enabled)
- `audit_checkpoints_adaptive.jsonl` (if audit signing enabled)
- `audit_fhir_adaptive.jsonl` (if audit signing enabled)
- `dcpg_snapshot.json` (if enabled)
- `dcpg_crdt_demo.json` (if enabled)
- `rl_reward_stats.json` (if enabled)
- `sample_dag.dot` / `sample_dag.json`
- `privacy_utility_curve.png` (if matplotlib installed)
