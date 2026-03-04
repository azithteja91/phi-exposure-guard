# Experiment Report

## Privacy-Utility Results

| Policy | Leak Total | Utility Proxy | Mean Latency (ms) | P90 Latency (ms) |
| --- | --- | --- | --- | --- |
| raw | 3.0256 | 1.0 | 0.127 | 0.148 |
| weak | 2.0 | 0.512195 | 0.164 | 0.228 |
| pseudo | 0.5128 | 1.0 | 0.17 | 0.207 |
| redact | 0.5128 | 0.505051 | 0.168 | 0.195 |
| adaptive | 0.5641 | 1.0 | 138.514 | 6.858 |

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
| raw | 0.127 | 0.126 | 0.148 |
| weak | 0.164 | 0.153 | 0.228 |
| pseudo | 0.17 | 0.164 | 0.207 |
| redact | 0.168 | 0.169 | 0.195 |
| adaptive | 138.514 | 2.555 | 6.858 |

## Adaptive Policy Notes

Cross-modal synergy triggered localized retokenization 2 time(s).

### Output Files
- `audit_log_signed_adaptive.jsonl` (if audit signing enabled)
- `audit_checkpoints_adaptive.jsonl` (if audit signing enabled)
- `audit_fhir_adaptive.jsonl` (if audit signing enabled)
- `dcpg_snapshot.json` (if enabled)
- `dcpg_crdt_demo.json` (if enabled)
- `rl_reward_stats.json` (if enabled)
- `sample_dag.dot` / `sample_dag.json` / `sample_dag.png`
- `privacy_utility_curve.png`
