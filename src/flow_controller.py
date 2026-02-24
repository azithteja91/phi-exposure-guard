from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional

from .masking_ops import apply_masking

Modality       = Literal["text", "asr", "image_proxy", "waveform_proxy", "audio_proxy"]
ResolvedPolicy = Literal["raw", "weak", "pseudo", "redact", "synthetic"]

@dataclass(frozen=True)
class PolicyContract:
    modality: Modality
    chosen_policy: ResolvedPolicy
    patient_token: str = "PATIENT_0_V0"
    risk_score: float = 0.0
    consent_level: str = "standard"
    policy_version: str = "v1"  # cache key

@dataclass
class DAGNode:
    node_id: str
    cmo_name: str
    modality: str
    policy: str
    predicate: Optional[str] = None
    fallback: bool = False

@dataclass
class DAGEdge:
    src: str
    dst: str
    label: str = ""

@dataclass
class MaskingDAG:
    contract_hash: str
    nodes: List[DAGNode] = field(default_factory=list)
    edges: List[DAGEdge] = field(default_factory=list)

    def to_dot(self) -> str:
        lines = [f'digraph masking_{self.contract_hash[:8]} {{', '  rankdir=LR;']
        for n in self.nodes:
            label  = f"{n.cmo_name}\\n({n.policy})"
            shape  = "diamond" if n.predicate else "box"
            style  = 'style=dashed,' if n.fallback else ''
            lines.append(f'  {n.node_id} [label="{label}" shape={shape} {style}];')
        for e in self.edges:
            lbl = f' [label="{e.label}"]' if e.label else ""
            lines.append(f'  {e.src} -> {e.dst}{lbl};')
        lines.append('}')
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            {
                "contract_hash": self.contract_hash,
                "nodes": [
                    {
                        "id": n.node_id,
                        "cmo": n.cmo_name,
                        "modality": n.modality,
                        "policy": n.policy,
                        "predicate": n.predicate,
                        "fallback": n.fallback,
                    }
                    for n in self.nodes
                ],
                "edges": [{"src": e.src, "dst": e.dst, "label": e.label} for e in self.edges],
            },
            indent=2,
        )

_POLICY_CMO: Dict[str, Dict[str, str]] = {
    "text": {
        "raw": "PassThrough", "weak": "GeneralizeDate",
        "pseudo": "PseudonymizeID", "redact": "RedactTextSpan",
        "synthetic": "SyntheticReplacement",
    },
    "asr": {
        "raw": "PassThrough", "weak": "GeneralizeDate",
        "pseudo": "PseudonymizeID", "redact": "RedactTextSpan",
        "synthetic": "SyntheticReplacement",
    },
    "image_proxy": {
        "raw": "PassThrough", "weak": "BlurImageRegion",
        "pseudo": "BlurImageRegion", "redact": "RedactImageOverlay",
        "synthetic": "BlurImageRegion",   # same as pseudo
    },
    "waveform_proxy": {
        "raw": "PassThrough", "weak": "MaskWaveformHeader",
        "pseudo": "MaskWaveformHeader", "redact": "MaskWaveformHeader",
        "synthetic": "MaskWaveformHeader",
    },
    "audio_proxy": {
        "raw": "PassThrough", "weak": "VoiceObfuscation",
        "pseudo": "VoiceObfuscation", "redact": "VoiceObfuscation",
        "synthetic": "VoiceObfuscation",
    },
}

_CONSENT_POLICY_CAP: Dict[str, str] = {
    "minimal":  "raw",     # opt out
    "research": "pseudo",  # pseudo ok
    "standard": "redact",  # full ok
    "full":     "redact",
}

_POLICY_RANK = {"raw": 0, "weak": 1, "pseudo": 2, "synthetic": 2, "redact": 3}

def _apply_consent_cap(policy: str, consent_level: str) -> str:
    """Consent cap."""
    cap = _CONSENT_POLICY_CAP.get(consent_level, "redact")
    if _POLICY_RANK.get(policy, 0) > _POLICY_RANK.get(cap, 3):
        return cap
    return policy

def _contract_hash(contract: PolicyContract) -> str:
    import hashlib
    key = (
        f"{contract.modality}:{contract.chosen_policy}:{contract.consent_level}"
        f":{contract.risk_score:.3f}:{contract.policy_version}"
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]

_DAG_CACHE: Dict[str, MaskingDAG] = {}
_DAG_CACHE_MAX = 128

def build_dag(contract: PolicyContract) -> MaskingDAG:
    """
    Compile DAG.
    """
    ch = _contract_hash(contract)
    if ch in _DAG_CACHE:
        return _DAG_CACHE[ch]

    effective_policy = _apply_consent_cap(contract.chosen_policy, contract.consent_level)
    cmo = _POLICY_CMO.get(contract.modality, {}).get(effective_policy, "RedactTextSpan")

    dag = MaskingDAG(contract_hash=ch)

    if contract.risk_score > 0.7 and contract.modality == "image_proxy":
        branch_node = DAGNode(
            node_id="risk_gate", cmo_name="RiskGate",
            modality=contract.modality, policy=effective_policy,
            predicate="risk_score > 0.7",
        )
        hi_node = DAGNode(
            node_id="redact_overlay", cmo_name="RedactImageOverlay",
            modality=contract.modality, policy="redact",
        )
        lo_node = DAGNode(
            node_id="blur_region", cmo_name="BlurImageRegion",
            modality=contract.modality, policy="pseudo",
        )
        fallback_node = DAGNode(
            node_id="fallback_redact", cmo_name="RedactTextSpan",
            modality=contract.modality, policy="redact", fallback=True,
        )
        dag.nodes = [branch_node, hi_node, lo_node, fallback_node]
        dag.edges = [
            DAGEdge("risk_gate", "redact_overlay", "risk > 0.7"),
            DAGEdge("risk_gate", "blur_region", "risk <= 0.7"),
            DAGEdge("redact_overlay", "fallback_redact", "on_error"),
            DAGEdge("blur_region", "fallback_redact", "on_error"),
        ]
    else:
        main_node = DAGNode(
            node_id="main_cmo", cmo_name=cmo,
            modality=contract.modality, policy=effective_policy,
        )
        fallback_node = DAGNode(
            node_id="fallback_redact", cmo_name="RedactTextSpan",
            modality=contract.modality, policy="redact", fallback=True,
        )
        dag.nodes = [main_node, fallback_node]
        dag.edges = [DAGEdge("main_cmo", "fallback_redact", "on_error")]

    if len(_DAG_CACHE) >= _DAG_CACHE_MAX:
        oldest = next(iter(_DAG_CACHE))
        del _DAG_CACHE[oldest]
    _DAG_CACHE[ch] = dag
    return dag

def build_pipeline(contract: PolicyContract) -> Callable[[Any, str], Any]:
    """Build pipeline."""
    dag = build_dag(contract)
    effective_policy = _apply_consent_cap(contract.chosen_policy, contract.consent_level)

    def pipeline(payload: Any, patient_token: str = "PATIENT_0_V0") -> Any:
        try:
            return apply_masking(
                modality=contract.modality,
                policy=effective_policy,
                payload=payload,
                patient_token=patient_token,
            )
        except Exception:
            return apply_masking(
                modality=contract.modality,
                policy="redact",
                payload=payload,
                patient_token=patient_token,
            )

    pipeline._dag = dag  # type: ignore
    return pipeline

def export_dag(contract: PolicyContract, fmt: str = "dot") -> str:
    dag = build_dag(contract)
    return dag.to_json() if fmt == "json" else dag.to_dot()
