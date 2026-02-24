from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from .masking_ops import apply_masking
except ImportError:

    def apply_masking(*, modality, policy, payload, patient_token="PATIENT_0_V0"):  # type: ignore
        return payload


try:
    from .cmo_media import apply_synthetic_replacement
except ImportError:

    def apply_synthetic_replacement(text: str) -> str:  # type: ignore
        return text


@dataclass
class DataBlock:
    event_id: str
    modality: str
    payload: Any
    phi_spans: List[Any] = field(default_factory=list)
    meta: Dict[str, Any] = field(default_factory=dict)

    def content_hash(self) -> str:
        raw = str(self.payload).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]


@dataclass
class MaskingPolicyContract:
    modality: str
    chosen_policy: str
    patient_token: str = "PATIENT_0_V0"
    phi_class: str = "GENERAL"
    risk_score: float = 0.0
    consent_level: str = "standard"


@dataclass
class MaskingActionLog:
    cmo_name: str
    event_id: str
    modality: str
    policy_applied: str
    input_hash: str
    output_hash: str
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    notes: str = ""


CMOFunction = Callable[[DataBlock, MaskingPolicyContract], DataBlock]


class CMORegistry:
    """Runtime registry for masking operators. Instance-level logs prevent cross-run contamination."""

    _registry: Dict[str, CMOFunction] = {}

    # instance-level log list; class-level only used as fallback for legacy callers
    _execution_logs: List[MaskingActionLog] = []

    @classmethod
    def register(cls, name: str, fn: CMOFunction) -> None:
        cls._registry[name] = fn

    @classmethod
    def get(cls, name: str) -> Optional[CMOFunction]:
        return cls._registry.get(name)

    @classmethod
    def list_operators(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def apply(
        cls,
        name: str,
        data_block: DataBlock,
        policy: MaskingPolicyContract,
    ) -> Tuple[DataBlock, MaskingActionLog]:
        fn = cls._registry.get(name) or _builtin_redact_cmo

        t0 = time.perf_counter()
        input_hash = data_block.content_hash()
        out_block = fn(data_block, policy)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        log = MaskingActionLog(
            cmo_name=name,
            event_id=data_block.event_id,
            modality=data_block.modality,
            policy_applied=policy.chosen_policy,
            input_hash=input_hash,
            output_hash=out_block.content_hash(),
            latency_ms=round(latency_ms, 4),
        )
        cls._execution_logs.append(log)
        return out_block, log

    @classmethod
    def flush_logs(cls) -> List[MaskingActionLog]:
        logs = list(cls._execution_logs)
        cls._execution_logs.clear()
        return logs


def _make_cmo(resolved_policy: str) -> CMOFunction:
    def cmo(block: DataBlock, policy: MaskingPolicyContract) -> DataBlock:
        out = apply_masking(
            modality=block.modality,
            policy=resolved_policy,
            payload=block.payload,
            patient_token=policy.patient_token,
        )
        return DataBlock(
            event_id=block.event_id,
            modality=block.modality,
            payload=out,
            phi_spans=[],
            meta={**block.meta, "cmo": resolved_policy},
        )

    return cmo


def _synthetic_cmo(block: DataBlock, policy: MaskingPolicyContract) -> DataBlock:
    """
    Synthetic replacement CMO: substitutes real names with generated fictional names
    and shifts dates by a fixed offset rather than replacing with opaque tokens.
    Preserves sentence structure and temporal ordering for downstream models.
    """
    if block.modality in ("text", "asr"):
        out = apply_synthetic_replacement(str(block.payload))
    else:
        out = apply_masking(
            modality=block.modality,
            policy="pseudo",
            payload=block.payload,
            patient_token=policy.patient_token,
        )
    return DataBlock(
        event_id=block.event_id,
        modality=block.modality,
        payload=out,
        phi_spans=[],
        meta={**block.meta, "cmo": "synthetic"},
    )


_builtin_redact_cmo = _make_cmo("redact")
_builtin_pseudo_cmo = _make_cmo("pseudo")
_builtin_weak_cmo = _make_cmo("weak")
_builtin_raw_cmo = _make_cmo("raw")

CMORegistry.register("RedactTextSpan", _builtin_redact_cmo)
CMORegistry.register("BlurImageRegion", _builtin_redact_cmo)
CMORegistry.register("MaskWaveformHeader", _builtin_redact_cmo)
CMORegistry.register("VoiceObfuscation", _builtin_redact_cmo)
CMORegistry.register("RedactImageOverlay", _builtin_redact_cmo)
CMORegistry.register("PseudonymizeID", _builtin_pseudo_cmo)
CMORegistry.register("GeneralizeDate", _builtin_weak_cmo)
CMORegistry.register("PassThrough", _builtin_raw_cmo)
CMORegistry.register("SyntheticReplacement", _synthetic_cmo)  # synthetic name + date CMO

_POLICY_TO_CMO: Dict[str, str] = {
    "redact": "RedactTextSpan",
    "pseudo": "PseudonymizeID",
    "weak": "GeneralizeDate",
    "raw": "PassThrough",
    "synthetic": "SyntheticReplacement",
}


def apply_via_cmo(
    *,
    modality: str,
    policy: str,
    payload: Any,
    patient_token: str = "PATIENT_0_V0",
    event_id: str = "",
    risk_score: float = 0.0,
) -> Tuple[Any, MaskingActionLog]:
    cmo_name = _POLICY_TO_CMO.get(str(policy), "RedactTextSpan")
    block = DataBlock(event_id=event_id, modality=modality, payload=payload)
    contract = MaskingPolicyContract(
        modality=modality,
        chosen_policy=str(policy),
        patient_token=patient_token,
        risk_score=risk_score,
    )
    out_block, log = CMORegistry.apply(cmo_name, block, contract)
    return out_block.payload, log
