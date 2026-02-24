# amphi_rl_dpgraph/schemas.py

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass(frozen=True)
class PHISpan:
    start: int
    end: int
    phi_type: str
    confidence: float = 1.0

@dataclass
class DataEvent:
    event_id: str
    patient_key: str
    timestamp: float
    modality: str  # "text", "asr", "image_proxy", "waveform_proxy", "audio_proxy"
    payload: Any
    phi_spans: List[PHISpan] = field(default_factory=list)
    phi_units: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class DecisionRecord:
    event_id: str
    patient_key: str
    policy_run: str
    chosen_policy: str
    reason: str

    risk_pre: float
    risk_post: Optional[float]
    risk_source: str

    provisional_risk: float = 0.0

    localized_remask: Dict[str, Any] = field(default_factory=dict)
    thresholds: Dict[str, float] = field(default_factory=dict)
    candidates: List[str] = field(default_factory=list)

    cross_modal_matches: List[str] = field(default_factory=list)

    decision_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

@dataclass
class AuditRecord:
    event_id: str
    patient_key: str
    modality: str

    policy_run: str
    chosen_policy: str
    reason: str

    risk: float
    localized_remask_trigger: bool

    latency_ms: float
    leaks_after: float

    policy_version: str = "v1"

    extra: Dict[str, Any] = field(default_factory=dict)
    decision_blob: Dict[str, Any] = field(default_factory=dict)
